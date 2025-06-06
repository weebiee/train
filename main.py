import json
import os
import pathlib
from typing import Union, Any, Optional
from functools import cache, cached_property

import torch
import torch.nn.functional as F
from peft import LoraConfig, TaskType
from torch import nn
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    AutoModel,
    BitsAndBytesConfig
)
from tokenizers import Tokenizer


class EmbeddingDataset(Dataset):
    def __init__(self, data_path: str, tokenizer, sentiments: list[str], max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length

        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        self.queries = []
        self.responses = []
        self.sentiments = sentiments

        for item in self.data:
            self.queries.append(item['query'])
            self.responses.append(item['response'])

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        query = self.queries[idx]
        response_idx = self.sentiments.index(self.responses[idx])

        # Tokenize query and positive
        query_encoded = self.tokenizer(
            query,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'query_input_ids': query_encoded['input_ids'].squeeze(),
            'query_attention_mask': query_encoded['attention_mask'].squeeze(),
            'response_sentiment_idx': response_idx
        }


class EmbeddingModel(torch.nn.Module):
    def __init__(self, model_name: str, tokenizer: Tokenizer, sentiments: list[str], max_length: int = 512):
        super().__init__()
        self.tokenizer = tokenizer
        self.sentiments = sentiments
        self.max_length = max_length

        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            quantization_config=BitsAndBytesConfig(load_in_8bit=True)
        )
        self.device = self.model.device

        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            inference_mode=False,
            r=8,  # Rank
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
        )
        self.model.add_adapter(lora_config, adapter_name="weebiee")

    def forward(self, input_ids, attention_mask, *args):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, *args)
        embeddings = outputs.last_hidden_state
        attention_mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
        sum_embeddings = torch.sum(embeddings * attention_mask_expanded, 1)
        sum_mask = torch.clamp(attention_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask


class EmbeddingTrainer(Trainer):
    @cached_property
    def sentiment_tokens(self):
        return self.model.tokenizer(self.model.sentiments, return_tensors='pt', padding='max_length',
                                    max_length=self.model.max_length).to(device=self.model.device)

    def prediction_step(
            self,
            model: nn.Module,
            inputs: dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[list[str]] = None,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        has_labels = all(key in inputs for key in ['query_input_ids', 'response_input_ids'])

        inputs = self._prepare_inputs(inputs)

        if ignore_keys is None:
            ignore_keys = []

        with torch.no_grad():
            with self.compute_loss_context_manager():
                query_embeddings = model(
                    input_ids=inputs['query_input_ids'],
                    attention_mask=inputs['query_attention_mask']
                )

                response_idx = inputs['response_sentiment_idx']
                response_embeddings = model(
                    input_ids=self.sentiment_tokens['input_ids'][response_idx],
                    attention_mask=self.sentiment_tokens['attention_mask'][response_idx]
                )

                # Normalize embeddings for cosine similarity
                query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
                response_embeddings = F.normalize(response_embeddings, p=2, dim=1)

                # Compute similarity scores (these serve as "logits")
                similarities = F.cosine_similarity(query_embeddings, response_embeddings, dim=1)

                # Compute loss if we have labels
                loss = None
                if has_labels:
                    # Use margin-based loss for better separation
                    loss = torch.clamp(1.0 - similarities, min=0.0).mean()

                # Create labels (all ones since these are positive pairs)
                labels = torch.ones_like(similarities)

        if prediction_loss_only:
            return loss, None, None

        # Return similarity scores as logits
        logits = similarities.unsqueeze(-1)  # Add dimension for consistency

        return loss, logits, labels

    def compute_loss(self, model, inputs, num_items_in_batch, return_outputs=False):
        # Extract query and positive inputs
        query_embeddings = model(
            input_ids=inputs['query_input_ids'],
            attention_mask=inputs['query_attention_mask']
        )

        label_idx = inputs['response_sentiment_idx']
        response_embeddings = model(
            input_ids=self.sentiment_tokens['input_ids'],
            attention_mask=self.sentiment_tokens['attention_mask']
        )

        # Normalize embeddings
        query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
        response_embeddings = F.normalize(response_embeddings, p=2, dim=1)

        # Cosine similarity loss
        similarities = F.cosine_similarity(query_embeddings, response_embeddings, dim=1)
        loss = 1 - similarities.mean()  # Convert similarity to loss

        return (loss, {"loss": loss}) if return_outputs else loss


def main():
    # Model and tokenizer
    model_name = "Alibaba-NLP/gte-Qwen2-1.5B-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    sentiments = ['positive', 'negative', 'neutral']

    # Add padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Initialize model
    model = EmbeddingModel(model_name, tokenizer, sentiments)

    # Dataset
    train_dataset = EmbeddingDataset("dataset/train.json", tokenizer, sentiments)
    # eval_dataset = EmbeddingDataset("dataset/test.json", tokenizer, sentiments)

    # Training arguments
    training_args = TrainingArguments(
        output_dir="output",
        num_train_epochs=1,
        per_device_train_batch_size=1,
        # eval_strategy="steps",
        # eval_steps=50,
        save_steps=50,
        save_total_limit=5,
        logging_steps=50,
        warmup_ratio=0.05,
        learning_rate=5e-6,
        bf16=True,
        dataloader_num_workers=4,
        dataloader_drop_last=True,
        remove_unused_columns=False,
        report_to=None,
    )

    # Initialize trainer
    trainer = EmbeddingTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        # eval_dataset=eval_dataset,
    )

    # Train
    output_path = pathlib.Path('output')
    trainer.train(resume_from_checkpoint=output_path.exists() and any(
        name for name in os.listdir(output_path) if name.startswith('checkpoint')))

    # Save the final model
    trainer.model.model.save_pretrained('weebiee')
    tokenizer.save_pretrained("output")


if __name__ == "__main__":
    main()
