import os
import pathlib

import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)

from model import EmbeddingModel, EmbeddingDataset


class EmbeddingTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, *args, **kwargs):
        # Extract query and positive inputs
        y_pred = model(
            input_ids=inputs['query_input_ids'],
            attention_mask=inputs['query_attention_mask']
        )

        # cross-entropy loss
        y = inputs['response_sentiment_idx']
        loss = F.cross_entropy(y_pred, y)

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
    model = EmbeddingModel(model_name, len(sentiments))

    # Dataset
    train_dataset = EmbeddingDataset("dataset/train.json", tokenizer, sentiments)
    # eval_dataset = EmbeddingDataset("dataset/test.json", tokenizer, sentiments)

    # Training arguments
    training_args = TrainingArguments(
        output_dir="output",
        num_train_epochs=1,
        per_device_train_batch_size=5,
        gradient_checkpointing=False,
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
    trainer.model.save('output/addition.pt')
    tokenizer.save_pretrained("output")


if __name__ == "__main__":
    main()
