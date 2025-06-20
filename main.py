import csv
import os
import pathlib
from os import PathLike
from typing import Optional, Union

import torch
import torch.nn.functional as F
import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from sklearn import metrics

from model.model import EmbeddingModel, EmbeddingDataset


class TestDataset(Dataset):
    def __init__(self, csv_file: str | PathLike[str], tokenizer, sentiments: list[str], max_length: int = 512):
        super().__init__()
        self.sentiments = sentiments
        self.__tokenizer = tokenizer
        self.__max_length = max_length

        with open(csv_file, 'rt') as fd:
            reader = csv.reader(fd)
            self.__rows = list(reader)

    def __len__(self):
        return len(self.__rows)

    def __getitem__(self, item):
        row = self.__rows[item]
        tokens = self.__tokenizer(
            row[-2],
            max_length=self.__max_length,
            return_tensors='pt',
            padding='max_length',
            truncation=True
        )
        return {
            'input_ids': tokens['input_ids'].reshape((self.__max_length, )),
            'attention_mask': tokens['attention_mask'].reshape((self.__max_length, )),
            'sentiment_idx': self.sentiments.index(row[-1])
        }


class EmbeddingTrainer(Trainer):
    def evaluate(
            self,
            eval_dataset: Optional[Union[Dataset, dict[str, Dataset]]] = None,
            ignore_keys: Optional[list[str]] = None,
            metric_key_prefix: str = "eval",
    ) -> dict[str, float]:
        dl = DataLoader(eval_dataset or self.eval_dataset, batch_size=self.args.eval_batch_size,
                        num_workers=self.args.dataloader_num_workers)
        labels_true = []
        labels_pred = []
        for batch in tqdm.tqdm(iter(dl)):
            labels_true += batch['sentiment_idx'].tolist()
            outputs = self.model(input_ids=batch['input_ids'].to(self.accelerator.device),
                                 attention_mask=batch['attention_mask'].to(self.accelerator.device))
            labels_pred += torch.argmax(outputs, dim=1).tolist()

        met = {
            'accuracy': metrics.accuracy_score(labels_true, labels_pred),
            'f1': metrics.f1_score(labels_true, labels_pred, average='weighted')
        }
        self.log(met)
        return met

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
    sentiments_cn = ['积极', '消极', '中性']

    # Add padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Initialize model
    model = EmbeddingModel(model_name, len(sentiments))

    # Dataset
    train_dataset = EmbeddingDataset("dataset/train.json", tokenizer, sentiments)
    eval_dataset = TestDataset("dataset/test.csv", tokenizer, sentiments_cn)

    # Training arguments

    # Initialize trainer
    training_args = TrainingArguments(
        output_dir="output",
        num_train_epochs=3,
        per_device_train_batch_size=10,
        gradient_accumulation_steps=10,
        gradient_checkpointing=False,
        eval_strategy="steps",
        eval_steps=20,
        save_steps=20,
        logging_steps=5,
        save_total_limit=5,
        learning_rate=5e-5,
        optim='schedule_free_adamw',
        lr_scheduler_type='constant',
        warmup_ratio=0.05,
        bf16=True,
        dataloader_num_workers=4,
        dataloader_drop_last=True,
        remove_unused_columns=False,
        report_to=None,
    )
    trainer = EmbeddingTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # Train
    output_path = pathlib.Path('output')
    trainer.train(resume_from_checkpoint=output_path.exists() and any(
        name for name in os.listdir(output_path) if name.startswith('checkpoint')))

    # Save the final model
    trainer.model.save('model/addition.pt')


if __name__ == "__main__":
    main()
