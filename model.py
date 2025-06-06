from tokenizers import Tokenizer
from torch import nn
from torch.utils.data import Dataset
import torch
import json
from os import PathLike

from transformers import AutoModel, BitsAndBytesConfig


class EmbeddingDataset(Dataset):
    def __init__(self, data_path: str, tokenizer, sentiments: list[str], max_length: int = 512,
                 indicates: slice | None = None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.indicates = indicates

        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
            if self.indicates:
                self.data = self.data[indicates]

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


class PoolerModel(torch.nn.Module):
    def __init__(self, model_name: str, max_length: int = 512):
        super().__init__()

        self.max_length = max_length
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            quantization_config=BitsAndBytesConfig(load_in_8bit=True)
        )
        self.device = self.model.device

    def forward(self, input_ids, attention_mask, *args):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, *args)
        embeddings = outputs.last_hidden_state
        attention_mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
        sum_embeddings = torch.sum(embeddings * attention_mask_expanded, 1)
        sum_mask = torch.clamp(attention_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def gradient_checkpointing_enable(self, *args, **kwargs):
        self.model.gradient_checkpointing_enable(*args, **kwargs)


class EmbeddingModel(torch.nn.Module):
    def __init__(self, model_name: str, sentiments: int, max_length: int = 512,
                 addition_path: str | PathLike[str] | None = None):
        super().__init__()
        self.sentiments = sentiments
        self.max_length = max_length

        self.base_model = PoolerModel(model_name, max_length)

        for para in self.base_model.parameters():
            para.requires_grad = False

        self.device = self.base_model.device
        embedding_space = 1536
        self.addition = nn.Sequential(
            nn.Linear(embedding_space, embedding_space),
            nn.ReLU(),
            nn.Linear(embedding_space, embedding_space),
            nn.ReLU(),
            nn.Linear(embedding_space, sentiments)
        ).to(device=self.device)
        if addition_path:
            self.addition.load_state_dict(
                torch.load(addition_path, map_location=self.device, weights_only=True)
            )

    def load(self, file_name: str | PathLike[str]):
        self.addition = torch.load(file_name)

    def forward(self, input_ids, attention_mask, *args, **kwargs):
        embedding_output = self.base_model(input_ids, attention_mask, *args, **kwargs)
        return self.addition(embedding_output)

    def gradient_checkpointing_enable(self, *args, **kwargs):
        self.base_model.gradient_checkpointing_enable(*args, **kwargs)

    def save(self, file_path: str | PathLike[str]):
        torch.save(self.addition.state_dict(), file_path)
