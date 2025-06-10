import numpy as np
import torch
from torch import Tensor
from transformers import (AutoTokenizer)

import evaluate
from model.model import BEST_PROMPT, EmbeddingModel


def last_token_pool(last_hidden_states: Tensor,
                    attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def main():
    base_model_name = "Alibaba-NLP/gte-Qwen2-1.5B-instruct"
    max_length = 8192

    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    model = EmbeddingModel(base_model_name, 3, addition_path='model/addition.pt')

    def compute(queries: np.ndarray, classes: np.ndarray):
        queries = np.char.add(np.array(list(BEST_PROMPT for _ in queries)), queries)
        tokens = (tokenizer(queries.tolist(), max_length=max_length, padding=True, truncation=True, return_tensors='pt')
                  .to(model.device))
        scores = model(**tokens)

        return classes[np.argmax(scores.cpu().detach().numpy(), axis=-1)]

    with open('dataset/test.csv', 'rt') as fd:
        f1, cm = evaluate.posts(
            fd=fd,
            total_size=1000,
            chunk_size=5,
            compute=compute
        )
    print(cm)
    print(f'f1: {f1}')


if __name__ == '__main__':
    main()
