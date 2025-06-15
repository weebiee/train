# Train

Finetune to get a three-class text classification model, based on `Alibaba-NLP/gte-Qwen2-1.5B-instruct`.

## Getting started
This project is managed using uv.

```shell
uv sync && uv venv
```

The training script is able to determine which device to train on.
If CUDA is available on your system, that's good, but 10GiB of vRAM
is required, while it can be reduced to 4GiB if lowering the batch size
to 1.
