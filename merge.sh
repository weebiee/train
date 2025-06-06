#!/usr/bin/env bash

USE_HF=1 \
swift export \
	--adapters output/v4-20250604-215333/checkpoint-84 \
	--merge_lora true
