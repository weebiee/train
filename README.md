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

### Preparing dataset
Put your set under `${project_root}/dataset` directory.

- train.json: `[{"query": "I love bingchilling!!", "response": "positive"}, ...]`
- test.csv: `6人为钓大鱼竟操控无人艇进保护区,易看达州,#6人为钓大鱼竟操控无人艇进保护区#【#央视揭洞庭湖禁捕区违规
钓鱼乱象#】禁钓区大肆垂钓，还“叫卖”现做野生鱼；无视劝导，和管理人员“打游击”；为钓更大的鱼，使用无人
艇和杀伤力十足的鱼钩；甚至有人钓起正在繁殖期的母鱼后拒不放生，还大肆炫耀……眼下正值鱼类繁殖期，记者
发现，洞庭湖周边仍有违规钓鱼现象。一些钓鱼人使用的工具五花八门，不少违规钓具甚至会对水生生物带来毁
灭性伤害。而他们丢弃的鱼线鱼网，质地坚韧、隐蔽性强，不容易被江豚的声呐系统识别。一旦江豚被丝线缠住
，它们不断挣扎，丝线就会切入皮肤，最终因伤口感染而死亡。转发倡议：共同保护生态环境，守护好一江碧水
！,中性`

### Start training

```shell
source .venv/bin/activate
python main.py
```
