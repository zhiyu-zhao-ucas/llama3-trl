# Llama DPO
The project trains Meta-Llama-3-8B with DPO using TRL.

Build Dockerfile
```shell
docker build -t zhiyu/trl .
```
Run docker container
```shell
docker run -p 8022:22 --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v /path/to/llama3_trl/:/workspace/code -v /path/to/Meta-Llama-3-8B:/workspace/model -v /path/to/Anthropic___hh-rlhf:/workspace/data -it --rm zhiyu/trl
```
Install flash-attn
```shell
pip install flash-attn
```
Login wandb
```shell
wandb login
```
Train Llama3 with DPO
```shell
python /workspace/code/main.py
```
