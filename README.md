# Llama DPO Project

This project focuses on training Meta-Llama-3-8B with Deterministic Policy Optimization (DPO) using the following steps:

## Building Docker Image
To build the Docker image, execute the following command:

```shell
docker build -t zhiyu/trl .
```

## Running Docker Container
To run the Docker container with necessary configurations and volumes, execute the following command:

```shell
docker run -p 8022:22 --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v /path/to/llama3_trl/:/workspace/code -v /path/to/Meta-Llama-3-8B:/workspace/model -v /path/to/Anthropic___hh-rlhf:/workspace/data -it --rm zhiyu/trl
```

## Logging in to Wandb
To use Wandb for logging and visualization, login to your Wandb account:

```shell
wandb login
```

## Training Llama3 with DPO
Once everything is set up, you can initiate the training of Llama3 with DPO:

```shell
python /workspace/code/main.py
```