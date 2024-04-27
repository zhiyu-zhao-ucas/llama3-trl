FROM huggingface/trl-source-gpu

RUN conda init bash \
    && echo "conda activate trl" >> ~/.bashrc

RUN pip install wandb