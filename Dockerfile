FROM huggingface/trl-source-gpu

RUN conda init bash \
    && echo "conda activate trl" >> ~/.bashrc
RUN source ~/.bashrc
RUN source activate trl

RUN pip install wandb