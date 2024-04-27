FROM huggingface/trl-source-gpu

# Activate our bash shell
RUN chsh -s /bin/bash
SHELL ["/bin/bash", "-c"]
ENV PATH /opt/conda/bin:$PATH

RUN chsh -s /bin/bash
SHELL ["/bin/bash", "-c"]
RUN source activate trl && \ 
    python3 -m pip install wandb flash-attn

RUN conda init bash \
    && echo "conda activate trl" >> ~/.bashrc
RUN source ~/.bashrc
RUN source activate trl