FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04
ARG DEBIAN_FRONTEND=noninteractive
RUN mkdir /workspace
WORKDIR /workspace
ADD ./ /workspace
RUN apt-get update && apt-get install -y \
    curl \
    git \
    make \
    nodejs \
    npm \
    tzdata \
    wget \
    && rm -rf /var/lib/apt/lists/*
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda
ENV PATH $CONDA_DIR/bin:$PATH
RUN conda init bash && . ~/.bashrc && conda install -y python=3.10
RUN git config --global --add safe.directory /workspace
RUN npm install n -g
RUN n 15.14.0
RUN apt purge -y nodejs npm
RUN pip install -r requirements.txt
