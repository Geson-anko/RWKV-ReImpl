FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime
ADD ./ /workspace
RUN apt-get update && apt-get install -y \
    curl \
    git \
    make \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*
RUN git config --global --add safe.directory /workspace
RUN npm install n -g
RUN n 15.14.0
RUN apt purge -y nodejs npm
RUN pip install -r requirements.txt
