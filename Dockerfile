FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel
ADD ./ /workspace
RUN apt update && apt install -y git nodejs
RUN git config --global --add safe.directory /workspace
RUN pip install -r requirements.txt