FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

RUN apt update && apt install -y pip

RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118

RUN pip install pytest pyrootutils hydra-core lightning hydra-colorlog