FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel
ADD ./ /workspace
RUN pip install -r requirement.txt