FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel
ADD ./ /workspace
RUN apt update && apt install -y git curl
RUN git config --global --add safe.directory /workspace
RUN apt install -y nodejs npm
RUN npm install n -g
RUN n 15.14.0
RUN pip install -r requirements.txt