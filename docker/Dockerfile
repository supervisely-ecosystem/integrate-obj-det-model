FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y build-essential
RUN apt-get install -y git ffmpeg libsm6 libxext6 ninja-build

RUN python3 -m pip install "git+https://github.com/facebookresearch/detectron2.git"
RUN python3 -m pip install "git+https://github.com/open-mmlab/cocoapi.git#subdirectory=pycocotools"
RUN pip3 install supervisely==6.69.70