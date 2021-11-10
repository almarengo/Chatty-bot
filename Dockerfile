FROM nvidia/cuda:11.1.1-cudnn8-runtime-ubuntu18.04
#FROM nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    python3.8 python3-pip python3-setuptools python3-dev &&\
    apt-get install -y python3.8-tk &&\
    apt-get remove python3-matplotlib &&\
    apt-get install -y x11-apps &&\
    apt-get install -y python3-matplotlib
 
RUN pip3 install torch==1.9.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html

WORKDIR /src

COPY requirements.txt ./requirements.txt

RUN python3 -m pip install --no-cache-dir -r requirements.txt

COPY . /src

#CMD [ "python3", "-u", "train.py" ]