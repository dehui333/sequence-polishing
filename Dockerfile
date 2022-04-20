FROM nvidia/cuda:11.2.2-cudnn8-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-o", "pipefail", "-c"]


RUN \
    apt update \
    && apt install -yq --no-install-recommends \
        python3-all-dev python3-pip \
        build-essential \
        libcurl4-gnutls-dev libssl-dev \
    && pip install numpy pysam biopython h5py \
       torch==1.10.2 pytorch-lightning wandb jsonargparse docstring-parser \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY . /roko

RUN \
    cd /roko/Dependencies/htslib-1.9 \
    && ./configure CFLAGS=-fpic --disable-bz2 --disable-lzma --without-libdeflate \
    && make \
    && cd /roko \
    && python3 setup.py install


