#!/bin/bash -e

PROJECT_ROOT=$(git rev-parse --show-toplevel)
cd ${PROJECT_ROOT}

source scripts/hl

echo "[$(emph 1/1)] Installing build essentials ..."

sudo apt-get update
sudo apt-get install -y --no-install-recommends git \
        wget \
        curl \
        vim \
        build-essential \
        python3-dev \
        python-is-python3 \
        python3-virtualenv \
        lsb-release \
        gnupg2 \
        curl \
        ca-certificates \
        libboost-filesystem-dev \
        rustc \
        cargo \
        libssl-dev \
        csvtool
wget --backups=1 https://bootstrap.pypa.io/get-pip.py
python get-pip.py
rm -f get-pip.py

pip install cmake matplotlib pytest numpy
# PyTorch
pip install pyyaml typing_extensions
# Quik-Fix
pip install nvtx cloudpickle
# HuggingFace
pip install filelock regex sacremoses tokenizers==0.12.0 \
        huggingface-hub==0.1.0 tqdm
# Wav2Vec2
pip install datasets==1.18.3 librosa jiwer
