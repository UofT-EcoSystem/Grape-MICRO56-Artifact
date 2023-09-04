#!/bin/bash -e

export NVARCH=x86_64
export CUDA_VERSION="12-0"
export NSIGHT_SYSTEMS_VERSION="2022.4.2"

UBUNTU_VERSION=$(lsb_release -sr)
CUDA_VERSION_DOT_SEP=${CUDA_VERSION/-/.}

PROJECT_ROOT=$(git rev-parse --show-toplevel)
cd ${PROJECT_ROOT}

source scripts/hl

if [ ${UBUNTU_VERSION} == "22.04" ]; then
        # https://gitlab.com/nvidia/container-images/cuda/-/blob/master/dist/12.0.0/ubuntu2204/base/Dockerfile?ref_type=heads#L25-26
        curl -fsSLO https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/${NVARCH}/cuda-keyring_1.0-1_all.deb
        sudo dpkg -i cuda-keyring_1.0-1_all.deb
elif [ ${UBUNTU_VERSION} == "20.04" ]; then
        # https://gitlab.com/nvidia/container-images/cuda/-/blob/master/dist/12.0.0/ubuntu2004/base/Dockerfile?ref_type=heads#L25-26
        curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/${NVARCH}/3bf863cc.pub | apt-key add -
        echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/${NVARCH} /" >/etc/apt/sources.list.d/cuda.list
else
        negative "Unknown UBUNTU_VERSION=${UBUNTU_VERSION}"
fi

sudo apt-get update
# https://gitlab.com/nvidia/container-images/cuda/-/blob/master/dist/12.0.0/ubuntu2204/devel/Dockerfile?ref_type=heads#L55-67
sudo apt-get install -y cuda-cudart-${CUDA_VERSION} \
        cuda-compat-${CUDA_VERSION} \
        cuda-cudart-dev-${CUDA_VERSION} \
        cuda-command-line-tools-${CUDA_VERSION} \
        cuda-minimal-build-${CUDA_VERSION} \
        cuda-libraries-dev-${CUDA_VERSION} \
        cuda-nvml-dev-${CUDA_VERSION} \
        cuda-nvprof-${CUDA_VERSION} \
        libnpp-dev-${CUDA_VERSION} \
        libcusparse-dev-${CUDA_VERSION} \
        libcublas-dev-${CUDA_VERSION} \
        libnccl2=*+cuda${CUDA_VERSION_DOT_SEP} \
        libnccl-dev=*+cuda${CUDA_VERSION_DOT_SEP} \
        libcudnn8=*+cuda${CUDA_VERSION_DOT_SEP} \
        libcudnn8-dev=*+cuda${CUDA_VERSION_DOT_SEP}
sudo apt-get install -y nsight-systems-${NSIGHT_SYSTEMS_VERSION}

cd /usr/local/cuda
sudo rm -rf samples
sudo git clone https://github.com/NVIDIA/cuda-samples samples
cd samples
sudo git checkout v12.0

echo "\
export PATH=\${PATH}:/usr/local/cuda/bin
export LD_LIBRARY_PATH=\${LD_LIBRARY_PATH}:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
export LIBRARY_PATH=\${LIBRARY_PATH}:/usr/local/cuda/lib64/stubs" >>~/.bashrc
