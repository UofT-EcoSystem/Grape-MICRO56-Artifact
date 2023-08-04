#!/bin/bash -e

while [[ $# -gt 0 ]]; do
        case $1 in
        --cudnn)
                CUDNN_VERSION=$2
                shift
                shift
                ;;
        --cudnn=*)
                CUDNN_VERSION=${1#*=}
                shift
                ;;
        --tensorrt)
                TENSORRT_VERSION=$2
                shift
                shift
                ;;
        --tensorrt=*)
                TENSORRT_VERSION=${1#*=}
                shift
                ;;
        --nccl)
                NCCL_VERSION=$2
                shift
                shift
                ;;
        --nccl=*)
                NCCL_VERSION=${1#*=}
                shift
                ;;
        *)
                echo "Unrecognized argument: $1"
                exit -1
                ;;
        esac
done

apt-get update

export DEBIAN_FRONTEND=noninteractive

CUDA_VERSION_DOT_SEPARATED=$(printf ${CUDA_VERSION} | grep -oE '[0-9]+.[0-9]+')
CUDA_VERSION_DASH_SEPARATED=${CUDA_VERSION_DOT_SEPARATED//\./-}

# CUDA Profiling Toolkits
apt-get install -y --no-install-recommends qt5-gtk-platformtheme libnss3 libxtst6

apt-get install -y --no-install-recommends \
        cuda-nsight-compute-${CUDA_VERSION_DASH_SEPARATED} \
        cuda-nsight-systems-${CUDA_VERSION_DASH_SEPARATED}

pip install nvtx

# CUDA Samples
cd /usr/local/cuda
git clone https://github.com/nvidia/cuda-samples samples
cd samples
git checkout v${CUDA_VERSION_DOT_SEPARATED}

BOLD_PREFIX="\033[1m"
BOLD_SUFFIX="\033[0m"

# cuDNN
if [ ! -z ${CUDNN_VERSION} ]; then
        echo -e "Installing ${BOLD_PREFIX}cuDNN ver.${CUDNN_VERSION}${BOLD_SUFFIX} ..."
        CUDNN_MAJOR_VERSION=$(printf ${CUDNN_VERSION} | grep -oE '[0-9]+' | head -1)

        apt-get install -y --no-install-recommends \
                libcudnn${CUDNN_MAJOR_VERSION}=${CUDNN_VERSION}-1+* \
                libcudnn${CUDNN_MAJOR_VERSION}-dev=${CUDNN_VERSION}-1+*
        apt-mark hold libcudnn${CUDNN_MAJOR_VERSION} \
                libcudnn${CUDNN_MAJOR_VERSION}-dev
fi

# TensorRT
if [ ! -z ${TENSORRT_VERSION} ]; then
        echo -e "Installing ${BOLD_PREFIX}TensorRT ver.${TENSORRT_VERSION}${BOLD_SUFFIX} ..."
        TENSORRT_MAJOR_VERSION=$(printf ${TENSORRT_VERSION} | grep -oE '[0-9]+' | head -1)

        apt-get install -y --no-install-recommends \
                libnvinfer${TENSORRT_MAJOR_VERSION}=${TENSORRT_VERSION}* \
                libnvinfer-dev=${TENSORRT_VERSION}* \
                libnvinfer-plugin${TENSORRT_MAJOR_VERSION}=${TENSORRT_VERSION}* \
                libnvinfer-plugin-dev=${TENSORRT_VERSION}* \
                libnvparsers${TENSORRT_MAJOR_VERSION}=${TENSORRT_VERSION}* \
                libnvparsers-dev=${TENSORRT_VERSION}* \
                libnvonnxparsers${TENSORRT_MAJOR_VERSION}=${TENSORRT_VERSION}* \
                libnvonnxparsers-dev=${TENSORRT_VERSION}* \
                python3-libnvinfer=${TENSORRT_VERSION}* \
                python3-libnvinfer-dev=${TENSORRT_VERSION}*
        apt-mark hold libnvinfer${TENSORRT_MAJOR_VERSION} \
                libnvinfer-dev \
                libnvinfer-plugin${TENSORRT_MAJOR_VERSION} \
                libnvinfer-plugin-dev \
                libnvparsers${TENSORRT_MAJOR_VERSION} \
                libnvparsers-dev \
                libnvonnxparsers${TENSORRT_MAJOR_VERSION} \
                libnvonnxparsers-dev \
                python3-libnvinfer \
                python3-libnvinfer-dev
fi

# NCCL
if [ ! -z ${NCCL_VERSION} ]; then
        echo -e "Installing ${BOLD_PREFIX}NCCL ver.${NCCL_VERSION}${BOLD_SUFFIX} ..."
        apt-get install -y --no-install-recommends libnccl-dev=${NCCL_VERSION}*
        apt-mark hold libnccl-dev
fi

# Cleanup
rm -rf /var/lib/apt/lists/*
