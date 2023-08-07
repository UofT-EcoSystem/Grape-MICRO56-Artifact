#!/bin/bash -e

UBUNTU_VERSION=$(lsb_release -sr)

PROJECT_ROOT=$(git rev-parse --show-toplevel)
cd ${PROJECT_ROOT}

source scripts/hl

IS_REBUILD=0

while [[ $# -gt 0 ]]; do
        case $1 in
        --rebuild)
                IS_REBUILD=1
                shift
                ;;
        *)
                negative "Usage: $(basename ${BASH_SOURCE[0]}) [--rebuild]"
                exit -1
                ;;
        esac
done

if [ ${IS_REBUILD} == "0" ]; then
        echo "[$(emph 1/3)] Checking out the $(bold pytorch) submodule ..."
        # Temporarily increase the largest buffer size of git to ensure a
        # successful checkout.
        #
        # Reference: https://stackoverflow.com/a/52487332/6320608
        git config --global http.postBuffer 157286400
        git submodule update --init --recursive submodules/pytorch
fi

cd submodules/pytorch

if [ ${IS_REBUILD} == "0" ]; then
        echo -n "[$(emph 2/3)] "
else
        echo -n "[$(emph 1/2)] "
fi
echo "Querying the NVCC_ARCHS from __nvcc_device_query ..."

if [ -x $(command -v __nvcc_device_query) ]; then
        NVCC_ARCHS=$(__nvcc_device_query)
        NVCC_ARCHS="${NVCC_ARCHS::-1}.${NVCC_ARCHS: -1}"
else
        negative "__nvcc_device_query not found. Have you installed CUDA properly?"
fi

if [ ${IS_REBUILD} == "0" ]; then
        echo -n "[$(emph 3/3)] "
else
        echo -n "[$(emph 2/2)] "
fi
echo "Building PyTorch with NVCC_ARCHS=$(bold ${NVCC_ARCHS}) ..."

export USE_CUDA=1 USE_CUDNN=1 \
        TORCH_CUDA_ARCH_LIST="${NVCC_ARCHS}" \
        BUILD_CAFFE2=0 \
        BUILD_CAFFE2_OPS=0 \
        BUILD_TEST=0 \
        USE_NNPACK=0 \
        USE_QNNPACK=0 \
        USE_XNNPACK=0 \
        USE_FBGEMM=0 \
        USE_DISTRIBUTED=0 \
        USE_MKLDNN=0 \
        MAX_JOBS=${MAX_JOBS}

if [ ${UBUNTU_VERSION} == "22.04" ]; then
        python setup.py develop --install-dir $(python -m site --user-site)
elif [ ${UBUNTU_VERSION} == "20.04" ]; then
        python setup.py develop
else
        negative "Unknown UBUNTU_VERSION=${UBUNTU_VERSION}"
fi
