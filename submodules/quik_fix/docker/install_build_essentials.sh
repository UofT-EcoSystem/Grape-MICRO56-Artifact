#!/bin/bash -e

sed -i 's/#force_color_prompt=yes/force_color_prompt=yes/' ~/.bashrc

while [[ $# -gt 0 ]]; do
        case $1 in
        --bazel)
                BAZEL_VERSION=$2
                shift
                shift
                ;;
        --bazel=*)
                BAZEL_VERSION=${1#*=}
                shift
                ;;
        --llvm)
                LLVM_VERSION=$2
                shift
                shift
                ;;
        --llvm=*)
                LLVM_VERSION=${1#*=}
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

apt-get install -y --no-install-recommends git \
        wget \
        curl \
        vim \
        build-essential \
        ccache \
        gdb \
        clang-format \
        python3-dev \
        python-is-python3

echo "ttf-mscorefonts-installer msttcorefonts/accepted-mscorefonts-eula select true" | debconf-set-selections
apt-get install -y --no-install-recommends ttf-mscorefonts-installer
apt-get install -y --no-install-recommends texlive-full
apt-get install -y --no-install-recommends tmux

git config --global user.name "ubuntu"
git config --global user.email "ubuntu@cs.toronto.edu"
git config --global --add safe.directory "*"
git config --global credential.helper store

echo "" >>~/.bashrc
echo "alias gti=git" >>~/.bashrc

wget https://bootstrap.pypa.io/get-pip.py
python get-pip.py && rm -f get-pip.py

pip install cmake cmake-format
pip install matplotlib
pip install black pytest

BOLD_PREFIX="\033[1m"
BOLD_SUFFIX="\033[0m"

if [ ! -z ${BAZEL_VERSION} ]; then
        echo -e "Installing ${BOLD_PREFIX}bazel ver.${BAZEL_VERSION}${BOLD_SUFFIX} ..."

        apt-get install -y --no-install-recommends unzip

        wget "https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh" \
                -O /tmp/bazel_installer.sh
        chmod +x /tmp/bazel_installer.sh
        /tmp/bazel_installer.sh
        rm -f /tmp/bazel_installer.sh

        apt-get purge -y unzip
        apt-get autoremove -y
fi

if [ ! -z ${LLVM_VERSION} ]; then
        echo -e "Installing ${BOLD_PREFIX}LLVM ver.${LLVM_VERSION}${BOLD_SUFFIX} ..."

        apt-get install -y --no-install-recommends lsb-release software-properties-common

        wget https://apt.llvm.org/llvm.sh -O /tmp/llvm.sh
        chmod +x /tmp/llvm.sh
        /tmp/llvm.sh ${LLVM_VERSION} all

        apt-get purge -y lsb-release software-properties-common
        apt-get autoremove -y
fi

# Cleanup
rm -rf /var/lib/apt/lists/*
