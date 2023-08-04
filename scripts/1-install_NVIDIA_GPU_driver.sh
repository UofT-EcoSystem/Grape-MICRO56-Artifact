#!/bin/bash -e

# The NVIDIA GPU driver has a kernel space and a user space component. We will
# be using a customized kernel module while keeping the user space component as
# it is in the official release.

export NVIDIA_DRIVER_VERSION=525.89.02

PROJECT_ROOT=$(cd $(dirname ${BASH_SOURCE[0]})/.. && pwd)
cd ${PROJECT_ROOT}

source scripts/hl.sh

IS_REINSTALL=0

while [[ $# -gt 0 ]]; do
        case $1 in
        --reinstall)
                IS_REINSTALL=1
                shift
                ;;
        *)
                negative "Usage: $(basename ${BASH_SOURCE[0]}) [--reinstall]"
                exit -1
                ;;
        esac
done

if [ ${IS_REINSTALL} == "0" ]; then
        echo -n "[$(emph 1/6)] "
else
        echo -n "[$(emph 1/3)] "
fi
echo "Uninstalling all existing NVIDIA GPU driver components ..."
if [ -x $(command -v nvidia-uninstall) ]; then
        sudo nvidia-uninstall
fi
if [ ${IS_REINSTALL} == "0" ]; then
        sudo apt-get purge -y "^nvidia*" && sudo apt-get autoremove -y

        echo "[$(emph 2/6)] Checking out the $(bold open-gpu-kernel-modules) submodule ..."
        git submodule update --init submodules/open-gpu-kernel-modules
fi

cd submodules/open-gpu-kernel-modules

if [ ${IS_REINSTALL} == "0" ]; then
        echo "[$(emph 3/6)] Downloading the official driver installer ..."
        wget https://us.download.nvidia.com/XFree86/Linux-x86_64/${NVIDIA_DRIVER_VERSION}/NVIDIA-Linux-x86_64-${NVIDIA_DRIVER_VERSION}.run

        echo "[$(emph 4/6)] Configuring the Linux kernel modules to avoid potential installation and runtime errors ..."

        # Resolve issue: "No such file or directory: bss_file.c"
        #
        # Reference: https://superuser.com/q/1214116
        echo -e "[ req ] \n\
        default_bits = 4096 \n\
        distinguished_name = req_distinguished_name \n\
        prompt = no \n\
        x509_extensions = myexts \n\

        [ req_distinguished_name ] \n\
        CN = Modules \n\
        \n\
        [ myexts ] \n\
        basicConstraints=critical,CA:FALSE \n\
        keyUsage=digitalSignature \n\
        subjectKeyIdentifier=hash \n\
        authorityKeyIdentifier=keyid" >x509.genkey

        openssl req -new -nodes -utf8 -sha512 -days 36500 -batch -x509 -config x509.genkey -outform DER -out signing_key.x509 -keyout signing_key.pem
        sudo mv signing_key.pem signing_key.x509 $(find /usr/src/linux-*/certs)

        # Resolve issue: "The Nouveau kernel driver is currently in use" when the user
        # space component is being installed.
        #
        # Reference: https://askubuntu.com/q/841876
        sudo echo "\
        blacklist nouveau
        options nouveau modeset=0" >/etc/modprobe.d/blacklist-nouveau.conf
        sudo update-initramfs -u

        # Resolve issue: "NVRM: GPU 0000:01:00.0: RmInitAdapter failed!" when when the
        # driver is initialized at runtime (for desktop or workstation GPUs such as
        # NVIDIA RTX 3090).
        #
        # Reference: https://github.com/NVIDIA/open-gpu-kernel-modules#compatible-gpus
        sudo echo "options nvidia NVreg_OpenRmEnableUnsupportedGpus=1" >/etc/modprobe.d/whitelist-unsupported-gpus.conf
fi

if [ ${IS_REINSTALL} == "0" ]; then
        echo -n "[$(emph 5/6)] "
else
        echo -n "[$(emph 2/3)] "
fi
echo "Building and installing the kernel space component of the driver ..."
make modules -j
sudo make modules_install -j

if [ ${IS_REINSTALL} == "0" ]; then
        echo -n "[$(emph 6/6)] "
else
        echo -n "[$(emph 3/3)] "
fi
echo "Installing the user space component ..."
sudo sh NVIDIA-Linux-x86_64-${NVIDIA_DRIVER_VERSION}.run --no-kernel-modules
