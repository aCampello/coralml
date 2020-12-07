
FROM ubuntu:18.04
LABEL maintainer="Johannes Blaschke <jpblaschke@lbl.gov>"
# adapted from Rollin Thomas <rcthomas@lbl.gov>
# and Kelly Rowland <kellyrowland@lbl.gov>

# Base Ubuntu packages

ENV DEBIAN_FRONTEND noninteractive
ENV LANG C.UTF-8

RUN \
    apt-get update          &&                                                 \
    apt-get --yes upgrade   &&                                                 \
    apt-get --yes install                                                      \
        bzip2                                                                  \
        curl                                                                   \
        git                                                                    \
        libffi-dev                                                             \
        lsb-release                                                            \
        tzdata                                                                 \
        vim                                                                    \
        wget                                                                   \
        bash                                                                   \
        autoconf                                                               \
        automake                                                               \
        gcc                                                                    \
        g++                                                                    \
        make                                                                   \
        cmake                                                                  \
        gfortran                                                               \
        tar                                                                    \
        unzip                                                                  \
        strace                                                                 \
        patchelf                                                               \
        libgl1-mesa-dev                                                        \
        libgtk2.0-0                                                            \
        x11-xserver-utils                                                      \
        mpich                                                                  \
        python3                                                                \
        python3-pip                                                            \
        software-properties-common


#-------------------------------------------------------------------------------
# ADD CUDA to the image
#

RUN apt-get update
RUN add-apt-repository ppa:graphics-drivers
RUN apt-key adv --fetch-keys  http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
RUN bash -c 'echo "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/cuda.list'
RUN bash -c 'echo "deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/cuda_learn.list'

RUN \
    apt-get update                                                          && \
    apt-get --yes install                                                      \
        cuda-10-2                                                              \
        libcudnn7-dev                                                          \
        libnccl-dev

ENV CUDAHOME /usr/local/cuda-10.1

#-------------------------------------------------------------------------------


# Timezone to Berkeley

ENV TZ=America/Los_Angeles
RUN \
    ln -snf /usr/share/zoneinfo/$TZ /etc/localtime  &&  \
    echo $TZ > /etc/timezone


#-------------------------------------------------------------------------------
# The /opt/ scripts require source => switch `RUN` to execute bash (instead sh)
SHELL ["/bin/bash", "-c"]

#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------
# Copy the projects into the image
#

COPY coralml /img/coralml
COPY data /img/data
COPY requirements.txt /img/requirements.txt
COPY setup.py /img/setup.py
RUN mkdir -p /img/coralml/src
COPY docker/pytorch-deeplab-xception /img/coralml/src/pytorch-deeplab-xception

#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------
# Install DEPENDENCIES
#

RUN cd /img                                                                 && \
    python3 -m pip install --upgrade pip                                    && \
    python3 -m pip install -r requirements.txt

#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------
# Install DETR dependencies
#

RUN git clone https://github.com/facebookresearch/detr.git

RUN python3 -m pip install cython scipy
RUN python3 -m pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
RUN python3 -m pip install git+https://github.com/cocodataset/panopticapi.git
RUN python3 -m pip install submitit

#-------------------------------------------------------------------------------
# LDCONFIG
#
# We recommend running an /sbin/ldconfig as part of the image build (e.g. in
# the Dockerfile) to update the cache after installing any new libraries in in
# the image build.
#

RUN /sbin/ldconfig

#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------
# ENTRYPOINT
#

RUN mkdir -p /img
ADD docker/entrypoint.sh /img

RUN mkdir -p /img/tests
COPY docker/tests/*.py /img/tests/

WORKDIR /img

RUN chmod +x entrypoint.sh

ENTRYPOINT ["/img/entrypoint.sh"]

#-------------------------------------------------------------------------------
