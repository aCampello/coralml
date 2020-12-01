
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

# RUN cd /tmp                                                                                                                                 && \
#     wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_9.0.176-1_amd64.deb                 && \
#     wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/libcudnn7_7.0.5.15-1+cuda9.0_amd64.deb       && \
#     wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/libcudnn7-dev_7.0.5.15-1+cuda9.0_amd64.deb   && \
#     wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/libnccl2_2.1.4-1+cuda9.0_amd64.deb           && \
#     wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/libnccl-dev_2.1.4-1+cuda9.0_amd64.deb

RUN apt-get update
RUN add-apt-repository ppa:graphics-drivers
RUN apt-key adv --fetch-keys  http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
RUN bash -c 'echo "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/cuda.list'
RUN bash -c 'echo "deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/cuda_learn.list'

# RUN cd /tmp && \
#     apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub

# RUN cd /tmp                                                                 && \
#     dpkg -i cuda-repo-ubuntu1604_9.0.176-1_amd64.deb                        && \
#     dpkg -i libcudnn7_7.0.5.15-1+cuda9.0_amd64.deb                          && \
#     dpkg -i libcudnn7-dev_7.0.5.15-1+cuda9.0_amd64.deb                      && \
#     dpkg -i libnccl2_2.1.4-1+cuda9.0_amd64.deb                              && \
#     dpkg -i libnccl-dev_2.1.4-1+cuda9.0_amd64.deb

RUN \
    apt-get update                                                          && \
    apt-get --yes install                                                      \
        cuda-10-1                                                              \
        libcudnn7-dev                                                          \
        libnccl-dev

# Timezone to Berkeley

ENV TZ=America/Los_Angeles
RUN \
    ln -snf /usr/share/zoneinfo/$TZ /etc/localtime  &&  \
    echo $TZ > /etc/timezone


#-------------------------------------------------------------------------------
# The /opt/ scripts require source => switch `RUN` to execute bash (instead sh)
SHELL ["/bin/bash", "-c"]

#-------------------------------------------------------------------------------


# #-------------------------------------------------------------------------------
# # CONDA
# #
# # Build miniconda and (possibly) MPI4PY (linking with manually-install MPICH
# # library above)
# #
# 
# RUN mkdir -p /img/conda.local
# COPY conda.local /img/conda.local
# 
# RUN cd /img/conda.local                                                     && \
#     . sites/default.sh                                                      && \
#     export SKIP_MPI4PY=true                                                 && \
#     ./install_conda.sh
# 
# RUN source /img/conda.local/env.sh                                          && \
#     conda install mpi4py
# 
# RUN source /img/conda.local/env.sh                                          && \
#     python /img/conda.local/util/patch-rpath.py /img/conda.local/miniconda3/lib
# 
# #-------------------------------------------------------------------------------


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


# #-------------------------------------------------------------------------------
# # CONDA environment and dependencies
# #
# 
# RUN source /img/conda.local/env.sh                                          && \
#     conda create --name coral_reef python==3.6.7                            && \
#     source activate coral_reef                                              && \
#     conda install mpi4py                                                    && \
#     conda install pytorch-cpu torchvision-cpu -c pytorch                    && \
#     cd /img/imageclef-2019-code                                             && \
#     pip install --upgrade pip                                               && \
#     pip install -r requirements.txt
# 
# #-------------------------------------------------------------------------------
# 
# 
# #-------------------------------------------------------------------------------
# # PATCH CCTBX's conda env's rpaths
# #
# 
# RUN source /img/conda.local/env.sh                                          && \
#     source activate coral_reef                                              && \
#     python /img/conda.local/util/patch-rpath.py $CONDA_PREFIX/lib
# 
# #-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------
# Install DEPENDENCIES
#

RUN cd /img                                                                 && \
    python3 -m pip install --upgrade pip                                    && \
    python3 -m pip install -r requirements.txt

#-------------------------------------------------------------------------------


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
