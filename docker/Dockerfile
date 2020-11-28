
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
        mpich

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
# CONDA
#
# Build miniconda and (possibly) MPI4PY (linking with manually-install MPICH
# library above)
#

RUN mkdir -p /img/conda.local
COPY conda.local /img/conda.local

RUN cd /img/conda.local                                                     && \
    . sites/default.sh                                                      && \
    export SKIP_MPI4PY=true                                                 && \
    ./install_conda.sh

RUN source /img/conda.local/env.sh                                          && \
    conda install mpi4py

RUN source /img/conda.local/env.sh                                          && \
    python /img/conda.local/util/patch-rpath.py /img/conda.local/miniconda3/lib

#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------
# Copy the projects into the image
#

COPY imageclef-2019-code /img/imageclef-2019-code
RUN mkdir -p /img/imageclef-2019-code/src
COPY pytorch-deeplab-xception /img/imageclef-2019-code/src/pytorch-deeplab-xception

#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------
# CONDA environment and dependencies
#

RUN source /img/conda.local/env.sh                                          && \
    conda create --name coral_reef python==3.6.7                            && \
    source activate coral_reef                                              && \
    conda install mpi4py                                                    && \
    conda install pytorch-cpu torchvision-cpu -c pytorch                    && \
    cd /img/imageclef-2019-code                                             && \
    pip install --upgrade pip                                               && \
    pip install -r requirements.txt

#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------
# PATCH CCTBX's conda env's rpaths
#

RUN source /img/conda.local/env.sh                                          && \
    source activate coral_reef                                              && \
    python /img/conda.local/util/patch-rpath.py $CONDA_PREFIX/lib

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
ADD entrypoint.sh /img

RUN mkdir -p /img/tests
COPY tests/*.py /img/tests/

WORKDIR /img

RUN chmod +x entrypoint.sh

ENTRYPOINT ["/img/entrypoint.sh"]

#-------------------------------------------------------------------------------
