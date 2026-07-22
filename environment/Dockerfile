# hash:sha256:3a00a2bc08e500660717fdfff02f6cee82dd45a3954fe3e321b9c9d11307981c
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/mambaforge3:24.5.0-0-python3.12.4-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-ask-pass /

ENV EXTRACTION_URL=https://github.com/AllenNeuralDynamics/aind-ophys-extraction
ENV VERSION="15.0"

RUN mamba install -y \
        caiman \
    && mamba clean -ya

RUN pip3 install -U --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cpu

RUN pip3 install -U --no-cache-dir \
    aind-data-schema==1.1.0 \
    cellpose==4.2.1.1 \
    h5py \
    pydantic==2.8.2 \
    pydantic_settings \
    sbxreader \
    scikit-image \
    sparse \
    aind-ophys-utils \
    git+https://github.com/j-friedrich/suite2p.git@save_cellpose_output_v1.1.0 \
    aind-log-utils \
    aind-qcportal-schema

RUN caimanmanager install
