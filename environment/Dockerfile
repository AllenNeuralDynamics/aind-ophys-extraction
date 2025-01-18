# hash:sha256:4bf2c6f19bdc43d3a84be1e8718a293abfbe18f34ed46fef82ed714f8d43c2ac
FROM registry.codeocean.allenneuraldynamics.org/codeocean/mambaforge3:23.1.0-4-python3.10.12-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

ENV EXTRACTION_URL=https://github.com/AllenNeuralDynamics/aind-ophys-extraction-suite2p
ENV PIPELINE_URL="https://codeocean.allenneuraldynamics.org/capsule/5619253/tree"
ENV PIPELINE_VERSION="1.0"
ENV VERSION="5.0"

RUN ["apt-get", "update"]
RUN ["apt-get", "install", "-y", "vim"]

RUN mamba install -y \
        caiman \
    && mamba clean -ya

RUN pip3 install -U --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu

RUN pip3 install -U --no-cache-dir \
    aind-data-schema==1.1.0 \
    cellpose==3.0 \
    h5py \
    pydantic==2.8.2 \
    sbxreader \
    scikit-image \
    sparse \
    aind-ophys-utils \
    git+https://github.com/j-friedrich/suite2p.git@save_cellpose_output \
    aind-log-utils \
    aind-qcportal-schema