FROM registry.codeocean.allenneuraldynamics.org/codeocean/jupyterlab:3.4.4-miniconda4.12.0-python3.9-ubuntu20.04

ENV EXTRACTION_URL=https://github.com/AllenNeuralDynamics/aind-ophys-extraction-suite2p

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip3 install -U --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu

RUN pip3 install -U --no-cache-dir \
    aind-data-schema==0.22.0 \
    cellpose \
    h5py \
    pydantic==2.8.2 \
    sbxreader \
    scikit-image \
    git+https://github.com/j-friedrich/suite2p.git@save_cellpose_output
