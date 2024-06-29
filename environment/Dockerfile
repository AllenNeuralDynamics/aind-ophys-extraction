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
    cellpose==2.1.0 \
    sbxreader \
    scikit-image \
    suite2p==0.10.2
