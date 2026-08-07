# hash:sha256:21e1c60535a4af16d5fb869da69dd45c8d5177d030a0d56870ce32e39024bfae
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/mambaforge3:23.1.0-4-python3.10.12-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-ask-pass /

ENV CAIMAN_TEMP=/scratch

RUN mamba install -y \
        caiman==1.10.3 \
    && mamba clean -ya

RUN --mount=type=secret,id=git-access-token \
    GIT_ACCESS_TOKEN="$(cat /run/secrets/git-access-token)" \
    pip3 install -U --no-cache-dir \
    -e git+https://github.com/AllenNeuralDynamics/aind-ophys-extraction-library.git@6dc3b7#egg=aind-ophys-extraction-library
