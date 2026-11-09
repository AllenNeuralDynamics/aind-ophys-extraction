# hash:sha256:649aa40b5b8fe011939369eb45e09d30a6bed17592239e8ff28807b2fb2a6841
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
    -e git+https://github.com/AllenNeuralDynamics/aind-ophys-extraction-library.git@643e296#egg=aind-ophys-extraction-library
