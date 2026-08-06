# hash:sha256:6c2bfc26a96c8b9e5a35ad6868bc43469ae773c41b77d9b203256fe1be5e4bef
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
    aind-data-schema==2.9.0 \
    aind-ophys-utils==0.2.0 \
    aind-qcportal-schema==0.6.4 \
    cellpose==3.0.10 \
    h5py==3.11.0 \
    pydantic==2.8.2 \
    pydantic-settings==2.14.2 \
    sbxreader==0.2.3 \
    scikit-image==0.24.0 \
    sparse==0.15.4 \
    torch==2.4.0 \
    -e git+https://github.com/AllenNeuralDynamics/aind-ophys-extraction-library.git@57b97b0#egg=aind-ophys-extraction-library \
    -e git+https://github.com/j-friedrich/suite2p.git@save_cellpose_output#egg=suite2p
