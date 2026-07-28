# hash:sha256:ad3d2c0b4a28642f711e01c0ef499af68156aec8031b5c3b81de21a7c5dbefe3
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/mambaforge3:23.1.0-4-python3.10.12-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-ask-pass /

RUN mamba install -y \
        caiman==1.10.3 \
    && mamba clean -ya

RUN --mount=type=secret,id=git-access-token \
    GIT_ACCESS_TOKEN="$(cat /run/secrets/git-access-token)" \
    pip3 install -U --no-cache-dir \
    aind-data-schema==2.8.1 \
    aind-ophys-utils==0.2.0 \
    aind-qcportal-schema \
    cellpose==3.0.10 \
    h5py==3.11.0 \
    pydantic==2.8.2 \
    pydantic-settings \
    sbxreader \
    scikit-image==0.24.0 \
    sparse==0.15.4 \
    torch==2.4.0 \
    -e git+https://github.com/AllenNeuralDynamics/aind-ophys-extraction-library.git@3a2abda#egg=aind-ophys-extraction-library \
    -e git+https://github.com/j-friedrich/suite2p.git@save_cellpose_output#egg=suite2p

COPY postInstall /
RUN --mount=type=secret,id=secrets . /run/secrets/secrets \
    && /postInstall
