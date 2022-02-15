# syntax=docker/dockerfile:1
# base our image on the tensorflow gpu jupyter image
FROM tensorflow/tensorflow:2.7.0-gpu-jupyter

# start in the /code directory
WORKDIR /code

# install system requirements
RUN apt update && \
    apt install \
        git \
        graphviz \
        tmux \
        vim \
        --yes && \
    apt clean

# Installing Node
ENV NODE_VERSION 12.14.1
SHELL ["/bin/bash", "--login", "-i", "-c"]
RUN curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.35.2/install.sh | bash
RUN source /root/.bashrc && nvm install ${NODE_VERSION}
SHELL ["/bin/bash", "--login", "-c"]
ENV NVM_DIR /root/.nvm
ENV PATH $PATH:${NVM_DIR}/versions/node/v${NODE_VERSION}/bin

# install python requirements
RUN pip install \
    black \
    imageio \
    jupyterlab \
    matplotlib \
    onnxruntime \
    pandas \
    plotly \
    pydot \
    pyyaml \
    scikit-learn \
    seaborn \
    tensorflow_addons \
    tf2onnx \
    tqdm

# setup jupyterlab
RUN jupyter-lab build -y

ENTRYPOINT [ "bash" ]