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

# install newest node version
RUN curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.35.3/install.sh | bash; \
    source ~/.bashrc; \
    nvm install node;

# install python requirements
RUN pip install \
    black \
    jupyterlab \
    matplotlib \
    pandas \
    plotly \
    pydot \
    scikit-learn \
    seaborn \
    tensorflow_addons \
    tf2onnx \
    tqdm

# setup jupyterlab
RUN jupyter lab build

ENTRYPOINT [ "bash" ]