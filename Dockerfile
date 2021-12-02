# syntax=docker/dockerfile:1
# base our image on the tensorflow gpu jupyter image
FROM tensorflow/tensorflow:2.7.0-gpu-jupyter

# start in the /code directory
WORKDIR /code

# install system requirements
RUN apt update && apt install vim tmux git graphviz --yes && apt clean

# install newest node version
RUN curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.35.3/install.sh | bash; \
source ~/.bashrc; \
nvm install node;

# install python requirements
RUN pip install tqdm jupyterlab scikit-learn pandas matplotlib seaborn plotly pydot

# setup jupyterlab
RUN jupyter lab build

ENTRYPOINT [ "bash" ]