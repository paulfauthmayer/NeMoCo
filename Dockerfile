# syntax=docker/dockerfile:1
# base our image on the tensorflow gpu jupyter image
FROM tensorflow/tensorflow:2.7.0-gpu-jupyter

# start in the /code directory
WORKDIR /code

# install system requirements
RUN apt update && apt install vim tmux git graphviz --yes && apt clean

# install python requirements
RUN pip install tqdm jupyterlab scikit-learn pandas matplotlib seaborn plotly pydot

ENTRYPOINT [ "bash" ]