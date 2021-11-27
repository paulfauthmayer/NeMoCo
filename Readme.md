# Neural Motion Controller

## Setup

1. Install Anydesk on your machine and set up tcp tunneling:
    1. 9000 -> 22 for ssh
    2. 8888 -> 8889 for jupyterlab
2. Connect to the machine via ssh: `ssh paul@localhost:9000`
3. Install the Remote SSH extension for vscode and connect to the machine
4. If `docker ps -a` does not list a container named `nemoco-container`, run `./activate-container.sh`
5. If the container already exists, run `docker start nemoco-container -i`
6. Once inside container, run your scripts or start the jupyter server with `start-server.py`
    