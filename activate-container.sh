#! /bin/bash
CONTAINERNAME=nemoco-container
SCRIPT_PATH=$(dirname $(realpath -s $0))

read -p "If this container exists already, it will be overwritten. Are you sure?
[yY/nN] : " -n 1 -r
echo    # (optional) move to a new line
if [[ $REPLY =~ ^[Yy]$ ]]
then
    docker rm $CONTAINERNAME
fi

# create bash history for docker container
BASH_HISTORY=".docker_bash_history"
if ! test -f "$BASH_HISTORY"; then
    touch $BASH_HISTORY
fi

# start the container
docker run \
    --name $CONTAINERNAME \
    -it \
    -p 8889:8889 \
    --gpus all \
    --mount type=bind,source=/home/paul/repos/nemoco-tf,target=/code \
    --mount type=bind,source=/home/paul/mi_cloud/csm/nemoco,target=/cloud \
    -v $SCRIPT_PATH/$BASH_HISTORY:/root/.bash_history \
    nemoco:latest
