#! /bin/bash
CONTAINERNAME=nemoco-container

read -p "If this container exists already, it will be overwritten. Are you sure?
[yY/nN] : " -n 1 -r
echo    # (optional) move to a new line
if [[ $REPLY =~ ^[Yy]$ ]]
then
    docker rm $CONTAINERNAME
fi

docker run \
    --name $CONTAINERNAME \
    -it \
    -p 8889:8889 \
    --gpus all \
    --mount type=bind,source=/home/paul/repos/nemoco-tf,target=/code \
    nemoco:latest
