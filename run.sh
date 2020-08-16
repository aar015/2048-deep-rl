#!/bin/sh

docker run \
--rm \
--pid=host \
-p 9000:8888 \
-v "$PWD"/jupyter:/home \
-v "$PWD"/lab-config:/root/.jupyter/lab/user-settings \
-v "$PWD"/ssl:/ssl \
--gpus all \
2048-jupyter:latest