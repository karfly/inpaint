#!/bin/sh

docker build -t inpaint_image .
docker kill inpaint && docker rm inpaint
docker run -it -p 5000:8003 --name inpaint inpaint_image
