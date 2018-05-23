#!/bin/sh

docker build -t inpaint_image . &&
docker run -it -p 5000:8003 --name inpaint inpaint_image
