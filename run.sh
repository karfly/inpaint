#!/bin/sh

rm -rf prometheus/client_data/*
docker-compose build
docker-compose up
