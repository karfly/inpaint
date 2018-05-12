#!/usr/bin/env bash

gunicorn "app:setup_app()" -b 0.0.0.0:8003 -w 4
