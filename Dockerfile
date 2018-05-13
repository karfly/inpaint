FROM ubuntu:xenial

RUN apt-get update && apt-get install -y \
        python3 python3-dev python3-pip

COPY . /inpaint
WORKDIR /inpaint

RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

# Download pretrained model
RUN python3 ./scripts/download_default_model.py --url https://www.dropbox.com/s/tghdqqkd6ytujsf/model_no_sigmoid_lr_84.pth?dl=1 --path app/static/models/model.state_dict

EXPOSE 8003

WORKDIR /inpaint/app
CMD ["gunicorn", "app:setup_app('static/models/model.state_dict')", "-b", "0.0.0.0:8003", "-w", "4"]
