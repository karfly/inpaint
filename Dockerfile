FROM ubuntu:xenial

RUN apt-get update && apt-get install -y \
        wget python3 python3-dev python3-pip
RUN pip3 install --upgrade pip

COPY requirements.txt /
RUN pip3 install -r /requirements.txt

ENV MODEL_URL https://www.dropbox.com/s/sfw94zc5u8zi1cl/model_selfie_13.pth?dl=1
RUN wget -O model.state_dict $MODEL_URL

COPY . /inpaint
RUN mv model.state_dict /inpaint/app/static/models/

EXPOSE 5000

WORKDIR /inpaint/app
CMD ["gunicorn", "app:setup_app('static/models/model.state_dict')", "-b", "0.0.0.0:5000", "-w", "4"]
