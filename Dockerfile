FROM ubuntu:xenial

ENV MODEL_URL https://www.dropbox.com/s/gb0yamqa6cnoomy/model_no_sigmoid_lr_all_50.pth?dl=1

RUN apt-get update && apt-get install -y \
        python3 python3-dev python3-pip

RUN pip3 install --upgrade pip
RUN pip3 install Flask==1.0.2 \
                 gunicorn==19.8.1 \
                 jsonlines==1.2.0 \
                 matplotlib==2.2.2 \
                 numpy==1.14.3 \
                 Pillow==5.1.0 \
                 pycodestyle==2.4.0 \
                 tqdm==4.23.3 \
                 pyflakes==1.6.0

COPY . /inpaint
RUN wget -O /inpaint/app/static/models/model.state_dict $MODEL_URL

EXPOSE 5000

WORKDIR /inpaint/app
CMD ["gunicorn", "app:setup_app('static/models/model.state_dict')", "-b", "0.0.0.0:5000", "-w", "4"]
