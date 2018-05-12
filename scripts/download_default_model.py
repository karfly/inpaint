#!/usr/bin/env python3
import argparse
import urllib.request

DEFAULT_URL = 'https://www.dropbox.com/s/9dode7w7y0y23rt/model_no_sigmoid_lr_31.pth?dl=1'
DEFAULT_PATH = 'default_model.state_dict'


def download(url, file_path):
    with open(file_path, 'wb') as stream:
        stream.write(urllib.request.urlopen(url, timeout=10).read())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', default=DEFAULT_PATH)
    parser.add_argument('-u', '--url', default=DEFAULT_URL)

    args = parser.parse_args()
    download(args.url, args.path)


if __name__ == '__main__':
    main()
