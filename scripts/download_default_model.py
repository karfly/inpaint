#!/usr/bin/env python3
import argparse
import urllib.request


URL = 'https://www.dropbox.com/s/2bsem24zj8mw66k/default_model.state_dict?dl=1'
DEFAULT_PATH = 'default_model.state_dict'


def download(url, file_path):
    with open(file_path, 'wb') as stream:
        stream.write(urllib.request.urlopen(url, timeout=10).read())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', default=DEFAULT_PATH)
    args = parser.parse_args()
    download(URL, args.path)


if __name__ == '__main__':
    main()
