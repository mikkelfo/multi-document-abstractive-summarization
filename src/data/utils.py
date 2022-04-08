import json, gzip
import os


def read_jsonl_gz(path):
    with gzip.open(path) as f:
        for l in f:
            yield json.loads(l)

def prepare_directory(dir):
    for source in ['text', 'summary']:
        if not os.path.isdir(f'data/processed/{dir}/{source}'):
            os.mkdir(f'data/processed/{dir}/{source}')
        for type in ['test', 'train', 'validation']:
            if not os.path.isdir(f'data/processed/{dir}/{source}/{type}'):
                os.mkdir(f'data/processed/{dir}/{source}/{type}')


if __name__ == '__main__':
    prepare_directory('cnn-dm')
    # prepare_directory('danewsroom/abstractive')
    # prepare_directory('wcep')