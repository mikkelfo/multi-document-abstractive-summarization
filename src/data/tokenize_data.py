import json
import torch
from transformers import ProphetNetTokenizer


def chunk_and_tokenize(chunk_size=1024):
    tokenizer = ProphetNetTokenizer.from_pretrained('microsoft/prophetnet-large-uncased')
    for name in ['train', 'test', 'validation']:
        file = open(f'data/processed/cnn-dm_{name}.json', 'r')
        data = json.load(file)

        for i in range(0, len(data), chunk_size):
            subset = data[i:(i+chunk_size)]
            text, summary = zip(*subset)

            tokenized_text = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            tokenized_summary = tokenizer(summary, return_tensors="pt", padding=True, truncation=True)

            torch.save(tokenized_text, f'data/processed/cnn-dm/text/{name}/chunk_{i // chunk_size}.pt')
            torch.save(tokenized_summary, f'data/processed/cnn-dm/summary/{name}/chunk_{i // chunk_size}.pt')


if __name__ == '__main__':
    chunk_and_tokenize()
