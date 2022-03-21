from transformers import ProphetNetTokenizer
import pandas as pd
import torch


def chunk_and_tokenize(type, chunk_size=1024):
    tokenizer = ProphetNetTokenizer.from_pretrained('microsoft/prophetnet-large-uncased')
    for name in ['train', 'test', 'validation']:
        df = pd.read_csv(f'data/processed/danewsroom/{type}/{name}.csv')

        for i in range(0, len(df), chunk_size):
            subset = df[i:(i+chunk_size)]
            text = subset.text.to_list()
            summary = subset.summary.to_list()

            tokenized_text = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            tokenized_summary = tokenizer(summary, return_tensors="pt", padding=True, truncation=True)

            torch.save(tokenized_text, f'data/processed/danewsroom/{type}/text/{name}/chunk_{i // chunk_size}.pt')
            torch.save(tokenized_summary, f'data/processed/danewsroom/{type}/summary/{name}/chunk_{i // chunk_size}.pt')


if __name__ == '__main__':
    chunk_and_tokenize('abstractive')
