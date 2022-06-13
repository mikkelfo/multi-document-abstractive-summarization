import json
import pandas as pd
import torch
from transformers import ProphetNetTokenizer, XLMProphetNetTokenizer
from utils import read_jsonl_gz, prepare_directory

def chunk_and_tokenize_json(dir, cased, chunk_size=512):
    prepare_directory(dir)
    if cased:
        tokenizer = XLMProphetNetTokenizer.from_pretrained("microsoft/xprophetnet-large-wiki100-cased")
    else:
        tokenizer = ProphetNetTokenizer.from_pretrained('microsoft/prophetnet-large-uncased')
    for name in ['train', 'test', 'validation']:
        with open(f'data/processed/{dir.split("_")[0]}/{name}.json', 'r') as file:
            data = json.load(file)        

        for i in range(0, len(data), chunk_size):
            subset = data[i:(i+chunk_size)]
            text, summary = zip(*subset)

            tokenized_text = tokenizer([t.lower() if not cased else t for t in text], return_tensors="pt", padding=True, truncation=True)
            tokenized_summary = tokenizer([s.lower() if not cased else s for s in summary], return_tensors="pt", padding=True, truncation=True)

            torch.save(tokenized_text, f'data/processed/{dir}/text/{name}/chunk_{i // chunk_size}.pt')
            torch.save(tokenized_summary, f'data/processed/{dir}/summary/{name}/chunk_{i // chunk_size}.pt')


def chunk_and_tokenize_df(dir, cased, chunk_size=512):
    prepare_directory(dir)
    if cased:
        tokenizer = XLMProphetNetTokenizer.from_pretrained("microsoft/xprophetnet-large-wiki100-cased")
    else:
        tokenizer = ProphetNetTokenizer.from_pretrained('microsoft/prophetnet-large-uncased')
    for name in ['train', 'test', 'validation']:
        df = pd.read_csv(f'data/processed/{dir}/{name}.csv')

        for i in range(0, len(df), chunk_size):
            subset = df[i:(i+chunk_size)]
            text = subset.text.to_list()
            summary = subset.summary.to_list()

            tokenized_text = tokenizer([t.lower() if not cased else t for t in text], return_tensors="pt", padding=True, truncation=True)
            tokenized_summary = tokenizer([s.lower() if not cased else s for s in summary], return_tensors="pt", padding=True, truncation=True)

            torch.save(tokenized_text, f'data/processed/{dir}/text/{name}/chunk_{i // chunk_size}.pt')
            torch.save(tokenized_summary, f'data/processed/{dir}/summary/{name}/chunk_{i // chunk_size}.pt')


def chunk_and_tokenize_wcep(dir, cased, chunk_size=64):
    prepare_directory(dir)
    if cased:
        tokenizer = XLMProphetNetTokenizer.from_pretrained("microsoft/xprophetnet-large-wiki100-cased")
    else:
        tokenizer = ProphetNetTokenizer.from_pretrained('microsoft/prophetnet-large-uncased')
    for name in ['train', 'test', 'validation']:
        with open(f'data/processed/{dir.split("_")[0]}/{name}.json', 'r') as file:
            data = json.load(file)

        for i in range(0, len(data), chunk_size):
            subset = data[i:(i+chunk_size)]
            chunk_text, chunk_summary = [], []
            for summary, text in subset:
                tokenized_text = tokenizer([t.lower() if not cased else t for t in text], return_tensors="pt", padding=True, truncation=True)
                tokenized_summary = tokenizer(summary.lower() if cased else summary, return_tensors="pt", padding=True, truncation=True)

                chunk_text.append(tokenized_text)
                chunk_summary.append(tokenized_summary)

            torch.save(chunk_text, f'data/processed/{dir}/text/{name}/chunk_{i // chunk_size}.pt')
            torch.save(chunk_summary, f'data/processed/{dir}/summary/{name}/chunk_{i // chunk_size}.pt')


if __name__ == '__main__':
    # chunk_and_tokenize_json('cnn-dm', cased=False)
    # chunk_and_tokenize_json('cnn-dm_cased', cased=True)
    # chunk_and_tokenize_df('danewsroom/abstractive', cased=True)
    chunk_and_tokenize_wcep('wcep', cased=False)
    # chunk_and_tokenize_wcep('wcep_cased', cased=True)

