import os
import pickle
import random
import torch
from transformers import ProphetNetTokenizer
from pathlib import Path

def tokenize_chunks(dirs: list, shuffle=True):
    text, summary = [], []

    for dir in dirs:
        files = list(Path(dir).glob('*.pickle'))
        for file in files:
            with open(file, 'rb') as f:
                dic = pickle.load(f)
            stories, highlights = dic.values()
            text += stories
            summary += highlights

    if shuffle:
        zipped = list(zip(text, summary))
        random.shuffle(zipped)
        text, summary = zip(*zipped)

    tokenizer = ProphetNetTokenizer.from_pretrained('microsoft/prophetnet-large-uncased')    
    
    tokenized_text = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    tokenized_summary = tokenizer(summary, return_tensors="pt", padding=True, truncation=True)

    torch.save(tokenized_text, 'data/processed/tokenized/text.pt')
    torch.save(tokenized_summary, 'data/processed/tokenized/summary.pt')

def chunk_tokenized_data(file, chunk_size=1000):
    type = file.split("/")[-1].split(".")[0]    # summary/text
    dic = torch.load(file)
    vals = list(dic.values())

    for i in range(0, len(vals[0]), chunk_size):
        chunk = {}
        for j, key in enumerate(dic.keys()):
            chunk[key] = vals[j][i:(i+chunk_size)].clone()
        torch.save(chunk, f'data/processed/tokenized/{type}/chunk_{i // chunk_size}.pt')

if __name__ == '__main__':
    # tokenize_chunks(['data/raw/chunks/cnn', 'data/raw/chunks/dailymail'])
    # chunk_tokenized_data('data/processed/tokenized/summary.pt')
    chunk_tokenized_data('data/processed/tokenized/text.pt')
