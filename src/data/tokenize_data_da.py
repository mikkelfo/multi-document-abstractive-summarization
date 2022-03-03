from transformers import ProphetNetTokenizer
import pandas as pd
import torch
import string
import os


def tokenize_danewsroom(file, shuffle=True):
    df = pd.read_csv(file)
    # Remove "empty" summaries
    df = df[df.summary.str.translate(str.maketrans('', '', string.punctuation)) != '']

    tokenizer = ProphetNetTokenizer.from_pretrained('microsoft/prophetnet-large-uncased')

    # Split into the 3 subsets (key, dataframe)
    for name, dataframe in df.groupby('density_bin'):
        # Shuffles dataframe
        if shuffle:
            dataframe = dataframe.sample(frac=1)
        text = dataframe.text.to_list()
        summary = dataframe.summary.to_list()

        tokenized_text = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        tokenized_summary = tokenizer(summary, return_tensors="pt", padding=True, truncation=True)

        torch.save(tokenized_text, f'data/processed/tokenized/danewsroom/{name}/text.pt')
        torch.save(tokenized_summary, f'data/processed/tokenized/danewsroom/{name}/summary.pt')


def chunk_tokenized_data(path, chunk_size=1000):
    for subset in os.listdir(path):
        dir = path + "/" + subset
        for form in ['summary', 'text']:
            # Create folder
            if not os.path.isdir(dir + "/" + form):
                os.mkdir(dir + "/" + form)
            # Load file
            file = dir + "/" + form + ".pt"
            dic = torch.load(file)
            vals = list(dic.values())

            for i in range(0, len(vals[0]), chunk_size):
                chunk = {}
                for j, key in enumerate(dic.keys()):
                    chunk[key] = vals[j][i:(i+chunk_size)].clone()
                torch.save(chunk, f'{dir}/{form}/chunk_{i // chunk_size}.pt')


if __name__ == '__main__':
    # tokenize_danewsroom('data/raw/danewsroom.csv')
    chunk_tokenized_data('data/processed/tokenized/danewsroom')
