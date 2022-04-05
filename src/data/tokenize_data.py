import json
import torch
from transformers import ProphetNetTokenizer, XLMProphetNetTokenizer
from utils import read_jsonl_gz, prepare_directory

def chunk_and_tokenize_json(dir, chunk_size=512, xlm=False):
    prepare_directory(dir)
    if xlm:
        tokenizer = XLMProphetNetTokenizer.from_pretrained("microsoft/xprophetnet-large-wiki100-cased")
    else:
        tokenizer = ProphetNetTokenizer.from_pretrained('microsoft/prophetnet-large-uncased')
    for name in ['train', 'test', 'validation']:
        file = open(f'data/processed/cnn-dm/{name}.json', 'r')
        data = json.load(file)

        for i in range(0, len(data), chunk_size):
            subset = data[i:(i+chunk_size)]
            text, summary = zip(*subset)

            tokenized_text = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            tokenized_summary = tokenizer(summary, return_tensors="pt", padding=True, truncation=True)

            torch.save(tokenized_text, f'data/processed{dir}/text/{name}/chunk_{i // chunk_size}.pt')
            torch.save(tokenized_summary, f'data/processed{dir}/summary/{name}/chunk_{i // chunk_size}.pt')


def chunk_and_tokenize_wcep():
    tokenizer = ProphetNetTokenizer.from_pretrained('microsoft/prophetnet-large-uncased')
    for split in ['train', 'test', 'validation']:
        data = list(read_jsonl_gz(f'data/raw/wcep/{split}.jsonl.gz'))
        # First tokenize the summaries (no chunking)
        summaries = [x['summary'] for x in data]
        output = tokenizer(summaries, return_tensors="pt", padding=True, truncation=True)
        torch.save(output, f'../../data/processed/wcep/summary/{split}/summary.pt')
        # Next tokenize and chunk the articles. One chunk = one cluster
        for i, cluster in enumerate(data):
            articles = cluster['articles']
            text = [a['text'] for a in articles]
            output = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            torch.save(output, f'data/processed/wcep/text/{split}/cluster_{i}.pt')


if __name__ == '__main__':
    chunk_and_tokenize_json('cnn-dm')
    # chunk_and_tokenize_multi()
