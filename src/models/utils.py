import torch
import os
import gzip, json


def process_chunk(split, chunk_idx, args):
    summary = torch.load(f'data/processed/{args.dir}/summary/{split}/chunk_{chunk_idx}.pt')
    text = torch.load(f'data/processed/{args.dir}/text/{split}/chunk_{chunk_idx}.pt')

    input_ids, attention_mask, = text['input_ids'][:, :args.token_length].to('cuda'), text['attention_mask'][:, :args.token_length].to('cuda')
    labels = summary['input_ids'][:, :args.token_length].to('cuda')
    labels = labels.masked_fill_(labels == 0, -100)

    N = len(input_ids)  # 512
    for i in range(0, N, args.batch_size):
        batch = input_ids[i:(i+args.batch_size)], attention_mask[i:(i+args.batch_size)], labels[i:(i+args.batch_size)]
        yield batch


def get_chunk_size(split, chunk_idx, args):
    chunk = torch.load(f'data/processed/{args.dir}/summary/{split}/chunk_{chunk_idx}.pt')
    return len(chunk.input_ids)


def read_jsonl_gz(path):
    with gzip.open(path) as f:
        for l in f:
            yield json.loads(l)


def target_summaries(split):
    # VM's doesnt like this import
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    vectorizer = TfidfVectorizer()
    data = list(read_jsonl_gz(f'data/raw/wcep/{split}.jsonl.gz'))

    targets = []
    for cluster in data:
        articles = cluster['articles']
        text = [a['text'] for a in articles][:16]
        X = vectorizer.fit_transform(text)
        targets.append(cosine_similarity(X).sum(1).argmax())

    return targets


def concat_chunks(dir):
    input_ids = torch.tensor([])
    attention_mask = torch.tensor([])
    for i in range(len(os.listdir(dir))):
        chunk = torch.load(f'{dir}/chunk_{i}.pt')
        input_ids = torch.cat((input_ids, chunk.input_ids))
        attention_mask = torch.cat((attention_mask, chunk.attention_mask))

    return list(zip(input_ids, attention_mask))
