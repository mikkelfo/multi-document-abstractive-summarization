import torch
from torch.cuda.amp import autocast
import os
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
import gzip, json


def process_chunk(chunk_idx, token_length, batch_size, split):
    summary = torch.load(f'data/processed/cnn-dm/summary/{split}/chunk_{chunk_idx}.pt')
    text = torch.load(f'data/processed/cnn-dm/text/{split}/chunk_{chunk_idx}.pt')

    input_ids, attention_mask, = text['input_ids'][:, :token_length], text['attention_mask'][:, :token_length]
    decoder_input_ids, _ = summary['input_ids'][:, :token_length], summary['attention_mask'][:, :token_length]

    N = len(input_ids)  # 1024
    for i in range(0, N, batch_size):
        batch = input_ids[i:(i+batch_size)], attention_mask[i:(i+batch_size)], decoder_input_ids[i:(i+batch_size)]
        yield batch


def process_chunk_multi(chunk_idx, token_length, batch_size, split):
    summary = torch.load(f'data/processed/cnn-dm-multi/summary/{split}/chunk_{chunk_idx}.pt')
    text = torch.load(f'data/processed/cnn-dm-multi/text/{split}/chunk_{chunk_idx}.pt')

    input_ids, attention_mask, = text['input_ids'][:, :token_length], text['attention_mask'][:, :token_length]
    decoder_input_ids, _ = summary['input_ids'][:, :token_length], summary['attention_mask'][:, :token_length]

    N = len(input_ids)  # 1024
    for i in range(0, N, batch_size):
        batch = input_ids[i:(i+batch_size)], attention_mask[i:(i+batch_size)], decoder_input_ids[i:(i+batch_size)]
        yield batch


def process_chunk_da(chunk_idx, token_length, batch_size, split):
    summary = torch.load(f'data/processed/danewsroom/abstractive/summary/{split}/chunk_{chunk_idx}.pt')
    text = torch.load(f'data/processed/danewsroom/abstractive/text/{split}/chunk_{chunk_idx}.pt')

    input_ids, attention_mask, = text['input_ids'][:, :token_length], text['attention_mask'][:, :token_length]
    decoder_input_ids, _ = summary['input_ids'][:, :token_length], summary['attention_mask'][:, :token_length]

    N = len(input_ids)  # 1024
    for i in range(0, N, batch_size):
        batch = input_ids[i:(i+batch_size)], attention_mask[i:(i+batch_size)], decoder_input_ids[i:(i+batch_size)]
        yield batch


def validate(model, TOKEN_LENGTH, BATCH_SIZE):
    val_loss = 0
    model.eval()
    with torch.no_grad():
        for chunk_idx in range(len(os.listdir('data/processed/cnn-dm/text/validation'))):
            N = 0
            for batch in process_chunk(chunk_idx, TOKEN_LENGTH, BATCH_SIZE, 'validation'):
                input_ids, attention_mask, labels = batch
                with autocast():
                    loss = model(input_ids=input_ids, attention_mask = attention_mask, labels = labels)
                val_loss += loss.detach()
                N += len(input_ids)
            val_loss /= N
    model.train()
    return val_loss


def validate_multi(model, TOKEN_LENGTH, BATCH_SIZE):
    val_loss = 0
    model.eval()
    with torch.no_grad():
        for chunk_idx in range(len(os.listdir('data/processed/cnn-dm-multi/text/validation'))):
            N = 0
            for batch in process_chunk_multi(chunk_idx, TOKEN_LENGTH, BATCH_SIZE, 'validation'):
                input_ids, attention_mask, labels = batch
                with autocast():
                    loss = model(input_ids=input_ids, attention_mask = attention_mask, labels = labels)
                val_loss += loss.detach()
                N += len(input_ids)
            val_loss /= N
    model.train()
    return val_loss


def validate_da(model, TOKEN_LENGTH, BATCH_SIZE):
    val_loss = 0
    model.eval()
    with torch.no_grad():
        for chunk_idx in range(len(os.listdir('data/processed/danewsroom/abstractive/text/validation'))):
            N = 0
            for batch in process_chunk_da(chunk_idx, TOKEN_LENGTH, BATCH_SIZE, 'validation'):
                input_ids, attention_mask, labels = batch
                with autocast():
                    loss = model(input_ids=input_ids, attention_mask = attention_mask, labels = labels)
                val_loss += loss.detach()
                N += len(input_ids)
            val_loss /= N
    model.train()
    return val_loss

def validate_wcep(model, TOKEN_LENGTH, CLUSTER_SIZE):
    val_targets = torch.load('data/validation_targets_8.pt')
    N_CLUSTERS_VAL = len(os.listdir('data/processed/wcep/text/validation'))
    val_loss = 0
    model.eval()
    with torch.no_grad():
        for idx in range(N_CLUSTERS_VAL):
            cluster = torch.load(f'data/processed/wcep/text/validation/cluster_{idx}.pt').to('cuda')
            input_ids = cluster.input_ids[:CLUSTER_SIZE, :TOKEN_LENGTH]
            attention_mask = cluster.attention_mask[:CLUSTER_SIZE, :TOKEN_LENGTH]
            target = input_ids[val_targets[idx]]
            with autocast():
                loss = model(input_ids=input_ids, attention_mask=attention_mask, target=target)
            val_loss += loss.detach() / len(input_ids)
    model.train()
    return val_loss

def read_jsonl_gz(path):
    with gzip.open(path) as f:
        for l in f:
            yield json.loads(l)


def target_summaries(split):
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
