import torch
import os
import gzip, json


def custom_forward_mds(model, input_ids, attention_mask, labels, args):
    enc_output = model.prophetnet.encoder(input_ids=input_ids, attention_mask=attention_mask)

    if args.method == 'cat':
        enc_output.last_hidden_state = enc_output.last_hidden_state[attention_mask.bool()].unsqueeze(0)
    elif args.method == 'mean':
        enc_output.last_hidden_state = enc_output.last_hidden_state.mean(1).unsqueeze(0)
    elif args.method == 'serial':
        raise Exception('Not implemented yet')
    elif args.method == 'sds':
        output = model(encoder_outputs=enc_output, attention_mask=attention_mask, labels=labels, use_cache=False)
        return output
    output = model(encoder_outputs=enc_output, labels=labels, use_cache=False)
    return output


def process_chunk(split, chunk_idx, args):
    summary = torch.load(f'data/processed/{args.dir}/summary/{split}/chunk_{chunk_idx}.pt', map_location=torch.device('cuda'))
    text = torch.load(f'data/processed/{args.dir}/text/{split}/chunk_{chunk_idx}.pt', map_location=torch.device('cuda'))

    if args.mds:
        N = len(summary)  # ~64
        for i in range(N):
            input_ids = text[i].input_ids[:, :args.token_length]
            attention_mask = text[i].attention_mask[:, :args.token_length]
            labels = summary[i].input_ids[:, :args.token_length]
            labels = labels.masked_fill_(labels == 0, -100)
            if args.method == 'sds':
                labels = labels.expand(len(input_ids), -1)
            yield input_ids, attention_mask, labels

    else:
        input_ids, attention_mask, = text['input_ids'][:, :args.token_length], text['attention_mask'][:, :args.token_length]
        labels = summary['input_ids'][:, :args.token_length]
        labels = labels.masked_fill_(labels == 0, -100)

        N = len(input_ids)  # 512
        for i in range(0, N, args.batch_size):
            batch = input_ids[i:(i+args.batch_size)], attention_mask[i:(i+args.batch_size)], labels[i:(i+args.batch_size)]
            yield batch


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
