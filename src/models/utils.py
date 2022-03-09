import torch
from torch.cuda.amp import autocast
import os

from model import ProphetNetAutocast

def process_chunk(chunk_idx, token_length, batch_size, split):
    summary = torch.load(f'data/processed/cnn-dm/summary/{split}/chunk_{chunk_idx}.pt')
    text = torch.load(f'data/processed/cnn-dm/text/{split}/chunk_{chunk_idx}.pt')

    input_ids, attention_mask, = text['input_ids'][:, :token_length].to('cuda'), text['attention_mask'][:, :token_length].to('cuda')
    decoder_input_ids, _ = summary['input_ids'][:, :token_length].to('cuda'), summary['attention_mask'][:, :token_length].to('cuda')

    N = len(input_ids)  # 1024
    for i in range(0, N, batch_size):
        batch = input_ids[i:(i+batch_size)], attention_mask[i:(i+batch_size)], decoder_input_ids[i:(i+batch_size)]
        yield batch


def process_chunk_da(chunk_idx, token_length, batch_size):
    summary = torch.load(f'data/processed/tokenized/danewsroom/abstractive/summary/chunk_{chunk_idx}.pt')
    text = torch.load(f'data/processed/tokenized/danewsroom/abstractive/text/chunk_{chunk_idx}.pt')

    input_ids, attention_mask, = text['input_ids'][:, :token_length].to('cuda'), text['attention_mask'][:, :token_length].to('cuda')
    decoder_input_ids, _ = summary['input_ids'][:, :token_length].to('cuda'), summary['attention_mask'][:, :token_length].to('cuda')

    N = len(input_ids)  # 1024
    for i in range(0, N, batch_size):
        batch = input_ids[i:(i+batch_size)], attention_mask[i:(i+batch_size)], decoder_input_ids[i:(i+batch_size)]
        yield batch


def validate(model, TOKEN_LENGTH, BATCH_SIZE):
    val_loss = 0
    model.eval()
    with torch.no_grad():
        for chunk_idx in range(len(os.listdir('data/processed/cnn-dm/text/train'))):
            for batch in process_chunk(chunk_idx, TOKEN_LENGTH, BATCH_SIZE, 'train'):
                input_ids, attention_mask, labels = batch
                with autocast():
                    loss = model(input_ids=input_ids, attention_mask = attention_mask, labels = labels)
                val_loss += loss.detach()
            break
    model.train()
    return val_loss

if __name__ == '__main__':
    model = ProphetNetAutocast(freeze_layers=False)
    loss = validate(model, 350, 4)
    print(loss)

