import torch
from torch.cuda.amp import autocast
import os


def process_chunk(chunk_idx, token_length, batch_size, split):
    summary = torch.load(f'data/processed/cnn-dm/summary/{split}/chunk_{chunk_idx}.pt')
    text = torch.load(f'data/processed/cnn-dm/text/{split}/chunk_{chunk_idx}.pt')

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
            for batch in process_chunk(chunk_idx, TOKEN_LENGTH, BATCH_SIZE, 'validation'):
                input_ids, attention_mask, labels = batch
                with autocast():
                    loss = model(input_ids=input_ids, attention_mask = attention_mask, labels = labels)
                val_loss += loss.detach()
    model.train()
    return val_loss


def validate_da(model, TOKEN_LENGTH, BATCH_SIZE):
    val_loss = 0
    model.eval()
    with torch.no_grad():
        for chunk_idx in range(len(os.listdir('data/processed/danewsroom/abstractive/text/validation'))):
            for batch in process_chunk_da(chunk_idx, TOKEN_LENGTH, BATCH_SIZE, 'validation'):
                input_ids, attention_mask, labels = batch
                with autocast():
                    loss = model(input_ids=input_ids, attention_mask = attention_mask, labels = labels)
                val_loss += loss.detach()
    model.train()
    return val_loss

