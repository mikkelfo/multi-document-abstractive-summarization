import torch
import os
from utils import process_chunk, get_chunk_size
from torch.cuda.amp import autocast


def validate(model, args):
    val_loss = 0
    N_CHUNKS_VALIDATION = len(os.listdir('data/processed/cnn-dm/text/validation'))
    model.eval()
    with torch.no_grad():
        for chunk_idx in range(N_CHUNKS_VALIDATION):
            chunk_loss = 0
            chunk_size = get_chunk_size('validation', chunk_idx, args)
            for batch in process_chunk('validation', chunk_idx, args):
                input_ids, attention_mask, labels = batch
                with autocast():
                    loss = model(input_ids=input_ids, attention_mask = attention_mask, labels = labels).loss.mean()
                chunk_loss += loss.item()
            val_loss += chunk_loss / (chunk_size / args.batch_size) 
            

    model.train()

    return val_loss / N_CHUNKS_VALIDATION

