import torch
import os
from utils import process_chunk, get_chunk_size
from torch.cuda.amp import autocast


def validate(model, args):
    val_loss = 0
    model.eval()
    with torch.no_grad():
        N_CHUNKS_VALIDATION = len(os.listdir('data/processed/cnn-dm/text/validation'))
        N = 0
        for chunk_idx in range(N_CHUNKS_VALIDATION):
            for batch in process_chunk('validation', chunk_idx, args):
                input_ids, attention_mask, labels = batch
                with autocast():
                    loss = model(input_ids=input_ids, attention_mask = attention_mask, labels = labels).loss.sum()
                val_loss += loss.detach()
                N += len(input_ids)

    model.train()

    return val_loss / N

