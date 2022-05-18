import torch
import os
from utils import process_chunk, custom_forward_mds
from torch.cuda.amp import autocast


def validate(model, args):
    val_loss = 0
    N_CHUNKS_VALIDATION = len(os.listdir(f'data/processed/{args.dir}/text/validation'))
    model.eval()
    with torch.no_grad():
        for chunk_idx in range(N_CHUNKS_VALIDATION):
            chunk_loss = 0

            for batch_idx, batch in enumerate(process_chunk('validation', chunk_idx, args)):
                input_ids, attention_mask, labels = batch
                with autocast():
                    if args.mds:
                        loss = custom_forward_mds(model, input_ids, attention_mask, labels, args).loss.mean()
                    else:
                        loss = model(input_ids=input_ids, attention_mask = attention_mask, labels = labels, use_cache=False).loss.mean()
                chunk_loss += loss.item()
            val_loss += chunk_loss / (batch_idx + 1)
            

    model.train()

    return val_loss / N_CHUNKS_VALIDATION

