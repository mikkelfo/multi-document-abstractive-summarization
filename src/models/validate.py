import torch
import os
from utils import process_chunk, custom_forward_mds
from torch.cuda.amp import autocast


def validate(model, args):
    val_loss = 0
    N_CHUNKS_VALIDATION = len(os.listdir(f'data/processed/{args.dir}/text/validation'))
    if args.unsupervised:
        targets = iter(torch.load('data/processed/wcep/wcep_validation_targets.pt'))
    model.eval()
    with torch.no_grad():
        for chunk_idx in range(N_CHUNKS_VALIDATION):
            chunk_loss = 0

            for batch_idx, batch in enumerate(process_chunk('validation', chunk_idx, args)):
                input_ids, attention_mask, labels = batch
                if args.unsupervised:
                    target_index = next(targets)
                    labels = input_ids[target_index].unsqueeze(0)                                   # Assign labels to the target summary
                    labels = labels.masked_select(labels != 0)                                      # Remove padding (Can be done because batch size is always 1)
                    input_ids = torch.cat((input_ids[:target_index], input_ids[target_index+1:]))   # Remove target summary from input
                    attention_mask = torch.cat((attention_mask[:target_index], attention_mask[target_index+1:]))   # Remove target summary from mask
                with autocast():
                    if args.mds:
                        loss = custom_forward_mds(model, input_ids, attention_mask, labels, args).loss.mean()
                    else:
                        loss = model(input_ids=input_ids, attention_mask = attention_mask, labels = labels, use_cache=False).loss.mean()
                chunk_loss += loss.item()
            val_loss += chunk_loss / (batch_idx + 1)
            

    model.train()

    return val_loss / N_CHUNKS_VALIDATION

