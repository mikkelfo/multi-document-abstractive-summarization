import os
from torch.cuda.amp import autocast, GradScaler
from utils import process_chunk, custom_forward_mds
import torch
import wandb
from validate import validate
import random
from generation import single_generation


def train(model, optimizer, scheduler, args):
    model.train()
    scaler = GradScaler()
    N_CHUNKS_TRAIN = len(os.listdir(f'data/processed/{args.dir}/text/train'))
    for epoch in range(args.epochs):
        chunk_indices = list(range(N_CHUNKS_TRAIN))
        if args.shuffle:
            random.shuffle(chunk_indices)
        for i, chunk_idx in enumerate(chunk_indices):
            log_step = (epoch*N_CHUNKS_TRAIN) + i + 1   # +1 since we start counting from 1
            chunk_loss = 0

            for batch_idx, batch in enumerate(process_chunk('train', chunk_idx, args)):
                input_ids, attention_mask, labels = batch
                with autocast():
                    if args.mds:
                        loss = custom_forward_mds(model, input_ids, attention_mask, labels, args).loss.mean()
                    else:
                        loss = model(input_ids=input_ids, attention_mask = attention_mask, labels = labels, use_cache=False).loss.mean()
                scaler.scale(loss).backward()
                # loss.backward()
                chunk_loss += loss.item()

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            scaler.step()
            # optimizer.step()
            scheduler.step()
            # optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()

            # Report chunk loss per article
            wandb.log({'Train loss': chunk_loss / (batch_idx + 1)}, step=log_step)

            # Checkpoint and validate
            if (i + 1) % args.checkpointing == 0:
                torch.save(model.state_dict(), f'checkpoints/{wandb.run.name}/epoch{epoch}_step{log_step}.pt')
                validation_loss = validate(model, args)
                wandb.log({'Validation loss': validation_loss}, step=log_step)
                single_generation(model, args, log_step)

        # Save model end of epoch and validate
        torch.save(model.state_dict(), f'checkpoints/{wandb.run.name}/epoch{epoch}_end.pt')
        validation_loss = validate(model, args)
        wandb.log({'Validation loss': validation_loss}, step=log_step)
        single_generation(model, args, log_step)
    
