import os
from torch.cuda.amp import autocast
from utils import process_chunk, get_chunk_size
import torch
import wandb
from validate import validate


def train(model, optimizer, args):
    model.train()
    N_CHUNKS_TRAIN = len(os.listdir(f'data/processed/{args.dir}/text/train'))
    for epoch in range(args.epochs):
        for chunk_idx in range(N_CHUNKS_TRAIN):
            log_step = (epoch*N_CHUNKS_TRAIN) + chunk_idx + 1   # +1 since we start counting from 1
            chunk_loss = 0
            
            for batch_idx, batch in enumerate(process_chunk('train', chunk_idx, args)):
                input_ids, attention_mask, labels = batch
                with autocast():
                    loss = model(input_ids=input_ids, attention_mask = attention_mask, labels = labels)
                loss.backward()
                chunk_loss += loss.detach()

                # Every 512 samples we step
                if (batch_idx + 1) % args.gradient_accumulation_step == 0:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()

            # Report chunk loss per article
            chunk_size = get_chunk_size('train', chunk_idx, args)
            wandb.log({'Train loss': chunk_loss / chunk_size}, step=log_step)

            # Checkpoint and validate
            if (chunk_idx + 1) % args.checkpointing == 0:
                torch.save(model.state_dict(), f'checkpoints/{wandb.run.name}/epoch{epoch}_step{chunk_idx+1}.pt')
                validation_loss = validate(model, args)
                wandb.log({'Validation loss': validation_loss}, step=log_step)

        # Save model end of epoch and validate
        torch.save(model.state_dict(), f'checkpoints/{wandb.run.name}/epoch{epoch}_end.pt')
        validation_loss = validate(model, args)
        wandb.log({'Validation loss': validation_loss}, step=log_step)
    
