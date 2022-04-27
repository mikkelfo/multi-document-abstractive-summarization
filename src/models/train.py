import os
from torch.cuda.amp import autocast
from utils import process_chunk, custom_forward_mds
import torch
import wandb
from validate import validate


def train(model, optimizer, args):
    model.train()
    N_CHUNKS_TRAIN = len(os.listdir(f'data/processed/{args.dir}/text/train'))
    for epoch in range(args.epochs):
        if args.unsupervised:
            targets = iter(torch.load('data/processed/wcep/wcep_train_targets.pt'))
        for chunk_idx in range(N_CHUNKS_TRAIN):
            log_step = (epoch*N_CHUNKS_TRAIN) + chunk_idx + 1   # +1 since we start counting from 1
            chunk_loss = 0

            for batch_idx, batch in enumerate(process_chunk('train', chunk_idx, args)):
                input_ids, attention_mask, labels = batch
                if args.unsupervised:
                    target_index = next(targets)
                    labels = input_ids[target_index].unsqueeze(0)                                   # Assign labels to the target summary
                    input_ids = torch.cat((input_ids[:target_index], input_ids[target_index+1:]))   # Remove target summary from input
                    attention_mask = torch.cat((attention_mask[:target_index], attention_mask[target_index+1:]))   # Remove target summary from mask
                with autocast():
                    if args.mds:
                        loss = custom_forward_mds(model, input_ids, attention_mask, labels, args).loss.mean()
                    else:
                        loss = model(input_ids=input_ids, attention_mask = attention_mask, labels = labels, use_cache=False).loss.mean()
                loss.backward()
                chunk_loss += loss.item()

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()

            # Report chunk loss per article
            wandb.log({'Train loss': chunk_loss / (batch_idx + 1)}, step=log_step)

            # Checkpoint and validate
            if (chunk_idx + 1) % args.checkpointing == 0:
                torch.save(model.state_dict(), f'checkpoints/{wandb.run.name}/epoch{epoch}_step{chunk_idx+1}.pt')
                validation_loss = validate(model, args)
                wandb.log({'Validation loss': validation_loss}, step=log_step)

        # Save model end of epoch and validate
        torch.save(model.state_dict(), f'checkpoints/{wandb.run.name}/epoch{epoch}_end.pt')
        validation_loss = validate(model, args)
        wandb.log({'Validation loss': validation_loss}, step=log_step)
    
