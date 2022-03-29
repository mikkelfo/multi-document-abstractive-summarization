import torch
import os
from utils import process_chunk_multi, validate_multi
import wandb
from torch.cuda.amp import autocast
from model import ProphetNetAutocast

''' CONSTANTS '''
EPOCHS = 10
BATCH_SIZE = 2
TOKEN_LENGTH = 275
N_CHUNKS = len(os.listdir('data/processed/cnn-dm-multi/summary/train'))
N_CHUNKS_VALIDATION = len(os.listdir('data/processed/cnn-dm-multi/text/validation'))
GRADIENT_ACCUMULATION_STEP = 256
CHECKPOINTING_STEP = 50
TRAIN_LOG_STEP = 5

''' INITIALIZATION '''
model = ProphetNetAutocast(language='da', freeze_layers=False)
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.01)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

# For model checkpointing
if not os.path.isdir('checkpoints'):
    os.mkdir('checkpoints')
run_num = len(next(os.walk('checkpoints'))[1])
if not os.path.isdir(f'checkpoints/{run_num}'):
    os.mkdir(f'checkpoints/{run_num}')

''' WANDB '''
wandb.init(project="abstractive-summarization-runs", entity="mikkelfo")
wandb.watch(model)

model.train()
for epoch in range(EPOCHS):
    epoch_loss = 0
    aggr_loss = 0
    for chunk_idx in range(N_CHUNKS):
        train_loss = 0
        N = 0
        for batch_idx, batch in enumerate(process_chunk_multi(chunk_idx, TOKEN_LENGTH, BATCH_SIZE, 'train')):
            input_ids, attention_mask, labels = batch
            with autocast():
                loss = model(input_ids=input_ids, attention_mask = attention_mask, labels = labels)
            loss.backward()

            # Gradient accumulation
            if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEP == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
            N += len(input_ids)
            train_loss += loss.detach()
        train_loss /= N
        wandb.log({'Train loss': train_loss}, step=(epoch*N_CHUNKS)+chunk_idx)
        aggr_loss += train_loss
        epoch_loss += train_loss

        # Logging train loss every 5 chunks
        if (chunk_idx + 1) % TRAIN_LOG_STEP == 0:
            wandb.log({f'Train loss {TRAIN_LOG_STEP}': aggr_loss / TRAIN_LOG_STEP}, step=(epoch*N_CHUNKS)+chunk_idx)
            aggr_loss = 0

        # Checkpointing and validation every 50 steps
        if (chunk_idx + 1) % CHECKPOINTING_STEP == 0:
            torch.save(model.state_dict(), f'checkpoints/{run_num}/epoch{epoch}_step{chunk_idx}')
            validation_loss = validate_multi(model, TOKEN_LENGTH, BATCH_SIZE)
            wandb.log({'Validation loss': validation_loss / N_CHUNKS_VALIDATION}, step=(epoch*N_CHUNKS)+chunk_idx)

    torch.save(model.state_dict(), f'checkpoints/{run_num}/epoch{epoch}_end')
    wandb.log({'Epoch train loss': epoch_loss / N_CHUNKS}, step=(epoch*N_CHUNKS)+chunk_idx)
    
    
