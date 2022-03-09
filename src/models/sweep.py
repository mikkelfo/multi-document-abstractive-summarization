import torch
import os
from utils import process_chunk, validate
import wandb
from torch.cuda.amp import autocast
from model import ProphetNetAutocast

''' CONSTANTS '''
BATCH_SIZE = 4
EPOCHS = 1
TOKEN_LENGTH = 350
N_CHUNKS = 50
AGGR_LOGGING_STEP = 5
CHECKPOINTING_STEP = 50
VALIDATION_LOGGING_STEP = 25

''' WANDB'''
wandb.init(project="abstractive-summarization-sweep", entity="mikkelfo")
wandb.config.learning_rate = 0.001
wandb.config.momentum = 0.9
wandb.config.gradient_accumulation_steps = 16

''' INITIALIZATION '''
model = ProphetNetAutocast(freeze_layers=False)
optimizer = torch.optim.SGD(model.parameters(), lr=wandb.config.learning_rate, momentum=wandb.config.momentum)

# For model checkpointing
if not os.path.isdir('checkpoints'):
    os.mkdir('checkpoints')
run_num = len(os.listdir('checkpoints'))

''' WANDB '''
wandb.watch(model)


for epoch in range(EPOCHS):
    epoch_loss = 0
    aggr_loss = 0

    for chunk_idx in range(N_CHUNKS):
        train_loss = 0
        for batch_idx, batch in enumerate(process_chunk(chunk_idx, TOKEN_LENGTH, BATCH_SIZE, 'train')):
            input_ids, attention_mask, labels = batch

            with autocast():
                loss = model(input_ids=input_ids, attention_mask = attention_mask, labels = labels)
                loss.backward()

            # Gradient accumulation
            if (batch_idx + 1) % wandb.config.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            train_loss = loss.detach()

        # Cleans up after chunk 
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        # Logging aggregated chunk loss
        aggr_loss += train_loss
        if (chunk_idx + 1) % AGGR_LOGGING_STEP:
            wandb.log({'5 Chunk train loss': aggr_loss})
            aggr_loss = 0

        wandb.log({'Train loss': train_loss})
        epoch_loss += train_loss

        # Checkpointing every 50 steps
        # if (chunk_idx + 1) % 50:
        #     torch.save(model.state_dict(), f'checkpoints/run-{run_num}_epoch{epoch}_step{chunk_idx}')

    # torch.save(model.state_dict(), f'checkpoints/run-{run_num}_epoch{epoch}_end')
    wandb.log({'Epoch train loss': epoch_loss})

    torch.cuda.empty_cache()

    validation_loss = validate(model, TOKEN_LENGTH, BATCH_SIZE)
    wandb.log({'Epoch validation loss': validation_loss})


    
