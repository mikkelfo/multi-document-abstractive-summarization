import torch
import os
from utils import process_chunk
import wandb
from torch.cuda.amp import autocast
from model import ProphetNetAutocast

''' CONSTANTS '''
BATCH_SIZE = 4
EPOCHS = 10
TOKEN_LENGTH = 350
N_CHUNKS = len(os.listdir('data/processed/tokenized/cnn-dm/summary'))
HYPERPARAM_DEFAULTS = dict(
    learning_rate = 0.001,
    momentum = 0.9,
    gradient_accumulation_steps = 16
)


''' WANDB'''
wandb.init(project="abstractive-summarization", entity="mikkelfo", config=HYPERPARAM_DEFAULTS)

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
    print(f'*****     EPOCH {epoch}     *****')
    epoch_loss = 0
    # for each file
    for chunk_idx in range(50):
        chunk_loss = 0
        for batch_idx, batch in enumerate(process_chunk(chunk_idx, TOKEN_LENGTH, BATCH_SIZE)):
            # batch = [r.to('cuda') for r in batch]
            input_ids, attention_mask, labels = batch

            with autocast():
                loss = model(input_ids=input_ids, attention_mask = attention_mask, labels = labels)
                loss.backward()

            # Gradient accumulation
            if batch_idx % wandb.config.gradient_accumulation_steps == 0 and batch_idx != 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            chunk_loss = loss.detach()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if chunk_idx % 10 == 0 and chunk_idx != 0:
            print(f'     Chunk {chunk_idx} loss: {chunk_loss}')

        wandb.log({'Chunk loss': chunk_loss})
        epoch_loss += chunk_loss

    wandb.log({'Epoch loss': epoch_loss})
    print(f'----------     Epoch {epoch} loss: {epoch_loss}     ----------')
    print()

    torch.save(model.state_dict(), f'checkpoints/run-{run_num}_epoch{epoch}')
