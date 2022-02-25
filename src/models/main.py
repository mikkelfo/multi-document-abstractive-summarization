import torch
import os
from utils import process_chunk
import wandb
from torch.cuda.amp import autocast
from model import ProphetNetAutocast

''' CONSTANTS '''
BATCH_SIZE = 4
EPOCHS = 10
TOKEN_LENGTH = 400
GRADIENT_ACCUMULATION_STEPS = 16
N_CHUNKS = len(os.listdir('data/processed/tokenized/summary'))


''' INITIALIZATION '''
model = ProphetNetAutocast()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# For model checkpointing
if not os.path.isdir('checkpoints'):
    os.mkdir('checkpoints')
run_num = len(os.listdir('checkpoints'))

''' WANDB '''
wandb.init(project="abstractive-summarization", entity="mikkelfo")
wandb.watch(model)


for epoch in range(EPOCHS):
    print(f'*****     EPOCH {epoch}     *****')
    epoch_loss = 0
    # for each file
    for chunk_idx in range(N_CHUNKS):
        chunk_loss = 0
        for batch_idx, batch in enumerate(process_chunk(chunk_idx, TOKEN_LENGTH, BATCH_SIZE)):
            # batch = [r.to('cuda') for r in batch]
            input_ids, attention_mask, labels = batch

            with autocast():
                loss = model(input_ids=input_ids, attention_mask = attention_mask, labels = labels)
                loss.backward()

            # Gradient accumulation
            if batch_idx % GRADIENT_ACCUMULATION_STEPS == 0 and batch_idx != 0:
                optimizer.step()
                # optimizer.zero_grad()
                # The below code should be more efficient than optimizer.zero_grad()
                for param in model.parameters():
                    param.grad = None

            chunk_loss = loss.detach()
        optimizer.step()
        # optimizer.zero_grad()
        # The below code should be more efficient than optimizer.zero_grad()
        for param in model.parameters():
            param.grad = None

        if chunk_idx % 10 == 0 and chunk_idx != 0:
            print(f'     Chunk {chunk_idx} loss: {chunk_loss}')

        wandb.log({'Chunk loss': chunk_loss})
        epoch_loss += chunk_loss

    wandb.log({'Epoch loss': epoch_loss})
    print(f'----------     Epoch {epoch} loss: {epoch_loss}     ----------')
    print()

    torch.save(model.state_dict(), f'checkpoints/run-{run_num}_epoch{epoch}')
