import torch
from transformers import ProphetNetForConditionalGeneration
import os
from utils import process_chunk
import wandb

''' CONSTANTS '''
BATCH_SIZE = 4
EPOCHS = 10
TOKEN_LENGTH = 400
GRADIENT_ACCUMULATION_STEPS = 16
N_CHUNKS = len(os.listdir('data/processed/tokenized/cnn-dm/summary'))


''' INITIALIZATION '''
model = ProphetNetForConditionalGeneration.from_pretrained('microsoft/prophetnet-large-uncased')
model = torch.nn.DataParallel(model)
model.to('cuda')
for param in list(model.parameters())[:-1]:
    param.requires_grad = False
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
            input_ids, attention_mask, labels = batch
            loss = model(input_ids=input_ids, attention_mask = attention_mask, labels = labels).loss.sum()
            loss.backward()

            # Gradient accumulation
            if batch_idx % GRADIENT_ACCUMULATION_STEPS == 0 and batch_idx != 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            chunk_loss = loss.item()
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
