import torch
import os
import wandb
from model import ProphetNetMulti
from utils import target_summaries, validate_wcep
from torch.cuda.amp import autocast


''' CONSTANTS '''
EPOCHS = 10
TOKEN_LENGTH = 275
GRADIENT_ACCUMULATION_STEP = 32
CHECKPOINTING_STEP = 1024
TRAIN_LOG_STEP = 64
CLUSTER_SIZE = 16
N_CLUSTERS_TRAIN = len(os.listdir('data/processed/wcep/text/train'))
N_CLUSTERS_VAL = len(os.listdir('data/processed/wcep/text/validation'))

''' INITIALIZATION '''
model = ProphetNetMulti()
optimizer = torch.optim.Adam(model.get_params(), lr=1e-4, weight_decay=0.01)

# For model checkpointing
if not os.path.isdir('checkpoints'):
    os.mkdir('checkpoints')
run_num = len(next(os.walk('checkpoints'))[1])
if not os.path.isdir(f'checkpoints/{run_num}'):
    os.mkdir(f'checkpoints/{run_num}')

''' WANDB '''
wandb.init(project="MDS-runs", entity="mikkelfo")
wandb.watch(model)

model.train()

''' PRODUCE TARGET SUMMARIES'''
train_targets = torch.load('data/train_targets.pt')

for epoch in range(EPOCHS):
    epoch_loss = 0
    aggr_loss = 0

    for idx in range(N_CLUSTERS_TRAIN):
        cluster = torch.load(f'data/processed/wcep/text/train/cluster_{idx}.pt').to('cuda')

        input_ids = cluster.input_ids[:CLUSTER_SIZE, :TOKEN_LENGTH]
        attention_mask = cluster.attention_mask[:CLUSTER_SIZE, :TOKEN_LENGTH]
        target = input_ids[train_targets[idx]]

        with autocast():
            loss = model(input_ids=input_ids, attention_mask=attention_mask, target=target)
        loss.backward()

        if (idx + 1) % GRADIENT_ACCUMULATION_STEP == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        c_loss = loss.detach() / len(input_ids)

        wandb.log({'Train loss': c_loss}, step=(epoch*N_CLUSTERS_TRAIN)+idx)
        aggr_loss += c_loss
        epoch_loss += c_loss

        # Logging train loss every 64 chunks
        if (idx + 1) % TRAIN_LOG_STEP == 0:
            wandb.log({f'Train loss {TRAIN_LOG_STEP}': aggr_loss / TRAIN_LOG_STEP}, step=(epoch*N_CLUSTERS_TRAIN)+idx)
            aggr_loss = 0
        
        if (idx + 1) % CHECKPOINTING_STEP == 0:
            torch.save(model.state_dict(), f'checkpoints/{run_num}/epoch{epoch}_step{idx}')
            validation_loss = validate_wcep(model, TOKEN_LENGTH, CLUSTER_SIZE)
            wandb.log({'Validation loss': validation_loss / N_CLUSTERS_VAL}, step=(epoch*N_CLUSTERS_TRAIN)+idx)
        
    torch.save(model.state_dict(), f'checkpoints/{run_num}/epoch{epoch}_end')
    wandb.log({'Epoch train loss': epoch_loss / N_CLUSTERS_TRAIN}, step=(epoch*N_CLUSTERS_TRAIN)+idx)