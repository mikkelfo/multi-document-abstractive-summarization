import os
import torch
from transformers import BertModel, ProphetNetEncoder
from torch.cuda.amp import autocast

from models.utils import process_chunk


def create_db(dir, token_length=275, batch_size=64):
    encoder = ProphetNetEncoder.from_pretrained("patrickvonplaten/prophetnet-large-uncased-standalone")

    database = torch.tensor([])
    with torch.no_grad():
        for chunk_idx in range(len(os.listdir(dir))):
            for batch in process_chunk(chunk_idx, token_length, batch_size, 'train'):
                input_ids, attention_mask, _ = batch
                with autocast():
                    output = encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
                database = torch.cat((database, output))
    return database


def create_db_bert(dir, token_length=512, batch_size=64):
    bert = BertModel.from_pretrained("bert-base-uncased")

    database = torch.tensor([])
    with torch.no_grad():
        for chunk_idx in range(len(os.listdir(dir))):
            for batch in process_chunk(chunk_idx, token_length, batch_size, 'train'):
                input_ids, attention_mask, _ = batch
                with autocast():
                    output = bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
                cls = output[:, 0, :] #extract CLS token
                database = torch.cat((database, cls))
    return database


def ranking(database, k=10):
    clusters = []
    for article in database:
        values, indices = torch.matmul(article, database.T).sort()
        score = values[:k].sum()
        ind = indices[:k]
        clusters.append((score, ind))
    return clusters


if __name__ == '__main__':
    database = torch.randn(27000, 512)
    ranking(database)
    # create_db('data/processed/cnn-dm/text/train')
    # create_db('data/processed/danewsroom/abstractive/text/train')