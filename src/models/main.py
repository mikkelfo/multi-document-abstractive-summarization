import torch
from transformers import ProphetNetForConditionalGeneration

CHUNK_SIZE = 1000
BATCH_SIZE = 100
model = ProphetNetForConditionalGeneration.from_pretrained('microsoft/prophetnet-large-uncased')
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.0001)

for epoch in range(10):
    print(f'*****     EPOCH {epoch}     *****')
    # for each file
    epoch_loss = 0
    for i in range(312):
        if i % 10 == 0:
            print(f"    chunk {i}")
        summary = torch.load(f'data/processed/tokenized/summary/chunk_{i}.pt')
        text = torch.load(f'data/processed/tokenized/text/chunk_{i}.pt')

        # TODO: Why cant it handle 512 length (reduced to 500)?
        input_ids, attention_mask, decoder_input_ids, decoder_attention_mask = text['input_ids'][:, :500], text['attention_mask'][:, :500], summary['input_ids'][:, :500], summary['attention_mask'][:, :500]
        # for each batch in file
        for j in range(0, 1000, BATCH_SIZE):
            foo = input_ids[j:(j+BATCH_SIZE)]
            output = model(
                input_ids=input_ids[j:(j+BATCH_SIZE)], 
                attention_mask = attention_mask[j:(j+BATCH_SIZE)], 
                labels = decoder_input_ids[j:(j+BATCH_SIZE)], 
            )
            epoch_loss += output.loss
            output.loss.backward()
            optimizer.step()
    print()
    print(f'----------     Total loss {epoch_loss}     ----------')
    print()
            


if __name__ == '__main__':
    pass