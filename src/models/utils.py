import torch

def process_chunk(chunk_idx, token_length, batch_size):
    summary = torch.load(f'data/processed/tokenized/cnn-dm/summary/chunk_{chunk_idx}.pt')
    text = torch.load(f'data/processed/tokenized/cnn-dm/text/chunk_{chunk_idx}.pt')

    input_ids, attention_mask, = text['input_ids'][:, :token_length].to('cuda'), text['attention_mask'][:, :token_length].to('cuda')
    decoder_input_ids, _ = summary['input_ids'][:, :token_length].to('cuda'), summary['attention_mask'][:, :token_length].to('cuda')

    N = len(input_ids)  # Since it's not exactly 1000 due to the discarded stories
    for i in range(0, N, batch_size):
        batch = input_ids[i:(i+batch_size)], attention_mask[i:(i+batch_size)], decoder_input_ids[i:(i+batch_size)]
        yield batch


def process_chunk_da(chunk_idx, token_length, batch_size):
    summary = torch.load(f'data/processed/tokenized/danewsroom/abstractive/summary/chunk_{chunk_idx}.pt')
    text = torch.load(f'data/processed/tokenized/danewsroom/abstractivetext/chunk_{chunk_idx}.pt')

    input_ids, attention_mask, = text['input_ids'][:, :token_length].to('cuda'), text['attention_mask'][:, :token_length].to('cuda')
    decoder_input_ids, _ = summary['input_ids'][:, :token_length].to('cuda'), summary['attention_mask'][:, :token_length].to('cuda')

    N = len(input_ids)  # Since it's not exactly 1000 due to the discarded stories
    for i in range(0, N, batch_size):
        batch = input_ids[i:(i+batch_size)], attention_mask[i:(i+batch_size)], decoder_input_ids[i:(i+batch_size)]
        yield batch