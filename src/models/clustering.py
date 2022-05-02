from transformers import BertTokenizer, BertModel
import torch
import json


def extract_cls(data_dir):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased").to('cuda')

    for name in ['train', 'test', 'validation']:
        with open(f'data/processed/{data_dir}/{name}.json', 'r') as file:
            data = json.load(file)
        text, _ = list(zip(*data))
        cls_tokens = torch.tensor([]).to('cuda')
        with torch.no_grad():
            for i in range(len(text)):
                inputs = tokenizer(text[i], return_tensors="pt").to('cuda')
                outputs = model(input_ids=inputs.input_ids[:, :512])

                last_hidden_states = outputs.last_hidden_state
                cls = last_hidden_states[:, 0, :]
                cls_tokens = torch.cat((cls_tokens, cls))
        torch.save(cls_tokens, f'data/processed/{data_dir}/cls_{name}.pt')


if __name__ == '__main__':
    extract_cls('cnn-dm')