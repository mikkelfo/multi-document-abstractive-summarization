import os
import torch
from transformers import ProphetNetForConditionalGeneration, XLMProphetNetForConditionalGeneration
from transformers import ProphetNetTokenizer, XLMProphetNetTokenizer
import json
from utils import process_chunk
import argparse

def setup():
    parser = argparse.ArgumentParser(description='Training script for SDS ProphetNet')
    parser.add_argument('dir', type=str)
    parser.add_argument('checkpoints', type=str)
    parser.add_argument('--xlm', const=True, default=False, nargs="?", help="Use XLMProphetNet (Default: ProphetNet)")
    parser.add_argument('--mds', const=True, default=False, nargs="?", help="Activate MDS setup")
    parser.add_argument('--batch_size', default=1, type=int, help="Micro-batch size")
    parser.add_argument('--token_length', default=256, type=int, help="Number of tokens")

    args = parser.parse_args()

    if args.xlm:
        tokenizer = XLMProphetNetTokenizer.from_pretrained("microsoft/xprophetnet-large-wiki100-cased")
        model = XLMProphetNetForConditionalGeneration.from_pretrained("microsoft/xprophetnet-large-wiki100-cased")
    else:
        tokenizer = ProphetNetTokenizer.from_pretrained('microsoft/prophetnet-large-uncased')
        model = ProphetNetForConditionalGeneration.from_pretrained('microsoft/prophetnet-large-uncased')

    model.to('cuda')

    return model, tokenizer, args

def generate_summaries():
    model, tokenizer, args = setup()
    os.mkdir(f'summaries/{args.checkpoints}')
    N_chunks = len(os.listdir(args.dir + '/summary/test'))
    for checkpoint in os.listdir(args.checkpoints):
        if 'end' not in checkpoint:
            continue
        print("Starting:", checkpoint)
        dic = torch.load(f'{args.checkpoints}/{checkpoint}')
        model.load_state_dict(dic)

        summaries = []
        for chunk_idx in range(N_chunks):
            for batch in process_chunk('test', chunk_idx, args):
                input_ids, attention_mask, _ = batch
                output = model.generate(input_ids=input_ids, attention_mask=attention_mask, min_length=45, max_length=110, num_beams=5, no_repeat_ngram_size=3, length_penalty=1.2)
                gen_summary = tokenizer.batch_decode(output, skip_special_tokens=True)
                summaries += gen_summary
        with open(f'summaries/{args.checkpoints}/{checkpoint}.json', 'w') as file:
            json.dump(summaries, file, indent=4)


# def generate_summaries(model_dir, data_dir='data/processed/cnn-dm/text/test'):
#     model = ProphetNetForConditionalGeneration.from_pretrained('microsoft/prophetnet-large-uncased')
#     model.to('cuda')
#     tokenizer = ProphetNetTokenizer.from_pretrained('microsoft/prophetnet-large-uncased')

#     if not os.path.isdir('models/summary-unfrozen'):
#         os.mkdir('models/summary-unfrozen')
#     for checkpoint in os.listdir(model_dir):
#         print("Starting:", checkpoint)
#         dic = torch.load(f'{model_dir}/{checkpoint}')
#         dic = clean_dic(dic)
#         model.load_state_dict(dic)

#         summaries = []
#         N_chunks = len(os.listdir(data_dir))
#         for chunk_idx in range(N_chunks):
#             print("    Chunk:", chunk_idx)
#             text = torch.load(f'data/processed/cnn-dm/text/test/chunk_{chunk_idx}.pt').input_ids.to('cuda')
#             for i in range(0, len(text), 8):
#                 foo = text[i:(i+8)]
#                 output = model.generate(input_ids=foo, min_length=45, max_length=110, num_beams=5, no_repeat_ngram_size=3, length_penalty=1.2)
#                 gen_summary = tokenizer.batch_decode(output, skip_special_tokens=True)
#                 summaries += gen_summary
        
#         with open(f'models/summary-unfrozen/{checkpoint}.json', 'w') as file:
#             json.dump(summaries, file, indent=4)
#         print()

# def generate_original(data_dir='data/processed/cnn-dm/text/test'):
#     model = ProphetNetForConditionalGeneration.from_pretrained('microsoft/prophetnet-large-uncased')
#     model.to('cuda')
#     tokenizer = ProphetNetTokenizer.from_pretrained('microsoft/prophetnet-large-uncased')

#     summaries = []
#     N_chunks = len(os.listdir(data_dir))
#     for chunk_idx in range(N_chunks):
#         print("    Chunk:", chunk_idx)
#         text = torch.load(f'data/processed/cnn-dm/text/test/chunk_{chunk_idx}.pt').input_ids.to('cuda')
#         for i in range(0, len(text), 8):
#             foo = text[i:(i+8)]
#             output = model.generate(input_ids=foo, min_length=45, max_length=110, num_beams=5, no_repeat_ngram_size=3, length_penalty=1.2)
#             gen_summary = tokenizer.batch_decode(output, skip_special_tokens=True)
#             summaries += gen_summary
    
#     with open(f'models/original_summary.json', 'w') as file:
#         json.dump(summaries, file, indent=4)


# Removes part of the key
def clean_dic(dic):
    dic2 = {}
    for key, val in dic.items():
        new_key = key[len('model.module.'):]
        dic2[new_key] = val
    return dic2
    


if __name__ == '__main__':
    generate_summaries()