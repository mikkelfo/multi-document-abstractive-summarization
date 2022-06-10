import os
import torch
from transformers import ProphetNetForConditionalGeneration, XLMProphetNetForConditionalGeneration
from transformers import ProphetNetTokenizer, XLMProphetNetTokenizer
import json
from utils import process_chunk
import argparse
from generation_utils import setup_serial_generation, get_forward, revert_forwards
from prophetnet_fixes import prophetnet_fixes
import wandb

def setup():
    parser = argparse.ArgumentParser(description='Training script for SDS ProphetNet')
    parser.add_argument('dir', type=str)
    parser.add_argument('checkpoints', type=str)
    parser.add_argument('--xlm', const=True, default=False, nargs="?", help="Use XLMProphetNet (Default: ProphetNet)")
    parser.add_argument('--mds', const=True, default=False, nargs="?", help="Activate MDS setup")
    parser.add_argument('--method', type=str, help="Which MDS method to use")
    parser.add_argument('--batch_size', default=1, type=int, help="Micro-batch size")
    parser.add_argument('--token_length', default=512, type=int, help="Number of tokens")
    parser.add_argument('--serial_strat', type=str, help="Which serial strategy to use (shuffle/prio)")

    args = parser.parse_args()
    assert args.serial_strat == 'prio' or args.serial_strat == 'shuffle' or args.serial_strat == None

    if args.xlm:
        tokenizer = XLMProphetNetTokenizer.from_pretrained("microsoft/xprophetnet-large-wiki100-cased")
        model = XLMProphetNetForConditionalGeneration.from_pretrained("microsoft/xprophetnet-large-wiki100-cased")
    else:
        tokenizer = ProphetNetTokenizer.from_pretrained('microsoft/prophetnet-large-uncased')
        model = ProphetNetForConditionalGeneration.from_pretrained('microsoft/prophetnet-large-uncased')

    model.to('cuda')

    if args.method == 'serial':
        model = setup_serial_generation(model)

    model = prophetnet_fixes(model)

    return model, tokenizer, args


def generate_summaries(model, tokenizer, args):
    if not os.path.isdir(f'summaries/{args.checkpoints}'):
        os.mkdir(f'summaries/{args.checkpoints}')
    N_chunks = len(os.listdir(f'data/processed/{args.dir}/text/test'))
    for checkpoint in [None] + os.listdir(f'checkpoints/{args.checkpoints}'):
        print("Starting:", checkpoint)
        if checkpoint is not None:
            dic = torch.load(f'checkpoints/{args.checkpoints}/{checkpoint}')
            model.load_state_dict(dic)

        summaries = []
        for chunk_idx in range(N_chunks):
            chunk_summ = []
            for batch in process_chunk('test', chunk_idx, args):
                input_ids, attention_mask, _ = batch
                if args.method == 'mean':
                    enc_output = model.prophetnet.encoder(input_ids=input_ids, attention_mask=attention_mask)
                    enc_output.last_hidden_state = enc_output.last_hidden_state.mean(1).unsqueeze(0)
                    output = model.generate(encoder_outputs=enc_output, min_length=45, max_length=110, num_beams=5, no_repeat_ngram_size=3, length_penalty=1.2)
                if args.method == 'serial':
                    enc_output = model.prophetnet.encoder(input_ids=input_ids, attention_mask=attention_mask)
                    output = model.generate(encoder_outputs=enc_output, attention_mask=attention_mask, min_length=45, max_length=110, num_beams=5, no_repeat_ngram_size=3, length_penalty=1.2)
                else:
                    output = model.generate(input_ids=input_ids, attention_mask=attention_mask, min_length=45, max_length=110, num_beams=5, no_repeat_ngram_size=3, length_penalty=1.2)
                gen_summary = tokenizer.batch_decode(output, skip_special_tokens=True)
                if args.mds:
                    chunk_summ.append(gen_summary)
                else:
                    chunk_summ += gen_summary
            summaries.append(chunk_summ)
        with open(f'summaries/{args.checkpoints}/{checkpoint}.json', 'w') as file:
            json.dump(summaries, file, indent=4)


def single_generation(model, args, log_step):
    model.eval()
    if args.xlm:
        tokenizer = XLMProphetNetTokenizer.from_pretrained("microsoft/xprophetnet-large-wiki100-cased")
    else:
        tokenizer = ProphetNetTokenizer.from_pretrained('microsoft/prophetnet-large-uncased')

    if args.method == 'serial':
        forwards = get_forward(model)
        model = setup_serial_generation(model)

    summaries = []
    for chunk_idx in range(N_chunks):
        for batch in process_chunk('test', chunk_idx, args):
            input_ids, attention_mask, _ = batch
            enc_output = model.prophetnet.encoder(input_ids=input_ids, attention_mask=attention_mask)
            if args.method == 'mean':
                enc_output.last_hidden_state = enc_output.last_hidden_state.mean(1).unsqueeze(0)
                attention_mask = None
            output = model.generate(encoder_outputs=enc_output, attention_mask=attention_mask, min_length=45, max_length=110, num_beams=5, no_repeat_ngram_size=3, length_penalty=1.2)
            gen_summary = tokenizer.batch_decode(output, skip_special_tokens=True)
            summaries += gen_summary

    summaries = ['\n'.join(sent_tokenize(summ)) for summ in summaries]
    with open(f'summaries/{wandb.run.name}/{step}.json', 'w') as file:
        json.dump(summaries, file, indent=4)

    with open(f'data/processed/references/{args.dir.split("/")[0]}.json', 'r') as f:
        references = json.load(f)

    evaluator = rouge.Rouge(metrics=['rouge-n', 'rouge-l'], max_n=2, apply_avg=True, alpha=0.5,
                            stemming=True if 'danewsroom' not in args.dir else False)

    scores = evaluator.get_scores(summaries, reference)
    r1 = s['rouge-1']['f']
    r2 = s['rouge-2']['f']
    rl = s['rouge-l']['f']
    wandb.log({'R-1': r1}, step=log_step)
    wandb.log({'R-2': r2}, step=log_step)
    wandb.log({'R-L': r3}, step=log_step)

    model.train()
    if args.serial:
        model = revert_forwards(model, forwards)

if __name__ == '__main__':
    model, tokenizer, args = setup()
    generate_summaries(model, tokenizer, args)