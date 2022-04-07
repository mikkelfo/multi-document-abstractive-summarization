import argparse
from transformers import XLMProphetNetForConditionalGeneration, ProphetNetForConditionalGeneration
import torch
import wandb
import os

def setup():
    parser = argparse.ArgumentParser(description='Training script for SDS ProphetNet')
    parser.add_argument('lang', type=str)
    parser.add_argument('--xlm', const=True, default=False, nargs="?", help="Use XLMProphetNet (Default: ProphetNet)")
    parser.add_argument('--gpus', default=1, type=int, help="Number of GPUs to utilise")
    parser.add_argument('--epochs', default=10, type=int, help="Number of epochs")
    parser.add_argument('--batch_size', default=1, type=int, help="Micro-batch size")
    parser.add_argument('--token_length', default=256, type=int, help="Number of tokens")
    parser.add_argument('--checkpointing', default=50, type=int, help="How often the model is saved ")
    parser.add_argument('--dir', type=str, help="Change the directory")

    args = parser.parse_args()

    assert args.epochs > 0
    assert args.batch_size > 0 and args.batch_size <= 512
    assert args.token_length > 0 and args.token_length < 512
    assert args.checkpointing > 0
    assert args.gpus >= 0
    assert args.lang == 'da' or args.lang == 'en', "Only supports 'da' or 'en'"

    if args.gpus > torch.cuda.device_count():
        raise Exception(f'Not enough GPUs available (args: {args.gpus}) > (available: {torch.cuda.device_count()})')
    if args.gpus > 1 and args.batch_size == 1:
        raise Exception('Multiple GPUs in use with batch_size 1')
    if args.lang == 'da' and not args.xlm:
        raise Exception('Danish language is only suported with XLM enabled')

    # args.gradient_accumulation_step = 512 / args.batch_size
    
    if args.xlm:
        print("Initializing XLM model")
        model = XLMProphetNetForConditionalGeneration.from_pretrained("microsoft/xprophetnet-large-wiki100-cased")
    else:
        print("Initializing ProphetNet model")
        model = ProphetNetForConditionalGeneration.from_pretrained('microsoft/prophetnet-large-uncased')
    model.gradient_checkpointing_enable()
    if args.gpus > 0:
        model.to('cuda')
        print("Running on GPU")
        if args.gpus > 1:
            model = torch.nn.DataParallel(model)
            print("DataParallel activated")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.01)

    if args.dir is None:
        if args.lang == 'da':
            args.dir = 'danewsroom/abstractive'
        elif args.lang == 'en':
            args.dir = 'cnn-dm'
    print("Directory:", args.dir)

    wandb.init(project="abstractive-summarization-runs", entity="mikkelfo")
    # wandb.watch(model)    # Causes memory spikes (2GB on XLM)
    wandb.config.update(args)

    return model, optimizer, args
    

def setup_checkpointing():
    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')
    run_name = wandb.run.name
    if not os.path.isdir(f'checkpoints/{run_name}'):
        os.mkdir(f'checkpoints/{run_name}')

