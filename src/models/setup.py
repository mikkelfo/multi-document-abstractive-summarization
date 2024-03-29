import argparse
from transformers import XLMProphetNetForConditionalGeneration, ProphetNetForConditionalGeneration
import torch
import random
import wandb
import os
from utils import implement_serial_input
from prophetnet_fixes import prophetnet_fixes
from scheduler import InverseSqrtScheduler


def setup():
    parser = argparse.ArgumentParser(description='Training script for SDS ProphetNet')

    # General
    parser.add_argument('dir', type=str)
    parser.add_argument('--xlm', const=True, default=False, nargs="?", help="Use XLMProphetNet (Default: ProphetNet)")
    parser.add_argument('--gpus', default=1, type=int, help="Number of GPUs to utilise")
    parser.add_argument('--epochs', default=10, type=int, help="Number of epochs")
    parser.add_argument('--batch_size', default=1, type=int, help="Micro-batch size")
    parser.add_argument('--token_length', default=256, type=int, help="Number of tokens")
    parser.add_argument('--checkpointing', default=100, type=int, help="How often the model is saved")
    parser.add_argument('--shuffle', const=True, default=False, nargs="?", help="Shuffle chunks during training")
    parser.add_argument('--watch', const=True, default=False, nargs="?", help="Activate wandb.watch")
    parser.add_argument('--fix', const=True, default=False, nargs="?", help="Activate fix for batch issues")
    parser.add_argument('--generation', const=True, default=False, nargs="?", help="Generate summaries after validation step")
    parser.add_argument('--seed', type=int)

    # Multi-document strategies
    parser.add_argument('--mds', const=True, default=False, nargs="?", help="Activate MDS setup")
    parser.add_argument('--method', type=str, help="Which MDS method to use")
    parser.add_argument('--serial_strat', type=str, help="Which serial strategy to use (shuffle/prio)")

    # Scheduler
    parser.add_argument('--scheduler', const=True, default=False, nargs="?", help='Use InverseSqrtScheduler')
    parser.add_argument('--clip_norm', type=float, help="Gradient clip grad norm")
    parser.add_argument('--clip_value', type=float, help="Gradient clip grad value")
    parser.add_argument('--warmup_updates', default=1000, type=int, help="Scheduler warmup updates")
    parser.add_argument('--warmup_init_lr', default=1e-07, type=float, help="Scheduler init lr")
    parser.add_argument('--warmup_end_lr', default=1e-4, type=float, help="Scheduler end lr")

    args = parser.parse_args()

    assert args.epochs > 0
    assert args.batch_size > 0 and args.batch_size <= 512
    assert args.token_length > 0 and args.token_length < 512
    assert args.checkpointing > 0
    assert args.gpus >= 0
    assert args.method is None or args.method == 'mean' or args.method == 'serial' or args.method == 'sds'
    assert args.serial_strat == 'prio' or args.serial_strat == 'shuffle' or args.serial_strat == None
    assert args.warmup_updates >= 0
    assert args.warmup_init_lr > 0
    assert args.warmup_init_lr > 0

    if args.mds and args.method is None:
        raise Exception('--method required with --mds ')
    if args.gpus > torch.cuda.device_count():
        raise Exception(f'Not enough GPUs available (args: {args.gpus}) > (available: {torch.cuda.device_count()})')
    if args.gpus > 1 and args.batch_size == 1:
        print('Multiple GPUs in use with batch_size 1')
    
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

    if args.method == 'serial':
        model = implement_serial_input(model)
    
    if args.fix:
        model = prophetnet_fixes(model)
    
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=0.01)
    if args.scheduler:
        scheduler = InverseSqrtScheduler(optimizer, warmup_updates=args.warmup_updates, warmup_init_lr=args.warmup_init_lr, warmup_end_lr=args.warmup_end_lr)
    else:
        scheduler = None
    print("Directory:", args.dir)

    if args.seed:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    wandb.init(project="abstractive-summarization-runs", entity="mikkelfo")
    wandb.config.update(args)
    if args.watch:
        wandb.watch(model)

    return model, optimizer, scheduler, args
    

def setup_checkpointing():
    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')
    run_name = wandb.run.name
    if not os.path.isdir(f'checkpoints/{run_name}'):
        os.mkdir(f'checkpoints/{run_name}')


def setup_summaries():
    if not os.path.isdir('summaries'):
        os.mkdir('summaries')
    run_name = wandb.run.name
    if not os.path.isdir(f'summaries/{run_name}'):
        os.mkdir(f'summaries/{run_name}')

