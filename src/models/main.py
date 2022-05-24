from setup import setup, setup_checkpointing, setup_summaries
from train import train

def main():
    model, optimizer, scheduler, args = setup()
    setup_checkpointing()
    setup_summaries()

    train(model, optimizer, scheduler, args)

if __name__ == '__main__':
    main()

