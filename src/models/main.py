from setup import setup, setup_checkpointing, setup_summaries
from train import train

def main():
    model, optimizer, args = setup()
    setup_checkpointing()
    setup_summaries()

    train(model, optimizer, args)

if __name__ == '__main__':
    main()

