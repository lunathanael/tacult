from src import Trainer, Args
import utac
import tyro

def main():
    args = tyro.cli(Args)
    trainer = Trainer(args)
    trainer.train()

if __name__ == "__main__":
    main()