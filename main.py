from src import Trainer, Args, utacenv

def main():
    args = Args()
    trainer = Trainer(args)
    trainer.train()

if __name__ == "__main__":
    main()