# Tacult - Ultimate Tic Tac Toe AI

A Python implementation of an AI system for Ultimate Tic Tac Toe (UTAC) using deep learning and Monte Carlo Tree Search (MCTS). This project provides tools for training, evaluating, and playing against AI agents in the Ultimate Tic Tac Toe game.

A live playable version of the AI is available [here](https://tacult.lunathanael.dev/).

## Prerequisites

Before installing this package, you need to install the `utac-gym` package first:

```bash
pip install git+https://github.com/lunathanael/utac-gym
```

## Installation

Install the package using pip:

```bash
pip install tacult
```

## Requirements

- Python >= 3.10
- PyTorch
- NumPy
- ONNX
- ONNX Runtime
- tqdm
- coloredlogs

## Features

- Deep learning model for Ultimate Tic Tac Toe
- Monte Carlo Tree Search (MCTS) implementation
- Training pipeline for self-play and supervised learning
- Model export to ONNX format
- Tournament system for evaluating different MCTS configurations
- CUDA support for GPU acceleration

## Usage

### Training

To train a new model:

```python
from tacult import train, UtacGame
from tacult.utils import dotdict

# Configure training parameters
args = dotdict({
    "numIters": 300,
    "minNumEps": 128,
    "numEnvs": 128,
    "tempThreshold": 9,
    "updateThreshold": 0.6,
    "maxlenOfQueue": 500_000,
    "numMCTSSims": 4000,
    "cpuct": 2.4,
    "model_args": {
        "channels": 32,
        "num_residual_blocks": 3,
        "embedding_size": 256,
        "dropout": 0.2,
        "cuda": True,  # Set to True if you have a GPU
    },
    "checkpoint_folder": "./temp/resnet2cc/",
})

# Start training
train(args)
```

### Exporting Model to ONNX

To export a trained model to ONNX format:

```python
from tacult import export_model
from pathlib import Path

args = {
    "policy_value_net_path": Path("path/to/your/model.pt"),
    "policy_value_net_onnx_path": Path("path/to/save/model.onnx")
}

export_model(args)
```

### Running Tournaments

To evaluate different MCTS configurations:

```python
from tacult import UtacGame, UtacNN
from tacult.pit import Pit
from tacult.base.nn_wrapper import load_network

# Load your trained model
game = UtacGame()
nnet = UtacNN(args)
nnet = load_network("path/to/model/folder", "best.pt")

# Create and run tournament
pit = Pit.create_mcts_tournament(
    game=game,
    nnet=nnet,
    num_sims_list=[100, 400, 1600],
    games_per_match=100,
    num_rounds=1
)

pit.run()
```

## Project Structure

- `src/tacult/`
  - `base/` - Core components (MCTS, neural network wrapper)
  - `network.py` - Neural network architecture
  - `trainer.py` - Training pipeline
  - `export_model.py` - ONNX export functionality
  - `utac_game.py` - Game rules and mechanics
  - `pit.py` - Tournament system

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

- Lunathanel (info@lunathanel.dev)
