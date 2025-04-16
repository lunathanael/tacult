# Tacult: A Reinforcement Learning Agent for Ultimate Tic-Tac-Toe

## Overview

**Tacult** is a scalable deep reinforcement learning agent developed to play Ultimate Tic-Tac-Toe using self-play. It leverages a combination of vectorized Monte Carlo Tree Search (MCTS) and neural network policy/value estimation to iteratively improve performance, drawing inspiration from methods like AlphaZero and MuZero.

The system is composed of three main components:

1. **utac** – A high-performance C++ library implementing the logic of Ultimate Tic-Tac-Toe, including move generation, terminal condition checking, and a fast board representation.
2. **utac-gym** – A Python Gym-compatible wrapper that exposes the environment to training and evaluation loops, facilitating interaction with reinforcement learning frameworks.
3. **tacult** – The core training engine, which integrates a scalable MCTS-based actor-learner pipeline, shared replay buffer, and neural network training using PyTorch.

This project is a research-oriented approach to explore scalable self-play RL training on a game with hierarchical structure and partial move constraints, requiring careful handling of game rules and strategic foresight.

## Screenshots

Example analysis of a board position.
![image](https://github.com/user-attachments/assets/d139a555-c57e-4072-9b73-064ec053e9cb)

Example Arena Playout, elo ratings in the picture depicted represent 1:100 odds with only 15 training iterations and 2 simulations! 
![Screenshot 2025-02-04 021404](https://github.com/user-attachments/assets/e7091fad-450f-4f90-a834-0036834818a6)

Picturization of Model
![Screenshot 2025-01-27 214801](https://github.com/user-attachments/assets/9549ca13-23f1-4adf-af96-ddfd32dc4d0c)

Figures from early training using tensorboard
![Screenshot 2025-01-09 180732](https://github.com/user-attachments/assets/92388d1f-ebf1-4da5-8093-4c239186313b)

## Deeper Dive

Tacult fuses the AlphaZero research paradigm with high‑performance engineering. At its core is **utac**, a C++ engine that represents the 9×9 Ultimate board via bitmasking and perfect‑hash lookup tables, yielding **> 5 000 MCTS node evaluations per second**. To scale self‑play, Tacult employs **vectorized MCTS**, processing hundreds of trees in parallel and using NumPy‑accelerated replay buffers for rapid data ingestion and sampling. Selection in each tree is governed by the PUCT formula:

$` 
\mathrm{UCT}(s,a) \;=\; Q(s,a)\;+\;c_{\text{puct}}\;P(s,a)\;\frac{\sqrt{\sum_b N(s,b)}}{1 + N(s,a)} 
`$

where

$`
Q(s,a) \;=\;\frac{W(s,a)}{N(s,a)},\quad 
P(s,a)\;\text{is the neural‐net prior},\quad 
N(s,a)\;\text{is the visit count}.
`$

After MCTS completes, moves are sampled according to the root‑visit distribution:

$`
\pi(a\mid s)\;=\;\frac{N(s,a)^{1/\tau}}{\sum_b N(s,b)^{1/\tau}},
`$

and stored alongside observations and final outcomes in a replay buffer. Network training minimizes the combined loss:

$`
\ell(\theta)\;=\;(z - v)^2 \;-\;\boldsymbol{\pi}^\top\!\log \, \boldsymbol{p}\;+\;\lambda\|\theta\|^2,
`$

with parameters updated by stochastic gradient descent:

$`
\theta \;\leftarrow\;\theta \;-\;\eta\,\nabla_\theta \,\ell(\theta).
`$

---

Bridging C++ throughput and Python flexibility is **utac‑gym**, built on Nanobind for near‑zero overhead calls. This interface drives PyTorch training on GPUs, while retaining the C++ engine’s speed. The neural model follows a deep ResNet backbone—with initial convolutions, multiple residual blocks, and separate policy/value heads—mirroring the architecture from Silver et al. (2017). The **policy head** outputs logits which are normalized via softmax:

$`
p_i\;=\;\frac{\exp(z_i)}{\sum_j \exp(z_j)},
`$

and the **value head** predicts a scalar $`v\in[-1,1]`$. Candidate networks are validated through an **Elo‑style arena**, where ratings evolve by:

$`
E_A\;=\;\frac{1}{1 + 10^{(R_B - R_A)/400}},
`$

$`
R'_A\;=\;R_A \;+\;K\,(S_A - E_A),
`$

ensuring only statistically significant improvements (ΔElo ≥ 10) survive. Together, these elements deliver a research‑grade platform—grounded in DeepMind’s pioneering RL algorithms and optimized for blazing performance.

## Network Architecture

The neural network is a dual-headed architecture that processes observations from the Ultimate Tic-Tac-Toe board and outputs two quantities:

- A **policy vector**, representing the agent's probability distribution over valid moves.
- A **value scalar**, representing the expected outcome (win, draw, or loss) from the current game state.

### Heuristic Design of the Network

The input to the network encodes the current board state using multiple binary planes. These include:

- A plane for the positions of X pieces.
- A plane for the positions of O pieces.
- A plane for valid next subgrids (due to the rules of Ultimate Tic-Tac-Toe).
- A plane indicating the current player's identity.
- A historical stack of previous board states to give temporal context (e.g., 8 frames for temporal unrolling).

These planes form a tensor of shape (C, 9, 9), where C is the number of input channels.

The neural network itself begins with a stack of convolutional layers designed to extract spatial and hierarchical features across the 9x9 macro grid. A series of residual blocks follow to facilitate deeper learning without vanishing gradients, similar to ResNet-based AlphaZero models.

Once a high-dimensional representation is computed, the network bifurcates into two heads:

- The **policy head** maps the shared representation to a logits vector of shape (9×9 = 81), which is then masked and normalized using a softmax function during training and MCTS rollouts.
- The **value head** reduces the shared representation through dense layers and outputs a scalar in the range [-1, 1], representing the expected result from the perspective of the current player.

Batch normalization and ReLU activations are used throughout the network, and weight initialization follows He normal initialization for stability.

## Reinforcement Learning Framework

### Self-Play and MCTS

Tacult uses a vectorized Monte Carlo Tree Search (vMCTS) for policy improvement. During training, multiple environments are run in parallel, each performing MCTS rollouts per move using the latest neural network.

The MCTS uses a UCB1-like selection function to balance exploration and exploitation, backed by neural network predictions. Upon expansion of a node, the network outputs are stored in the node’s statistics, and backpropagation of values occurs up the tree.

The selected action is then sampled proportionally to visit counts (softmax of root visit counts) during training, and greedily during evaluation.

### Replay Buffer

A prioritized replay buffer stores self-play episodes in the form of:

- The observation tensor.
- The search-derived policy target.
- The game outcome.

The buffer is periodically sampled for training in batches. The agent is trained using a composite loss function:

- Cross-entropy loss for the policy head.
- Mean squared error loss for the value head.
- Optional entropy regularization.

### Vectorization

A key design choice was the implementation of vectorized environments and batched network inference. By processing multiple MCTS trees in parallel, the system achieves high GPU utilization and faster training throughput.

The Gym wrapper (`utac-gym`) supports this batching by returning stacked observations and handling step/reset across vectorized envs.

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
## References

This project draws inspiration from:

- Silver, D., Schrittwieser, J., et al. "Mastering the game of Go without human knowledge." *Nature* 550.7676 (2017): 354-359.
- Schrittwieser, J., et al. "Mastering Atari, Go, Chess and Shogi by planning with a learned model." *Nature* 588.7839 (2020): 604–609.
- Pettinger, A. "Monte Carlo Tree Search and Deep Reinforcement Learning." [Online Article] (https://apettit.github.io/blog/rl/2024/03/29/mcts.html)

These references underpin the agent’s architectural decisions, training loop design, and data representation.

## License

MIT License. See `LICENSE` file for more details.
