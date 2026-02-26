# MuJoGo Runtime

## Prerequisites

- Python 3.11
- Conda (Miniconda or Anaconda)
- Homebrew (macOS)

## Setup

### 1. Install GNU Go

```bash
brew install gnu-go
```

### 2. Create conda environment

```bash
conda create -n mujogo python=3.11 -y
conda activate mujogo
pip install -r requirements.txt
```

### 3. Clone the source

```bash
git clone --depth 1 --filter=blob:none --sparse https://github.com/google-deepmind/deepmind-research.git
cd deepmind-research
git sparse-checkout set physics_planning_games
```

## Running

```bash
conda activate mujogo
PYTHONPATH=deepmind-research python deepmind-research/run_mujogo.py --environment_name=go_7x7
```

### Available environments

| Name | Description |
|------|-------------|
| `go_7x7` | 7x7 Go with GnuGo opponent |
| `tic_tac_toe_markers_features` | Tic-tac-toe with random opponent |
| `tic_tac_toe_mixture_opponent_markers_features` | Tic-tac-toe with mixture opponent |
| `tic_tac_toe_optimal_opponent_markers_features` | Tic-tac-toe with optimal opponent |

## Notes

- The gnugo binary path is configured in `deepmind-research/physics_planning_games/board_games/go_logic.py` (set to `/opt/homebrew/bin/gnugo` for macOS).
- The viewer opens a MuJoCo interactive window where a Jaco robot arm interacts with the board.
- Only board games are supported by `run_mujogo.py`. The mujoban environment requires additional dependencies (`six`, `boxoban`).
