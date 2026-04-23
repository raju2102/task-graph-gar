# Task-Graph GAR — Planner Implementation

Implementation of the Planner component from **Task-Graph GAR: Dynamic Multi-Agent Mathematical Reasoning via Learned Problem Decomposition and Parallel Execution**.

---

## Files

| File | Description |
|---|---|
| `task_graph.py` | `TaskNode` and `TaskGraph` data structures. Handles DAG validation, topological sort, frontier computation, and critical path length. |
| `planner.py` | `Planner` class. Wraps a pretrained HuggingFace LLM (default: Qwen2.5-1.5B-Instruct). Prompts the model to decompose a problem into a JSON DAG, parses the output into a `TaskGraph`. |
| `rewards.py` | The four planner reward functions from the paper: `r_correctness` (R^m), `r_parallelism` (R^par), `r_node_validity` (R^val), `r_redundancy` (R^red), plus the combined `planner_reward`. |
| `data.py` | Loads GSM8K from HuggingFace. Parses chain-of-thought steps into linear DAGs for SFT warm-up. Also loads raw problems with ground-truth answers for RL training. |
| `train.py` | Two-stage training: (1) SFT warm-up on GSM8K CoT DAGs, (2) GRPO reinforcement learning using the planner reward signal. |
| `demo.py` | Runs the planner on three sample math problems and prints the task graph with reward scores. |

---

## Setup

### 1. Create a virtual environment (recommended)

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. HuggingFace model download

The default model is `Qwen/Qwen2.5-1.5B-Instruct`. It will be downloaded automatically on first run (~3 GB). Make sure you have enough disk space and a stable internet connection.

To use a different model, pass `model_name=` to `Planner(...)` in `planner.py` or as an argument to `train()` in `train.py`.

---

## Running the demo (no training required)

```bash
python demo.py
```

This loads the base pretrained model (untrained Planner) and runs it on three sample math problems. It prints the generated task graph, dependency edges, and reward scores for each problem.

To run with a saved trained model:

```bash
python demo.py ./planner_model
```

---

## Training

```bash
python train.py
```

This runs both training stages in sequence:

**Stage 1 — SFT warm-up**
- Loads 1000 GSM8K problems
- Converts each chain-of-thought solution into a linear DAG
- Fine-tunes the Planner with standard cross-entropy loss so it learns to produce valid JSON DAGs

**Stage 2 — GRPO reinforcement learning**
- Loads 500 GSM8K problems (no labels used, only the question)
- For each problem, generates G=8 candidate task graphs
- Computes the planner reward for each graph
- Updates the model using group-relative policy optimisation (GRPO / MAGRPO)

The trained model is saved to `./planner_model/`.

### Adjusting training scale

Edit the `train()` call at the bottom of `train.py`:

```python
train(
    model_name="Qwen/Qwen2.5-1.5B-Instruct",  # swap for a larger model on the supercomputer
    save_path="./planner_model",
    sft_samples=1000,   # increase for more SFT data
    rl_problems=500,    # increase for more RL training
)
```

---

## Running on the supercomputer

1. Copy the entire folder to the supercomputer.
2. Install dependencies (same `requirements.txt`).
3. Change `model_name` to a larger model (e.g. `Qwen/Qwen2.5-7B-Instruct` or `meta-llama/Llama-3.1-8B-Instruct`).
4. Increase `sft_samples` and `rl_problems` for a full training run.
5. Run `python train.py`.

The code automatically uses CUDA if a GPU is available.

---

## Reward functions

| Reward | Formula | What it incentivises |
|---|---|---|
| R^m (correctness) | 1 if answer matches ground truth, else 0 | Solving the problem correctly |
| R^par (parallelism) | 1 − (critical path / total nodes) | Shorter critical path = more parallel execution |
| R^val (node validity) | Fraction of nodes with non-trivial subproblem text | Well-formed decompositions |
| R^red (redundancy penalty) | Fraction of highly overlapping node pairs | Penalises repeated or redundant sub-tasks |
| Combined | λ_m·R^m + λ_par·R^par + λ_val·R^val − λ_red·R^red | All of the above jointly |

Default weights: λ_m=1.0, λ_par=0.3, λ_val=0.5, λ_red=0.2. These can be changed in `rewards.py`.

---

## Notes

- The Planner is **not** trained on labelled (problem → DAG) pairs. It learns decomposition structure entirely from the reward signal, as described in the paper.
- The SFT warm-up uses GSM8K CoT as a structural bootstrap only — it teaches the model to output valid JSON DAGs, not necessarily good decompositions.
- R^m during RL requires the executor outputs to verify correctness. In this implementation, ground-truth answers from GSM8K are used as a proxy. A full system would pass executor traces to the discriminator.
