from functools import partial
from typing import List, Optional, Tuple

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup

from data import load_gsm8k_rl_problems, load_gsm8k_sft_data, load_gsm8k_parallel_sft_data
from planner import Planner
from rewards import planner_reward
from task_graph import TaskGraph


class SFTDataset(Dataset):
    def __init__(self, data: List[Tuple[str, TaskGraph]], planner: Planner, max_length: int = 512):
        self.samples = []
        for problem, graph in data:
            prompt = planner.build_prompt(problem)
            target = graph.to_json() + planner.tokenizer.eos_token
            self.samples.append(prompt + target)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def sft_collate(batch: List[str], tokenizer, max_length: int) -> dict:
    encodings = tokenizer(
        batch,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    labels = encodings["input_ids"].clone()
    labels[encodings["attention_mask"] == 0] = -100
    encodings["labels"] = labels
    return encodings


def validate_sft(planner: Planner, val_problems: List[str], num_samples: int = 50) -> Tuple[float, float]:
    problems = val_problems[:num_samples]
    parsed = 0
    valid_dag = 0

    planner.model.eval()
    for problem in problems:
        try:
            graph = planner.generate(problem, temperature=0.0)
            parsed += 1
            if graph.is_valid():
                valid_dag += 1
        except Exception:
            pass

    parse_rate = parsed / len(problems)
    validity_rate = valid_dag / len(problems)
    return parse_rate, validity_rate


def validate_parallel_sft(planner: Planner, val_problems: List[str], num_samples: int = 50) -> Tuple[float, float, float, float]:
    problems = val_problems[:num_samples]
    parsed = 0
    valid_dag = 0
    parallel_count = 0
    rpar_total = 0.0

    planner.model.eval()
    for problem in problems:
        try:
            graph = planner.generate(problem, temperature=0.0)
            parsed += 1
            if graph.is_valid():
                valid_dag += 1
                node_ids = set(graph.node_ids())
                has_dep = {v for _, v in graph.edges}
                is_dep_of = {u for u, _ in graph.edges}
                roots = [n for n in node_ids if n not in has_dep]
                has_parallel = len(roots) > 1
                if has_parallel:
                    parallel_count += 1
                crit = graph.critical_path_length()
                n = len(graph.nodes)
                rpar_total += 1.0 - crit / n
        except Exception:
            pass

    n_valid = valid_dag if valid_dag > 0 else 1
    parse_rate = parsed / len(problems)
    validity_rate = valid_dag / len(problems)
    parallelism_rate = parallel_count / len(problems)
    mean_rpar = rpar_total / n_valid
    return parse_rate, validity_rate, parallelism_rate, mean_rpar


def run_sft(
    planner: Planner,
    max_samples: int = 1000,
    epochs: int = 10,
    batch_size: int = 1,
    lr: float = 2e-6,
    max_length: int = 512,
    parse_rate_threshold: float = 0.90,
    validity_rate_threshold: float = 0.90,
):
    all_data = load_gsm8k_sft_data(max_samples=max_samples + 100)
    train_data = all_data[:max_samples]
    val_problems = [problem for problem, _ in all_data[max_samples:max_samples + 100]]

    dataset = SFTDataset(train_data, planner, max_length)
    collate_fn = partial(sft_collate, tokenizer=planner.tokenizer, max_length=max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    optimizer = AdamW(planner.model.parameters(), lr=lr)
    total_steps = len(dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=total_steps // 10, num_training_steps=total_steps
    )

    for epoch in range(epochs):
        planner.model.train()
        total_loss = 0.0
        for step, batch in enumerate(dataloader):
            batch = {k: v.to(planner.device) for k, v in batch.items()}
            outputs = planner.model(**batch)
            loss = outputs.loss
            if torch.isnan(loss):
                print(f"  [SFT] NaN loss at Epoch {epoch+1}, Step {step} — skipping batch.")
                optimizer.zero_grad()
                continue
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(planner.model.parameters(), 0.3)
            if torch.isnan(grad_norm):
                print(f"  [SFT] NaN gradients at Epoch {epoch+1}, Step {step} — skipping update.")
                optimizer.zero_grad()
                continue
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.item()
            if step % 50 == 0:
                print(f"  [SFT] Epoch {epoch+1}, Step {step}, Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        parse_rate, validity_rate = validate_sft(planner, val_problems)
        print(f"  [SFT] Epoch {epoch+1} done. Avg loss: {avg_loss:.4f} | Parse rate: {parse_rate:.2%} | DAG validity: {validity_rate:.2%}")

        if parse_rate >= parse_rate_threshold and validity_rate >= validity_rate_threshold:
            print(f"  [SFT] Early stopping: both thresholds reached at epoch {epoch+1}.")
            break


def _parallel_ratio_from_metrics(parallelism_rate: float) -> float:
    if parallelism_rate < 0.15:
        return 0.30
    elif parallelism_rate < 0.30:
        return 0.50
    elif parallelism_rate < 0.50:
        return 0.70
    else:
        return 0.85


def run_parallel_sft(
    planner: Planner,
    parallel_pool_size: int = 2000,
    linear_pool_size: int = 1000,
    epochs: int = 10,
    batch_size: int = 1,
    lr: float = 1e-6,
    max_length: int = 512,
    parse_rate_threshold: float = 0.90,
    validity_rate_threshold: float = 0.90,
):
    import random

    parallel_data = load_gsm8k_parallel_sft_data(max_samples=parallel_pool_size + 50)
    train_parallel_pool = parallel_data[:parallel_pool_size]
    val_problems = [p for p, _ in parallel_data[parallel_pool_size:parallel_pool_size + 50]]
    linear_pool = load_gsm8k_sft_data(max_samples=linear_pool_size)

    optimizer = AdamW(planner.model.parameters(), lr=lr)
    collate_fn = partial(sft_collate, tokenizer=planner.tokenizer, max_length=max_length)

    par_ratio = 0.30
    print(f"  [PAR-SFT] Starting with parallel ratio: {par_ratio:.0%}")

    for epoch in range(epochs):
        n_total = len(train_parallel_pool) + len(linear_pool)
        n_par = int(n_total * par_ratio)
        n_lin = n_total - n_par
        par_sample = random.sample(train_parallel_pool, min(n_par, len(train_parallel_pool)))
        lin_sample = random.sample(linear_pool, min(n_lin, len(linear_pool)))
        mixed = par_sample + lin_sample
        random.shuffle(mixed)

        print(f"  [PAR-SFT] Epoch {epoch+1}: {len(par_sample)} parallel + {len(lin_sample)} linear samples")

        dataset = SFTDataset(mixed, planner, max_length)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=len(dataloader) // 10, num_training_steps=len(dataloader)
        )

        planner.model.train()
        total_loss = 0.0
        for step, batch in enumerate(dataloader):
            batch = {k: v.to(planner.device) for k, v in batch.items()}
            outputs = planner.model(**batch)
            loss = outputs.loss
            if torch.isnan(loss):
                print(f"  [PAR-SFT] NaN loss at Epoch {epoch+1}, Step {step} — skipping batch.")
                optimizer.zero_grad()
                continue
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(planner.model.parameters(), 0.3)
            if torch.isnan(grad_norm):
                print(f"  [PAR-SFT] NaN gradients at Epoch {epoch+1}, Step {step} — skipping update.")
                optimizer.zero_grad()
                continue
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.item()
            if step % 50 == 0:
                print(f"  [PAR-SFT] Epoch {epoch+1}, Step {step}, Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        parse_rate, validity_rate, parallelism_rate, mean_rpar = validate_parallel_sft(planner, val_problems)
        print(
            f"  [PAR-SFT] Epoch {epoch+1} done. "
            f"Avg loss: {avg_loss:.4f} | "
            f"Parse rate: {parse_rate:.2%} | "
            f"DAG validity: {validity_rate:.2%} | "
            f"Parallelism rate: {parallelism_rate:.2%} | "
            f"Mean R^par: {mean_rpar:.4f}"
        )

        new_ratio = _parallel_ratio_from_metrics(parallelism_rate)
        if new_ratio != par_ratio:
            print(f"  [PAR-SFT] Adjusting parallel ratio: {par_ratio:.0%} → {new_ratio:.0%}")
        par_ratio = new_ratio

        if parse_rate >= parse_rate_threshold and validity_rate >= validity_rate_threshold:
            print(f"  [PAR-SFT] Early stopping at epoch {epoch+1}: parse+validity thresholds reached.")
            break


def compute_grpo_loss(
    planner: Planner,
    problem: str,
    graphs: List[Optional[TaskGraph]],
    rewards: List[float],
) -> torch.Tensor:
    valid_pairs = [(g, r) for g, r in zip(graphs, rewards) if g is not None]
    if not valid_pairs:
        return torch.tensor(0.0, device=planner.device, requires_grad=False)

    valid_graphs, valid_rewards = zip(*valid_pairs)
    reward_tensor = torch.tensor(valid_rewards, dtype=torch.float32)
    advantages = (reward_tensor - reward_tensor.mean()) / (reward_tensor.std() + 1e-8)

    prompt = planner.build_prompt(problem)
    prompt_len = len(planner.tokenizer(prompt, return_tensors="pt")["input_ids"][0])

    total_loss = torch.zeros(1, device=planner.device, requires_grad=True)

    for graph, advantage in zip(valid_graphs, advantages):
        target = graph.to_json() + planner.tokenizer.eos_token
        inputs = planner.tokenizer(
            prompt + target,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        ).to(planner.device)

        labels = inputs["input_ids"].clone()
        labels[0, :prompt_len] = -100

        outputs = planner.model(input_ids=inputs["input_ids"], labels=labels)
        log_prob = -outputs.loss
        total_loss = total_loss + (-advantage.to(planner.device) * log_prob)

    return total_loss / len(valid_pairs)


def run_grpo(
    planner: Planner,
    max_problems: int = 500,
    epochs: int = 2,
    G: int = 8,
    lr: float = 1e-5,
):
    problems = load_gsm8k_rl_problems(max_samples=max_problems)
    optimizer = AdamW(planner.model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_reward = 0.0
        count = 0
        for i, problem_dict in enumerate(problems):
            question = problem_dict["question"]
            ground_truth = problem_dict["answer"]

            graphs = planner.generate_batch(question, G=G)
            rewards = [
                planner_reward(g, ground_truth=ground_truth)
                if g is not None and g.is_valid()
                else 0.0
                for g in graphs
            ]

            planner.model.train()
            loss = compute_grpo_loss(planner, question, graphs, rewards)

            if loss.requires_grad:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(planner.model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            total_reward += sum(rewards) / len(rewards)
            count += 1

            if i % 20 == 0:
                print(f"  [GRPO] Epoch {epoch+1}, Problem {i+1}/{len(problems)}, Avg reward: {total_reward/count:.4f}")

        print(f"  [GRPO] Epoch {epoch+1} done. Mean reward: {total_reward/count:.4f}")


def train(
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    save_path: str = "./planner_model",
    sft_samples: int = 1000,
    rl_problems: int = 500,
    gradient_checkpointing: bool = True,
    device: str = "cpu",
    batch_size: int = 1,
):
    print(f"Loading base model: {model_name} on {device}")
    planner = Planner(model_name=model_name, gradient_checkpointing=gradient_checkpointing, device=device)

    print("\nStage 1: SFT warm-up on GSM8K chain-of-thought...")
    run_sft(planner, max_samples=sft_samples, batch_size=batch_size)

    # Stage 2 (GRPO) requires the Executor and Discriminator to be implemented first.
    # print("\nStage 2: GRPO reinforcement learning...")
    # run_grpo(planner, max_problems=rl_problems)

    planner.save(save_path)
    print(f"\nModel saved to {save_path}")


def train_parallel(
    checkpoint_path: str = "./planner_model",
    save_path: str = "./planner_model_v2",
    gradient_checkpointing: bool = False,
    device: str = "cpu",
    batch_size: int = 4,
):
    print(f"Loading checkpoint: {checkpoint_path} on {device}")
    planner = Planner.load(checkpoint_path, device=device)
    if gradient_checkpointing:
        planner.model.gradient_checkpointing_enable()

    print("\nParallel SFT: curriculum fine-tuning on mixed parallel + linear data...")
    run_parallel_sft(planner, batch_size=batch_size)

    planner.save(save_path)
    print(f"\nModel saved to {save_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="sft", choices=["sft", "parallel"],
                        help="sft: initial training from base model. parallel: fine-tune from checkpoint.")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--checkpoint_path", type=str, default="./planner_model")
    parser.add_argument("--save_path", type=str, default="./planner_model")
    parser.add_argument("--sft_samples", type=int, default=1000)
    parser.add_argument("--rl_problems", type=int, default=500)
    parser.add_argument("--gradient_checkpointing", type=lambda x: x.lower() != "false", default=False)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()

    if args.mode == "parallel":
        train_parallel(
            checkpoint_path=args.checkpoint_path,
            save_path=args.save_path,
            gradient_checkpointing=args.gradient_checkpointing,
            device=args.device,
            batch_size=args.batch_size,
        )
    else:
        train(
            model_name=args.model_name,
            save_path=args.save_path,
            sft_samples=args.sft_samples,
            rl_problems=args.rl_problems,
            gradient_checkpointing=args.gradient_checkpointing,
            device=args.device,
            batch_size=args.batch_size,
        )
