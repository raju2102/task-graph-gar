import re
from typing import List, Tuple, Optional

from datasets import load_dataset

from task_graph import TaskGraph, TaskNode


def parse_socratic_steps(solution: str) -> List[str]:
    lines = [l.strip() for l in solution.split("\n") if l.strip()]
    steps = []
    for line in lines:
        if line.startswith("####"):
            continue
        if "**" in line:
            question = line.split("**")[0].strip()
        else:
            question = line.strip()
        if question and len(question) > 5:
            steps.append(question)
    return steps if steps else []


def steps_to_linear_dag(steps: List[str]) -> TaskGraph:
    nodes = [TaskNode(id=f"v{i+1}", subproblem=step) for i, step in enumerate(steps)]
    edges = [(f"v{i+1}", f"v{i+2}") for i in range(len(steps) - 1)]
    return TaskGraph(nodes=nodes, edges=edges)


def extract_answer(solution: str) -> Optional[str]:
    match = re.search(r"####\s*([\d,.\-]+)", solution)
    if match:
        return match.group(1).replace(",", "")
    return None


def load_gsm8k_sft_data(
    split: str = "train", max_samples: int = 1000
) -> List[Tuple[str, TaskGraph]]:
    dataset = load_dataset("openai/gsm8k", "socratic", split=split)
    data = []
    for i, example in enumerate(dataset):
        if i >= max_samples:
            break
        steps = parse_socratic_steps(example["answer"])
        if len(steps) >= 2:
            graph = steps_to_linear_dag(steps)
            data.append((example["question"], graph))
    return data


def load_gsm8k_rl_problems(
    split: str = "train", max_samples: int = 5000
) -> List[dict]:
    dataset = load_dataset("openai/gsm8k", "main", split=split)
    problems = []
    for i, example in enumerate(dataset):
        if i >= max_samples:
            break
        problems.append({
            "question": example["question"],
            "answer": extract_answer(example["answer"]),
        })
    return problems
