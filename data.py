import random
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


def combine_two_problems(
    q1: str, steps1: List[str], q2: str, steps2: List[str]
) -> Tuple[str, TaskGraph]:
    combined_problem = (
        f"Solve both parts below.\n"
        f"Part 1: {q1}\n"
        f"Part 2: {q2}"
    )
    nodes = []
    for i, step in enumerate(steps1):
        nodes.append(TaskNode(id=f"a{i+1}", subproblem=f"[Part 1] {step}"))
    for i, step in enumerate(steps2):
        nodes.append(TaskNode(id=f"b{i+1}", subproblem=f"[Part 2] {step}"))
    sink = TaskNode(id="sink", subproblem="State the final answers to Part 1 and Part 2.")
    nodes.append(sink)

    edges = []
    for i in range(len(steps1) - 1):
        edges.append((f"a{i+1}", f"a{i+2}"))
    for i in range(len(steps2) - 1):
        edges.append((f"b{i+1}", f"b{i+2}"))
    edges.append((f"a{len(steps1)}", "sink"))
    edges.append((f"b{len(steps2)}", "sink"))

    return combined_problem, TaskGraph(nodes=nodes, edges=edges)


def load_gsm8k_parallel_sft_data(
    split: str = "train", max_samples: int = 500, seed: int = 42
) -> List[Tuple[str, TaskGraph]]:
    dataset = load_dataset("openai/gsm8k", "socratic", split=split)
    rng = random.Random(seed)

    pool = []
    for example in dataset:
        steps = parse_socratic_steps(example["answer"])
        if len(steps) >= 2:
            pool.append((example["question"], steps))

    rng.shuffle(pool)
    pairs = [(pool[i], pool[i + 1]) for i in range(0, min(max_samples * 2, len(pool) - 1), 2)]

    data = []
    for (q1, s1), (q2, s2) in pairs[:max_samples]:
        problem, graph = combine_two_problems(q1, s1, q2, s2)
        data.append((problem, graph))
    return data


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
