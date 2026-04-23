from collections import defaultdict
from typing import Optional

from task_graph import TaskGraph


def r_correctness(assembled_answer: Optional[str], ground_truth: Optional[str]) -> float:
    if assembled_answer is None or ground_truth is None:
        return 0.0
    return 1.0 if assembled_answer.strip() == ground_truth.strip() else 0.0


def r_parallelism(graph: TaskGraph) -> float:
    n = len(graph.nodes)
    if n == 0:
        return 0.0
    if n == 1:
        return 1.0
    critical = graph.critical_path_length()
    return 1.0 - (critical / n)


def r_node_validity(graph: TaskGraph) -> float:
    if not graph.nodes:
        return 0.0
    if not graph.is_valid():
        return 0.0
    valid_count = sum(1 for node in graph.nodes if len(node.subproblem.strip()) > 10)
    return valid_count / len(graph.nodes)


def r_redundancy(graph: TaskGraph) -> float:
    n = len(graph.nodes)
    if n <= 1:
        return 0.0
    subproblems = [set(node.subproblem.lower().split()) for node in graph.nodes]
    redundant_pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            si, sj = subproblems[i], subproblems[j]
            if not si or not sj:
                continue
            overlap = len(si & sj) / min(len(si), len(sj))
            if overlap > 0.8:
                redundant_pairs += 1
    max_pairs = n * (n - 1) / 2
    return redundant_pairs / max_pairs


def planner_reward(
    graph: TaskGraph,
    assembled_answer: Optional[str] = None,
    ground_truth: Optional[str] = None,
    lambda_m: float = 1.0,
    lambda_par: float = 0.3,
    lambda_val: float = 0.5,
    lambda_red: float = 0.2,
) -> float:
    if not graph.is_valid():
        return 0.0
    rm = r_correctness(assembled_answer, ground_truth)
    rpar = r_parallelism(graph)
    rval = r_node_validity(graph)
    rred = r_redundancy(graph)
    return lambda_m * rm + lambda_par * rpar + lambda_val * rval - lambda_red * rred
