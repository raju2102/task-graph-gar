import sys

from planner import Planner
from rewards import planner_reward, r_parallelism, r_node_validity, r_redundancy

SAMPLE_PROBLEMS = [
    "If a train travels at 60 mph for 2 hours and then at 80 mph for 3 hours, what is the total distance traveled?",
    "A rectangle has a length that is twice its width. If the perimeter is 48 cm, what is the area?",
    "Find all integer solutions to x^2 + y^2 = 25.",
]


def print_graph_report(problem: str, graph, idx: int):
    print(f"\n{'='*60}")
    print(f"Problem {idx}: {problem}")
    print(f"{'='*60}")
    print(f"Nodes ({len(graph.nodes)}):")
    for node in graph.nodes:
        print(f"  [{node.id}] {node.subproblem}")
    if graph.edges:
        print("Edges (dependencies):")
        for u, v in graph.edges:
            print(f"  {u} -> {v}")
    else:
        print("Edges: none (all nodes run in parallel)")
    print(f"\nValid DAG         : {graph.is_valid()}")
    print(f"Critical path     : {graph.critical_path_length()} / {len(graph.nodes)} nodes")
    print(f"R_parallelism     : {r_parallelism(graph):.3f}")
    print(f"R_node_validity   : {r_node_validity(graph):.3f}")
    print(f"R_redundancy      : {r_redundancy(graph):.3f}")
    print(f"Planner reward    : {planner_reward(graph):.3f}  (no executor, R_m=0)")


def run_demo(model_path: str = None, model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"):
    print("Loading planner...")
    planner = Planner(model_name=model_path if model_path else model_name)
    print("Planner loaded.\n")

    for i, problem in enumerate(SAMPLE_PROBLEMS, start=1):
        try:
            graph = planner.generate(problem)
            print_graph_report(problem, graph, i)
        except Exception as e:
            print(f"\nProblem {i}: {problem}")
            print(f"  Failed to parse graph: {e}")


if __name__ == "__main__":
    model_path = sys.argv[1] if len(sys.argv) > 1 else None
    run_demo(model_path=model_path)
