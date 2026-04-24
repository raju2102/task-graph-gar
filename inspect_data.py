from datasets import load_dataset
from data import parse_socratic_steps, steps_to_linear_dag, load_gsm8k_sft_data
from planner import Planner
from task_graph import TaskGraph
import torch
import json

DIVIDER = "=" * 70


def inspect_data(n: int = 3):
    print(f"\n{DIVIDER}")
    print("SECTION 1: RAW GSM8K SAMPLES")
    print(DIVIDER)

    dataset = load_dataset("openai/gsm8k", "socratic", split="train")
    for i, example in enumerate(dataset):
        if i >= n:
            break
        print(f"\n--- Sample {i+1} ---")
        print(f"Question:\n  {example['question']}")
        print(f"\nRaw solution:\n  {example['answer']}")
        steps = parse_socratic_steps(example["answer"])
        print(f"\nParsed steps ({len(steps)}):")
        for j, step in enumerate(steps):
            print(f"  Step {j+1}: {step}")
        graph = steps_to_linear_dag(steps)
        print(f"\nConverted DAG (JSON):")
        print(graph.to_json())
        print(f"Valid DAG: {graph.is_valid()}")


def inspect_prompt(n: int = 1):
    print(f"\n{DIVIDER}")
    print("SECTION 2: FULL PROMPT SENT TO MODEL")
    print(DIVIDER)

    data = load_gsm8k_sft_data(max_samples=n)
    planner = Planner()

    for i, (problem, graph) in enumerate(data):
        print(f"\n--- Sample {i+1} ---")
        prompt = planner.build_prompt(problem)
        print(f"Prompt:\n{prompt}")
        print(f"\nExpected target (DAG JSON):\n{graph.to_json()}")


def inspect_model_output(n: int = 2, model_path: str = None):
    print(f"\n{DIVIDER}")
    label = f"SECTION 3: MODEL OUTPUT ({'trained: ' + model_path if model_path else 'untrained base model'})"
    print(label)
    print(DIVIDER)

    data = load_gsm8k_sft_data(max_samples=n)
    planner = Planner.load(model_path) if model_path else Planner()
    planner.model.eval()

    for i, (problem, expected_graph) in enumerate(data):
        print(f"\n--- Sample {i+1} ---")
        print(f"Problem: {problem}")

        prompt = planner.build_prompt(problem)
        inputs = planner.tokenizer(prompt, return_tensors="pt").to(planner.device)

        with torch.no_grad():
            output_ids = planner.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                pad_token_id=planner.tokenizer.eos_token_id,
            )
        generated = output_ids[0][inputs["input_ids"].shape[1]:]
        raw_text = planner.tokenizer.decode(generated, skip_special_tokens=True)

        print(f"\nRaw model output:\n{raw_text}")

        print(f"\nParsing attempt:")
        try:
            graph = planner._parse_output(raw_text)
            print(f"  Parse succeeded")
            print(f"  Valid DAG: {graph.is_valid()}")
            print(f"  Nodes: {len(graph.nodes)}, Edges: {len(graph.edges)}")
            print(f"  Generated graph:\n{graph.to_json()}")
        except Exception as e:
            print(f"  Parse FAILED: {e}")

        print(f"\nExpected graph:\n{expected_graph.to_json()}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=None, help="Path to trained model. If not set, loads base pretrained model.")
    parser.add_argument("--n", type=int, default=2)
    args = parser.parse_args()

    inspect_data(n=args.n)
    inspect_prompt(n=1)
    inspect_model_output(n=args.n, model_path=args.model_path)
