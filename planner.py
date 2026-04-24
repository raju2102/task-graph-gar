import json
import re
from typing import List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from task_graph import TaskGraph, TaskNode

SYSTEM_PROMPT = (
    "You are a mathematical problem decomposition planner. "
    "Given a math problem, decompose it into a directed acyclic graph (DAG) of sub-problems.\n\n"
    "Output a JSON object with exactly this format:\n"
    "{\n"
    '  "nodes": [\n'
    '    {"id": "v1", "subproblem": "natural language description of sub-task"},\n'
    "    ...\n"
    "  ],\n"
    '  "edges": [["v1", "v2"], ...]\n'
    "}\n\n"
    "Rules:\n"
    "- Each node is a self-contained sub-problem stated as a question\n"
    "- An edge [u, v] means v requires the output of u before it can be solved\n"
    "- Nodes with no shared dependency edge can be solved in parallel\n"
    "- Together the nodes must fully solve the original problem\n"
    "- Output ONLY the JSON object, nothing else"
)


def _get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _get_dtype(device: str):
    if device == "cuda":
        return torch.bfloat16
    if device == "mps":
        return torch.bfloat16
    return torch.float32


class Planner:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        device: Optional[str] = None,
        gradient_checkpointing: bool = True,
    ):
        self.device = device or _get_device()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=_get_dtype(self.device),
        )
        if gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        self.model.to(self.device)

    def build_prompt(self, problem: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Problem: {problem}"},
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def _parse_output(self, text: str) -> TaskGraph:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise ValueError(f"No JSON found in model output: {text[:200]}")
        return TaskGraph.from_json(match.group())

    def generate(
        self,
        problem: str,
        temperature: float = 0.8,
        max_new_tokens: int = 512,
    ) -> TaskGraph:
        prompt = self.build_prompt(problem)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        greedy = temperature == 0.0
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if not greedy else 1.0,
                do_sample=not greedy,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated = output_ids[0][inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        return self._parse_output(text)

    def generate_batch(
        self,
        problem: str,
        G: int = 8,
        temperature: float = 0.8,
        max_new_tokens: int = 512,
    ) -> List[Optional[TaskGraph]]:
        results = []
        for _ in range(G):
            try:
                graph = self.generate(problem, temperature=temperature, max_new_tokens=max_new_tokens)
                results.append(graph)
            except (ValueError, KeyError, json.JSONDecodeError):
                results.append(None)
        return results

    def save(self, path: str):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> "Planner":
        instance = cls.__new__(cls)
        instance.device = device or _get_device()
        instance.tokenizer = AutoTokenizer.from_pretrained(path)
        instance.model = AutoModelForCausalLM.from_pretrained(
            path,
            dtype=_get_dtype(instance.device),
        )
        instance.model.to(instance.device)
        return instance
