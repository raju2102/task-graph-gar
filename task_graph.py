from dataclasses import dataclass, field
from typing import List, Tuple, Set
from collections import defaultdict, deque
import json


@dataclass
class TaskNode:
    id: str
    subproblem: str


@dataclass
class TaskGraph:
    nodes: List[TaskNode]
    edges: List[Tuple[str, str]]

    def node_ids(self) -> List[str]:
        return [n.id for n in self.nodes]

    def is_valid(self) -> bool:
        if len(self.nodes) < 2:
            return False
        ids = set(self.node_ids())
        for u, v in self.edges:
            if u not in ids or v not in ids:
                return False
        return self._is_acyclic()

    def _is_acyclic(self) -> bool:
        in_degree = defaultdict(int)
        adj = defaultdict(list)
        for n in self.node_ids():
            in_degree[n] = in_degree.get(n, 0)
        for u, v in self.edges:
            adj[u].append(v)
            in_degree[v] += 1
        queue = deque([n for n in self.node_ids() if in_degree[n] == 0])
        visited = 0
        while queue:
            node = queue.popleft()
            visited += 1
            for neighbor in adj[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        return visited == len(self.nodes)

    def topological_order(self) -> List[str]:
        in_degree = defaultdict(int)
        adj = defaultdict(list)
        for n in self.node_ids():
            in_degree[n] = in_degree.get(n, 0)
        for u, v in self.edges:
            adj[u].append(v)
            in_degree[v] += 1
        queue = deque([n for n in self.node_ids() if in_degree[n] == 0])
        order = []
        while queue:
            node = queue.popleft()
            order.append(node)
            for neighbor in adj[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        return order

    def get_frontier(self, completed: Set[str]) -> List[str]:
        predecessors = defaultdict(set)
        for u, v in self.edges:
            predecessors[v].add(u)
        return [
            n for n in self.node_ids()
            if n not in completed and predecessors[n].issubset(completed)
        ]

    def critical_path_length(self) -> int:
        if not self.nodes:
            return 0
        adj = defaultdict(list)
        for u, v in self.edges:
            adj[u].append(v)
        order = self.topological_order()
        dp = {}
        for node in reversed(order):
            if not adj[node]:
                dp[node] = 1
            else:
                dp[node] = 1 + max(dp[s] for s in adj[node])
        return max(dp.values())

    def to_dict(self) -> dict:
        return {
            "nodes": [{"id": n.id, "subproblem": n.subproblem} for n in self.nodes],
            "edges": [[u, v] for u, v in self.edges],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TaskGraph":
        nodes = [TaskNode(id=n["id"], subproblem=n["subproblem"]) for n in d["nodes"]]
        edges = [(e[0], e[1]) for e in d["edges"]]
        return cls(nodes=nodes, edges=edges)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, s: str) -> "TaskGraph":
        return cls.from_dict(json.loads(s))

    def __str__(self) -> str:
        lines = [f"TaskGraph ({len(self.nodes)} nodes, {len(self.edges)} edges)"]
        for node in self.nodes:
            lines.append(f"  [{node.id}] {node.subproblem}")
        if self.edges:
            lines.append("  Edges:")
            for u, v in self.edges:
                lines.append(f"    {u} -> {v}")
        return "\n".join(lines)
