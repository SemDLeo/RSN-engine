# rsn/core/node.py

from typing import Any, Dict, List, Optional


class Node:
    """
    Generic Tree Node for RSN Engine

    This node is domain-agnostic:
    - It does NOT assume finance, price, or ML
    - It only stores state + structure + evaluation

    You can attach ANY data into `state`
    """

    def __init__(
        self,
        state: Any,
        parent: Optional["Node"] = None,
        depth: int = 0,
        metadata: Optional[Dict] = None,
    ):
        self.state = state                # Core representation (feature / price / embedding / etc.)
        self.parent = parent
        self.children: List["Node"] = []
        self.depth = depth

        # Evaluation (used by MCTS / scoring / strategy)
        self.value: float = 0.0
        self.visits: int = 0

        # Optional extra info (timestamps, actions, etc.)
        self.metadata: Dict = metadata or {}

    # -------------------------
    # Tree Structure
    # -------------------------
    def add_child(self, child: "Node"):
        self.children.append(child)

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def is_root(self) -> bool:
        return self.parent is None

    # -------------------------
    # Traversal
    # -------------------------
    def get_path(self) -> List["Node"]:
        node = self
        path = []
        while node is not None:
            path.append(node)
            node = node.parent
        return list(reversed(path))

    def get_ancestors(self) -> List["Node"]:
        node = self.parent
        ancestors = []
        while node:
            ancestors.append(node)
            node = node.parent
        return ancestors

    # -------------------------
    # Statistics (for MCTS)
    # -------------------------
    def update(self, reward: float):
        self.visits += 1
        self.value += reward

    def average_value(self) -> float:
        if self.visits == 0:
            return 0.0
        return self.value / self.visits

    # -------------------------
    # Debug
    # -------------------------
    def __repr__(self):
        return f"<Node depth={self.depth} children={len(self.children)} value={self.value:.4f}>"