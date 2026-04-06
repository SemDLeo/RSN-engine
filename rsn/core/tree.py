# rsn/core/tree.py

from typing import Callable, List
from rsn.core.node import Node


class Tree:
    """
    Generic Tree Structure

    This class:
    - Manages root
    - Provides traversal
    - Provides aggregation utilities

    It does NOT depend on:
    - ML models
    - Financial logic
    """

    def __init__(self, root: Node):
        self.root = root

    # -------------------------
    # Traversal
    # -------------------------
    def traverse_dfs(self, node: Node = None) -> List[Node]:
        node = node or self.root
        nodes = [node]
        for child in node.children:
            nodes.extend(self.traverse_dfs(child))
        return nodes

    def traverse_bfs(self) -> List[Node]:
        queue = [self.root]
        result = []

        while queue:
            current = queue.pop(0)
            result.append(current)
            queue.extend(current.children)

        return result

    # -------------------------
    # Leaf nodes
    # -------------------------
    def get_leaves(self) -> List[Node]:
        return [node for node in self.traverse_dfs() if node.is_leaf()]

    # -------------------------
    # Depth
    # -------------------------
    def max_depth(self) -> int:
        return max(node.depth for node in self.traverse_dfs())

    # -------------------------
    # Aggregation
    # -------------------------
    def aggregate(self, fn: Callable[[Node], float]) -> float:
        values = [fn(node) for node in self.traverse_dfs()]
        return sum(values) / len(values) if values else 0.0

    # -------------------------
    # Best Node (for decision)
    # -------------------------
    def best_leaf(self, key: Callable[[Node], float]) -> Node:
        leaves = self.get_leaves()
        if not leaves:
            return None
        return max(leaves, key=key)

    # -------------------------
    # Backpropagation (MCTS)
    # -------------------------
    def backpropagate(self, node: Node, reward: float):
        current = node
        while current:
            current.update(reward)
            current = current.parent

    # -------------------------
    # Debug
    # -------------------------
    def summary(self):
        nodes = self.traverse_dfs()
        print(f"Total nodes: {len(nodes)}")
        print(f"Max depth: {self.max_depth()}")