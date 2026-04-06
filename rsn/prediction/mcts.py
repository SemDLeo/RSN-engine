# rsn/prediction/mcts.py

import math
import random


def ucb(node, c=1.4):
    if node.visits == 0:
        return float("inf")
    return node.average_value() + c * math.sqrt(math.log(node.parent.visits + 1) / node.visits)


def select(node):
    while not node.is_leaf():
        node = max(node.children, key=ucb)
    return node


def simulate(node):
    return node.metadata.get("cum_return", 1.0)


def backprop(node, reward):
    while node:
        node.update(reward)
        node = node.parent


def run_mcts(root, iterations=50):

    for _ in range(iterations):
        leaf = select(root)
        reward = simulate(leaf)
        backprop(leaf, reward)

    best = max(root.children, key=lambda n: n.average_value())
    return best