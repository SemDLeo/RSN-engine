# rsn/prediction/tree.py

import torch
import numpy as np
from rsn.core.node import RSNNode

def expand_node(node, transition_model, value_model, branch_factor=3, sigma=0.05):

    for _ in range(branch_factor):

        x = torch.tensor(node.features, dtype=torch.float32)

        mu = transition_model(x).detach().numpy()

        noise = np.random.normal(0, sigma, size=mu.shape)
        next_features = mu + noise

        child = RSNNode(next_features, parent=node, depth=node.depth + 1)

        value = value_model(torch.tensor(next_features, dtype=torch.float32)).item()
        child.value = value

        node.children.append(child)


def build_tree(root, transition_model, value_model, depth_limit=3):

    if root.depth >= depth_limit:
        return

    expand_node(root, transition_model, value_model)

    for child in root.children:
        build_tree(child, transition_model, value_model, depth_limit)