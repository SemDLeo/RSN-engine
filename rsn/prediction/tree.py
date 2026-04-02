# rsn/prediction/tree.py

import torch
import numpy as np
from rsn.core.node import RSNNode

def expand_node(node, transition_model, value_model, branch_factor=3):

    for _ in range(branch_factor):

        x = torch.tensor(node.features, dtype=torch.float32)

        mu, logvar = transition_model(x)

        mu = mu.detach().numpy()
        sigma = np.exp(0.5 * logvar.detach().numpy())

        # Sample next features from the predicted distribution
        next_features = np.random.normal(mu, sigma)

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