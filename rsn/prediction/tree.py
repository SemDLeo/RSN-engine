# rsn/prediction/tree.py

import torch
import numpy as np
from rsn.core.node import Node


def expand_node(node, transition_model, value_model, branch_factor=3):

    x = torch.tensor(node.state, dtype=torch.float32).unsqueeze(0)

    for _ in range(branch_factor):

        mu, logvar = transition_model(x)

        mu = mu.item()
        sigma = np.exp(0.5 * logvar.item())

        # 🔥 sample return
        r = np.random.normal(mu, sigma)

        current_price = node.metadata["price"]
        next_price = current_price * (1 + r)

        next_state = node.state.copy()
        next_state[0] = next_price  # 更新Close

        child = Node(
            state=next_state,
            parent=node,
            depth=node.depth + 1,
            metadata={
                "price": next_price,
                "return": r,
                "cum_return": node.metadata.get("cum_return", 1.0) * (1 + r)
            }
        )

        value = value_model(torch.tensor(next_state, dtype=torch.float32)).item()
        child.value = value

        node.add_child(child)


def build_tree(root, transition_model, value_model, depth_limit=3):

    if root.depth >= depth_limit:
        return

    expand_node(root, transition_model, value_model)

    for child in root.children:
        build_tree(child, transition_model, value_model, depth_limit)