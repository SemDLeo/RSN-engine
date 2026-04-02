# examples/rsn_inference.py
import sys   
import os     
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np

from rsn.core.node import RSNNode
from rsn.prediction.transition_model import TransitionModel
from rsn.prediction.value_model import ValueModel
from rsn.prediction.tree import build_tree


def main():

    transition_model = TransitionModel()
    value_model = ValueModel()

    transition_model.load_state_dict(torch.load("transition.pth"))
    value_model.load_state_dict(torch.load("value.pth"))

    transition_model.eval()
    value_model.eval()

    initial_state = np.random.rand(6)

    root = RSNNode(initial_state)

    build_tree(root, transition_model, value_model, depth_limit=3)

    print("Tree built successfully.")


if __name__ == "__main__":
    main()