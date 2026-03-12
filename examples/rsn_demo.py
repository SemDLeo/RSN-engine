# examples/rsn_demo.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import random
from rsn.spatial.mapping import get_unique_position
from rsn.storage.hash_storage import NodeStorage
from rsn.prediction.predictor import NodePredictor
from rsn.prediction.scoring import ScoreEvaluator
from rsn.prediction.optimizer import PathOptimizer


# =============================
# Node Class
# =============================
class RSNNode:
    """
    Basic RSN node
    """
    def __init__(self, depth=0, parent=None):
        self.depth = depth
        self.parent = parent
        self.children = []
        # Example feature vector: 6D
        self.features = [random.uniform(0, 1) for _ in range(6)]
        # Node position
        self.x, self.y, self.z = get_unique_position(depth, random.randint(0,100), sum(self.features))
        # Placeholder value
        self.value = None

# =============================
# Setup modules
# =============================
storage = NodeStorage()
predictor = NodePredictor()
scorer = ScoreEvaluator()
optimizer = PathOptimizer()

# =============================
# Recursive tree generation
# =============================
def generate_tree(node, depth_limit=3, branch_factor=2):
    """
    Recursively generate RSN nodes
    """
    if node.depth >= depth_limit:
        return

    storage.save(f"{node.depth}-{id(node)}", node)

    for _ in range(branch_factor):
        child = RSNNode(depth=node.depth+1, parent=node)
        node.children.append(child)
        generate_tree(child, depth_limit, branch_factor)

# =============================
# Demo execution
# =============================
def main():
    root = RSNNode()
    print(f"Root Node: depth={root.depth}, pos=({root.x:.2f},{root.y:.2f},{root.z:.2f})")

    # Generate tree
    generate_tree(root, depth_limit=3, branch_factor=3)

    # Predict and score nodes
    all_nodes = storage.store.values()
    for node in all_nodes:
        predictor.predict(node)
        score = scorer.evaluate(node)
        print(f"Node depth={node.depth}, pos=({node.x:.2f},{node.y:.2f},{node.z:.2f}), value={node.value:.2f}, score={score:.2f}")

    # Optimize path: select best node
    best_node = optimizer.select_best(all_nodes)
    print("\nBest node selected:")
    print(f"Depth={best_node.depth}, pos=({best_node.x:.2f},{best_node.y:.2f},{best_node.z:.2f}), value={best_node.value:.2f}")

if __name__ == "__main__":
    main()