# rsn/core/tree.py
from rsn.core.node import RSNNode
import time

class RSNTree:
    """
    RSNTree manages the recursive spatial network tree.
    
    Attributes:
        root (RSNNode): Root node of the tree.
        nodes (list): Flat list of all nodes for traversal, visualization, or storage.
        max_depth (int): Maximum recursion depth.
        child_per_node (int): Branching factor (number of children per node).
        sleep_time (float): Optional delay for dynamic generation (visualization purposes).
    """
    
    def __init__(self, root_features=None, max_depth=10, child_per_node=2, sleep_time=0.0):
        self.root = RSNNode(depth=0, features=root_features)
        self.nodes = [self.root]
        self.max_depth = max_depth
        self.child_per_node = child_per_node
        self.sleep_time = sleep_time

    def generate(self):
        """
        Recursively generate the tree starting from the root node.
        """
        self._generate_recursive(self.root)

    def _generate_recursive(self, node):
        """
        Internal recursive function to expand children nodes.
        """
        if node.depth >= self.max_depth:
            return

        for _ in range(self.child_per_node):
            # Create a new node with the same features as parent by default
            child = RSNNode(depth=node.depth + 1, parent=node, features=node.features.copy())
            node.children.append(child)
            self.nodes.append(child)

            # Optional sleep for animation / visualization
            if self.sleep_time > 0:
                time.sleep(self.sleep_time)

            # Recursive call
            self._generate_recursive(child)

    def traverse(self):
        """
        Returns a generator for all nodes in the tree.
        """
        for node in self.nodes:
            yield node