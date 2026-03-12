# rsn/core/node.py
import numpy as np

class RSNNode:
    """
    RSNNode represents a node in the Recursive Spatial Network (RSN).
    
    Attributes:
        depth (int): The recursion depth of the node.
        parent (RSNNode or None): Parent node reference.
        hist_index (int): Historical index or step in the input data.
        children (list): List of child RSNNode objects.
        features (np.ndarray): Feature vector representing node state (default: 6D zero vector).
        value (float): Optional evaluation value associated with the node.
        action (any): Optional action associated with the node (domain-specific).
        x, y, z (float): Unique spatial coordinates for visualization or distributed mapping.
    """

    GRID_SIZE = 0.1  # Default grid size for spatial hashing
    used_positions = set()  # Global set to track used positions

    def __init__(self, depth, parent=None, hist_index=0, features=None):
        self.depth = depth
        self.parent = parent
        self.hist_index = hist_index
        self.children = []

        # Node features (default 6D vector if not provided)
        self.features = features if features is not None else np.zeros(6)

        # Node evaluation value (can be used by prediction module)
        self.value = 0.0

        # Node action placeholder (user-defined for any domain)
        self.action = None

        # Unique spatial position for visualization or distributed storage
        self.x, self.y, self.z = self._get_unique_position()

    def _get_unique_position(self):
        """
        Generate a unique 3D spatial position for this node
        using a hash of depth, hist_index, and features.
        """
        attempt = 0
        while True:
            node_id = f"{self.depth}-{self.hist_index}-{hash(self.features.tostring())}-{attempt}"
            h = hash(node_id)
            gx = (h & 0xFFFF) % 1000
            gy = ((h >> 16) & 0xFFFF) % 1000
            gz = ((h >> 32) & 0xFFFF) % 1000
            x = gx * self.GRID_SIZE + np.random.uniform(-0.03, 0.03)
            y = gy * self.GRID_SIZE + np.random.uniform(-0.03, 0.03)
            z = gz * self.GRID_SIZE + np.random.uniform(-0.03, 0.03)
            pos = (round(x, 4), round(y, 4), round(z, 4))
            if pos not in RSNNode.used_positions:
                RSNNode.used_positions.add(pos)
                return pos
            attempt += 1