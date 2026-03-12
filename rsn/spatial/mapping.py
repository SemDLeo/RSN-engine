# rsn/spatial/mapping.py
"""
Mapping algorithms for nodes in RSN space
"""

import numpy as np

used_positions = set()
GRID_SIZE = 0.1

def get_unique_position(depth: int, hist_index: int, price: float):
    """
    Generate a deterministic unique 3D position for a node.
    Uses hash + grid offset to avoid collisions.

    Parameters:
        depth (int): Node depth in tree
        hist_index (int): Historical index
        price (float): Node price / value

    Returns:
        tuple: (x, y, z) coordinates
    """
    attempt = 0
    while True:
        node_id = f"{depth}-{hist_index}-{price:.4f}-{attempt}"
        h = hash(node_id)
        gx = (h & 0xFFFF) % 1000
        gy = ((h >> 16) & 0xFFFF) % 1000
        gz = ((h >> 32) & 0xFFFF) % 1000
        x = gx * GRID_SIZE + np.random.uniform(-0.03, 0.03)
        y = gy * GRID_SIZE + np.random.uniform(-0.03, 0.03)
        z = gz * GRID_SIZE + np.random.uniform(-0.03, 0.03)
        pos = (round(x, 4), round(y, 4), round(z, 4))
        if pos not in used_positions:
            used_positions.add(pos)
            return pos
        attempt += 1