# rsn/spatial/mapping.py

import numpy as np

used_positions = set()
GRID_SIZE = 0.1

def get_unique_position(node_id):
    attempt = 0
    while True:
        h = hash(f"{node_id}-{attempt}")

        x = ((h & 0xFFFF) % 1000) * GRID_SIZE
        y = (((h >> 16) & 0xFFFF) % 1000) * GRID_SIZE
        z = (((h >> 32) & 0xFFFF) % 1000) * GRID_SIZE

        pos = (round(x, 4), round(y, 4), round(z, 4))

        if pos not in used_positions:
            used_positions.add(pos)
            return pos

        attempt += 1