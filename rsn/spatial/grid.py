# rsn/spatial/grid.py
"""
Grid management for RSN spatial module
"""

class GridManager:
    """
    Manage nodes in discrete 3D grid for indexing and retrieval
    """
    def __init__(self, grid_size=0.1):
        self.grid_size = grid_size
        self.grid = {}

    def add_node(self, node, pos):
        key = tuple(int(c / self.grid_size) for c in pos)
        self.grid[key] = node

    def get_node(self, pos):
        key = tuple(int(c / self.grid_size) for c in pos)
        return self.grid.get(key, None)