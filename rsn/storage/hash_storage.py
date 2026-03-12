# rsn/storage/hash_storage.py
"""
Simple hash-based storage for RSN nodes
"""

class NodeStorage:
    """
    In-memory storage using node_id as key
    """
    def __init__(self):
        self.store = {}

    def save(self, node_id, node):
        self.store[node_id] = node

    def load(self, node_id):
        return self.store.get(node_id, None)