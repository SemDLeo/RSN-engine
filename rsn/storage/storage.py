# rsn/storage/storage.py

class RSNStorage:
    def __init__(self):
        self.nodes = {}

    def store(self, node_id, node):
        self.nodes[node_id] = node

    def get(self, node_id):
        return self.nodes.get(node_id)

    def all_nodes(self):
        return list(self.nodes.values())