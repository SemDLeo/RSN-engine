# rsn/core/node.py

class RSNNode:
    def __init__(self, features, parent=None, depth=0):
        self.features = features
        self.parent = parent
        self.depth = depth
        self.children = []
        self.value = None

    def add_child(self, child):
        self.children.append(child)