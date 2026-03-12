# rsn/storage/retrieval.py
"""
Node retrieval tools
"""

class NodeRetrieval:
    """
    Provides query methods for RSN nodes
    """
    @staticmethod
    def nearest_node(nodes, target_pos):
        """
        Return node closest to target_pos
        """
        min_dist = float('inf')
        nearest = None
        for node in nodes:
            dist = sum((a-b)**2 for a,b in zip(node.x, target_pos))**0.5
            if dist < min_dist:
                min_dist = dist
                nearest = node
        return nearest