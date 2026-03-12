# rsn/spatial/collision.py
"""
Collision detection for spatial node placement
"""

def check_collision(pos, existing_positions: set):
    """
    Check if a position collides with existing nodes

    Parameters:
        pos (tuple): (x, y, z)
        existing_positions (set): set of existing positions

    Returns:
        bool: True if collides
    """
    return pos in existing_positions