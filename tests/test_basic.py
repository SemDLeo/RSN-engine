# tests/test_basic.py

from rsn.core.node import RSNNode

def test_node():
    node = RSNNode([1,2,3])
    assert node.depth == 0