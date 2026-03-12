import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import random
import plotly.graph_objects as go

from rsn.spatial.mapping import get_unique_position
from rsn.storage.hash_storage import NodeStorage
from rsn.prediction.predictor import NodePredictor


# ==========================
# RSN Node
# ==========================

class RSNNode:

    def __init__(self, depth=0, parent=None):
        self.depth = depth
        self.parent = parent
        self.children = []

        # 6D feature vector
        self.features = [random.uniform(0,1) for _ in range(6)]

        # spatial position
        self.x, self.y, self.z = get_unique_position(
            depth,
            random.randint(0,100),
            sum(self.features)
        )

        self.value = None


# ==========================
# Setup modules
# ==========================

storage = NodeStorage()
predictor = NodePredictor()


# ==========================
# Generate RSN tree
# ==========================

def generate_tree(node, depth_limit=3, branch_factor=3):

    storage.save(f"{node.depth}-{id(node)}", node)

    if node.depth >= depth_limit:
        return

    for _ in range(branch_factor):

        child = RSNNode(depth=node.depth + 1, parent=node)
        node.children.append(child)

        generate_tree(child, depth_limit, branch_factor)


# ==========================
# Collect nodes
# ==========================

def collect_nodes():

    nodes = list(storage.store.values())

    xs = []
    ys = []
    zs = []
    colors = []

    for n in nodes:

        predictor.predict(n)

        xs.append(n.x)
        ys.append(n.y)
        zs.append(n.z)

        colors.append(n.value)

    return nodes, xs, ys, zs, colors


# ==========================
# Collect edges
# ==========================

def collect_edges(nodes):

    edge_x = []
    edge_y = []
    edge_z = []

    for n in nodes:

        if n.parent is not None:

            edge_x += [n.parent.x, n.x, None]
            edge_y += [n.parent.y, n.y, None]
            edge_z += [n.parent.z, n.z, None]

    return edge_x, edge_y, edge_z


# ==========================
# Visualization
# ==========================

def visualize():

    nodes, xs, ys, zs, colors = collect_nodes()

    edge_x, edge_y, edge_z = collect_edges(nodes)

    edge_trace = go.Scatter3d(
        x=edge_x,
        y=edge_y,
        z=edge_z,
        mode='lines',
        line=dict(width=2),
        hoverinfo='none'
    )

    node_trace = go.Scatter3d(
        x=xs,
        y=ys,
        z=zs,
        mode='markers',
        marker=dict(
            size=6,
            color=colors,
            colorscale='Viridis',
            colorbar=dict(title="Node Value"),
        ),
        text=[f"depth:{n.depth}" for n in nodes],
        hoverinfo='text'
    )

    fig = go.Figure(data=[edge_trace, node_trace])

    fig.update_layout(
        title="RSN Spatial Network Visualization",
        showlegend=False,
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
        )
    )

    fig.show()


# ==========================
# Main
# ==========================

def main():

    root = RSNNode()

    generate_tree(root, depth_limit=3, branch_factor=3)

    visualize()


if __name__ == "__main__":
    main()