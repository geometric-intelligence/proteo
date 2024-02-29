import graph_tool as gt  # need to import before pytorch
import graph_tool.topology as top
import matplotlib.pyplot as plt
import networkx as nx
import torch


def get_rings(edge_index, max_k):
    if isinstance(edge_index, torch.Tensor):
        edge_index = edge_index.numpy()

    edge_list = edge_index.T
    graph_gt = gt.Graph(directed=False)
    graph_gt.add_edge_list(edge_list)
    gt.stats.remove_self_loops(graph_gt)
    gt.stats.remove_parallel_edges(graph_gt)
    # We represent rings with their original node ordering
    # so that we can easily read out the boundaries
    # The use of the `sorted_rings` set allows to discard
    # different isomorphisms which are however associated
    # to the same original ring – this happens due to the intrinsic
    # symmetries of cycles
    rings = set()
    sorted_rings = set()
    for k in range(3, max_k + 1):
        pattern = nx.cycle_graph(k)
        pattern_edge_list = list(pattern.edges)
        pattern_gt = gt.Graph(directed=False)
        pattern_gt.add_edge_list(pattern_edge_list)
        sub_isos = top.subgraph_isomorphism(
            pattern_gt, graph_gt, induced=True, subgraph=True, generator=False
        )
        print("k", k, "len(sub_isos)", len(sub_isos))
        # sub_iso_sets = map(lambda isomorphism: tuple(isomorphism.a), sub_isos)
        list_sub_iso_sets = [tuple(isomorphism.a) for isomorphism in sub_isos]

        for iso in list_sub_iso_sets:
            if tuple(sorted(iso)) not in sorted_rings:
                rings.add(iso)
                sorted_rings.add(tuple(sorted(iso)))
    rings = list(rings)
    return rings


# Your id_maps
id_maps = [
    {
        (0,): 0,
        (1,): 1,
        (2,): 2,
        (3,): 3,
        (4,): 4,
        (5,): 5,
        (6,): 6,
        (7,): 7,
        (8,): 8,
        (9,): 9,
        (10,): 10,
        (11,): 11,
        (12,): 12,
        (13,): 13,
        (14,): 14,
        (15,): 15,
        (16,): 16,
        (17,): 17,
        (18,): 18,
        (19,): 19,
        (20,): 20,
        (21,): 21,
    },
    {
        (0, 1): 0,
        (1, 2): 1,
        (2, 3): 2,
        (2, 7): 3,
        (3, 4): 4,
        (4, 5): 5,
        (5, 6): 6,
        (5, 7): 7,
        (7, 8): 8,
        (8, 9): 9,
        (8, 10): 10,
        (10, 11): 11,
        (11, 12): 12,
        (11, 16): 13,
        (12, 13): 14,
        (13, 14): 15,
        (14, 15): 16,
        (15, 16): 17,
        (16, 17): 18,
        (17, 18): 19,
        (18, 19): 20,
        (19, 20): 21,
        (19, 21): 22,
    },
]

edge_index = torch.tensor(
    [
        [
            0,
            1,
            1,
            2,
            2,
            2,
            3,
            3,
            4,
            4,
            5,
            5,
            5,
            6,
            7,
            7,
            7,
            8,
            8,
            8,
            9,
            10,
            10,
            11,
            11,
            11,
            12,
            12,
            13,
            13,
            14,
            14,
            15,
            15,
            16,
            16,
            16,
            17,
            17,
            18,
            18,
            19,
            19,
            19,
            20,
            21,
        ],
        [
            1,
            0,
            2,
            1,
            3,
            7,
            2,
            4,
            3,
            5,
            4,
            6,
            7,
            5,
            2,
            5,
            8,
            7,
            9,
            10,
            8,
            8,
            11,
            10,
            12,
            16,
            11,
            13,
            12,
            14,
            13,
            15,
            14,
            16,
            11,
            15,
            17,
            16,
            18,
            17,
            19,
            18,
            20,
            21,
            19,
            19,
        ],
    ]
)

makegraph = True

if makegraph:
    G = nx.DiGraph()

    # Add nodes
    for node_id, label in id_maps[0].items():
        G.add_node(label, label=str(node_id))

    # Add edges
    for edge_id, label in id_maps[1].items():
        print(edge_id)
        print(label)
        node1, node2 = edge_id
        print(node1)
        print(node2)
        print("we are adding edge between ", node1, " and ", node2)
        G.add_edge(node1, node2, label=str(label))

    # Plot the graph
    pos = nx.spring_layout(G)  # You can use other layout algorithms as well
    nx.draw(
        G,
        pos,
        with_labels=True,
        font_weight='bold',
        node_size=70,
        node_color='skyblue',
        font_size=10,
        alpha=0.5,
    )

    # # Add edge labels
    # edge_labels = {(id_maps[0][node1], id_maps[0][node2]): str(edge_id) for edge_id, (node1, node2) in id_maps[1].items()}
    # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', font_size=6)

    # Save the plot
    plt.savefig('graph2.png')

# see if get_rings works
max_k = 18
rings = get_rings(edge_index, max_k)
print("rings", rings)
