"""Communication-graph construction for the resilient MARL agents.

train.py's build_neighbor_list() calls construct_r_redundant_graph(n, r, seed) once per
training batch to get a fresh graph over which agents exchange candidate weights; r
(= 2*F_BUDGET_G + 1, see train.py) is the robustness parameter the resilient/projection/
trimmed_mean defenses rely on to outvote or out-weigh up to F_BUDGET_G corrupted
neighbors -- every node needs at least r independent neighbors for those defenses'
trim/vote thresholds to be meaningful.
"""

import numpy as np
import networkx as nx

def adjacency_matrix(edge, n):
    """Convert an edge list/view into a symmetric n x n 0/1 adjacency matrix (sets both
    A[i, j] and A[j, i] for every edge)."""

    A = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1,n):
            if (i,j) in edge or (j,i) in edge:
                A[i, j] = 1
                A[j, i] = 1
    return A


def find_r(A, upper_r, lower_r):
    """Check the (upper_r, lower_r)-redundancy of a given adjacency matrix A. 
    Not currently called anywhere in this codebase -- kept as a standalone way to 
    verify a graph's redundancy property."""
    n = A.shape[0]

    # Check if the 2-hop graph is strongly connected
    A_connectivity = (A @ A) + A

    # Check if the each element is in betweeen upper and lower thresholds, which means it is not (upper_r, lower_r)-redundant
    first_check = np.any((A_connectivity > lower_r) & (A_connectivity < upper_r))


    # return False if there is any such element
    if first_check:
        return False

    # Check if the 2-hop graph is connected
    A_last_checks = np.where(A_connectivity>=upper_r, 1, 0)

    G = nx.Graph(A_last_checks)
    return nx.is_connected(G)


def construct_r_redundant_graph(n, r, seed=None):
    """Build a fixed topology -- the first r structural nodes form a clique, and every
    other node connects to all r of them -- so every node has at least r neighbors,
    then randomly permute which agent index lands on which structural node. Returns the
    resulting n x n adjacency matrix."""

    rng = np.random.default_rng(seed)

    # Construct the graph topology
    G = nx.Graph()
    G.add_nodes_from(range(n))

    # First r structural nodes form a clique
    for i in range(r):
        for j in range(i + 1, r):
            G.add_edge(i, j)

    # Every remaining structural node connects to all clique nodes
    for i in range(r, n):
        for j in range(r):
            G.add_edge(i, j)

    # Randomly assign actual agent indices to structural nodes
    permutation = rng.permutation(n)
    mapping = {i: permutation[i] for i in range(n)}

    G = nx.relabel_nodes(G, mapping)

    return adjacency_matrix(G.edges(), n)
