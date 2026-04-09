"""Graph construction utilities for BioKGSuite."""

import networkx as nx
import pandas as pd


def build_graph(kg: pd.DataFrame, nodes: pd.DataFrame) -> nx.Graph:
    """Build an undirected NetworkX graph from edge and node tables.

    Parameters
    ----------
    kg : pd.DataFrame
        Edge table with columns x_index, y_index (integer node indices).
    nodes : pd.DataFrame
        Node table with columns idx, type, name.

    Returns
    -------
    nx.Graph
        Undirected graph with node attributes 'node_type' and 'name'.
    """
    G = nx.Graph()

    # Add nodes with attributes (itertuples is ~50x faster than iterrows)
    G.add_nodes_from(
        (row.idx, {'node_type': row.type, 'name': row.name})
        for row in nodes.itertuples(index=False)
    )

    # Add edges in bulk
    G.add_edges_from(zip(kg['x_index'].values, kg['y_index'].values))

    return G


def build_lookup_maps(nodes: pd.DataFrame) -> dict:
    """Build quick-lookup dictionaries from the node table.

    Parameters
    ----------
    nodes : pd.DataFrame
        Node table with columns idx, type, name.

    Returns
    -------
    dict with keys:
        'node_type_map' : {idx -> type}
        'node_name_map' : {idx -> name}
    """
    return {
        'node_type_map': dict(zip(nodes['idx'], nodes['type'])),
        'node_name_map': dict(zip(nodes['idx'], nodes['name'])),
    }


def find_node(name: str, node_name_map: dict, node_type_map: dict,
              node_type: str):
    """Find a node index by name and type (case-insensitive).

    Parameters
    ----------
    name : str
        Node name to search for.
    node_name_map : dict {idx -> name}
    node_type_map : dict {idx -> type}
    node_type : str
        Expected node type (e.g. 'drug', 'disease', 'gene/protein').

    Returns
    -------
    int or None — node index if found, else None.
    """
    name_lower = name.lower()
    for idx, n in node_name_map.items():
        if str(n).lower() == name_lower and node_type_map.get(idx) == node_type:
            return idx
    return None
