from mbi import Domain

import jax
import jax.numpy as jnp
from jax import lax

Parents = jax.Array
Ranks = jax.Array
Edges = jax.Array
AdjMatrix = jax.Array


def _attr_to_idx(domain, attr_name: str) -> int:
    """Converts an attribute name (string) to its corresponding integer index."""
    return domain.attrs.index(attr_name)


def _idx_to_attr(domain, idx: int) -> str:
    """Converts an integer index to its corresponding attribute name (string)."""
    return domain.attrs[idx]


def _edges_to_jax(edges: list[tuple[str]], domain) -> Edges:
    """Converts a list of string-based edges to a JAX array of integer-indexed edges."""
    jax_edges = []
    for u, v in edges:
        jax_edges.append([_attr_to_idx(domain, u), _attr_to_idx(domain, v)])
    return jnp.array(jax_edges, dtype=jnp.int32)


def _edges_from_jax(jax_edges: Edges, domain) -> list[tuple[str]]:
    """Converts a JAX array of integer-indexed edges to a list of string-based edges."""
    string_edges = []
    for u_idx, v_idx in jax_edges:
        string_edges.append((_idx_to_attr(domain, u_idx), _idx_to_attr(domain, v_idx)))
    return string_edges


def _find_root(parents: Parents, i: int) -> tuple[int, Parents]:
    """Finds the root of a node and performs path compression."""

    def find_root_cond(state: tuple[int, Parents]) -> bool:
        """Loop condition: continue while the current node is not its own parent."""
        current_node, parents_arr = state
        return parents_arr[current_node] != current_node

    def find_root_body(state: tuple[int, Parents]) -> tuple[int, Parents]:
        """Loop body: move to the parent of the current node."""
        current_node, parents_arr = state
        return parents_arr[current_node], parents_arr

    root, _ = lax.while_loop(find_root_cond, find_root_body, (i, parents))

    def path_compress_cond(state: tuple[int, Parents]) -> bool:
        """Loop condition: continue while the current node is not the root."""
        current_node, _ = state
        return current_node != root

    def path_compress_body(state: tuple[int, Parents]) -> tuple[int, Parents]:
        """Loop body: point the current node to the root and move to the next."""
        current_node, parents_arr = state
        next_node = parents_arr[current_node]
        parents_arr = parents_arr.at[current_node].set(root)
        return next_node, parents_arr

    _, parents = lax.while_loop(path_compress_cond, path_compress_body, (i, parents))

    return root, parents


def _union(
    parents: Parents,
    ranks: Ranks,
    i: int,
    j: int,
) -> tuple[Parents, Ranks, bool]:
    """Performs the union operation on two nodes in the Union-Find structure."""
    root_i, parents = _find_root(parents, i)
    root_j, parents = _find_root(parents, j)

    def merge() -> tuple[Parents, Ranks, bool]:
        def rank_i_greater() -> tuple[Parents, Ranks]:
            return parents.at[root_j].set(root_i), ranks

        def rank_j_greater() -> tuple[Parents, Ranks]:
            return parents.at[root_i].set(root_j), ranks

        def ranks_equal() -> tuple[Parents, Ranks]:
            new_parents = parents.at[root_i].set(root_j)
            new_ranks = ranks.at[root_j].add(1)
            return new_parents, new_ranks

        new_parents, new_ranks = lax.switch(
            jnp.sign(ranks[root_i] - ranks[root_j]) - 1,
            [rank_j_greater, ranks_equal, rank_i_greater],
        )

        return new_parents, new_ranks, True

    def no_op() -> tuple[Parents, Ranks, bool]:
        return parents, ranks, False

    return lax.cond(root_i != root_j, merge, no_op)


def _jax_kruskal(edges: Edges, num_nodes: int) -> Edges:
    """
    Computes the Minimum Spanning Tree (MST) using Kruskal's algorithm.

    This implementation is fully JAX-native. It iterates over a sorted list of
    edges using `jax.lax.scan` and uses a functional disjoint set data structure
    (Union-Find via `_find_root` and `_union`) to build the MST.

    Args:
        edges: A JAX array of shape (num_edges, 2) representing the graph
               edges, pre-sorted by weight in ascending order. The nodes
               in the edges should be integer indices.
        num_nodes: The total number of nodes in the graph.

    Returns:
        A JAX array of shape (num_nodes - 1, 2) containing the edges
        that form the MST.
    """

    def loop_body(
        carry: tuple[Parents, Ranks, Edges, int], edge: Edges
    ) -> tuple[tuple[Parents, Ranks, Edges, int], None]:
        """The body of the scan operation, processes one edge at a time."""
        parents, ranks, mst_edges, edge_count = carry
        u, v = edge[0], edge[1]

        new_parents, new_ranks, success = _union(parents, ranks, u, v)

        def add_edge_to_mst() -> Edges:
            return mst_edges.at[edge_count].set(edge)

        def keep_mst_same() -> Edges:
            return mst_edges

        updated_mst = lax.cond(success, add_edge_to_mst, keep_mst_same)
        new_edge_count = edge_count + success.astype(jnp.int32)

        return (new_parents, new_ranks, updated_mst, new_edge_count), None

    initial_parents = jnp.arange(num_nodes, dtype=jnp.int32)
    initial_ranks = jnp.zeros(num_nodes, dtype=jnp.int32)

    initial_mst = jnp.full((num_nodes - 1, 2), -1, dtype=jnp.int32)
    initial_edge_count = 0
    init_carry = (initial_parents, initial_ranks, initial_mst, initial_edge_count)

    (final_parents, _, final_mst, _), _ = lax.scan(loop_body, init_carry, edges)

    return final_mst


def kruskal(edges: list[tuple[str]], domain: Domain) -> list[tuple[str]]:
    """
    Computes the Minimum Spanning Tree (MST) using a JAX-native Kruskal's algorithm.

    This function acts as a user-friendly wrapper, converting string-based
    attribute names to integer indices for the JAX backend and converting
    the results back to string-based attribute names.

    Args:
        edges: A list of (attribute1, attribute2) tuples representing the edges.
               These edges are assumed to be pre-sorted by weight in ascending order
               for MST construction.
        domain: The Domain object defining the attributes and their mapping.

    Returns:
        A list of (attribute1, attribute2) tuples representing the edges
        that form the Minimum Spanning Tree.
    """
    jax_edges = _edges_to_jax(edges, domain)
    num_nodes = len(domain.attrs)

    mst_jax_edges = _jax_kruskal(jax_edges, num_nodes)

    valid_edges = mst_jax_edges[mst_jax_edges[:, 0] != -1]
    return _edges_from_jax(valid_edges, domain)
