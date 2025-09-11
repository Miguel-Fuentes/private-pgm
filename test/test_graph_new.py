import pytest
import jax
import jax.numpy as jnp
from mbi import Domain
from dev.graph import kruskal, sampling_order, _attr_to_idx, _idx_to_attr, _edges_to_jax, _edges_from_jax

# Helper for creating a simple Domain
def create_test_domain(attrs_list):
    # Assuming each attribute has 2 possible values (binary)
    data = {attr: 2 for attr in attrs_list}
    return Domain.fromdict(data)

def test_attr_to_idx():
    domain = create_test_domain(['A', 'B', 'C'])
    assert _attr_to_idx(domain, 'A') == 0
    assert _attr_to_idx(domain, 'B') == 1
    assert _attr_to_idx(domain, 'C') == 2

def test_idx_to_attr():
    domain = create_test_domain(['A', 'B', 'C'])
    assert _idx_to_attr(domain, 0) == 'A'
    assert _idx_to_attr(domain, 1) == 'B'
    assert _idx_to_attr(domain, 2) == 'C'

def test_edges_to_jax():
    domain = create_test_domain(['A', 'B', 'C'])
    edges = [('A', 'B'), ('B', 'C')]
    jax_edges = _edges_to_jax(edges, domain)
    expected_jax_edges = jnp.array([[0, 1], [1, 2]], dtype=jnp.int32)
    assert jnp.array_equal(jax_edges, expected_jax_edges)

def test_edges_from_jax():
    domain = create_test_domain(['A', 'B', 'C'])
    jax_edges = jnp.array([[0, 1], [1, 2]], dtype=jnp.int32)
    edges = _edges_from_jax(jax_edges, domain)
    expected_edges = [('A', 'B'), ('B', 'C')]
    assert edges == expected_edges

def test_kruskal_simple():
    domain = create_test_domain(['A', 'B', 'C'])
    # Edges sorted by implicit weight (e.g., A-B is "cheaper" than B-C)
    edges = [('A', 'B'), ('B', 'C'), ('A', 'C')]
    mst = kruskal(edges, domain)
    # For a simple 3-node graph, MST should have 2 edges
    assert len(mst) == 2
    # The MST should connect all nodes
    assert ('A', 'B') in mst or ('B', 'A') in mst
    assert ('B', 'C') in mst or ('C', 'B') in mst

def test_kruskal_disconnected_graph():
    domain = create_test_domain(['A', 'B', 'C', 'D'])
    # A-B, C-D are disconnected components
    edges = [('A', 'B'), ('C', 'D')]
    mst = kruskal(edges, domain)
    # Kruskal's will return edges that form a forest (multiple trees) if graph is disconnected
    # In this case, it should return both edges as they are the only ones
    assert len(mst) == 2
    assert ('A', 'B') in mst or ('B', 'A') in mst
    assert ('C', 'D') in mst or ('D', 'C') in mst

def test_sampling_order_simple():
    edge_list = [('A', 'B'), ('B', 'C')]
    order = sampling_order(edge_list)
    # BFS order starting from 'A' could be ['A', 'B', 'C']
    # or ['A', 'C', 'B'] depending on adjacency list iteration.
    # For this simple case, ['A', 'B', 'C'] is expected.
    assert order == ['A', 'B', 'C']

def test_sampling_order_complex():
    edge_list = [('A', 'B'), ('A', 'C'), ('B', 'D'), ('C', 'E')]
    order = sampling_order(edge_list)
    # Expected BFS order starting from 'A'
    # Level 0: A
    # Level 1: B, C (order depends on dict iteration, but B, C is common)
    # Level 2: D, E
    # So, ['A', 'B', 'C', 'D', 'E'] or ['A', 'C', 'B', 'E', 'D'] etc.
    # We need to check if it's a valid BFS order.
    # For simplicity, let's check if all nodes are present and it starts with root.
    assert len(order) == 5
    assert order[0] == 'A'
    assert set(order) == {'A', 'B', 'C', 'D', 'E'}
    # More robust test would involve checking levels, but for now, this is a start.
