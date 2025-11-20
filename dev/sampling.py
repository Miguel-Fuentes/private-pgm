from mbi import (
    Factor,
)

import jax
import jax.numpy as jnp

from typing import Iterable, Dict

Key = jax.Array
Edges = jax.Array
Probs = jax.Array
Counts = jax.Array
SamplingOrder = jax.Array
SampledValues = jax.Array
Marginals = Dict[tuple[int, int], Counts]


def _attr_to_idx(domain, attr_name: str) -> int:
    """Converts an attribute name (string) to its corresponding integer index."""
    return domain.attrs.index(attr_name)


def _idx_to_attr(domain, idx: int) -> str:
    """Converts an integer index to its corresponding attribute name (string)."""
    return domain.attrs[idx]


def _jax_best_permutation(edges: Edges, scores: jax.Array) -> Edges:
    """
    Deterministically permutes a list of edges based on their scores.
    """
    descending_indices = jnp.argsort(scores)[::-1]
    return edges[descending_indices]


def _jax_random_permutation(edges: Edges, scores: jax.Array, key: Key) -> Edges:
    """
    Randomly permutes a list of edges based on scores plus Gumbel noise.
    """
    noise = jax.random.gumbel(key, shape=scores.shape)
    noisy_scores = jnp.log(scores + 1e-20) + noise
    return _jax_best_permutation(edges, noisy_scores)


def condition(marg: Factor, conditions: dict[str, int]) -> Factor:
    """
    Conditions a marginal distribution (Factor) on specified attribute values.

    Args:
        marg: The marginal distribution as a Factor object.
        conditions: A dictionary mapping attribute names (strings) to their
                    fixed integer values.

    Returns:
        A new Factor object representing the conditioned marginal distribution.

    Raises:
        ValueError: If an attribute in conditions is not in the marginal's domain
                    or if a value is out of bounds for its attribute.
    """
    for attribute, value in conditions.items():
        if attribute not in marg.domain.attrs:
            raise ValueError(f"Attribute {attribute} not in marginal domain")

        attribute_size = marg.domain[attribute]
        if not (0 <= value < attribute_size):
            raise ValueError(f"Value {value} out of bounds for attribute {attribute}")

    slicer = [slice(None)] * len(marg.domain.attrs)
    for attribute, value in conditions.items():
        axis_idx = marg.domain.attrs.index(attribute)
        slicer[axis_idx] = value

    conditioned_values = marg.values[tuple(slicer)]
    new_domain = marg.domain.marginalize(list(conditions.keys()))
    return Factor(new_domain, conditioned_values)


def order(canonical_order: tuple[str], atts: Iterable[str]) -> tuple[str]:
    """
    Orders a subset of attributes according to a given canonical order.

    Args:
        canonical_order: A tuple of strings representing the desired canonical order of attributes.
        atts: An iterable of strings representing the subset of attributes to order.

    Returns:
        A tuple of strings containing the attributes from 'atts', ordered
        according to their appearance in 'canonical_order'.
    """
    return tuple(s for s in canonical_order if s in atts)


def best_permutation(options: list[str], scores: jnp.array) -> list[str]:
    """
    Deterministically permutes a list of options based on their scores.

    Args:
        options: A list of strings representing the items to be permuted.
        scores: A JAX array of scores corresponding to each option.

    Returns:
        A list of strings representing the options, sorted in descending order
        of their scores.
    """
    descending_indices = jnp.argsort(scores)[::-1]
    return [options[i] for i in descending_indices]


def random_permutation(scores: dict[tuple[str], float], key, domain) -> list[tuple]:
    """
    Randomly permutes a list of attribute pairs (edges) based on their scores
    using a JAX-native Gumbel-Max trick.

    Args:
        scores: A dictionary mapping (attribute1, attribute2) tuples to their
                corresponding float scores.
        key: A JAX random key.
        domain: The Domain object defining the attributes and their mapping.

    Returns:
        A list of (attribute1, attribute2) tuples representing the permuted
        attribute pairs.
    """
    # Convert dict[tuple[str], float] to Edges and scores
    string_edges = list(scores.keys())
    float_scores = jnp.array(list(scores.values()))

    # Convert string_edges to JAX Edges (integer indices)
    jax_edges = []
    for u, v in string_edges:
        jax_edges.append([_attr_to_idx(domain, u), _attr_to_idx(domain, v)])
    jax_edges = jnp.array(jax_edges, dtype=jnp.int32)

    # Call _jax_random_permutation
    permuted_jax_edges = _jax_random_permutation(jax_edges, float_scores, key)

    valid_edges = permuted_jax_edges[permuted_jax_edges[:, 0] != -1]

    # Convert permuted_jax_edges back to list[tuple[str]]
    permuted_string_edges = []
    for u_idx, v_idx in valid_edges:
        permuted_string_edges.append(
            (_idx_to_attr(domain, u_idx), _idx_to_attr(domain, v_idx))
        )

    return permuted_string_edges
