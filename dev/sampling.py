from mbi import (
    Factor,
    Domain,
    Dataset,
)

import jax
import jax.numpy as jnp
import pandas as pd

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


def _factor_to_counts(marg: Factor) -> Counts:
    """Extracts the counts (values) from a Factor object as a JAX array."""
    return marg.values


def _margs_to_jax_margs(margs: dict[str, Factor], domain) -> Marginals:
    """
    Converts a dictionary of string-keyed Factor objects to a JAX-compatible
    dictionary of marginals with integer-indexed keys and JAX array values.
    """
    jax_margs = {}
    for marg_key_str, factor_obj in margs.items():
        jax_marg_key = tuple(sorted([_attr_to_idx(domain, attr) for attr in marg_key_str]))
        jax_margs[jax_marg_key] = _factor_to_counts(factor_obj)
    return jax_margs


def _sampling_order_to_jax(sampling_order: list[str], domain) -> SamplingOrder:
    """Converts a list of string-based attribute names to a JAX array of integer indices."""
    return jnp.array([_attr_to_idx(domain, attr) for attr in sampling_order], dtype=jnp.int32)


def _jax_record_to_dict(jax_record: jax.Array, canonical_order: tuple[str], domain) -> dict[str, int]:
    """
    Converts a single JAX-array record (integer values) back to a dictionary
    with string attribute names as keys.
    """
    result_dict = {}
    for i, attr_name in enumerate(canonical_order):
        result_dict[attr_name] = int(jax_record[i])
    return result_dict


def _jax_best_permutation(edges: Edges, scores: jax.Array) -> Edges:
    """
    Deterministically permutes a list of edges based on their scores.
    """
    descending_indices = jnp.argsort(scores)[::-1]
    return edges[descending_indices]


def _jax_random_permutation(
    edges: Edges, scores: jax.Array, key: Key
) -> Edges:
    """
    Randomly permutes a list of edges based on scores plus Gumbel noise.
    """
    noise = jax.random.gumbel(key, shape=scores.shape)
    noisy_scores = jnp.log(scores + 1e-20) + noise
    return _jax_best_permutation(edges, noisy_scores)


def _jax_sample(counts: Counts, key: Key) -> tuple[int, ...]:
    """
    Samples a single outcome from an n-dimensional counts table.
    """
    original_shape = counts.shape
    flat_counts = counts.flatten()
    total = jnp.sum(flat_counts)
    probs = flat_counts / (total + 1e-10)
    sampled_idx = jax.random.choice(key, a=flat_counts.size, p=probs).astype(jnp.int32)
    return jnp.unravel_index(sampled_idx, original_shape)


def _get_conditional_sampler(counts_table_2d: Counts):
    """
    A factory that creates a JIT-compiled, vmapped function to sample a
    column of child values conditioned on a column of parent values.
    """
    def _sample_one_child(parent_val: int, key: Key) -> jnp.int32:
        conditional_counts = counts_table_2d[parent_val, :]
        probs = conditional_counts / (conditional_counts.sum() + 1e-8)
        return jax.random.choice(key, a=probs.size, p=probs).astype(jnp.int32)

    return jax.jit(jax.vmap(_sample_one_child, in_axes=(0, 0)))


def _jax_generate_data_from_tree(
    margs: Marginals,
    sampling_order: SamplingOrder,
    num_nodes: int,
    total: int,
    key: Key,
) -> jax.Array:
    """
    Generates a full synthetic dataset using a column-by-column approach.
    """
    data = jnp.zeros((total, num_nodes), dtype=jnp.int32)

    key, subkey = jax.random.split(key)
    att1, att2 = int(sampling_order[0]), int(sampling_order[1])

    root_key = tuple(sorted((att1, att2)))
    root_counts = margs[root_key]

    if att1 > att2:
        root_counts = root_counts.T

    flat_probs = root_counts.flatten() / (root_counts.sum() + 1e-8)
    sampled_indices = jax.random.choice(
        subkey, a=flat_probs.size, shape=(total,), p=flat_probs
    )
    v1, v2 = jnp.unravel_index(sampled_indices, root_counts.shape)

    data = data.at[:, att1].set(v1.astype(jnp.int32))
    data = data.at[:, att2].set(v2.astype(jnp.int32))

    for i in range(1, num_nodes - 1):
        key, subkey = jax.random.split(key)
        parent_att = int(sampling_order[i])
        child_att = int(sampling_order[i + 1])

        marg_key = tuple(sorted((parent_att, child_att)))
        counts_table = margs[marg_key]
        if parent_att > child_att:
            counts_table = counts_table.T

        conditional_sampler = _get_conditional_sampler(counts_table)
        parent_column = data[:, parent_att]
        keys = jax.random.split(subkey, total)
        child_column = conditional_sampler(parent_column, keys)

        data = data.at[:, child_att].set(child_column)

    return data


def sample(marg: Factor, key: jax.Array):
    """
    Samples a single outcome from a given marginal distribution (Factor).

    Args:
        marg: The marginal distribution as a Factor object.
        key: A JAX random key.

    Returns:
        A tuple of integer indices representing the sampled coordinates
        within the marginal's domain.
    """
    counts = _factor_to_counts(marg)
    return _jax_sample(counts, key)


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


def flatten_record(canonical_order: tuple[str], values: dict[str,int]) -> jax.Array:
    """
    Converts a dictionary of attribute values into a flattened JAX array
    based on a canonical order.

    Args:
        canonical_order: A tuple of strings representing the canonical order of attributes.
        values: A dictionary mapping attribute names (strings) to their integer values.

    Returns:
        A 1D JAX array of integer values, ordered according to 'canonical_order'.
    """
    return jnp.array([values[att] for att in canonical_order])


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
        permuted_string_edges.append((_idx_to_attr(domain, u_idx), _idx_to_attr(domain, v_idx)))

    return permuted_string_edges


def sample_from_tree(margs: dict[str, Factor],
                     sampling_order: list[str],
                     key, domain: Domain,
                     total: int) -> jax.Array:
    """
    Generates a full synthetic dataset from a tree-structured probabilistic graphical model.

    Args:
        margs: A dictionary mapping (attribute1, attribute2) tuples to their
               corresponding Factor objects representing pairwise marginals.
        sampling_order: A list of strings representing the attribute names
                        in the desired sampling order (e.g., BFS order of the MST).
        canonical_order: A tuple of strings representing the canonical order of all attributes.
        key: A JAX random key.
        domain: The Domain object defining the attributes and their mapping.
        total: The number of samples to generate.

    Returns:
        A JAX array of shape (total, num_attributes) representing the synthetic dataset.
    """
    jax_margs = _margs_to_jax_margs(margs, domain)
    jax_sampling_order = _sampling_order_to_jax(sampling_order, domain)
    num_nodes = len(domain)
    synth = _jax_generate_data_from_tree(jax_margs, jax_sampling_order, num_nodes, total, key)
    synth = pd.DataFrame(synth, columns=domain.attributes)

    return Dataset(synth, domain)