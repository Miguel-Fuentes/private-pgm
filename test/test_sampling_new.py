import pytest
import jax
import jax.numpy as jnp
from mbi import Domain, Factor
from dev.sampling import (
    sample, condition, order, flatten_record, best_permutation,
    random_permutation, sample_from_tree,
    _attr_to_idx, _idx_to_attr, _factor_to_counts, _margs_to_jax_margs,
    _sampling_order_to_jax, _jax_record_to_dict
)

# Helper for creating a simple Domain
def create_test_domain(attrs_list):
    # Assuming each attribute has 2 possible values (binary)
    data = {attr: 2 for attr in attrs_list}
    return Domain.fromdict(data)

# Helper for creating a simple Factor
def create_test_factor(domain, attrs, values):
    return Factor(domain.project(attrs), jnp.array(values))

def test_attr_to_idx():
    domain = create_test_domain(['X', 'Y', 'Z'])
    assert _attr_to_idx(domain, 'X') == 0
    assert _attr_to_idx(domain, 'Y') == 1

def test_idx_to_attr():
    domain = create_test_domain(['X', 'Y', 'Z'])
    assert _idx_to_attr(domain, 0) == 'X'
    assert _idx_to_attr(domain, 1) == 'Y'

def test_factor_to_counts():
    domain = create_test_domain(['A', 'B'])
    factor = create_test_factor(domain, ['A', 'B'], [[1, 2], [3, 4]])
    counts = _factor_to_counts(factor)
    assert jnp.array_equal(counts, jnp.array([[1, 2], [3, 4]]))

def test_margs_to_jax_margs():
    domain = create_test_domain(['A', 'B', 'C'])
    marg_ab = create_test_factor(domain, ['A', 'B'], [[1, 2], [3, 4]])
    marg_bc = create_test_factor(domain, ['B', 'C'], [[5, 6], [7, 8]])
    margs = {('A', 'B'): marg_ab, ('B', 'C'): marg_bc}
    
    jax_margs = _margs_to_jax_margs(margs, domain)
    
    # Expected keys are sorted integer tuples
    expected_key_ab = (0, 1) # A=0, B=1
    expected_key_bc = (1, 2) # B=1, C=2
    
    assert expected_key_ab in jax_margs
    assert expected_key_bc in jax_margs
    assert jnp.array_equal(jax_margs[expected_key_ab], jnp.array([[1, 2], [3, 4]]))
    assert jnp.array_equal(jax_margs[expected_key_bc], jnp.array([[5, 6], [7, 8]]))

def test_sampling_order_to_jax():
    domain = create_test_domain(['A', 'B', 'C'])
    sampling_order_str = ['A', 'C', 'B']
    jax_order = _sampling_order_to_jax(sampling_order_str, domain)
    expected_jax_order = jnp.array([0, 2, 1], dtype=jnp.int32)
    assert jnp.array_equal(jax_order, expected_jax_order)

def test_jax_record_to_dict():
    domain = create_test_domain(['A', 'B', 'C'])
    canonical_order = ('A', 'B', 'C')
    jax_record = jnp.array([0, 1, 0], dtype=jnp.int32) # A=0, B=1, C=0
    result_dict = _jax_record_to_dict(jax_record, canonical_order, domain)
    expected_dict = {'A': 0, 'B': 1, 'C': 0}
    assert result_dict == expected_dict

def test_sample():
    key = jax.random.PRNGKey(0)
    domain = create_test_domain(['X', 'Y'])
    marg = create_test_factor(domain, ['X', 'Y'], [[10, 0], [0, 10]]) # X=0, Y=0 or X=1, Y=1
    
    # Since it's random, we can't assert exact value, but we can check properties
    sampled_coords = sample(marg, key)
    assert len(sampled_coords) == 2
    assert all(isinstance(c, jnp.ndarray) for c in sampled_coords)
    # Check if the sampled value is one of the possible outcomes
    assert (sampled_coords[0] == 0 and sampled_coords[1] == 0) or \
           (sampled_coords[0] == 1 and sampled_coords[1] == 1)

def test_condition():
    domain = create_test_domain(['A', 'B', 'C'])
    marg = create_test_factor(domain, ['A', 'B', 'C'], 
                              jnp.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])) # A, B, C binary
    
    # Condition on A=0
    conditioned_marg = condition(marg, {'A': 0})
    assert conditioned_marg.domain.attrs == ('B', 'C')
    assert jnp.array_equal(conditioned_marg.values, jnp.array([[1, 2], [3, 4]]))

    # Condition on B=1
    conditioned_marg = condition(marg, {'B': 1})
    assert conditioned_marg.domain.attrs == ('A', 'C')
    assert jnp.array_equal(conditioned_marg.values, jnp.array([[3, 4], [7, 8]]))

    # Test error for invalid attribute
    with pytest.raises(ValueError):
        condition(marg, {'D': 0})
    
    # Test error for out of bounds value
    with pytest.raises(ValueError):
        condition(marg, {'A': 2})

def test_order():
    canonical = ('A', 'B', 'C', 'D')
    atts = ['C', 'A']
    assert order(canonical, atts) == ('A', 'C')

def test_flatten_record():
    canonical = ('X', 'Y', 'Z')
    values = {'X': 1, 'Y': 0, 'Z': 1}
    flattened = flatten_record(canonical, values)
    assert jnp.array_equal(flattened, jnp.array([1, 0, 1]))

def test_best_permutation():
    options = ['A', 'B', 'C']
    scores = jnp.array([0.5, 0.9, 0.2])
    perm = best_permutation(options, scores)
    assert perm == ['B', 'A', 'C']

def test_random_permutation():
    key = jax.random.PRNGKey(0)
    domain = create_test_domain(['A', 'B', 'C', 'D'])
    scores = {('A', 'B'): 0.8, ('A', 'C'): 0.6, ('B', 'D'): 0.9}
    
    perm = random_permutation(scores, key, domain)
    assert len(perm) == 3
    # Check if all original edges are present in the permutation (order might vary due to randomness) # This comment is incorrect, the order is deterministic given the key
    assert set(perm) == {('A', 'B'), ('A', 'C'), ('B', 'D')}

def test_sample_from_tree():
    key = jax.random.PRNGKey(0)
    domain = create_test_domain(['A', 'B', 'C'])
    
    # Create dummy marginals for a simple tree A-B-C
    # Marginals are (A,B) and (B,C)
    marg_ab = create_test_factor(domain, ['A', 'B'], [[10, 5], [5, 10]]) # A=0, B=0 or A=1, B=1 more likely
    marg_bc = create_test_factor(domain, ['B', 'C'], [[10, 5], [5, 10]]) # B=0, C=0 or B=1, C=1 more likely
    
    margs = {('A', 'B'): marg_ab, ('B', 'C'): marg_bc}
    sampling_order_list = ['A', 'B', 'C']
    canonical_order = ('A', 'B', 'C')
    
    sampled_record = sample_from_tree(margs, sampling_order_list, canonical_order, key, domain)
    
    assert len(sampled_record) == 3
    assert 'A' in sampled_record and 'B' in sampled_record and 'C' in sampled_record
    
    # Since it's random, we can't assert exact values, but we can check if they are valid
    assert sampled_record['A'] in [0, 1]
    assert sampled_record['B'] in [0, 1]
    assert sampled_record['C'] in [0, 1]
