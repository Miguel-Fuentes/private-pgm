"""
Core neural generator and loss utilities for the “GEM” (Generative Networks with the
Exponential Mechanism) query-release pipeline of Liu et al. (2021).

This module focuses on these portion of GEM:
    • a neural generator that maps latent noise to product distributions over a discrete domain,
    • a MixtureComponents cache describing the mixture induced by a fixed latent batch,
    • differentiable loss computation on query marginals,
    • parameter-update and sampling helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import string

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from ..domain import Domain
from ..factor import Factor

__all__ = [
    "ResidualBlock",
    "Generator",
    "MixtureComponents",
    "compute_gem_loss",
    "gem_update",
    "sample_records",
]


class ResidualBlock(eqx.Module):
    """Linear → LayerNorm → ReLU with a skip connection."""

    fc: eqx.nn.Linear
    norm: eqx.nn.LayerNorm
    activation: Callable[[jax.Array], jax.Array]

    def __init__(self, dim: int, key: jax.random.PRNGKey):
        """
        Parameters
        ----------
        dim
            Feature dimension of the input/output.
        key
            PRNG key used to initialise weights.
        """
        k_fc, _ = jax.random.split(key)
        self.fc = eqx.nn.Linear(dim, dim, key=k_fc)
        self.norm = eqx.nn.LayerNorm((dim,))
        self.activation = jax.nn.relu

    def __call__(self, x: jax.Array, *, inference: bool) -> jax.Array:
        """Forward pass using LayerNorm"""
        out = self.fc(x)
        out = self.norm(out)
        out = self.activation(out)
        return x + out


class Generator(eqx.Module):
    """
    GEM-style generator: latent noise → per-attribute categorical probabilities.

    Architecture
    ------------
    z --(Linear)--> hidden --[ResidualBlocks]--> hidden --(Linear)--> logits
    -> partition logits per attribute -> softmax -> categorical distributions.
    """

    proj: eqx.nn.Linear
    layers: List[ResidualBlock]
    head: eqx.nn.Linear

    split_sizes: Tuple[int, ...] = eqx.field(static=True)
    domain_map: Dict[str, int] = eqx.field(static=True)

    def __init__(
        self,
        z_dim: int,
        hidden_dim: int,
        num_blocks: int,
        domain: Domain,
        key: jax.random.PRNGKey,
    ):
        """
        Parameters
        ----------
        z_dim
            Dimension of the latent noise vector.
        hidden_dim
            Width of hidden representations.
        num_blocks
            Number of residual blocks.
        domain
            `mbi.domain.Domain` describing attribute names and cardinalities.
        key
            PRNG key for initialisation.
        """
        self.domain_map = {name: i for i, name in enumerate(domain.attrs)}
        self.split_sizes = tuple(domain.shape)

        keys = jax.random.split(key, num_blocks + 2)
        self.proj = eqx.nn.Linear(z_dim, hidden_dim, key=keys[0])
        self.layers = [ResidualBlock(hidden_dim, key=k) for k in keys[1:-1]]
        self.head = eqx.nn.Linear(hidden_dim, sum(self.split_sizes), key=keys[-1])

    def __call__(
        self,
        z: jax.Array,
        *,
        inference: bool,
    ) -> List[jax.Array]:
        """
        Evaluate the generator on a single latent vector.

        Parameters
        ----------
        z
            Latent noise vector.
        inference
            Present for API compatibility; LayerNorm does not distinguish between
            training and inference.

        Returns
        -------
        probs
            List of length `num_attributes`; each entry is a probability vector for one
            attribute (shape = (cardinality,)).
        """
        x = jax.nn.relu(self.proj(z))
        for layer in self.layers:
            x = layer(x, inference=inference)

        logits = self.head(x)

        indices = list(np.cumsum(self.split_sizes)[:-1])
        chunks = jnp.split(logits, indices)
        probs = [jax.nn.softmax(chunk, axis=-1) for chunk in chunks]

        return probs


@dataclass
class MixtureComponents:
    """
    Cached mixture components induced by a fixed latent batch.

    Attributes
    ----------
    column_probs
        List whose i-th entry has shape (K, |X_i|): row k gives the categorical
        distribution over attribute i for latent seed k.
    latent_batch
        The latent seeds used (shape = (K, z_dim)).
    domain_map
        Mapping from attribute name to column index.
    split_sizes
        Tuple of attribute cardinalities.
    """

    column_probs: List[jax.Array]
    latent_batch: jax.Array
    domain_map: Dict[str, int]
    split_sizes: Tuple[int, ...]

    @classmethod
    def from_generator(
        cls,
        model: Generator,
        latent_batch: jax.Array,
        *,
        inference: bool,
    ) -> "MixtureComponents":
        """
        Evaluate `model` on `latent_batch` and cache the resulting mixture.

        Parameters
        ----------
        model
            Generator instance.
        latent_batch
            Latent seeds reused across training/sampling (shape = (K, z_dim)).
        inference
            Passed through to the generator (no effect for LayerNorm).
        """
        column_probs = cls._run_generator(
            model, latent_batch, inference=inference
        )
        return cls(
            column_probs=list(column_probs),
            latent_batch=latent_batch,
            domain_map=model.domain_map,
            split_sizes=model.split_sizes,
        )

    @staticmethod
    def _run_generator(
        model: Generator,
        latent_batch: jax.Array,
        *,
        inference: bool,
    ) -> List[jax.Array]:
        """Vectorised generator evaluation used by `from_generator`."""
        def _call(z):
            return model(z, inference=inference)

        vmap_gen = jax.vmap(_call)
        return vmap_gen(latent_batch)


    @property
    def num_components(self) -> int:
        """Number of mixture components (latent seeds)."""
        return int(self.latent_batch.shape[0])

    @property
    def num_attributes(self) -> int:
        """Number of discrete attributes."""
        return len(self.column_probs)

    
    def marginal(self, attrs: Sequence[str]) -> jax.Array:
        """
        Compute the exact marginal over the given attribute names.

        Parameters
        ----------
        attrs
            Sequence of attribute names in the generator domain.

        Returns
        -------
        marginal
            Tensor whose axes correspond to the requested attributes.
        """
        if not attrs:
            raise ValueError("MixtureComponents.marginal requires at least one attribute.")
        indices = [self.domain_map[attr] for attr in attrs]
        return self._marginal_from_indices(indices)

    def marginal_from_indices(self, indices: Sequence[int]) -> jax.Array:
        """
        Same as :meth:`marginal` but accepts column indices directly.
        """
        if not indices:
            raise ValueError("MixtureComponents.marginal_from_indices requires indices.")
        return self._marginal_from_indices(indices)

    def _marginal_from_indices(self, indices: Sequence[int]) -> jax.Array:
        """Internal einsum implementation shared by loss/sampling code."""
        relevant = [self.column_probs[i] for i in indices]

        dims = iter(string.ascii_lowercase)
        einsum_inputs = [f"z{next(dims)}" for _ in relevant]
        output = "".join(label[1] for label in einsum_inputs)

        summed = jnp.einsum(f"{','.join(einsum_inputs)}->{output}", *relevant)
        return summed / self.num_components


    def sample(self, num_records: int, key: jax.random.PRNGKey) -> jax.Array:
        """
        Sample synthetic records by choosing mixture components uniformly.

        Parameters
        ----------
        num_records
            Number of records to synthesise.
        key
            PRNG key.

        Returns
        -------
        samples
            Integer array of shape (num_records, num_attributes).
        """
        key_comp, key_cats = jax.random.split(key)
        component_idx = jax.random.randint(
            key_comp, (num_records,), 0, self.num_components
        )

        gathered = [probs[component_idx] for probs in self.column_probs]
        keys_per_attr = jax.random.split(key_cats, self.num_attributes)

        def _sample_attr(probs: jax.Array, k: jax.random.PRNGKey) -> jax.Array:
            return jax.random.categorical(k, jnp.log(probs + 1e-10), axis=-1)

        attr_samples = [_sample_attr(gathered[i], keys_per_attr[i]) for i in range(self.num_attributes)]
        return jnp.stack(attr_samples, axis=1)


# =============================================================================
# GEM loss & optimisation
# =============================================================================


def compute_gem_loss(
    model: Generator,
    fixed_noise: jax.Array,
    queries: Iterable[Factor],
) -> jax.Array:
    """
    Average L1 discrepancy between model marginals and provided query answers.

    Parameters
    ----------
    model
        Generator to be trained.
    fixed_noise
        Latent seeds (shape = (K, z_dim)) reused across optimisation steps.
    queries
        Iterable of `mbi.factor.Factor` objects encoding target marginals.

    Returns
    -------
    loss
        Scalar loss (mean absolute error across queries).
    """
    mixture = MixtureComponents.from_generator(
        model, fixed_noise, inference=False
    )

    total_error = 0.0
    num_queries = 0

    for factor in queries:
        num_queries += 1
        try:
            col_indices = [mixture.domain_map[attr] for attr in factor.domain.attributes]
        except KeyError as err:
            raise ValueError(f"Query attribute {err} not found in generator domain.")

        model_marginal = mixture.marginal_from_indices(col_indices)
        total_error += jnp.mean(jnp.abs(model_marginal - factor.values))

    if num_queries == 0:
        raise ValueError("compute_gem_loss received an empty collection of queries.")

    return total_error / num_queries


@eqx.filter_jit
def gem_update(
    model: Generator,
    opt_state: optax.OptState,
    optimizer: optax.GradientTransformation,
    fixed_noise: jax.Array,
    queries: Iterable[Factor],
) -> Tuple[Generator, optax.OptState, jax.Array]:
    """
    Single GEM optimisation step (loss + gradients + parameter update)."""

    def loss_fn(m: Generator) -> jax.Array:
        return compute_gem_loss(m, fixed_noise, queries)

    loss, grads = eqx.filter_value_and_grad(loss_fn)(model)

    updates, new_opt_state = optimizer.update(grads, opt_state, model)
    new_model = eqx.apply_updates(model, updates)

    return new_model, new_opt_state, loss


def sample_records(
    model: Generator,
    latent_batch: jax.Array,
    num_records: int,
    key: jax.random.PRNGKey,
) -> jnp.ndarray:
    """
    Convenience wrapper: compute mixture components (with LayerNorm inference mode) and sample data.

    Returns
    -------
    samples
        Integer array (num_records, num_attributes).
    """
    mixture = MixtureComponents.from_generator(
        model, latent_batch, inference=True
    )
    return mixture.sample(num_records, key)