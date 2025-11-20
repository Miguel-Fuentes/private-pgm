from typing import List, Tuple, Dict
import string

import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import optax

from mbi.domain import Domain
from mbi.factor import Factor


class ResidualBlock(eqx.Module):
    fc: eqx.nn.Linear
    norm: eqx.nn.BatchNorm
    activation: callable

    def __init__(self, dim: int, key: jax.random.PRNGKey):
        k_fc, _ = jax.random.split(key)
        self.fc = eqx.nn.Linear(dim, dim, key=k_fc)
        self.norm = eqx.nn.BatchNorm(dim, axis_name="batch", mode="ema")
        self.activation = jax.nn.relu

    def __call__(
        self, x: jax.Array, state: eqx.nn.State
    ) -> Tuple[jax.Array, eqx.nn.State]:
        out = self.fc(x)
        out, state = self.norm(out, state)
        out = self.activation(out)
        return x + out, state


class Generator(eqx.Module):
    proj: eqx.nn.Linear
    layers: List[ResidualBlock]
    head: eqx.nn.Linear

    # Static metadata
    split_sizes: Tuple[int] = eqx.field(static=True)
    domain_map: Dict[str, int] = eqx.field(static=True)

    def __init__(
        self,
        z_dim: int,
        hidden_dim: int,
        num_blocks: int,
        domain: Domain,
        key: jax.random.PRNGKey,
    ):
        self.domain_map = {name: i for i, name in enumerate(domain.attrs)}
        self.split_sizes = tuple(domain.shape)

        keys = jax.random.split(key, num_blocks + 2)

        self.proj = eqx.nn.Linear(z_dim, hidden_dim, key=keys[0])
        self.layers = [ResidualBlock(hidden_dim, key=k) for k in keys[1:-1]]
        self.head = eqx.nn.Linear(hidden_dim, sum(self.split_sizes), key=keys[-1])

    def __call__(
        self, z: jax.Array, state: eqx.nn.State
    ) -> Tuple[List[jax.Array], eqx.nn.State]:
        x = jax.nn.relu(self.proj(z))

        for layer in self.layers:
            x, state = layer(x, state)

        logits = self.head(x)

        # Split flat logits into per-column chunks and apply softmax
        indices = list(np.cumsum(self.split_sizes)[:-1])
        chunks = jnp.split(logits, indices)
        probs = [jax.nn.softmax(chunk, axis=-1) for chunk in chunks]

        return probs, state

    def sample(
        self, z_batch: jax.Array, state: eqx.nn.State, key: jax.random.PRNGKey
    ) -> jax.Array:
        vmap_gen = jax.vmap(self, in_axes=(0, None), out_axes=(0, None))
        probs_list, _ = vmap_gen(z_batch, state)

        keys = jax.random.split(key, len(probs_list))
        samples = []

        # Gumbel-Max sampling per column
        for i, probs in enumerate(probs_list):
            sample = jax.random.categorical(keys[i], jnp.log(probs + 1e-10), axis=-1)
            samples.append(sample)

        return jnp.stack(samples, axis=1)


def compute_gem_loss(
    model: Generator,
    bn_state: eqx.nn.State,
    fixed_noise: jax.Array,
    queries: List[Factor],
) -> Tuple[jax.Array, eqx.nn.State]:
    K = fixed_noise.shape[0]

    # Vectorized generation with batch-synced normalization
    vmap_gen = jax.vmap(model, in_axes=(0, None), out_axes=(0, None), axis_name="batch")
    column_probs, batch_bn_states = vmap_gen(fixed_noise, bn_state)

    # Un-batch the state (all K states are identical due to sync)
    new_bn_state = jax.tree_util.tree_map(lambda s: s[0], batch_bn_states)

    total_error = 0.0

    for factor in queries:
        try:
            col_indices = [model.domain_map[attr] for attr in factor.domain.attributes]
        except KeyError as e:
            raise ValueError(f"Query attribute {e} not in model domain")

        relevant_probs = [column_probs[i] for i in col_indices]

        # Construct Einsum: "za,zb,zc->abc" where 'z' is the batch dim
        dims = iter(string.ascii_lowercase)
        inputs = [f"z{next(dims)}" for _ in relevant_probs]
        output = "".join(inp[1] for inp in inputs)

        # Sum over batch dimension 'z' to get unnormalized marginal
        summed_marginal = jnp.einsum(f"{','.join(inputs)}->{output}", *relevant_probs)
        model_marginal = summed_marginal / K

        total_error += jnp.mean(jnp.abs(model_marginal - factor.values))

    return total_error / len(queries), new_bn_state


@eqx.filter_jit
def gem_update(
    model: Generator,
    bn_state: eqx.nn.State,
    opt_state: optax.OptState,
    optimizer: optax.GradientTransformation,
    fixed_noise: jax.Array,
    queries: List[Factor],
):
    def loss_fn(m, s):
        return compute_gem_loss(m, s, fixed_noise, queries)

    (loss, new_bn_state), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(
        model, bn_state
    )

    updates, new_opt_state = optimizer.update(grads, opt_state, model)
    new_model = eqx.apply_updates(model, updates)

    return new_model, new_bn_state, new_opt_state, loss
