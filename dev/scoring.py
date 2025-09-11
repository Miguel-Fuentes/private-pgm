from mbi import Factor
import jax.numpy as jnp


def entropy(marg: Factor) -> float:
    """
    Computes the entropy of a marginal distribution.

    Args:
        marg: The marginal distribution as a Factor object.

    Returns:
        The entropy of the distribution as a float.

    Raises:
        ValueError: If the marginal contains negative counts or has a total count of zero.
    """
    if jnp.any(marg.values < 0):
        raise ValueError("marginal given with negative counts")

    total = jnp.sum(marg.values)
    if total == 0:
        raise ValueError("marginal given with 0 total count")

    probs = marg.values / total
    log_p = jnp.where(probs, jnp.log(probs), 0.0)
    p_log_p = probs * log_p

    return float(-jnp.sum(p_log_p))


def mutual_information(marg: Factor) -> float:
    """
    Computes the mutual information between two attributes in a 2-way marginal distribution.

    Args:
        marg: A 2-way marginal distribution as a Factor object.

    Returns:
        The mutual information between the two attributes as a float.

    Raises:
        ValueError: If the marginal is not a 2-way marginal.
    """
    if len(marg.domain.attrs) != 2:
        raise ValueError("mutual information onlu defined for 2-way marginals")

    att1, att2 = marg.domain.attrs[0], marg.domain.attrs[1]
    marg1, marg2 = marg.project(att1), marg.project(att2)

    ent1 = entropy(marg1)
    ent2 = entropy(marg2)
    ent_joint = entropy(marg)

    return ent1 + ent2 - ent_joint
