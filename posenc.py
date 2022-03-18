from __future__ import annotations

from jax import jit, numpy as jnp, vmap
from jax.nn import relu, sigmoid



N_POS = 32
N_PRE = 4 # Anything > 4 is absolutely indistinguishable from 4



# @jit
def encode(idx: jnp.ndarray, size: int, srate: int = 1) -> jnp.ndarray:
    """
    Translates a linear position to sin/cos pairs of exponentially increasing frequency, like hands on a clock.

    Returned dimensions:
        - ``-4`` and earlier: Same as input.
        - ``-3``: Frequencies.
        - ``-2``: Top-down or bottom-up, i.e. like an image (relative to the whole size) or like audio (relative to the sample rate).
        - ``-1``: Sine (1) or cosine (0).

    Argument ``n_extra`` adds frequencies with period greater than 1. Anything > 4 is indistinguishable from 4.
    """
    assert N_POS > 0
    assert N_PRE >= 0
    assert jnp.isscalar(size)
    assert jnp.isscalar(srate)
    both = 2 * jnp.pi * jnp.stack([idx / size, idx / srate], -1) # ``stack`` packages top-down & bottom-up versions
    scaled = both[..., None] @ jnp.exp2(jnp.arange(N_POS) - N_PRE)[None]
    trig = jnp.stack([jnp.cos(scaled), jnp.sin(scaled)], -1)
    return trig.reshape(*idx.shape, N_POS, 2, 2)



@jit
def similarity(A: jnp.ndarray, B: jnp.ndarray, big: jnp.ndarray, n_extra: int = 4) -> jnp.ndarray:
    """ Decreases with distance from A (one positional encoding) to B (many). """
    s = vmap(jnp.multiply, (None, 2), 2)(A, B) # Pointwise multiplication: sin with sin, cos with cos
    s = jnp.sum(s, -1) # = sin2 + cos2, at each scale individually. now a vector (1D).
    s = relu(s) # Crucial to eliminate Fourier "ripples"
    m = sigmoid(jnp.arange(s.shape[-1]) - n_extra - big[:, None])
    return jnp.prod(m + (1 - m) * s, -1)