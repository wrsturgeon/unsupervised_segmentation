from functools import partial
from jax import jit, numpy as jnp, vmap
from jax.nn import relu, sigmoid
from numpy.random import rand, randn

from ops import N

jnp.set_printoptions(suppress=True) # No scientific notation in printing



@partial(jit, static_argnums=[2])
def positional_encoding(pos: jnp.ndarray, srate: jnp.ndarray, n_scales: int = 32, n_extra: int = 4):
    """
    Translates a linear position to sin/cos pairs of exponentially increasing frequency, like hands on a clock.

    ``srate`` (sample rate) should be the size(s) of the input (e.g. dimensions of an image) if not audio or a similar datatype.

    Returned dimensions, from the back:
        - ``-1``: Sine (1) or cosine (0).
        - ``-2``: Top-down or bottom-up, i.e. like an image (relative to the whole size) or like audio (relative to the sample rate).
        - ``-3``: Frequencies.
    All else remain as inputted.

    Argument ``n_extra`` adds frequencies with period greater than 1. Anything > 4 is indistinguishable from 4.
    """
    assert n_scales > 0
    assert n_extra >= 0
    assert srate.ndim == 1
    assert pos.shape[-1] == srate.shape[0], f"{pos.shape}[-1] != {srate.shape}[0]"
    assert srate.dtype == jnp.uint32
    both = 2 * jnp.pi * jnp.stack([pos / srate, pos], -1) # ``stack`` packages top-down & bottom-up versions
    scaled = both[..., None] @ jnp.exp2(jnp.arange(n_scales) - n_extra)[None]
    trig = jnp.stack([jnp.cos(scaled), jnp.sin(scaled)], -1)
    return trig.reshape(*pos.shape, n_scales, 2, 2)

@jit
def positional_similarities(pos: jnp.ndarray, many_pos: jnp.ndarray, big: jnp.ndarray, n_extra: int = 4):
    s = vmap(jnp.multiply, (None, 2), 2)(pos, many_pos) # Pointwise multiplication: sin with sin, cos with cos
    s = jnp.sum(s, -1) # = sin2 + cos2, at each scale individually. now a vector (1D).
    s = relu(s) # Crucial to eliminate Fourier "ripples"
    m = sigmoid(jnp.arange(s.shape[-1]) - n_extra - big[:, None])
    return jnp.prod(m + (1 - m) * s, -1)


    
# @jit
def query_pixel(self, pos: jnp.ndarray) -> jnp.ndarray:
    assert pos.ndim == 1
    # assert pos.shape[0] == self.data_ndim # This apparently causes a JAX tracer error ... ?!?!?
    assert pos.dtype == jnp.float32
    posenc = positional_encoding(pos)
    return positional_similarities(posenc, self.pos, self.big)