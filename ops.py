from functools import partial
from time import time_ns
from jax import jit, numpy as jnp, vmap
from jax.lax import stop_gradient
from jax.nn import softmax#, normalize
from numpy.random import randn
from scipy.stats import ortho_group



# Classes can't JIT themselves--with a huge asterisk
# This file explores one way to work around that by separating parameters and functions
# ops_oo.py explores an object-oriented workaround that, of course, causes far more trouble
# Haiku is cool but it seems like way too much work for what it is, especially without JIT

# Parameters are stored as a list
# Two-element list at the highest level: [0] = parameters; [1] = running standard deviation



R_AMT = 0.05
N = 512
FRIC = 0.99



@jit
def update_running_std(running: jnp.float16, std: jnp.ndarray) -> jnp.float16:
    step_multiplier = jnp.exp2(-jnp.log2(stop_gradient(running)) * jnp.log2(std)) # Larger if the update points toward 1
    return FRIC * running + (1 - FRIC) * std * step_multiplier



def fork_params(indent: int = 0) -> list:
    print(*[" " for _ in range(indent)], "Initializing fork ...", end=" ")
    start = time_ns()
    p = [jnp.asarray(randn(4), jnp.float16), None]
    std = jnp.std(_fork(p[0], randn(N)))
    p[1] = jnp.ones_like(std, jnp.float16)
    p = fork_normalize(p, std)
    print((time_ns() - start) / 1e9, "s")
    return p
@jit
def fork_update(p: list, dLdp: list, lr: jnp.float32) -> list:
    p[0] -= lr * dLdp[0]
    return fork_normalize(p)
@jit
def fork_normalize(p: list, aux_std: jnp.ndarray = 1) -> list:
    p[0] = p[0].at[:2].divide(p[1] * aux_std + jnp.finfo(jnp.float16).smallest_normal)
    return p
def fork(p: list, x: jnp.ndarray, stochastic: bool) -> jnp.ndarray:
    out = _fork(p[0], x, stochastic)
    p[1] = update_running_std(p[1], jnp.std(out))
    return out
@partial(jit, static_argnames=["stochastic"])
def _fork(p: list, x: jnp.ndarray, stochastic: bool) -> jnp.ndarray:
    """ General form of a continuous pointwise linear function. """
    m1, m2, b1, b2 = p
    if stochastic:
        m1 += R_AMT * randn()
        m2 += R_AMT * randn()
        b1 += R_AMT * randn()
        b2 += R_AMT * randn()
    intersection = stop_gradient((b2 - b1) / (m1 - m2))
    condition = x < intersection
    return jnp.where(condition, m1 * x + b1, m2 * x + b2)



def linear_params(n_in: int = N, n_out: int = N, indent: int = 0) -> list:
    print(*[" " for _ in range(indent)], "Initializing linear ...", end=" ")
    start = time_ns()
    p = [jnp.asarray(ortho_group.rvs(max(n_out, n_in))[:n_out, :n_in], jnp.float16), None]
    std = jnp.std(_linear(p[0], randn(n_in, N)), 1)
    p[1] = jnp.ones_like(std, jnp.float16)
    p = linear_normalize(p, std)
    print((time_ns() - start) / 1e9, "s")
    return p
@jit
def linear_update(p: list, dLdp: list, lr: jnp.float32) -> list:
    p[0] -= lr * dLdp[0]
    return linear_normalize(p)
@jit
def linear_normalize(p: list, aux_std: jnp.ndarray = 1) -> list:
    p[0] /= (p[1] * aux_std)[:, None] + jnp.finfo(jnp.float16).smallest_normal
    return p
def linear(p: list, x: jnp.ndarray, stochastic: bool) -> jnp.ndarray:
    out = _linear(p[0], x, stochastic)
    p[1] = update_running_std(p[1], jnp.log2(jnp.std(out, 1)))
    return out
@partial(jit, static_argnames=["stochastic"])
def _linear(p: jnp.ndarray, x: jnp.ndarray, stochastic: bool) -> jnp.ndarray:
    """ Simple matrix multiplication. """
    return (p + R_AMT * randn(*p.shape) if stochastic else p) @ x



def feedforward_params(n: int = N, expand: int = 4, indent: int = 0) -> list:
    print(*[" " for _ in range(indent)], "Initializing feedforward ...")
    start = time_ns()
    p = [[
        fork_params(indent + 1),
        fork_params(indent + 1),
        linear_params(n, expand * n, indent + 1),
        linear_params(expand * n, n, indent + 1),
    ], jnp.float16(1)]
    p = feedforward_normalize(p, jnp.std(_feedforward(p[0], randn(n, N))))
    print(*[" " for _ in range(indent)], (time_ns() - start) / 1e9, "s")
    return p
@jit
def feedforward_update(p: list, dLdp: list, lr: jnp.float32) -> list:
    aux_std = jnp.sqrt(aux_std) * p[1]
    p[0][2] = linear_update(p[0][2], dLdp[0][2], lr, aux_std)
    p[0][3] = linear_update(p[0][3], dLdp[0][3], lr, aux_std)
    return feedforward_normalize(p)
@jit
def feedforward_normalize(p: list, aux_std: jnp.ndarray = 1) -> list:
    aux_std = jnp.sqrt(jnp.mean(aux_std)) * p[1]
    p[0][2] = linear_normalize(p[0][2], aux_std)
    p[0][3] = linear_normalize(p[0][3], aux_std)
    return p
def feedforward(p: list, x: jnp.ndarray, stochastic: bool) -> jnp.ndarray:
    out = _feedforward(p[0], x, stochastic)
    p[1] = update_running_std(p[1], jnp.std(out))
    return out
@partial(jit, static_argnames=["stochastic"])
def _feedforward(p: list, x: jnp.ndarray, stochastic: bool) -> jnp.ndarray:
    """ Linear -> Fork -> Linear -> Fork """
    fork_a, fork_b, linear_a, linear_b = p
    return fork(fork_b, linear(linear_b, fork(fork_a, linear(linear_a, x, stochastic), stochastic), stochastic), stochastic)



def mhsa_params(n: int = N, h: int = 8, indent: int = 0) -> list:
    print(*[" " for _ in range(indent)], "Initializing mhsa ...")
    start = time_ns()
    p = [[
        linear_params(n, 3 * n, indent + 1),
        linear_params(n, n, indent + 1),
        [fork_params(indent + 1) for _ in range(h)],
        [fork_params(indent + 1) for _ in range(h)],
        [fork_params(indent + 1) for _ in range(h)],
    ], jnp.float16(1)]
    p = mhsa_normalize(p, jnp.std(_mhsa(p[0], randn(n, N))))
    print(*[" " for _ in range(indent)], (time_ns() - start) / 1e9, "s")
    return p
@jit
def mhsa_update(p: list, dLdp: list, lr: jnp.float32) -> list:
    aux_std = jnp.sqrt(aux_std) * p[1]
    p[0][0] = linear_update(p[0][0], dLdp[0][0], lr, aux_std)
    p[0][1] = linear_update(p[0][1], dLdp[0][1], lr, aux_std)
    return mhsa_normalize(p)
@jit
def mhsa_normalize(p: list, aux_std: jnp.ndarray = 1) -> list:
    aux_std = jnp.sqrt(aux_std) * p[1]
    p[0][0] = linear_normalize(p[0][0], aux_std)
    p[0][1] = linear_normalize(p[0][1], aux_std)
    return p
def mhsa(p: list, x: jnp.ndarray, stochastic: bool) -> jnp.ndarray:
    out = _mhsa(p[0], x, stochastic)
    p[1] = update_running_std(p[1], jnp.std(out))
    return out
@partial(jit, static_argnames=["stochastic"])
def _mhsa(p: list, x: jnp.ndarray, stochastic: bool) -> jnp.ndarray:
    """ Multi-head self attention. Q, K, V are computed internally via linear transformation. """
    qkv_proj, final_proj, q_forks, k_forks, v_forks = p
    h = len(q_forks) # workaround (O(1)): don't want an extra parameter
    n = qkv_proj[0].shape[1] # ditto ^
    assert n % h == 0
    d_k = n // h
    q, k, v = linear(qkv_proj, x, stochastic).reshape(3, h, d_k, -1)
    fork_mh = lambda p, a: jnp.asarray([fork(pi, ai, stochastic) for pi, ai in zip(p, a)]) # Classic vmap but it doesn't like lists
    q, k, v = fork_mh(q_forks, q), fork_mh(k_forks, k), fork_mh(v_forks, v)
    scores = vmap(lambda a, b: a @ b.T)(q, k) / jnp.sqrt(d_k)
    reconcat = (softmax(scores) @ v).reshape(n, -1)
    return linear(final_proj, reconcat, stochastic)



def mhsa_res_params(n: int = N, h: int = 8, indent: int = 0) -> list:
    print(*[" " for _ in range(indent)], "Initializing mhsa_res ...")
    start = time_ns()
    p = [mhsa_params(n, h, indent + 1), jnp.float16(1)]
    p = mhsa_res_normalize(p, jnp.std(_mhsa_res(p[0], randn(n, N))))
    print(*[" " for _ in range(indent)], (time_ns() - start) / 1e9, "s")
    return p
@jit
def mhsa_res_update(p: list, dLdp: list, lr: jnp.float32) -> list:
    p[0] = mhsa_update(p[0], dLdp[0], lr)
    return mhsa_res_normalize(p)
@jit
def mhsa_res_normalize(p: list, aux_std: jnp.ndarray = 1) -> list:
    p[0] = mhsa_normalize(p[0], aux_std * p[1])
    return p
def mhsa_res(p: list, x: jnp.ndarray, stochastic: bool) -> jnp.ndarray:
    out = _mhsa_res(p[0], x, stochastic)
    p[1] = update_running_std(p[1], jnp.std(out))
    return out
@partial(jit, static_argnames=["stochastic"])
def _mhsa_res(p: list, x: jnp.ndarray, stochastic: bool) -> jnp.ndarray:
    """ Multi-head attention, summed with the original input. """
    return jnp.sqrt(0.5) * (x + mhsa(p, x, stochastic))



def feedforward_res_params(n: int = N, expand: int = 4, indent: int = 0) -> list:
    print(*[" " for _ in range(indent)], "Initializing feedforward_res ...")
    start = time_ns()
    p = [feedforward_params(n, expand, indent + 1), jnp.float16(1)]
    p = feedforward_res_normalize(p, jnp.std(_feedforward_res(p[0], randn(n, N))))
    print(*[" " for _ in range(indent)], (time_ns() - start) / 1e9, "s")
    return p
@jit
def feedforward_res_update(p: list, dLdp: list, lr: jnp.float32) -> list:
    p[0] = feedforward_update(p[0], dLdp[0], lr)
    return feedforward_res_normalize(p)
@jit
def feedforward_res_normalize(p: list, aux_std: jnp.ndarray = 1) -> list:
    p[0] = feedforward_normalize(p[0], aux_std * p[1])
    return p
def feedforward_res(p: list, x: jnp.ndarray, stochastic: bool) -> jnp.ndarray:
    out = _feedforward_res(p[0], x, stochastic)
    p[1] = update_running_std(p[1], jnp.std(out))
    return out
@partial(jit, static_argnames=["stochastic"])
def _feedforward_res(p: list, x: jnp.ndarray, stochastic: bool) -> jnp.ndarray:
    """ Feedforward layer, summed with the original input. """
    return jnp.sqrt(0.5) * (x + feedforward(p, x, stochastic))



def encoder_block_params(n: int = N, h: int = 8, expand: int = 4, indent: int = 0) -> list:
    print(*[" " for _ in range(indent)], "Initializing encoder_block ...")
    start = time_ns()
    p = [[
        mhsa_res_params(n, h, indent + 1),
        feedforward_res_params(n, expand, indent + 1),
    ], jnp.float16(1)]
    p = encoder_block_normalize(p, jnp.std(_encoder_block(p[0], randn(n, N))))
    print(*[" " for _ in range(indent)], (time_ns() - start) / 1e9, "s")
    return p
@jit
def encoder_block_update(p: list, dLdp: list, lr: jnp.float32) -> list:
    p[0] = (
        mhsa_res_update(p[0][0], dLdp[0][0], lr),
        feedforward_res_update(p[0][1], dLdp[0][1], lr),
    )
    return encoder_block_normalize(p)
@jit
def encoder_block_normalize(p: list, aux_std: jnp.ndarray = 1) -> list:
    # aux_std = jnp.sqrt(aux_std) * p[1]
    aux_std = aux_std * p[1]
    p[0] = (
        mhsa_res_normalize(p[0][0], aux_std),
        feedforward_res_normalize(p[0][1], aux_std),
    )
    return p
def encoder_block(p: list, x: jnp.ndarray, stochastic: bool) -> jnp.ndarray:
    out = _encoder_block(p[0], x, stochastic)
    p[1] = update_running_std(p[1], jnp.std(out))
    return out
@partial(jit, static_argnames=["stochastic"])
def _encoder_block(p: list, x: jnp.ndarray, stochastic: bool) -> jnp.ndarray:
    """ A single "block" in a Transformer encoder. """
    att_p, ffn_p = p
    return feedforward_res(ffn_p, mhsa_res(att_p, x, stochastic), stochastic)