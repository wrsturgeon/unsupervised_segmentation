from __future__ import annotations

from jax import jit, numpy as jnp
from numpy.random import randn, random as rand_unit
from os.path import join

import ops
import posenc



D_POS = posenc.N_POS << 2
assert ops.D > D_POS



def load_parameters(name: str, unloaded: list) -> None:
    def assign_rec(unassigned: list | jnp.ndarray, data: list, i: int) -> int:
        # dear god why couldn't we just use pointers ...
        for j, sub in enumerate(unassigned):
            if isinstance(sub, list):
                i = assign_rec(sub, data, i)
            else:
                unassigned[j] = data[i] # To avoid the [string of expletives] hidden pass-by-value
                i += 1
        return i
    with jnp.load(join("parameters", f"{name}.npz")) as f:
        assign_rec(unloaded, [f[s] for s in f.files], 0)

def save_parameters(name: str, data: list) -> None:
    # Depth-first traversal
    def flatten(x: list | jnp.ndarray) -> list:
        if isinstance(x, jnp.ndarray):
            return [x]
        if len(x) == 0:
            return []
        return [*flatten(x[0]), *flatten(x[1:])]
    f = flatten(data)
    # print([type(fi) for fi in f])
    jnp.savez(join("parameters", f"{name}.npz"), *f)
    


@jit
def init_things() -> jnp.ndarray:
    things = jnp.empty([ops.N, ops.D], jnp.float16)
    p = posenc.encode(rand_unit(ops.N), 1).reshape(ops.N, -1)
    print(p.shape)
    things = things.at[:, :D_POS      ].set(p)
    things = things.at[:,  D_POS      ].set(1 / (1 - rand_unit(ops.N)))
    things = things.at[:, (D_POS + 1):].set(randn(ops.N, ops.D - D_POS - 1))
    things = things.at[:, (D_POS + 1):].divide(jnp.std(things[:, (D_POS + 1):], 1, keepdims=True) + jnp.finfo(jnp.float16).smallest_normal)
    return things

print(init_things())



# @jit
def query_pixel(things: jnp.ndarray, idx: jnp.ndarray, size: int, srate: int = 1) -> jnp.ndarray:
    p = posenc.encode(idx, size, srate)
    return posenc.similarity(p, things[:D_POS], things[D_POS])