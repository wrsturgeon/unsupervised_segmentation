from __future__ import annotations

from jax.numpy import load, ndarray, savez
from os.path import join



def load_parameters(name: str, unloaded: list) -> None:
    def assign_rec(unassigned: list | ndarray, data: list, i: int) -> int:
        # dear god why couldn't we just use pointers ...
        for j, sub in enumerate(unassigned):
            if isinstance(sub, list):
                i = assign_rec(sub, data, i)
            else:
                unassigned[j] = data[i] # To avoid the [string of expletives] hidden pass-by-value
                i += 1
        return i
    with load(join("parameters", f"{name}.npz")) as f:
        assign_rec(unloaded, [f[s] for s in f.files], 0)



def save_parameters(name: str, data: list) -> None:
    # Depth-first traversal
    def flatten(x: list | ndarray) -> list:
        if isinstance(x, ndarray):
            return [x]
        if len(x) == 0:
            return []
        return [*flatten(x[0]), *flatten(x[1:])]
    f = flatten(data)
    # print([type(fi) for fi in f])
    savez(join("parameters", f"{name}.npz"), *f)
    


# # Test
# from ops import encoder_block_params
# P = encoder_block_params()
# save_parameters("test", P)
# p = encoder_block_params()
# load_parameters("test", p) # In place
# def eq(A: list | ndarray, B: list | ndarray) -> bool:
#     if isinstance(A, list):
#         return isinstance(B, list) and all([eq(a, b) for a, b in zip(A, B)])
#     return (not isinstance(B, list)) and (A == B).all()
# print(eq(p, P))