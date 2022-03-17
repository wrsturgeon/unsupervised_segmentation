from __future__ import annotations # Postponement: allows use of class's own type in its definitions

from functools import partial
from jax import jit, numpy as jnp, vmap
from jax.lax import stop_gradient
from jax.nn import softmax, normalize
from jax.tree_util import register_pytree_node_class
from numpy.random import randn



R_AMT = 0.05
N = 512



class Module:
    """ Overarching superclass to navigate JAX's pytree labyrinth. """

    def _init_slow(self, consts) -> None:
        try:
            assert isinstance(self.constants, tuple), f"``constants`` of {type(self).__name__} must be a tuple"
        except AttributeError:
            raise AttributeError(f"Define ``constants`` of {type(self).__name__}, even if it's only an empty tuple ()")
        try:
            assert isinstance(self.parameters, tuple), f"``parameters`` of {type(self).__name__} must be a tuple"
        except AttributeError:
            raise AttributeError(f"Define ``parameters`` of {type(self).__name__}")
        assert self.parameters is not None, "Undefined parameters"
        if len(self.constants) > 0:
            c_inits = self.init_constants()
            for k, v in zip(self.constants, c_inits):
                try:
                    v = consts[k]
                except KeyError:
                    pass
                self.__dict__[k] = v
            for k in consts.keys():
                assert k in self.constants, f"Argument \"{k}\" passed to {type(self).__name__} but not in its ``constant`` tuple"
            for c in self.constants:
                assert c in self.__dict__.keys(), f"{type(self).__name__} initialized without necessary constant {c}"
                assert self.__dict__[c] is not None, f"{type(self).__name__} initialized without necessary constant {c}"
        print(f"Initializing", type(self).__name__, "...")
        params = self.init_parameters()
        assert isinstance(params, tuple)
        for k, v in zip(self.parameters, params):
            # Magic. There's no fucking way this should work. Yet it does.
            self.__dict__[k] = v
    
    @partial(jit, static_argnums=[0]) # For reuse: see ``tree_unflatten``
    def _init_fast(self, params, consts) -> None:
        for k, v in zip(self.constants , consts): self.__dict__[k] = v
        for k, v in zip(self.parameters, params): self.__dict__[k] = v

    def __init__(self,
                from_tree   :   bool    = False,
                params      :   tuple   = (),
                consts      :   list    = [],
                **kwargs) -> None:
        if from_tree:
            self._init_fast(params, consts)
        else:
            self._init_slow(kwargs)
        assert self.__call__.__code__, f"{type(self).__name__}'s __call__ method cannot be compiled in its body"
        self.__call__ = jit(self.__call__) # CRUCIAL for performance, recompiled for each instance as a compile-time constant
    
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError(f"{type(self).__name__} doesn't implement __call__")

    # @partial(jit, static_argnums=[0])
    def get_parameters(self) -> list:
        print("HAPPILY")
        print(self.parameters)
        for p in self.parameters:
            try:
                print(self.__dict__[p])
            except KeyError as e:
                print(p)
                raise e
        return [self.__dict__[p] for p in self.parameters]
    
    @partial(jit, static_argnums=[0]) # Did I just JIT a function that does absolutely, completely nothing? Yes. Yes, I did.
    def init_parameters(self) -> tuple:
        """ In order of ``self.parameters``. """
        raise NotImplementedError(f"{type(self).__name__} doesn't implement ``init_parameters``")

    @partial(jit, static_argnums=[0])
    def get_constants(self) -> list:
        print("I'M EATING COCK")
        return [self.__dict__[p] for p in self.constants]

    @partial(jit, static_argnums=[0])
    def init_constants(self) -> tuple:
        """ In order of ``self.constants``. """
        if len(self.constants) > 0: raise NotImplementedError(f"{type(self).__name__} doesn't implement ``init_parameters``")
    
    @partial(jit, static_argnums=[0])
    def update_(self, dL: Module, lr: jnp.float64):
        for param in self.parameters:
            p = self.__dict__[param]
            if isinstance(p, Module):
                self.__dict__[param].update_(dL.__dict__[param], lr)
            else:
                self.__dict__[param] = p - lr * dL.__dict__[param]

    @partial(jit, static_argnums=[0])
    def tree_flatten(self) -> tuple:
        print("EAT COCK")
        p = self.get_parameters()
        c = self.get_constants()
        r = (p, c)
        print("I HAVE EATEN COCK")
        return r
        # return (self.get_parameters(), ()) # okay
        # return ((), self.get_constants())

    @classmethod
    @partial(jit, static_argnums=[0]) # This ordering does, apparently, matter
    def tree_unflatten(cls, aux, children) -> Module:
        return cls(from_tree=True, params=children, consts=aux)



@register_pytree_node_class
class Fork(Module):
    """ General form of a continuous pointwise linear function. """
    
    constants = ()

    parameters = "m1", "m2", "b1", "b2"
    def init_parameters(self) -> tuple:
        return (
            jnp.float16(jnp.exp2(0.1 * randn())),
            jnp.float16(jnp.exp2(0.1 * randn())),
            jnp.asarray(0.1 * randn(), jnp.float16),
            jnp.asarray(0.1 * randn(), jnp.float16),
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        m1, m2, b1, b2 = self.m1, self.m2, self.b1, self.b2
        # m1 += R_AMT * randn()
        # m2 += R_AMT * randn() # Slight randomization
        # b1 += R_AMT * randn()
        # b2 += R_AMT * randn()
        intersection = stop_gradient((b2 - b1) / (m1 - m2))
        condition = x < intersection
        return jnp.where(condition, m1 * x + b1, m2 * x + b2)



@register_pytree_node_class
class Linear(Module):
    """ Simple matrix multiplication. """

    constants = "n_in", "n_out"
    def init_constants(self) -> tuple:
        return N, N

    parameters = "w", # Yes, the comma is on purpose
    def init_parameters(self) -> tuple:
        return jnp.asarray(randn(self.n_out, self.n_in), jnp.float16), # Again, yes, on purpose
    
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return (self.w + R_AMT * randn(*self.w.shape)) @ x



@register_pytree_node_class
class FeedForward(Module):
    """ Linear -> Fork -> Linear -> Fork """

    constants = "n_in", "expand"
    def init_constants(self) -> tuple:
        return N, 4

    parameters = "ForkA", "ForkB", "LinearA", "LinearB"
    def init_parameters(self) -> tuple:
        return (
            Fork(),
            Fork(),
            Linear(n_in=self.n_in, n_out=self.expand * self.n_in),
            Linear(n_in=self.expand * self.n_in, n_out=self.n_in),
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.ForkB(self.LinearB(self.ForkA(self.LinearA(x))))



@register_pytree_node_class
class SelfAttention(Module):
    """ Self-attention. ``x`` is projected to vectors Q, K, V within this function. """

    constants = "n_in", "d"
    def init_constants(self) -> tuple:
        return N, N

    parameters = "QKVProj", "ForkQ", "ForkK", "ForkV"
    def init_parameters(self) -> tuple:
        return Linear(n_in=self.n_in, n_out=3 * self.d), Fork(), Fork(), Fork()

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # Linear projection to all three in one operation by reshaping
        q, k, v = self.QKVProj(x).reshape(3, self.d, -1)
        q = self.ForkQ(q)
        k = self.ForkK(k)
        v = self.ForkV(v)
        rt = jnp.sqrt(self.d) # Compile-time constant thanks to superclass Module's JIT after instantiation. Verified in Jaxpr.
        return softmax((q @ k.T) / rt) @ v



@register_pytree_node_class
class MHA(Module):
    """ Self-attention. ``x`` is projected to vectors Q, K, V within this function. """

    constants = "n_in", "d", "heads" # d := "d_k", i.e. of each HEAD, not the whole enchilada
    def init_constants(self) -> tuple:
        return N, N // 8, 8

    parameters = "QKVProj", "FinalProj", "QForks", "KForks", "VForks"
    def init_parameters(self) -> tuple:
        return (
            Linear(n_in=self.n_in, n_out=3 * self.d * self.heads),
            Linear(n_in=self.d * self.heads, n_out=self.n_in),
            [Fork() for _ in range(self.heads)],
            [Fork() for _ in range(self.heads)],
            [Fork() for _ in range(self.heads)],
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # Linear projection to all three in one operation by reshaping
        q, k, v = self.QKVProj(x).reshape(3, self.heads, self.d, -1)
        for h in range(self.heads):
            # q[h] = self.QForks[h](q[h]), and similarly
            q = q.at[h].set(self.QForks[h](q[h]))
            k = k.at[h].set(self.KForks[h](k[h]))
            v = v.at[h].set(self.VForks[h](v[h]))
        scores = vmap(lambda a, b: a @ b.T)(q, k) / jnp.sqrt(self.d)
        cat = (softmax(scores) @ v).reshape(self.d * self.heads, -1)
        return self.FinalProj(cat)



@register_pytree_node_class
class EncoderBlock(Module):
    """ Single "block" in a Transformer encoder. """

    constants = "n_in", "d", "heads", "expand"
    def init_constants(self) -> tuple:
        return N, N // 8, 8, 4

    parameters = "SA", "FF"
    def init_parameters(self) -> tuple:
        return MHA(n_in=self.n_in, d=self.d, heads=self.heads), FeedForward(n_in=self.n_in, expand=self.expand)
        
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x = normalize(x + self.SA(x))
        # x = normalize(x + self.FF(x))
        x += self.SA(x)
        x += self.FF(x)
        return x