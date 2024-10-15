from typing import Optional

import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrandom
from jaxtyping import Array, PRNGKeyArray


class MinGRUParallelLayer(eqx.Module):
    linear_z: eqx.nn.Linear
    linear_h: eqx.nn.Linear
    linear_out: eqx.nn.Linear

    input_size: int = eqx.field(static=True)
    hidden_size: int = eqx.field(static=True)
    output_size: int = eqx.field(static=True)
    use_bias: bool = eqx.field(static=True)

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        use_bias: bool = True,
        dtype=None,
        *,
        key: PRNGKeyArray,
    ):
        dtype = default_floating_dtype() if dtype is None else dtype
        key_z, key_h, key_out = jrandom.split(key, 3)

        self.linear_z = eqx.nn.Linear(
            input_size, hidden_size, use_bias, dtype, key=key_h
        )
        self.linear_h = eqx.nn.Linear(
            input_size, hidden_size, use_bias, dtype, key=key_z
        )
        self.linear_out = eqx.nn.Linear(
            hidden_size, output_size, use_bias, dtype, key=key_out
        )

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.use_bias = use_bias

    def __call__(self, xs, *, key: PRNGKeyArray | None = None):
        zs = jnn.sigmoid(jax.vmap(self.linear_z)(xs))
        hs = jax.vmap(self.linear_h)(xs)

        inputs = (prepend_ones(1 - zs), prepend_zeros(zs * hs))
        _, states = jax.lax.associative_scan(associative_scan_fn, inputs)

        return jax.vmap(self.linear_out)(states[1:])


def default_floating_dtype():
    if jax.config.jax_enable_x64:  # pyright: ignore
        return jnp.float64
    else:
        return jnp.float32


def prepend_ones(a: jax.Array) -> jax.Array:
    return jnp.insert(a, 0, jnp.ones_like(a[0]), axis=0)


def prepend_zeros(a: jax.Array) -> jax.Array:
    return jnp.insert(a, 0, jnp.zeros_like(a[0]), axis=0)


def associative_scan_fn(first, second):
    a_0, b_0 = first
    a_1, b_1 = second
    return a_0 * a_1, a_1 * b_0 + b_1


# We also define a non-parallel version for comparison purposes


class MinGRUCell(eqx.Module, strict=True):
    linear_z: eqx.nn.Linear
    linear_h: eqx.nn.Linear
    input_size: int = eqx.field(static=True)
    hidden_size: int = eqx.field(static=True)
    use_bias: bool = eqx.field(static=True)

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        use_bias: bool = True,
        dtype=None,
        *,
        key: PRNGKeyArray,
    ):
        dtype = default_floating_dtype() if dtype is None else dtype
        key_z, key_h = jrandom.split(key)
        self.linear_z = eqx.nn.Linear(
            input_size, hidden_size, use_bias, dtype, key=key_h
        )
        self.linear_h = eqx.nn.Linear(
            input_size, hidden_size, use_bias, dtype, key=key_z
        )
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_bias = use_bias

    def __call__(
        self, input: Array, hidden: Array, *, key: Optional[PRNGKeyArray] = None
    ):
        gate = jnn.sigmoid(self.linear_z(input))
        update = self.linear_h(input)
        hidden_new = (1 - gate) * hidden + gate * update
        return hidden_new


class MinGRULayer(eqx.Module, strict=True):
    cell: MinGRUCell
    linear_out: eqx.nn.Linear
    input_size: int = eqx.field(static=True)
    hidden_size: int = eqx.field(static=True)
    output_size: int = eqx.field(static=True)
    use_bias: bool = eqx.field(static=True)

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        use_bias: bool = True,
        dtype=None,
        *,
        key: PRNGKeyArray,
    ):
        dtype = default_floating_dtype() if dtype is None else dtype
        key_cell, key_out = jrandom.split(key)

        self.cell = MinGRUCell(input_size, hidden_size, use_bias, dtype, key=key_cell)
        self.linear_out = eqx.nn.Linear(
            hidden_size, output_size, use_bias, dtype, key=key_out
        )

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.use_bias = use_bias

    def __call__(self, xs, *, key: PRNGKeyArray | None = None):
        def scan_fn(state, input):
            new_state = self.cell(input, state)
            out = self.linear_out(new_state)
            return new_state, out

        init_state = jnp.zeros(self.cell.hidden_size)
        _, outs = jax.lax.scan(scan_fn, init_state, xs)
        return outs
