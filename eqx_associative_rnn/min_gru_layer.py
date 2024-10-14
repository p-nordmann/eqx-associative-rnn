import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrandom
from jaxtyping import PRNGKeyArray


class MinGRULayer(eqx.Module):
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
        zs = jnn.sigmoid(eqx.filter_vmap(self.linear_z)(xs))
        hs = eqx.filter_vmap(self.linear_h)(xs)

        inputs = (prepend_ones(1 - zs), prepend_zeros(zs * hs))
        _, states = jax.lax.associative_scan(scan_fn, inputs)

        return eqx.filter_vmap(self.linear_out)(states[1:])


def default_floating_dtype():
    if jax.config.jax_enable_x64:  # pyright: ignore
        return jnp.float64
    else:
        return jnp.float32


def prepend_ones(a: jax.Array) -> jax.Array:
    return jnp.insert(a, 0, jnp.ones_like(a[0]), axis=0)


def prepend_zeros(a: jax.Array) -> jax.Array:
    return jnp.insert(a, 0, jnp.zeros_like(a[0]), axis=0)


def scan_fn(first, second):
    a_0, b_0 = first
    a_1, b_1 = second
    return a_0 * b_0, b_0 * a_1 + b_1
