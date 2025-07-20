import jax.numpy as jnp

from eqx_mingru.mingru import prepend_ones, prepend_zeros


def test_prepend_zeros():
    eps = 1e-9
    xs = jnp.ones(shape=(3, 4))
    got = prepend_zeros(xs)
    want = jnp.array([[0, 0, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]])
    assert got.shape == want.shape
    assert jnp.allclose(got, want, atol=eps).item()


def test_prepend_ones():
    eps = 1e-9
    xs = jnp.zeros(shape=(3, 4))
    got = prepend_ones(xs)
    want = jnp.array([[1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    assert got.shape == want.shape
    assert jnp.allclose(got, want, atol=eps).item()
