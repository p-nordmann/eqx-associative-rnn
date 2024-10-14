import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import PRNGKeyArray

from eqx_associative_rnn.min_gru_layer import MinGRULayer


def test_training_sinusoid():
    """Training on a sinusoid to make sure it learns something."""
    seed = 0
    key = jax.random.PRNGKey(seed)

    # Make sinusoidal data.
    key, key_sinusoid = jax.random.split(key)
    data_train, data_test = generate_training_data(
        100_000, 80_000, 100, 0.3, key=key_sinusoid
    )

    # Make model.
    key, *keys_model = jax.random.split(key, 4)
    model = eqx.nn.Sequential(
        [
            MinGRULayer(1, 200, 100, key=keys_model[0]),
            MinGRULayer(100, 200, 100, key=keys_model[1]),
            MinGRULayer(100, 200, 1, key=keys_model[2]),
        ]
    )

    # Make optimizer.
    opt = optax.adam(learning_rate=1e-3)
    opt_state = opt.init(model)

    batch_size = 100
    window_size = 40

    # Eval model before training.
    key, key_epoch = jax.random.split(key)
    losses_before = []
    for inputs in make_epoch(
        data=data_test, window_size=window_size, batch_size=batch_size, key=key_epoch
    ):
        loss = make_eval_step(model, inputs)
        losses_before.append(loss)

    # Train model after training.
    key, key_epoch = jax.random.split(key)
    for inputs in make_epoch(
        data=data_train, window_size=window_size, batch_size=batch_size, key=key_epoch
    ):
        model, opt_state = make_step(model, inputs, opt, opt_state)

    # Eval model.
    key, key_epoch = jax.random.split(key)
    losses_after = []
    for inputs in make_epoch(
        data=data_test, window_size=window_size, batch_size=batch_size, key=key_epoch
    ):
        loss = make_eval_step(model, inputs)
        losses_after.append(loss)

    # Check that loss is good.
    before, after = jnp.mean(jnp.stack(losses_before)), jnp.mean(
        jnp.stack(losses_after)
    )

    expected_improvement_factor = 20  # Abitrary, loss should significatively improve
    assert before / expected_improvement_factor > after


def generate_training_data(
    total_size, train_size, sine_period, noise_std, *, key: PRNGKeyArray
):
    data = (
        jnp.sin(jnp.arange(total_size) * (2 * jnp.pi) / sine_period)
        + jax.random.normal(shape=(total_size,), key=key) * noise_std
    )
    data = (data - jnp.min(data)) / (jnp.max(data) - jnp.min(data))
    return data[:train_size, jnp.newaxis], data[train_size:, jnp.newaxis]


def make_epoch(data, window_size, batch_size, *, key):
    length = data.shape[0]
    dim = data.shape[1]

    # Make permutation of the data.
    key, key_permutation = jax.random.split(key)
    idx = jax.random.permutation(x=data.shape[0], key=key_permutation)

    # Build batches.
    for k in range(0, length - window_size, batch_size):
        if k + batch_size > length - window_size:
            break  # Skip the end
        batch = jnp.zeros((batch_size, window_size, dim), dtype=data.dtype)
        for j in range(window_size):
            batch = batch.at[:, j].set(data[idx[k : k + batch_size] + j])
        yield batch


def compute_loss(model, inputs):
    return jnp.mean(optax.l2_loss(jax.vmap(model)(inputs)[:, :-1], inputs[:, 1:]))


@eqx.filter_jit
def make_step(model, inputs, opt, opt_state):
    _, grads = eqx.filter_value_and_grad(compute_loss)(model, inputs)
    updates, opt_state = opt.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state


@eqx.filter_jit
def make_eval_step(model, inputs):
    return compute_loss(model, inputs)
