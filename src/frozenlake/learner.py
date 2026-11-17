"""One-step tabular Q-learning, compiled with JAX for independent parallel runs."""
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("JAX_ENABLE_X64", "true")
import jax
import jax.numpy as jnp
import numpy as np
from .initialization import initial_q


def synchronize(tree):
    for leaf in jax.tree_util.tree_leaves(tree):
        leaf.block_until_ready()


def initial_state(cfg, runs, terminal_states, device):
    count = len(runs)
    q = initial_q(cfg, terminal_states)
    keys = np.stack([np.array(jax.random.PRNGKey(r["map_seed"] * 100000 + r["training_seed"])) for r in runs])
    state = dict(q=q, visits=np.zeros_like(q, dtype=np.int32),
                 position=np.zeros(count, dtype=np.int32), length=np.zeros(count, dtype=np.int32),
                 episode_reward=np.zeros(count), episode_return=np.zeros(count),
                 td_sum=np.zeros(count), td_max=np.zeros(count),
                 episodes=np.zeros(count, dtype=np.int32), goals=np.zeros(count, dtype=np.int32),
                 keys=keys, step=np.array(0, dtype=np.int32))
    return jax.device_put(state, device)


def make_kernel(cfg, count):
    batch = jnp.arange(count, dtype=jnp.int32)
    train = cfg["training"]
    alpha_schedule, epsilon_schedule = train["alpha"], train["epsilon"]

    def kernel(carry, probability, next_state, reward, terminal):
        def step(c, _):
            split = jax.vmap(lambda key: jax.random.split(key, 2))(c["keys"])
            keys = split[:, 0, :]
            u = jax.vmap(lambda key: jax.random.uniform(key, (4,), dtype=jnp.float64))(split[:, 1, :])
            epsilon = epsilon_schedule["initial"] + (epsilon_schedule["final"] - epsilon_schedule["initial"]) * jnp.minimum(1., c["step"] / epsilon_schedule["decay_steps"])
            alpha = jnp.where(c["step"] >= alpha_schedule["switch_step"], alpha_schedule["final"], alpha_schedule["initial"])
            state = c["position"]
            values = c["q"][batch, state, :]
            mask = values == jnp.max(values, axis=1, keepdims=True)
            rank = jnp.floor(u[:, 2] * jnp.sum(mask, axis=1)).astype(jnp.int32) + 1
            greedy = jnp.argmax(mask & (jnp.cumsum(mask, axis=1) == rank[:, None]), axis=1)
            action = jnp.where(u[:, 0] < epsilon, jnp.floor(4 * u[:, 1]).astype(jnp.int32), greedy)
            probabilities = probability[batch, state, action, :]
            outcome = jnp.minimum(2, jnp.sum(u[:, 3, None] >= jnp.cumsum(probabilities, axis=1), axis=1)).astype(jnp.int32)
            nxt = next_state[batch, state, action, outcome]
            r = reward[batch, state, action, outcome]
            terminated = terminal[batch, state, action, outcome]
            old = c["q"][batch, state, action]
            continuation = jnp.max(c["q"][batch, nxt, :], axis=1)
            # Only simulator sampling reads P. The learner receives one observed
            # transition; neither model expectations nor evaluation feed updates.
            target = jnp.where(terminated, r, r + train["gamma"] * continuation)
            error = target - old
            q = c["q"].at[batch, state, action].set(old + alpha * error)
            visits = c["visits"].at[batch, state, action].add(1)
            length = c["length"] + 1
            episode_reward = c["episode_reward"] + r
            episode_return = c["episode_return"] + jnp.power(train["gamma"], c["length"]) * r
            td_sum = c["td_sum"] + jnp.abs(error)
            td_max = jnp.maximum(c["td_max"], jnp.abs(error))
            ended = terminated | (length >= cfg["environment"]["horizon"])
            kind = jnp.where(r > 0, 1, jnp.where(terminated, -1, 0)).astype(jnp.int32)
            event = (ended, length, episode_reward, episode_return, kind, td_sum / length, td_max)
            new = dict(q=q, visits=visits, position=jnp.where(ended, 0, nxt), length=jnp.where(ended, 0, length),
                       episode_reward=jnp.where(ended, 0., episode_reward), episode_return=jnp.where(ended, 0., episode_return),
                       td_sum=jnp.where(ended, 0., td_sum), td_max=jnp.where(ended, 0., td_max),
                       episodes=c["episodes"] + ended.astype(jnp.int32),
                       goals=c["goals"] + (ended & (r > 0)).astype(jnp.int32), keys=keys, step=c["step"] + 1)
            return new, event
        return jax.lax.scan(step, carry, xs=None, length=train["chunk_steps"])
    return kernel
