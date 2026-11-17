"""The only initialization hook changed by the planned optimistic control."""
import numpy as np


def initial_q(cfg, terminal_states):
    terminal_states = np.asarray(terminal_states, dtype=bool)
    q = np.full((*terminal_states.shape, 4), cfg["initialization"]["value"], dtype=np.float64)
    q[terminal_states] = 0.0
    return q
