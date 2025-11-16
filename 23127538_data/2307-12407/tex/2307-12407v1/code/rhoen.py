# define target functions
def goal_fn(eq_state):
    diff = (eq_state.lengths - 0.15)**2
    std = jnp.std(eq_state.lengths)
    return jnp.sum(diff) + 0.01 * std