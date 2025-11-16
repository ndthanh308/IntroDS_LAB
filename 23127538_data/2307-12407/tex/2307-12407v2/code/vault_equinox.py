import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import equinox as eqx
import jax_fdm.equilibrium as jfe


# load a structure from a json file
network = FDNetwork.from_json("vault.json")

# instantiate structure and model
vault = jfe.Structure.from_network(network)
model = jfe.Model.from_network(network)

# compute an equilibrium state
eq_state = model(vault)

# partition model
filter = jtu.tree_map(lambda _: False, model)
filter = eqx.tree_at(lambda tree: (tree.q), filter, replace=(True))


# define loss function
def loss_fn(diff, static):
    model = eqx.combine(diff, static)
    eq_state = model(vault)
    loss = (eq_state.xyz - xyz_hat)**2
    return jnp.mean(loss)

import optax

opt = optax.adam(learning_rate=0.001)
opt_state = opt.init(model)

@jax.jit
def opt_step(model, opt_state):
    diff, static = eqx.partition(model, filter)
    loss, grad = jax.value_and_grad(loss_fn)(diff, static)
    updates, opt_state = opt.update(grad, opt_state)
    model = optax.apply_updates(model, updates)
    return q, opt_state, loss
