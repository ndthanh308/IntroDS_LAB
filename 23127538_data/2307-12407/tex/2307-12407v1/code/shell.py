import jax_fdm.equilibrium as jfe
from jax_fdm.datastructures import FDNetwork

# load a structure from a COMPAS network 
net = FDNetwork.from_json("shell.json")
eqs = jfe.EquilibriumStructure
structure = eqs.from_network(net)
import jax.numpy as jnp

# set the initial force densities
q = jnp.ones(structure.num_edges) * -1.0

# compute an equilibrium state
fdm = jfe.EquilibriumModel()
params = (q, xyz_fixed, loads)
eq_state = fdm(params, structure)

# define loss and target functions
def goal_fn(eq_state):
    dist = (eq_state.xyz - xyz_target)**2
    return jnp.sum(dist) 

def loss_fn(q):
    params = (q, xyz_fixed, loads)
    eq_state = fdm(params, structure)
    return goal_fn(eq_state)

import optax
from jax import jit
from jax import value_and_grad

@jit
def opt_step(q, o_state):
    loss, grad = value_and_grad(loss_fn)(q)
    upd, o_state = opt.update(grad, o_state)
    q = optax.apply_updates(q, upd)
    return q, o_state, loss

# optimization loop
opt = optax.adam(learning_rate=0.01)
o_state = opt.init(q)

for i in range(5000):
    q, o_state, loss = opt_step(q, o_state)