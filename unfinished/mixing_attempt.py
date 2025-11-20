# %pip install --quiet phiflow
from phi.torch.flow import *
from tqdm.notebook import trange


# ------------------------------------------------------------
# System parameters
# ------------------------------------------------------------

N_COMP = 2
OMEGA = 5.0
THETA = 0.1
PRESSURE_COEFF = [1.0, 1.0]

DT = 0.05
STEPS = 300


# ------------------------------------------------------------
# Pressure function and gradients
# ------------------------------------------------------------

def pressure(rho, i):
    return PRESSURE_COEFF[i] * rho

def grad_pressure(rho, i):
    return gradient(pressure(rho, i))


# ------------------------------------------------------------
# Pipe geometry + mask
# ------------------------------------------------------------

PIPE_LENGTH = 100
PIPE_RADIUS = 20

grid = CenteredGrid(
    extrapolation=PERIODIC,          # for internal advection
    x=128, y=64,
    bounds=Box(x=PIPE_LENGTH, y=2*PIPE_RADIUS)
)

# Pipe mask: 1.0 inside the pipe, 0.0 outside
x, y = grid.vector_mesh()
mask = math.where(abs(y - PIPE_RADIUS) < PIPE_RADIUS, 1.0, 0.0)
pipe_mask = CenteredGrid(mask, grid.bounds)


# ------------------------------------------------------------
# Apply pipe boundary conditions
# ------------------------------------------------------------

def apply_pipe_bc(rho_list, q_list):
    """
    Solid walls: reflect momentum normal to the wall.
    Inflow at left, open outflow at right.
    """

    new_rhos = []
    new_qs = []

    for rho, q in zip(rho_list, q_list):

        # --- wall boundary (top/bottom): reflect normal momentum ---
        q_wall = q * pipe_mask + reflect(q) * (1 - pipe_mask)

        # --- inflow at left boundary ---
        inflow_rho = rho.shifted(x=1)
        inflow_q   = q_wall.shifted(x=1)
        q_wall = q_wall.with_boundary({'x-': inflow_q.values})
        rho    = rho.with_boundary({'x-': inflow_rho.values})

        # --- open outflow (right) ---
        q_wall = q_wall.with_boundary({'x+': q_wall.values})
        rho    = rho.with_boundary({'x+': rho.values})

        new_rhos.append(rho)
        new_qs.append(q_wall)

    return new_rhos, new_qs


# ------------------------------------------------------------
# One PDE time step
# ------------------------------------------------------------

@jit_compile
def step(state, dt=DT):

    rhos = state['rho']
    qs   = state['q']

    # Apply BCs at beginning
    rhos, qs = apply_pipe_bc(rhos, qs)

    # Compute velocities
    vs = [q / (rho + 1e-6) for q, rho in zip(qs, rhos)]

    total_rho = sum(rhos)
    total_q   = sum(qs)
    v_bar = total_q / (total_rho + 1e-6)

    new_rhos = []
    new_qs   = []

    for i in range(N_COMP):

        rho_i = rhos[i]
        q_i   = qs[i]
        v_i   = vs[i]

        # Continuity: rho_t + div(q_i) = 0
        rho_adv = advect.semi_lagrangian(rho_i, v_i, dt=dt)
        rho_new = rho_adv

        # Momentum equation
        grad_p = grad_pressure(rho_i, i)
        adv_v  = advect.semi_lagrangian(v_i, v_i, dt=dt)

        friction = -0.5 * THETA * v_i * field.l2_norm(v_i)
        coupling = -OMEGA * (v_i - v_bar)
        forces = friction + coupling - grad_p / (rho_i + 1e-6)

        v_new = adv_v + dt * forces
        q_new = v_new * rho_new

        new_rhos.append(rho_new)
        new_qs.append(q_new)

    # Apply BCs again after update
    new_rhos, new_qs = apply_pipe_bc(new_rhos, new_qs)

    return {'rho': new_rhos, 'q': new_qs}


# ------------------------------------------------------------
# Initial condition: slightly perturbed inflow
# ------------------------------------------------------------

rho_init = [grid + 1.0 + 0.2*Noise(), grid + 1.2 + 0.2*Noise()]
q_init   = [
    CenteredGrid(Noise(vector='x,y'), PERIODIC, x=128, y=64),
    CenteredGrid(Noise(vector='x,y'), PERIODIC, x=128, y=64)
]

state0 = {'rho': rho_init, 'q': q_init}


# ------------------------------------------------------------
# Time stepping using iterate()  (requested!)
# ------------------------------------------------------------

trajectory = iterate(step, batch(time=STEPS), state0)


# ------------------------------------------------------------
# Visualization
# ------------------------------------------------------------

# Show density of component 1
plot([s['rho'][0] for s in trajectory], animate='time')
