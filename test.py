"""Compare: pure StaggeredGrid vs centered↔staggered conversion each step."""

from phi.torch.flow import *
import matplotlib.pyplot as plt

domain = Box(x=100, y=100)
grid_kwargs = dict(x=64, y=64)
inflow = Sphere(x=50, y=9.5, radius=5)
inflow_rate = 0.2
buoyancy = 0.1
num_steps = 100

# Use 'auto' solver with suppressed errors
pressure_solve = Solve('auto', 1e-3, 1e-3, max_iterations=1000, suppress=[Diverged, NotConverged])

# =====================================================
# VERSION A: Pure StaggeredGrid (no conversion)
# =====================================================
print("=== VERSION A: Pure StaggeredGrid ===")

velocity_a = StaggeredGrid(0, 0, domain, **grid_kwargs)
smoke_a = CenteredGrid(0, ZERO_GRADIENT, domain, **grid_kwargs)

def step_pure(v, s, dt=1.):
    s = advect.mac_cormack(s, v, dt) + inflow_rate * resample(inflow, to=s, soft=True)
    buoyancy_force = resample(s * (0, buoyancy), to=v)
    v = advect.semi_lagrangian(v, v, dt) + buoyancy_force * dt
    v, _ = fluid.make_incompressible(v, solve=pressure_solve)
    return v, s

smoke_a_history = []
for i in range(num_steps):
    velocity_a, smoke_a = step_pure(velocity_a, smoke_a)
    smoke_a_history.append(float(math.max(smoke_a.values)))
    if i % 20 == 0:
        print(f"  Step {i}: smoke_max={smoke_a_history[-1]:.4f}")

# =====================================================
# VERSION B: Convert centered↔staggered each step
# =====================================================
print("\n=== VERSION B: Centered↔Staggered conversion each step ===")

vel_x_b = math.zeros(spatial(x=64, y=64))
vel_y_b = math.zeros(spatial(x=64, y=64))
smoke_b = math.zeros(spatial(x=64, y=64))

def step_convert(vel_x, vel_y, smoke_data, dt=1.):
    vel_centered = CenteredGrid(
        math.stack([vel_x, vel_y], channel(vector='x,y')),
        0, bounds=domain, **grid_kwargs
    )
    v = resample(vel_centered, to=StaggeredGrid(0, 0, domain, **grid_kwargs))
    s = CenteredGrid(smoke_data, ZERO_GRADIENT, domain, **grid_kwargs)
    
    s = advect.mac_cormack(s, v, dt) + inflow_rate * resample(inflow, to=s, soft=True)
    buoyancy_force = resample(s * (0, buoyancy), to=v)
    v = advect.semi_lagrangian(v, v, dt) + buoyancy_force * dt
    v, _ = fluid.make_incompressible(v, solve=pressure_solve)
    
    vel_at_centers = v.at_centers()
    return vel_at_centers.values.vector['x'], vel_at_centers.values.vector['y'], s.values

smoke_b_history = []
for i in range(num_steps):
    vel_x_b, vel_y_b, smoke_b = step_convert(vel_x_b, vel_y_b, smoke_b)
    smoke_b_history.append(float(math.max(smoke_b)))
    if i % 20 == 0:
        print(f"  Step {i}: smoke_max={smoke_b_history[-1]:.4f}")

# =====================================================
# Compare
# =====================================================
print("\n=== Comparison ===")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

ax = axes[0]
ax.plot(smoke_a_history, label='Pure StaggeredGrid', linewidth=2)
ax.plot(smoke_b_history, label='With conversion', linewidth=2, linestyle='--')
ax.set_xlabel('Step')
ax.set_ylabel('Max smoke')
ax.set_title('Smoke max over time')
ax.legend()
ax.grid(True)

ax = axes[1]
ax.imshow(smoke_a.values.numpy('y,x'), cmap='hot', origin='lower')
ax.set_title('Pure StaggeredGrid (final)')

ax = axes[2]
ax.imshow(smoke_b.numpy('y,x'), cmap='hot', origin='lower')
ax.set_title('With conversion (final)')

plt.tight_layout()
plt.savefig('staggered_comparison.png', dpi=150)
plt.show()

print("\nSaved to staggered_comparison.png")
print(f"Final smoke max difference: {abs(smoke_a_history[-1] - smoke_b_history[-1]):.6f}")