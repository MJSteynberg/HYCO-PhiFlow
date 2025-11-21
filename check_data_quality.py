"""Check if training data has noise or artifacts"""
from phi.flow import *
from phi.math import math
import matplotlib.pyplot as plt

# Load training data
sim_data = math.load("data/burgers/sim_0000.npz")
velocity = sim_data['velocity']

print(f"Velocity shape: {velocity.shape}")
print(f"Min: {float(math.min(velocity)):.6f}, Max: {float(math.max(velocity)):.6f}")

# Check for high-frequency noise in the data
# Compare gradient magnitudes
for t in [0, 5, 10]:
    v_t = velocity.time[t]
    grad = math.spatial_gradient(v_t, difference='forward')
    print(f"Time {t}: mean={float(math.mean(v_t)):.6f}, grad_magnitude={float(math.mean(math.abs(grad))):.6f}")

# Plot a few timesteps
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for i, t in enumerate([0, 5, 10]):
    v = velocity.time[t].vector[0].numpy('x')
    axes[i].plot(v)
    axes[i].set_title(f'Time {t}')
    axes[i].set_xlabel('x')
    axes[i].set_ylabel('velocity')
    axes[i].grid(True)

plt.tight_layout()
plt.savefig('data_quality_check.png')
print("\nSaved plot to data_quality_check.png")
print("Check if the data looks smooth or has oscillations")
