import matplotlib.pyplot as plt
import numpy as np

# Data
runs = ["Run 1", "Run 2", "Run 3"]
latent_dims = [64, 128, 192, 256]

gpu_usage = {
    64: [39, 42, 46],
    128: [43, 47, 68],
    192: [46, 61.5, 95.5],
    256: [50, 80, 0]  # OOM replaced with 0 for visualization
}

runtime = {
    64: [2*60 + 23, 2*60 + 43, 3*60 + 7],
    128: [2*60 + 40, 3*60 + 8, 3*60 + 52],
    192: [2*60 + 58, 3*60 + 40, 4*60 + 40],
    256: [3*60 + 13, 4*60 + 10, 0]  # OOM replaced with 0
}

# Plot setup
x = np.arange(len(runs))  # Indices for runs
bar_width = 0.2

fig, axs = plt.subplots(2, 1, figsize=(10, 8))

# --- GPU Usage Plot ---
ax = axs[0]
for i, latent in enumerate(latent_dims):
    values = gpu_usage[latent]
    ax.bar(x + i*bar_width - bar_width, values, bar_width, label=f"Latent {latent}")

ax.set_ylabel("GPU RAM Usage (%)")
ax.set_title("GPU RAM Usage per Run")
ax.set_xticks(x)
ax.set_xticklabels(runs)
ax.set_ylim(0,100)
ax.legend()
ax.grid(axis='y')

# --- Runtime Plot ---
ax = axs[1]
for i, latent in enumerate(latent_dims):
    values = runtime[latent]
    ax.bar(x + i*bar_width - bar_width, values, bar_width, label=f"Latent {latent}")

ax.set_ylabel("Runtime (s)")
ax.set_title("Runtime per Run")
ax.set_xticks(x)
ax.set_xticklabels(runs)
ax.legend()
ax.grid(axis='y')

plt.tight_layout()
plt.savefig("plot.png")
