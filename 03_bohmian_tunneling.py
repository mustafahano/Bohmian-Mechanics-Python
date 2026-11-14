import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_banded

# Natural units (hbar = m = 1)
hbar, m = 1.0, 1.0

# Grid and time stepping
N = 1200
x_min, x_max = -20.0, 20.0
x = np.linspace(x_min, x_max, N)
dx = x[1] - x[0]

dt = 0.005
t_max = 5.0
n_steps = int(t_max / dt)

# Initial Gaussian wave packet (E_kin = k0^2 / 2 = 4.5)
x0, sigma0, k0 = -6.0, 1.2, 3.0
norm = (2.0 * np.pi * sigma0**2)**(-0.25)
psi = norm * np.exp(-(x - x0)**2 / (4.0 * sigma0**2)) * np.exp(1j * k0 * x)
psi[0] = psi[-1] = 0.0

initial_norm = np.sum(np.abs(psi)**2) * dx

# Rectangular potential barrier (V0 = 5.2 > E_kin)
V = np.zeros(N)
V[(x >= 0.0) & (x <= 1.2)] = 5.2

# Crank-Nicolson banded matrix setup
alpha = 1j * hbar * dt / (4.0 * m * dx**2)
beta = 1j * dt / (2.0 * hbar)

diag_A = 1.0 + 2.0 * alpha + beta * V
off_A = -alpha * np.ones(N - 1)

ab = np.zeros((3, N), dtype=complex)
ab[0, 1:] = off_A
ab[1, :] = diag_A
ab[2, :-1] = off_A

# Initial Bohmian ensemble positions
n_particles = 24
x_particles = np.linspace(x0 - 1.5 * sigma0, x0 + 1.5 * sigma0, n_particles)

trajectories = np.zeros((n_steps, n_particles))
density_history = np.zeros((n_steps, N))
time_grid = np.linspace(0.0, t_max, n_steps)

for step in range(n_steps):
    trajectories[step] = x_particles
    density_history[step] = np.abs(psi)**2

    # Bohmian velocity field: v = (hbar/m) * Im(dpsi_dx / psi)
    dpsi_dx = np.zeros_like(psi)
    dpsi_dx[1:-1] = (psi[2:] - psi[:-2]) / (2.0 * dx)

    psi_safe = np.where(np.abs(psi) < 1e-12, 1e-12 + 0j, psi)
    v_field = (hbar / m) * np.imag(dpsi_dx / psi_safe)

    x_particles += np.interp(x_particles, x, v_field) * dt

    # Advance wave function via Crank-Nicolson
    b = np.zeros_like(psi)
    b[1:-1] = (
        alpha * psi[:-2]
        + (1.0 - 2.0 * alpha - beta * V[1:-1]) * psi[1:-1]
        + alpha * psi[2:]
    )
    b[0] = b[-1] = 0.0

    psi = solve_banded((1, 1), ab, b)

# Diagnostics
final_norm = np.sum(np.abs(psi)**2) * dx
T_prob = np.sum(np.abs(psi[x > 1.2])**2) * dx

print(f"Norm check: {initial_norm:.6f} -> {final_norm:.6f}")
print(f"Transmission T: {T_prob:.4f} ({T_prob:.2%})")

# Plot probability density heatmap and Bohmian trajectories
plt.figure(figsize=(9.5, 5.8))
plt.imshow(
    density_history,
    extent=[x_min, x_max, 0.0, t_max],
    origin="lower",
    aspect="auto",
    cmap="magma",
    alpha=0.92,
)
plt.colorbar(label=r"$|\psi(x,t)|^2$")

plt.axvline(0.0, color="cyan", linestyle="--", lw=1.2, label="Barrier (V0 = 5.2)")
plt.axvline(1.2, color="cyan", linestyle="--", lw=1.2)

for i in range(n_particles):
    plt.plot(trajectories[:, i], time_grid, color="white", lw=1.0, alpha=0.85)

plt.title(f"1D Quantum Tunneling & Bohmian Trajectories (T = {T_prob:.1%})")
plt.xlabel("x")
plt.ylabel("t")
plt.xlim(-12.0, 14.0)
plt.legend(loc="upper left")
plt.tight_layout()
plt.savefig("figures/bohmian_tunneling.png", dpi=300)
plt.show()