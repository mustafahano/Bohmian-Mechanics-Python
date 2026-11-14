import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh_tridiagonal

# Spatial grid (natural units: hbar = m = omega = 1)
N = 500
x = np.linspace(-6.0, 6.0, N)
dx = x[1] - x[0]

# Harmonic oscillator potential
V = 0.5 * x**2

# Tridiagonal Hamiltonian matrix elements
diag = (1.0 / dx**2) + V
off_diag = -0.5 * (1.0 / dx**2) * np.ones(N - 1)

# Solve for the lowest 4 eigenstates
eigvals, eigvecs = eigh_tridiagonal(
    diag, off_diag, select="i", select_range=(0, 3)
)

# Compare numerical eigenvalues against exact E_n = n + 0.5
exact_E = np.arange(4) + 0.5
for n in range(4):
    print(f"n={n} | Numerical E: {eigvals[n]:.6f} | Exact E: {exact_E[n]:.1f} | Error: {abs(eigvals[n] - exact_E[n]):.2e}")

# Plot potential and shifted eigenfunctions
plt.figure(figsize=(8.5, 5.5))
plt.plot(x, V, "k--", lw=1.4, label=r"$V(x) = \frac{1}{2}x^2$")

for n in range(4):
    # Normalize eigenvector so integral(|psi|^2 dx) = 1
    psi_n = eigvecs[:, n] / np.sqrt(dx)
    plt.plot(x, 0.4 * psi_n + eigvals[n], lw=1.8, label=f"n={n}, E={eigvals[n]:.3f}")
    plt.axhline(eigvals[n], color="gray", linestyle=":", alpha=0.4)

plt.xlim(-4.0, 4.0)
plt.ylim(0.0, 4.5)
plt.title("1D Harmonic Oscillator Eigenstates (FD Tridiagonal Solver)")
plt.xlabel("x")
plt.ylabel("Energy / Shifted Amplitude")
plt.legend(loc="upper right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("figures/harmonic_oscillator.png", dpi=300)
plt.show()