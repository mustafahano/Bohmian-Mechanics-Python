import numpy as np
import matplotlib.pyplot as plt

# Grid parameters
N = 200
x = np.linspace(-5.0, 5.0, N)
dx = x[1] - x[0]

# Gaussian test packet
f = np.exp(-x**2)

# Exact derivatives for benchmarking
df_exact = -2.0 * x * f
d2f_exact = (4.0 * x**2 - 2.0) * f

# Second-order central finite differences
df_fd = np.zeros_like(f)
d2f_fd = np.zeros_like(f)

df_fd[1:-1] = (f[2:] - f[:-2]) / (2.0 * dx)
d2f_fd[1:-1] = (f[2:] - 2.0 * f[1:-1] + f[:-2]) / (dx**2)

# Quick check on interior grid truncation error
print(f"1st deriv max error: {np.max(np.abs(df_fd[1:-1] - df_exact[1:-1])):.2e}")
print(f"2nd deriv max error: {np.max(np.abs(d2f_fd[1:-1] - d2f_exact[1:-1])):.2e}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

ax1.plot(x, df_exact, 'k-', lw=1.8, label='Analytical')
ax1.plot(x[1:-1], df_fd[1:-1], 'r--', lw=1.6, label='Central FD')
ax1.set_title('First Derivative (Velocity Field)')
ax1.set_xlabel('x')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(x, d2f_exact, 'k-', lw=1.8, label='Analytical')
ax2.plot(x[1:-1], d2f_fd[1:-1], 'b--', lw=1.6, label='Central FD')
ax2.set_title('Second Derivative (Kinetic & Q)')
ax2.set_xlabel('x')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("figures/finite_differences.png", dpi=300)
plt.show()