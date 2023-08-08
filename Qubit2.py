import numpy as np
from itertools import product

# Define the Pauli matrices
sigma_x = np.array([[0, 1], [1, 0]])
sigma_y = np.array([[0, -1j], [1j, 0]])
sigma_z = np.array([[1, 0], [0, -1]])

# Define the measurement operators
m_ops = [np.kron(sigma_z, sigma_z),
         np.kron(sigma_z, sigma_x),
         np.kron(sigma_x, sigma_z),
         np.kron(sigma_x, sigma_x)]

# Generate random state
psi = np.random.rand(4) + 1j * np.random.rand(4)
psi /= np.linalg.norm(psi)

# Generate random probabilities
p = np.random.rand(4)
p /= np.sum(p)

# Generate random measurement counts
num_shots = 1000
counts = np.random.multinomial(num_shots, p)

# Compute M matrix
M_mat = np.zeros((4, 4), dtype=np.complex128)
for i, j in product(range(4), repeat=2):
    if counts[i] != 0:
        M_i = m_ops[i] @ psi
        M_j = m_ops[j] @ psi
        M_mat[i, j] = np.sqrt(counts[i]) * M_i.conj() @ M_j / (p[i] * counts[i])

# Compute density matrix via linear inversion
rho = np.linalg.pinv(M_mat)

print(rho)

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

bloch_vector=rho
fig = plt.figure()
ax = Axes3D(fig, azim=-135, elev=35)

# Plot the Bloch sphere
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
x = np.outer(np.cos(u), np.sin(v))
y = np.outer(np.sin(u), np.sin(v))
z = np.outer(np.ones(np.size(u)), np.cos(v))
ax.plot_surface(x, y, z, rstride=4, cstride=4, color='gray', alpha=0.1, linewidth=0)

# Plot the Bloch vector
ax.scatter(bloch_vector[0], bloch_vector[1], bloch_vector[2], color='red', s=100)

# Set the axis limits and labels
ax.set_xlim3d(-1, 1)
ax.set_ylim3d(-1, 1)
ax.set_zlim3d(-1, 1)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()


