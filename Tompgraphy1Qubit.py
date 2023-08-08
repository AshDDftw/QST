import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the state vector
psi = (1/np.sqrt(2)) * np.array([[1], [1j]])

# Define the Pauli matrices
mx = np.array([[0, 1], [1, 0]])
my = np.array([[0, -1j], [1j, 0]])
mz = np.array([[1, 0], [0, -1]])

# Calculate the expectation values
x_exp = np.real(np.dot(np.conj(psi.T), np.dot(mx, psi)))
y_exp = np.real(np.dot(np.conj(psi.T), np.dot(my, psi)))
z_exp = np.real(np.dot(np.conj(psi.T), np.dot(mz, psi)))

# Construct the density matrix
rho = (np.identity(2) + x_exp * mx + y_exp * my + z_exp * mz) / 2

# Calculate the eigenvalues and eigenvectors of the density matrix
evals, evecs = np.linalg.eig(rho)

# Calculate the probabilities of measuring each eigenstate
probs = np.real(np.diag(np.dot(np.conj(evecs.T), np.dot(rho, evecs))))

# Calculate the Bloch vector
bloch = np.real([np.trace(np.dot(mx, rho)), np.trace(np.dot(my, rho)), np.trace(np.dot(mz, rho))])

print(rho)

# Define the sphere surface
u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
x = np.cos(u)*np.sin(v)
y = np.sin(u)*np.sin(v)
z = np.cos(v)

# Create the plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the sphere surface
ax.plot_wireframe(x, y, z, color='gray', alpha=0.3)

# Plot the Bloch vector
ax.scatter(bloch[0], bloch[1], bloch[2], s=100)

# Add the arrow to the plot
arrow_length = 0.5
ax.quiver(0, 0, 0, bloch[0], bloch[1], bloch[2],  color='black', alpha=0.8, arrow_length_ratio=0.1)

# Set the axis limits and labels
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(-1, 1)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Show the plot
plt.show()
