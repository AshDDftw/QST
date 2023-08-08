import numpy as np

# Define the Pauli matrices
sigma_x = np.array([[0, 1], [1, 0]])
sigma_y = np.array([[0, -1j], [1j, 0]])
sigma_z = np.array([[1, 0], [0, -1]])

# Define the measurement operators
m_00 = (np.kron(sigma_z, sigma_z) + np.eye(4)) / 4
m_01 = (np.kron(sigma_z, sigma_x)) / 4
m_10 = (np.kron(sigma_x, sigma_z)) / 4
m_11 = (np.kron(sigma_x, sigma_x) - np.kron(sigma_y, sigma_y)) / 4

# Generate the measurement outcomes
outcomes = []
for i in range(10000):
    # Generate a random state vector
    psi = np.random.rand(4) + 1j * np.random.rand(4)
    psi /= np.linalg.norm(psi)

    # Generate the measurement outcomes
    p_00 = np.real(np.dot(np.conj(psi), np.dot(m_00, psi)))
    p_01 = np.real(np.dot(np.conj(psi), np.dot(m_01, psi)))
    p_10 = np.real(np.dot(np.conj(psi), np.dot(m_10, psi)))
    p_11 = np.real(np.dot(np.conj(psi), np.dot(m_11, psi)))

    outcomes.append((p_00, p_01, p_10, p_11))

# Reconstruct the density matrix using maximum likelihood estimation
rho_ml = np.zeros((4, 4), dtype=np.complex128)
for outcome in outcomes:
    p_00, p_01, p_10, p_11 = outcome
    rho_ml += p_00 * np.kron(np.eye(2), np.eye(2)) + \
              p_01 * np.kron(np.eye(2), sigma_x) + \
              p_10 * np.kron(sigma_x, np.eye(2)) + \
              p_11 * np.kron(sigma_x, sigma_x)

rho_ml /= len(outcomes)

print(rho_ml)
