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

# Define the measurement outcomes
outcomes = [(35000, 360, 32800, 499),
            (16324, 17521, 13441, 16901),
            (17932, 32028, 15132, 17238),
            (13171, 17170, 16722, 33586)]

# Reconstruct the density matrix using maximum likelihood estimation
rho_ml = np.zeros((4, 4), dtype=np.complex128)
for outcome in outcomes:
    p_00 = outcome[0] / sum(outcome)
    p_01 = outcome[1] / sum(outcome)
    p_10 = outcome[2] / sum(outcome)
    p_11 = outcome[3] / sum(outcome)

    rho_ml += p_00 * np.kron(np.eye(2), np.eye(2)) + \
              p_01 * np.kron(np.eye(2), sigma_x) + \
              p_10 * np.kron(sigma_x, np.eye(2)) + \
              p_11 * np.kron(sigma_x, sigma_x)

print(rho_ml)