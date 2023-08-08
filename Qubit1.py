import numpy as np

def measure(qubit_state):
    # Define the Pauli matrices
    sigma_x = np.array([[0, 1], [1, 0]])
    sigma_y = np.array([[0, -1j], [1j, 0]])
    sigma_z = np.array([[1, 0], [0, -1]])

    # Define the Hadamard gate
    Hadamard = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]])

    # Apply the Hadamard gate to the qubit
    qubit_state = np.dot(sigma_x, qubit_state)

    # print(qubit_state)

    # Measure the qubit
    probabilities = np.abs(qubit_state) ** 2

    # print(probabilities)
    measurement = np.random.choice([0, 1], p=probabilities.ravel())

    # Print the measurement result
    print(measurement)

# Define the state vector of the qubit

# qubit_state = np.array([[0], [1]])  # qubit initialized to |0>


measure(np.array([[1], [0]]))
# measure(np.array([[1], [0]]))
# measure(np.array([[0], [1]]))
# measure(np.array([[0], [1]]))
# measure(np.array([[0], [1]]))
# measure(np.array([[1], [0]]))
# measure(np.array([[1], [0]]))