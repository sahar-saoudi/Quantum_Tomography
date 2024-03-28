import numpy as np
from numpy.typing import NDArray
from qiskit import QuantumCircuit
from qiskit.providers import Backen
from qiskit.quantum_info import PauliList, pauli_basis
from expectation_values import estimate_expectation_values


def create_density_matrix(
        coefficients: NDArray[np.float_], 
        paulis: PauliList
    ) -> NDArray[np.float_]:
        density_matrix = np.tensordot(paulis.to_matrix(array=True), coefficients, axes=([0],[0]))
        return density_matrix

def state_tomography(
state_circuit: QuantumCircuit,
backend: Backend,
execute_opts : dict = dict(),

)-> NDArray[np.complex_]:
    """
    Gets the tomography of an unknown quantum state

        Parameters :
            state_circuit (QuantumCircuit) : a QuantumCircuit that prepares our unknown state
            backend (Backend) : a backend on which to run the state_circuit
            execute_opts (dict) : a dict of optional parameter for the circuit execution

        Output :
            state_vector (NDArray[np.complex_]) : an array representing the state vector of our unknown state
    """

    num_of_qubits = state_circuit.num_qubits
    paulis = pauli_basis(num_of_qubits)

    expectation_values = estimate_expectation_values(paulis, state_circuit, backend, execute_opts)
    coefficients = expectation_values/(2**num_of_qubits)
    density_matrix = create_density_matrix(coefficients, paulis)

    probabilities, state_vectors = np.linalg.eigh(density_matrix)
    index = np.argmax(probabilities)
    state_vector = state_vectors[:, index]
        
    return state_vector
