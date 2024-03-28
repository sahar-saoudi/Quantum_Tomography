import numpy as np
from numpy.typing import NDArray
from qiskit import QuantumCircuit, transpile
from qiskit.providers import Backend
from qiskit.quantum_info import Pauli, PauliList
from typing import List
from diagonalization import diagonalize_pauli_with_circuit

def bitstring_to_bits(bit_string: str) -> List[bool] :
    """
    Converts a string of 1s and 0s to a list of boolean and reverses it

        Parameters :
            bit_string (str) : a string of 1s and 0s to convert

        Returns :
            bits (List[bool]) : a list of boolean values based on the digits of bit_string
    """
    bits = [(bit=='1') for bit in bit_string]
    reversed_bits = bits[::-1]
    return reversed_bits

def diag_pauli_eigenvalue(pauli: Pauli, state: str) -> int :
    """
    Calculates the eigenvalue of a diagonal Pauli matrix

        Parameters :
            bit_string (str) : a string of 1s and 0s to convert

        Returns :
            bits (List[bool]) : a list of boolean values based on the digits of bit_string
    """
    boolean_state = bitstring_to_bits(state)
    vector_product = np.logical_and(pauli.z, boolean_state)
    if (np.count_nonzero(vector_product) % 2) == 0 : return 1 
    return -1

def diag_pauli_expectation_value(pauli: Pauli, counts: dict) -> float:
    assert(np.all(~pauli.x))

    num_of_counts = sum(counts.values())
    expectation_value = 0

    for state in counts.keys() :
        state_probability = counts[state] / num_of_counts
        eigenvalue = diag_pauli_eigenvalue(pauli, state)
        expectation_value += state_probability * eigenvalue
    return expectation_value

def transpile_diag_state_circuit(state_circuit : QuantumCircuit, diag_circuit : QuantumCircuit, backend : Backend) -> QuantumCircuit :
    """
    Creates a transpiled circuit containing a copy of the state circuit and a circuit to diagonalize it

        Parameters :
            state_circuit (QuantumCircuit) : a quantum circuit that represent our quantum state
            diag_circuit (QuantumCircuit) : a quantum circuit to diagonalize it
            backend (Backend) : a backend to transpile the circuit to

        Output :
           transpile_diag_state_circuit (QuantumCircuit): a transpiled circuit 
    """
    circ1_copy = state_circuit
    circuits_combined = circ1_copy.compose(diag_circuit)
    circuits_combined.measure_all(add_bits=False)        
    transpiled_circuit = transpile(circuits_combined, backend)
    return transpiled_circuit

def estimate_expectation_values(
        paulis: PauliList,
        state_circuit: QuantumCircuit,
        backend: Backend,
        execute_opts: dict = dict()
    ) -> NDArray[np.float_]:
    """
    Estimates the expectation values of a Pauli

        Parameters :
            paulis (PauliList) : a list of Paulis to estimate the expectation values of
            state_circuit (QuantumCircuit) : a quantum circuit that represent our quantum state
            backend (Backend) : a backend on which to run the state_circuit
            execute_opts (dict) : a dict of optional parameter for the circuit execution

        Output :
            expectation_values (NDArray[np.float_]) : an array containing the estimated expectation values of each Pauli
    """

    diag_state_circuits = []
    diag_paulis = []
    for i, pauli in enumerate(paulis) : 
        diag_pauli, diag_pauli_circuit = diagonalize_pauli_with_circuit(pauli)
        diag_paulis.append(diag_pauli)
        transpiled_circuit = transpile_diag_state_circuit(state_circuit, diag_pauli_circuit, backend)
        diag_state_circuits.append(transpiled_circuit)

    results = backend.run(diag_state_circuits, **execute_opts).result()
    
    expectation_values = np.empty(paulis.size)
    for i, pauli in enumerate(diag_paulis) :
        counts = results.get_counts(i)
        expectation_values[i] = diag_pauli_expectation_value(pauli, counts)

    return expectation_values


