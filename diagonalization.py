from typing import Tuple, List
from numpy.typing import NDArray
from qiskit.providers import Backend
from qiskit.quantum_info import Pauli, PauliList
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.primitives import Sampler
import numpy as np
    
def diagonalize_circuit(pauli : Pauli) -> QuantumCircuit:
    num_of_qubits = pauli.num_qubits
    qreg = QuantumRegister(num_of_qubits)
    diagonalization_circuit = QuantumCircuit(qreg)

    for i in range(num_of_qubits):
        if pauli.x[i] :
            if pauli.z[i] :
                diagonalization_circuit.sdg(i)
            diagonalization_circuit.h(i)

    return diagonalization_circuit

def diagonalize_pauli_with_circuit(pauli : Pauli) -> Tuple[Pauli, QuantumCircuit]:  
    """
    Diagonalizes a Pauli matrix and gets the circuit that does so

        Parameters :
            pauli (Pauli) : a Pauli matrix to diagonalize

        Output :
            (diagonal_pauli, circuit) (Tuple[Pauli, QuantumCircuit]) : a diagonaliazed Pauli and a circuit that diagonalized it
    
    """ 
    new_zs = np.logical_or(pauli.z, pauli.x)
    diag_pauli = Pauli((new_zs, np.zeros(pauli.num_qubits)))
    diag_circuit = diagonalize_circuit(pauli)
         
    assert(np.all(~diag_pauli.x))
    return (diag_pauli, diag_circuit)
