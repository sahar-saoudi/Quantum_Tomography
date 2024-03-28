import unittest
import expectation_values as tested_file
import numpy as np
from numpy.typing import NDArray
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Pauli, PauliList, Statevector, pauli_basis, DensityMatrix

class Test_expectation_value(unittest.TestCase):
    def setUp(self):
        self.diag_pauli1 = Pauli('Z')
        self.diag_pauli2 = Pauli('ZIZ')

        self.state1 = '0'
        self.state2 = '110'

    def test_bitstring_to_bit(self):
        bits_expected = [False]
        bits = tested_file.bitstring_to_bits(self.state1)
        self.assertEqual(bits, bits_expected, 'Error converting bitstring to bit')

        bits_expected = [False, True, True]
        bits = tested_file.bitstring_to_bits(self.state2)
        self.assertEqual(bits, bits_expected, 'Error converting bitstring to bit')


    def test_diag_eigenvalue(self):
        self.assertEqual(tested_file.diag_pauli_eigenvalue(self.diag_pauli1, self.state1), 1, 'The eigenvalue is wrong')

        self.assertEqual(tested_file.diag_pauli_eigenvalue(self.diag_pauli2, self.state2), -1, 'The eigenvalue is wrong')

    def test_diag_expectation_value(self):
        counts = {'0' : 5, '1' : 5}
        self.assertEqual(tested_file.diag_pauli_expectation_value(self.diag_pauli1, counts), 0, 'Expectation value of the diagonal pauli is wrong')

        counts = {'000' : 1, '010' : 10, '101' : 5, '111' : 4}
        self.assertEqual(tested_file.diag_pauli_expectation_value(self.diag_pauli2, counts), 1, 'Expectation value of the diagonal pauli is wrong')

    def test_expectation_values_estimation(self):
        
        num_of_qubits = 2
        paulis = PauliList(['YI', 'IX'])
        qreg = QuantumRegister(num_of_qubits)
        creg = ClassicalRegister(num_of_qubits)
        state_circuit = QuantumCircuit(qreg, creg)
        state_circuit.ry(np.pi/2, 0)
        execute_opts = {'shots': 1000}

        expectation_values = tested_file.estimate_expectation_values(paulis, state_circuit, AerSimulator(), execute_opts)

        state_circuit = QuantumCircuit(qreg)
        state_circuit.ry(np.pi/2, 0)
        
        expected_answers = np.linalg.eigvals(state_circuit.to_gate().to_matrix())
        self.assertEqual(len(expectation_values), len(expected_answers))
        for expectation_value, expected_answer in zip(expectation_values, expected_answers) :
            self.assertAlmostEqual(expectation_value, expected_answer, delta=0.1)
        
if __name__ == '__main__':
    unittest.main()
    
        