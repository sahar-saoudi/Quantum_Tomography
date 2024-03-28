import unittest
import tomography as tested_file
import numpy as np
from numpy.typing import NDArray
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Pauli, PauliList, Statevector, pauli_basis, DensityMatrix

class Test_expectation_value(unittest.TestCase):
    def setUp(self):
        num_of_qubits = 3
        qreg = QuantumRegister(num_of_qubits)
        creg = ClassicalRegister(num_of_qubits)
        self.state_circuit = QuantumCircuit(qreg, creg)

        self.execute_opts = {'shots': 1000}


    def test_density_matrix(self):
        expected_answer1 = np.array([[1, 0],[0,0]], dtype=np.float_)
        coefficients1 = np.array([0.5, 0, 0, 0.5])
        density_matrix_created = tested_file.create_density_matrix(coefficients1, pauli_basis(1))
        self.assertTrue(np.all(np.equal(expected_answer1, density_matrix_created)))

        expected_answer2 = np.array([[1, 1, 1, 1],[1,1,1,1],[1,1,1,1], [1,1,1,1]], dtype=np.float_)
        coefficients2 = np.array([1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        density_matrix_created = tested_file.create_density_matrix(coefficients2, pauli_basis(2))
        self.assertTrue(np.all(np.equal(expected_answer2, density_matrix_created)))


    def test_tomography(self):
        self.state_circuit.x(1)
        expected_answers = Statevector(self.state_circuit)

        complex_result = tested_file.state_tomography(self.state_circuit, AerSimulator(), self.execute_opts)
        absolute_results = np.absolute(complex_result)
        for result, expected_answer in zip(absolute_results, expected_answers) :
            self.assertAlmostEqual(result, expected_answer, delta=0.1)


        self.state_circuit.h(0)
        self.state_circuit.cx(1, 0)
        self.state_circuit.x(1)

        expected_answers = Statevector(self.state_circuit)

        complex_result = tested_file.state_tomography(self.state_circuit, AerSimulator(), self.execute_opts)
        absolute_results = np.absolute(complex_result)
        for result, expected_answer in zip(absolute_results, expected_answers) :
            self.assertAlmostEqual(result, expected_answer, delta=0.1)

        
if __name__ == '__main__':
    unittest.main()
    
        
