import unittest
from diagonalization import diagonalize_pauli_with_circuit
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info import Pauli, Operator, Statevector

class Test_diagonalization(unittest.TestCase):
    def setUp(self):
        self.pauli1 = Pauli('YII')
        self.diag_pauli1 = Pauli('ZII')
        self.pauli2 = Pauli('XXY')
        self.diag_pauli2 = Pauli('ZZZ')

        self.qreg = QuantumRegister(3)
        self.diag_circuit1 = QuantumCircuit(self.qreg)
        self.diag_circuit1.sdg(self.qreg[2])
        self.diag_circuit1.h(self.qreg[2])
        self.circuit1_state_vector = Statevector.from_instruction(self.diag_circuit1)

        self.diag_circuit2 = QuantumCircuit(self.qreg)
        self.diag_circuit2.sdg(self.qreg[0])
        self.diag_circuit2.h(self.qreg)
        self.circuit2_state_vector = Statevector.from_instruction(self.diag_circuit2)
        

    def test_diagonalize_pauli_with_circuit(self):
       tested_result = diagonalize_pauli_with_circuit(self.pauli1)
       self.assertEqual(tested_result[0], self.diag_pauli1)

       state_vector_result = Statevector.from_instruction(tested_result[1])
       self.assertTrue(state_vector_result.equiv(self.circuit1_state_vector))
       
       tested_result = diagonalize_pauli_with_circuit(self.pauli2)
       self.assertEqual(tested_result[0], self.diag_pauli2)

       state_vector_result = Statevector.from_instruction(tested_result[1])
       self.assertTrue(state_vector_result.equiv(self.circuit2_state_vector))



        
if __name__ == '__main__':
    unittest.main()
    
        