"""
circuitutil.py - A module for extending Qiskit circuit functionality.
"""

import numpy as np
import pickle
from qiskit import Aer, BasicAer, QuantumCircuit, QuantumRegister, execute, assemble
from qiskit.extensions import *
from qiskit.compiler import transpile
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import BasicSwap, CXCancellation
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.aqua.algorithms import NumPyMinimumEigensolver, VQE
from qiskit.aqua import QuantumInstance
from qiskit.chemistry.algorithms.ground_state_solvers.minimum_eigensolver_factories import VQEUCCSDFactory
from qiskit.chemistry.algorithms.ground_state_solvers import GroundStateEigensolver
from qiskit.chemistry.transformations import FermionicTransformation, FermionicQubitMappingType
from qiskit.aqua.components.optimizers import COBYLA, SPSA, SLSQP

### CONSTANTS ###

# NOTICE: GATE_TO_PULSE_TIME is kept here for dependency reasons, but all
# future references to this dict and any other experimental constants
# should be kept in fqc/data/data.py.
# See Gate_Times.ipynb or Realistic_Pulses.ipynb for determination of these pulse times
GATE_TO_PULSE_TIME = {'h': 1.4, 'cx': 3.8, 'rz': 0.4, 'rx': 2.5, 'x': 2.5, 'swap': 7.4, 'id': 0.0}
GATE_TO_PULSE_TIME_REALISTIC = {'h': 20, 'cx': 45, 'rz': 1, 'rx': 31, 'x': 31, 'swap': 59, 'id': 0.0}
NUM_SHOTS = 1000

unitary_backend = BasicAer.get_backend('unitary_simulator')
state_backend = BasicAer.get_backend('statevector_simulator')

### FUNCTIONS ###

def get_unitary(circuit):
    """Given a qiskit circuit, produce a unitary matrix to represent it.
    Args:
    circuit :: qiskit.QuantumCircuit - an arbitrary quantum circuit

    Returns:
    matrix :: np.matrix - the unitary representing the circuit
    """
    job = execute(circuit, unitary_backend)
    unitary = job.result().get_unitary(circuit, decimals=10)
    return np.matrix(unitary)

def get_nearest_neighbor_coupling_list(width, height, directed=True):
    """Returns a coupling list for nearest neighbor (rectilinear grid) architecture.

    Qubits are numbered in row-major order with 0 at the top left and
    (width*height - 1) at the bottom right.

    If directed is True, the coupling list includes both  [a, b] and [b, a] for each edge.
    """
    coupling_list = []

    def _qubit_number(row, col):
        return row * width + col

    # horizontal edges
    for row in range(height):
        for col in range(width - 1):
            coupling_list.append((_qubit_number(row, col), _qubit_number(row, col + 1)))
            if directed:
                coupling_list.append((_qubit_number(row, col + 1), _qubit_number(row, col)))

    # vertical edges
    for col in range(width):
        for row in range(height - 1):
            coupling_list.append((_qubit_number(row, col), _qubit_number(row + 1, col)))
            if directed:
                coupling_list.append((_qubit_number(row + 1, col), _qubit_number(row, col)))

    return coupling_list


def _tests():
    """A function to run tests on the module"""
    pass

if __name__ == "__main__":
    _tests()

