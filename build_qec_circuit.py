import stim
from simulation_utils import logic_NA_prep, logic_circuit, logic_CSS_prep_ft_reuse
from convert_na_2_stim import Convert2STIM

import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.qasm3 import dumps, loads
from mqt.qecc import CSSCode
from mqt.qecc.codes import HexagonalColorCode, SquareOctagonColorCode

# This code is aimed at constructing the error correction cirucits. Theere are many parameters that can be modified 
#------------------------  inputs  ------------------------------
# Desired code and logical circuits

CSS_name = "Shor" #can be changed to any CSS name of MQT-QECC
code = CSSCode.from_code_name(CSS_name)
states = [True,True] #logical qubit initialization. True is the 0 logical, False is the + logical.
optimize_depth = True

#--- if a colour code is dedired here are the instructions ---  
#code = HexagonalColorCode(3)
#code = SquareOctagonColorCode(3)

#--------------------------- my circuit ---------------------------

my_circuit = QuantumCircuit(2)
my_circuit.x(0)
my_circuit.cx(0, 1)

#----------------------- .qasm circuit--------------------------
name = "circuit_data/squareoctagon3_RE_11_full_ft.qasm" # desired name of the circuit
name_data = "circuit_data/squareoctagon3_RE_11_full_ft_data.txt" #desired name of the data


prep_circ_qasm, prep_circ_obj, nv = logic_CSS_prep_ft_reuse(code, states, file_name = "prep_circuit_og.qasm", fault_tolerant = True, logic_qiskit = my_circuit)
na_circ = logic_NA_prep(prep_circ_qasm, file_name = name)

#w e save the circuit data
with open(name_data, "w") as f:
    f.write(str(nv))

