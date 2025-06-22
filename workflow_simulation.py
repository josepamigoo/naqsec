import stim
from simulation_utils import logic_NA_prep, logic_CSS_prep_ft, add_error_functions
from NA_decoder import NAdecoder
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
import os
import yaml
from scipy.io import savemat
from mqt.qecc import CSSCode
from mqt.qecc.codes import HexagonalColorCode, SquareOctagonColorCode
import time
import json

# This code provides an example of execution of the tool. Parameteres may be chnaged.
start = time.time()
#------------------------  inputs  ------------------------------
 # desired code and logic circuits
 # NORMAL
CSS_name = "Shor" 
code = CSSCode.from_code_name(CSS_name)

# IF COLOR COES WANT TO BE USED
#code = HexagonalColorCode(5)
#code = SquareOctagonColorCode(5)
states = [True] # True/ False -> Z/X basis
optimize_depth = True
circuit_name_info = "shor_RE_0_full_ft"

# #------------------------ logic circuit-------------------------

# Import data from the folder where they are stored

name_data = "circuit_data/{}_data.txt".format(circuit_name_info)
with open(name_data, "r") as f:
    circuit_data = eval(f.read()) 
print(circuit_data)
#----------------------- AZAC routing --------------------------

name = 'circuit_data/{}.naviz'.format(circuit_name_info) #CHANGE PATH IF NECESSARY
with open( name , 'r') as file:
    compiled_circ = file.read()

#--------------------- simulation & decoding ------------------
########## PARAMS ##########

dist_max = 2 # usually the standard paramter, can be double checked in architecture file of router
cz_zone = [(30, 302),(270, 372)] # read from architecture file of router

noise_levels = np.logspace(-4, -2, 6) # desired noise values 


######### ERROR MODEL ##########
error_model_name = "standard_depolarizing" 
error_dict_name = 'error_models/{}/error_channels.yaml'.format(error_model_name) #import desired error model from the folder error_models in the same parent folder
with open(error_dict_name, "r") as f:
    error_model = yaml.safe_load(f)

error_model = add_error_functions(error_model_name, error_model) #join dictionary and auxiliary functions
print(error_model)

#### SIMULATION ####

decoder_na = NAdecoder(code, states, compiled_circ, circuit_data, cz_zone, dist_max)

#ideal_qubits = decoder_na.get_ancillas() #cin case ancillas would like to be shielded

logic_errors, acceptance_rate = decoder_na.benchmark(noise_levels, states, error_model, ideal_qubits = [], nb_shots = 1_000_000, at_least_min_errors= True, min_errors = 100, joint_decoding = False, parallel_execution = False, num_cores = 4)

#save data in numpy array and .mat for matlab
name_simulation_py = "simulation_results/simulation_data_{}.npz".format(circuit_name_info)
name_simulation_mat = "simulation_results/simulation_data_{}.mat".format(circuit_name_info)
np.savez(name_simulation_py, noise_levels = noise_levels, logic_errors = logic_errors, acceptance_rate = acceptance_rate)
savemat(name_simulation_mat, {"noise_levels": noise_levels,"logic_errors": logic_errors, "acceptance_rate": acceptance_rate})
print(logic_errors, acceptance_rate)

# keep track of execution time
end = time.time()
elapsed = (end - start) / 60
print(f"Execution time: {elapsed:.4f} minutes")
