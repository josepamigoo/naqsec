
import numpy as np

import qiskit
from qiskit import QuantumCircuit, ClassicalRegister
from qiskit.qasm3 import dumps, loads

from mqt.qecc import codes
from mqt.qecc import CSSCode
from mqt.qecc.circuit_synthesis.simulation import _support
from mqt.qecc.circuit_synthesis import StatePrepCircuit, heuristic_prep_circuit, heuristic_verification_circuit,  depth_optimal_prep_circuit,  gate_optimal_verification_circuit,   gate_optimal_prep_circuit
from mqt.qecc.circuit_synthesis.synthesis_utils import measure_flagged, measure_stab_unflagged

# --------------------------AUXILIARY PARSING FUNCTIONS----------------------------------
def into_na_gates(qasm_text):
    """ Converts a QASM circuit with CNOT gates into a circuit with NA gates (H-CZ-H)."""

    circuit = loads(qasm_text)

    og_qreg = circuit.qregs
    og_creg = circuit.cregs
    circ_na = QuantumCircuit(*og_qreg, *og_creg)

    for instruction, qargs, cargs in  circuit.data:
        if instruction.name == "cx":
             circ_na.h(qargs[1])
             circ_na.cz(qargs[0], qargs[1])
             circ_na.h(qargs[1])
        else:
            circ_na.append(instruction, qargs, cargs)

    return dumps(circ_na)

def qubit_id(ops):
        """Extracts the qubit index from a QASM operation string."""
        aux = ops.split("[")
        aux = aux[1].split("]")
        return int(aux[0])

def op_name(ops):
        """Extracts the operation name from a QASM operation string."""
        aux = ops.split("[")
        return aux[0]

def remove_double_gates(qasm_text):
    """Removes consecutive Hadamard gates on the same qubit in a QASM circuit."""
    remove_list = []
    qasm_splitted = qasm_text.split(";\n")
    q = qubit_id(qasm_splitted[2])

    for jj in range(q):
        ind, last_is_h = 0, False   
        for ii, instruction in enumerate(qasm_splitted):
            ops = instruction.split(" ") 
            if ops[0] == "h" and qubit_id(ops[1]) == jj :
                if last_is_h == True : 
                    remove_list.append(ind) 
                    remove_list.append(ii) 
                    last_is_h = False
                else:
                    ind, last_is_h = ii, True 

            if ops[0] == "barrier" : last_is_h = False             
            if ops[0] == "cz" and (qubit_id(ops[1]) == jj or  qubit_id(ops[2]) == jj) : 
                last_is_h = False

    qasm_filtered = [line for i, line in enumerate(qasm_splitted) if i not in remove_list]
    return ";\n".join(qasm_filtered)

def remove_meas(ft_circ):
    """ Strips measurements from a given QASM circuit to avoid cofnclist when composing with ther circuits"""

    qasm_text = dumps(ft_circ)
    qasm_splitted = qasm_text.split(";\n")

    meas2remove = []
    
    for ii, instruction in enumerate(qasm_splitted):
        ops = instruction.split(" ")
        if len(ops)>= 3:
            if ops[2] == "measure": meas2remove.append(ii)
        if op_name(instruction) == "bit": meas2remove.append(ii)

    for i in sorted(meas2remove, reverse=True):
        qasm_splitted.pop(i)

    return loads(";\n".join(qasm_splitted))

def remove_registries(ft_circ):
    """ Strips registries from a given QASM circuit to avoid cofnclist when composing with ther circuits"""

    qasm_text = dumps(ft_circ)
    qasm_splitted = qasm_text.split(";\n")

    reg2remove = []
    reg_names = []
    
    for ii, instruction in enumerate(qasm_splitted):
        ops = instruction.split(" ")

        if op_name(instruction) == "qubit" and ops[1]!= "q":
            reg_qubits = qubit_id(instruction)
            reg2remove.append(ii)
            reg_names.append((ops[1], int(reg_qubits)))

    for reg_info in reg_names:
        qasm_splitted = kill_single_reg(qasm_splitted, reg_info)

    for i in sorted(reg2remove, reverse=True):
        qasm_splitted.pop(i)

    return loads(";\n".join(qasm_splitted))
       
def kill_single_reg(qasm_splitted, reg_info):
    """ 
    Auxiliary function of remove_registries to remove single qubit registries from a QASM circuit.
    For that the ntire circuit is parsed and the qubit indices are adjusted to account for the removed registry.
    """

    single_gates = ["h", "x", "z", "s", "sdg"]
    double_gates = ["cz", "cx"]
    

    for ii, instruction in enumerate(qasm_splitted):
        ops = instruction.split(" ")
        if op_name(instruction) == "qubit" and ops[1] == "q": 
            main_qubits = qubit_id(instruction)
            qasm_splitted[ii] = "qubit[" + str(main_qubits + reg_info[1]) + "] q"
        if ops[0] in single_gates:
            if op_name(ops[1]) == reg_info[0]:
                rel_num = int(qubit_id(instruction))
                qasm_splitted[ii] = ops[0] + " q[" + str(main_qubits + rel_num) + "]"
        if ops[0] in double_gates:      
            if op_name(ops[1]) == reg_info[0]:
                rel_num = int(qubit_id(ops[1]))
                qasm_splitted[ii] = ops[0] + " q[" + str(main_qubits + rel_num) + "], " + ops[2]
            elif op_name(ops[2]) == reg_info[0]:
                rel_num = int(qubit_id(ops[2]))
                qasm_splitted[ii] = ops[0] + " " + ops[1] + " q[" + str(main_qubits + rel_num) + "]" 

        if ops[0] == "barrier" :
            for kk in range(1, len(ops)):
                if op_name(ops[kk]) == reg_info[0]:
                    rel_num = int(qubit_id(ops[1]))
                    if kk != (len(ops) - 1):
                        ops[kk] = "q[" + str(main_qubits + rel_num) + "],"
                        qasm_splitted[ii] = " ".join(ops)
                    else:
                        ops[kk] = "q[" + str(main_qubits + rel_num) + "]"
                        qasm_splitted[ii] = " ".join(ops)
    return qasm_splitted

def fix_barriers(circuit):
    """"""
    og_qreg = circuit.qregs
    og_creg = circuit.cregs
    new_circ = QuantumCircuit(*og_qreg, *og_creg)

    for instr, qargs, cargs in circuit.data:

        if instr.name == "barrier": new_circ.barrier()
        else: new_circ.append(instr, qargs, cargs)

    return new_circ


#----------------------------- BUILDING THE CIRCUIT -------------------------------------------
def logic_NA_prep(qasm_text, file_name = None ):
     """ Global function that is called to convert a QASM circuit ans synthetise it for NA  platforms
     
     Args:
         qasm_text (str): QASM circuit text to be converted.
         file_name (str, optional): If provided, the resulting circuit will be saved to this
    """
     
     cz_circuit = into_na_gates(qasm_text)
     na_circuit = remove_double_gates(cz_circuit)

     if file_name is not None:
        with open(file_name, "w") as f:
            f.write(na_circuit)

     return na_circuit

def logic_circuit(prep_circ, logic_qiskit, code, ntot):
    """ Takes the logical gates of the logical circuits and implements to the encoded quibits when possible.
     Args:
         prep_circ (QuantumCircuit): State preparation circuit of the logical qubits to which the logical gates will be appended
         logic_qiskit (QuantumCircuit): Logical circuit containing the logical gates to be applied to the encoded qubits.
         code: CSSCode object from QECC containing the logical qubits and their stabilizers.
         ntot: List containing the total number of qubits in the circuit for each logical qubit."""

    logic_qasm = dumps(logic_qiskit)
    
    nq = code.n
    lq = code.k

    x_obs = code.Lx
    z_obs = code.Lz

    last_hadamard = False

    inds_x = [np.array(_support(logical)) for logical in x_obs]
    inds_z = [np.array(_support(logical)) for logical in z_obs]

    splitted_qasm = logic_qasm.split(";\n")

    for instruction in splitted_qasm:
        ops = instruction.split(" ")
        if ops[0] == "x":
            id = qubit_id(ops[1])
            ind = id // lq
            ref = sum(ntot[:ind])
            if last_hadamard :
                prep_circ.x(list(inds_z[id % lq] + ref))
            else: 
                prep_circ.x(list(inds_x[id % lq] + ref))
        if ops[0] == "z":
            id = qubit_id(ops[1])
            ind = id // lq
            ref = sum(ntot[:ind])
            if last_hadamard:
                prep_circ.z(list(inds_x[id % lq] + ref))
            else:
                prep_circ.z(list(inds_z[id % lq] + ref))
        if ops[0] == "h":
            id = qubit_id(ops[1])
            ref = sum(ntot[:id])
            for kk in range(nq): prep_circ.h(kk + ref)
            if not code.is_self_dual():
                AssertionError("Hadamard gates are not supported for non self-dual codes")
        if ops[0] == "s":
            id = qubit_id(ops[1])
            ref = sum(ntot[:id])
            for kk in range(nq): prep_circ.sdg(kk + ref)
            print("Warning: S gates are only correctly supported for some qubits, the result may not be correct. To be addressed")
        if ops[0] == "sdg":
            id = qubit_id(ops[1])
            ref = sum(ntot[:id])
            for kk in range(nq): prep_circ.s(kk + ref)
            print("Warning: Sdg gates are only correctly supported for some qubits, the result may not be correct. To be addressed")
        if ops[0] == "cx":
            if lq> 1: 
                print("Warning: CNOT gates in circuits with k>1 are only crrectly supported between different code circuits. To be addressed ")
            id_1 = qubit_id(ops[1])
            ref_1 = sum(ntot[:id_1])
            id_2 = qubit_id(ops[2])
            ref_2 = sum(ntot[:id_2])
            for kk in range(nq):
               prep_circ.cx(kk + ref_1, kk + ref_2)
            

    swap_stab = False # remanent feature

    return  prep_circ, swap_stab

def logic_CSS_prep_ft(code, states, optimize_depth = True, file_name = None, logic_qiskit = QuantumCircuit(0), fault_tolerant = True, visualization_barriers = False):
    """    Function to build the logical circuit for a given CSS code and a set of states. Ancillas are NOT reused.
           In this case, due to the unknwon number of flag measurements in advanvce, stape preparations are both constructed 
           at the same time and together with the logical circuit the three layers are composed.  
           
           The heuristic algorithm is used for the state preparation circuits and verification.
    Args: 
        code: CSSCode object from QECC.
        states: List of boolean values representing the states to be prepared. True is 0 state, False is the + state.
        optimize_depth (optional): If True, the circuit will be optimized for depth. Defaults to True.
        file_name (optional): If provided, the resulting circuit will be saved to this file.
        logic_qiskit (optional): Logical circuit containing the logical gates to be applied to the encoded qubits.
        fault_tolerant (optional): If True, the circuit will be built in a fault-tolerant manner. Defaults to True.
        visualization_barriers (optional): If True, barriers will be added to the circuit for visualization and debugging purposes. Defaults to False.

    Output:
        qasm_file: QASM  string of the final circuit.
        total_circuit: The final circuit as a QuantumCircuit object.
        circuit_data (tuple): A tuple containing the number of ancillas used for state preparation, syndrome extraction, total qubits, rounds, and whether fault tolerance was used.
    """

    
    z_measurements = list(code.Hz)
    x_measurements = list(code.Hx)

    nq = code.n
    nx = len(x_measurements)
    nz = len(z_measurements)
    lq = code.k
    nv_prep = []
    nv_synd = []
    ntotal = []

    if logic_qiskit.num_qubits != 0:
        assert logic_qiskit.num_qubits == len(states)*lq

    total_circuit = QuantumCircuit(0)
    correction_circuit = QuantumCircuit(0)
    clbit_num = 0
    

    for kk, base in enumerate(states):
        sp_circ = heuristic_prep_circuit(code, zero_state = base, optimize_depth = optimize_depth) 
        tx, tz = sp_circ.max_x_errors, sp_circ.max_z_errors
        tcode = min(tx, tz)

        if fault_tolerant:
            sp_circ = heuristic_verification_circuit(sp_circ) 
            sp_circ = remove_meas(sp_circ) 
            sp_circ = remove_registries(sp_circ)
            rounds = tcode + 1
        else: 
            sp_circ = sp_circ.circ
            rounds = 1

        prep_qubits = sp_circ.num_qubits
        nv_prep.append(sp_circ.num_qubits - nq)
        ntotal.append(nq + nv_prep[kk])
        
        total_circuit = sp_circ.tensor(total_circuit)
        total_circuit = QuantumCircuit((nx + nz)*rounds).tensor(total_circuit)   
        
        synd_circ = QuantumCircuit(prep_qubits + (nx + nz)*rounds)
        stab_m = ClassicalRegister((nx + nz)*len(states)*rounds*max(tx,tz), "s_mes") # max tx and tz
        synd_circ.add_register(stab_m)
        
        ref = nq + nv_prep[kk]
        ancilla = ref

        for jj in range(rounds):
            if visualization_barriers: synd_circ.barrier()

            for i, m in enumerate(z_measurements):
                stab = np.array(np.where(m != 0)[0])
                qubits = list(stab)
                if fault_tolerant:
                    measure_flagged(synd_circ, qubits, ancilla, stab_m[clbit_num], z_measurement = True, t = tz)
                else:
                    measure_stab_unflagged(synd_circ, qubits, ancilla, stab_m[clbit_num], z_measurement = True)
                ancilla += 1
                clbit_num += 1
                if visualization_barriers: synd_circ.barrier()
            
            for i, m in enumerate(x_measurements):
                stab = np.array(np.where(m != 0)[0])
                qubits =  list(stab)
                if fault_tolerant:
                    measure_flagged(synd_circ, qubits, ancilla, stab_m[clbit_num], z_measurement = False, t = tx)
                else:
                    measure_stab_unflagged(synd_circ, qubits, ancilla, stab_m[clbit_num], z_measurement = False)
                ancilla += 1
                clbit_num += 1
                if visualization_barriers: synd_circ.barrier()

            added_qubits = synd_circ.num_qubits - ntotal[kk] 
            ntotal[kk] += added_qubits
            
        if fault_tolerant: 
            nv_synd.append(added_qubits)
            total_circuit = QuantumCircuit(added_qubits*rounds).tensor(total_circuit) 
        else:
            nv_synd.append(0)
        
        synd_circ = remove_meas(synd_circ)
        synd_circ = remove_registries(synd_circ)
        correction_circuit = synd_circ.tensor(correction_circuit)


    total_circuit, swap_stab = logic_circuit(total_circuit, logic_qiskit, code, ntotal)
    total_circuit = total_circuit.compose(correction_circuit)
    total_circuit = fix_barriers(total_circuit)
    qasm_file = dumps(total_circuit)

    if file_name is not None:
        with open(file_name, "w") as f:
            f.write(qasm_file)

    return qasm_file, total_circuit, (nv_prep, nv_synd, ntotal, rounds, False)

def logic_CSS_prep_ft_reuse(code, states, optimize_depth = True, file_name = None, logic_qiskit = QuantumCircuit(0), fault_tolerant = True, visualization_barriers = False):
    
    """    Function to build the logical circuit for a given CSS code and a set of states. Ancillas are reused.
           In this case, due to the unknwon number of flag measurements in advanvce, stape preparations are both constructed 
           at the same time and together with the logical circuit the three layers are composed.  
           The heuristic algorithm is used for the state preparation circuits and verification.

           Args: 
            code: CSSCode object from QECC.
            states: List of boolean values representing the states to be prepared. True is 0 state, False is the + state.
            optimize_depth (optional): If True, the circuit will be optimized for depth. Defaults to True.
            file_name (optional): If provided, the resulting circuit will be saved to this file.
            logic_qiskit (optional): Logical circuit containing the logical gates to be applied to the encoded qubits.
            fault_tolerant (optional): If True, the circuit will be built in a fault-tolerant manner. Defaults to True.
            visualization_barriers (optional): If True, barriers will be added to the circuit for visualization and debugging purposes. Defaults to False.

           Output:
            qasm_file: QASM  string of the final circuit.
            total_circuit: The final circuit as a QuantumCircuit object.
            circuit_data (tuple): A tuple containing the number of ancillas used for state preparation, syndrome extraction, total qubits, rounds, and whether fault tolerance was used.
    """
    
    
    z_measurements = list(code.Hz)
    x_measurements = list(code.Hx)

    nq = code.n
    nx = len(x_measurements)
    nz = len(z_measurements)
    lq = code.k
    nv_prep = []
    nv_synd = []
    ntotal = []

    if logic_qiskit.num_qubits != 0:
        assert logic_qiskit.num_qubits == len(states)*lq

    total_circuit = QuantumCircuit(0)
    correction_circuit = QuantumCircuit(0)
    synd_circ_all = QuantumCircuit(0)
    
    clbit_num = 0
    

    for kk, base in enumerate(states):
        sp_circ = heuristic_prep_circuit(code, zero_state = base, optimize_depth = optimize_depth) 
        tx, tz = sp_circ.max_x_errors, sp_circ.max_z_errors
        tcode = min(tx, tz)

        if fault_tolerant:
            sp_circ = heuristic_verification_circuit(sp_circ) 
            sp_circ = remove_meas(sp_circ) 
            sp_circ = remove_registries(sp_circ)
            rounds = tcode + 1
        else: 
            sp_circ = sp_circ.circ
            rounds = 1

        prep_qubits = sp_circ.num_qubits
        nv_prep.append(sp_circ.num_qubits - nq)
        ntotal.append(nq + nv_prep[kk])
        
        total_circuit = sp_circ.tensor(total_circuit)  
        
        synd_circ = QuantumCircuit(prep_qubits + (nx + nz))
        reg_name = "s_mes_" + str(kk)
        stab_m = ClassicalRegister((nx + nz)*len(states)*rounds*max(tx,tz), reg_name) # max tx and tz
        synd_circ.add_register(stab_m)
        
        ref = nq + nv_prep[kk]
 
        ancilla = ref
        for i, m in enumerate(z_measurements):
            stab = np.array(np.where(m != 0)[0])
            qubits = list(stab)
            if fault_tolerant:
                measure_flagged(synd_circ, qubits, ancilla, stab_m[clbit_num], z_measurement = True, t = tz)
            else:    
                measure_stab_unflagged(synd_circ, qubits, ancilla, stab_m[clbit_num], z_measurement = True)
            ancilla += 1
            clbit_num += 1
            if visualization_barriers: synd_circ.barrier()
            
        for i, m in enumerate(x_measurements):
            stab = np.array(np.where(m != 0)[0])
            qubits =  list(stab)
            if fault_tolerant:
                measure_flagged(synd_circ, qubits, ancilla, stab_m[clbit_num], z_measurement = False, t = tx)
            else:
                measure_stab_unflagged(synd_circ, qubits, ancilla, stab_m[clbit_num], z_measurement = False)
            ancilla += 1
            clbit_num += 1
            if visualization_barriers: synd_circ.barrier()

        added_qubits = synd_circ.num_qubits - ntotal[kk] 
        ntotal[kk] += added_qubits
        nv_synd.append(added_qubits - (nx + nz))

        synd_circ = remove_meas(synd_circ)
        synd_circ = remove_registries(synd_circ)
        
        synd_circ_all = synd_circ.tensor(synd_circ_all)
        total_circuit = QuantumCircuit(added_qubits).tensor(total_circuit) 
    
    synd_circ_all.barrier()
    
    for _ in range(rounds):
            correction_circuit = synd_circ_all.compose(correction_circuit)

    if visualization_barriers: total_circuit.barrier()

    print("total circuit qubits: ", total_circuit.num_qubits, correction_circuit.num_qubits)
    total_circuit, swap_stab = logic_circuit(total_circuit, logic_qiskit, code, ntotal)
    total_circuit = total_circuit.compose(correction_circuit)
    total_circuit = fix_barriers(total_circuit)
    qasm_file = dumps(total_circuit)

    if file_name is not None:
        with open(file_name, "w") as f:
            f.write(qasm_file)

    return qasm_file, total_circuit, (nv_prep, nv_synd, ntotal, rounds, fault_tolerant)

def logic_CSS_prep_ft_optimal(code, states, optimize_depth = True, file_name = None, logic_qiskit = QuantumCircuit(0), fault_tolerant = True, visualization_barriers = False, max_time = 3600*24):
    """    Function to build the logical circuit for a given CSS code and a set of states using the optimal algorithms. Ancillas are reused.
           In this case, due to the unknwon number of flag measurements in advanvce, stape preparations are both constructed 
           at the same time and together with the logical circuit the three layers are composed.  

           Args: 
            code: CSSCode object from QECC.
            states: List of boolean values representing the states to be prepared. True is 0 state, False is the + state.
            optimize_depth (optional): If True, the circuit will be optimized for depth. Defaults to True.
            file_name (optional): If provided, the resulting circuit will be saved to this file.
            logic_qiskit (optional): Logical circuit containing the logical gates to be applied to the encoded qubits.
            fault_tolerant (optional): If True, the circuit will be built in a fault-tolerant manner. Defaults to True.
            visualization_barriers (optional): If True, barriers will be added to the circuit for visualization and debugging purposes. Defaults to False.
            max_time (optional) : Maximum time in seconds for the optimal algorithms to run. Defaults to 24 hours.

           Output:
            qasm_file: QASM  string of the final circuit.
            total_circuit: The final circuit as a QuantumCircuit object.
            circuit_data (tuple): A tuple containing the number of ancillas used for state preparation, syndrome extraction, total qubits, rounds, and whether fault tolerance was used.
    """
    
    z_measurements = list(code.Hz) #out of loop
    x_measurements = list(code.Hx)

    nq = code.n
    nx = len(x_measurements)
    nz = len(z_measurements)
    lq = code.k
    nv_prep = []
    nv_synd = []
    ntotal = []

    if logic_qiskit.num_qubits != 0:
        assert logic_qiskit.num_qubits == len(states)*lq

    total_circuit = QuantumCircuit(0)
    correction_circuit = QuantumCircuit(0)
    synd_circ_all = QuantumCircuit(0)
    
    clbit_num = 0
    

    for kk, base in enumerate(states):
        sp_circ = gate_optimal_prep_circuit(code, max_timeout= max_time) 
        tx, tz = sp_circ.max_x_errors, sp_circ.max_z_errors
        tcode = min(tx, tz)

        if fault_tolerant:
            sp_circ = gate_optimal_verification_circuit(sp_circ, max_timeout= max_time) 
            sp_circ = remove_meas(sp_circ) 
            sp_circ = remove_registries(sp_circ)
            rounds = tcode + 1
        else: 
            sp_circ = sp_circ.circ
            rounds = 1

        prep_qubits = sp_circ.num_qubits
        nv_prep.append(sp_circ.num_qubits - nq)
        ntotal.append(nq + nv_prep[kk])
        
        total_circuit = sp_circ.tensor(total_circuit)
        
        synd_circ = QuantumCircuit(prep_qubits + (nx + nz))
        reg_name = "s_mes_" + str(kk)
        stab_m = ClassicalRegister((nx + nz)*len(states)*rounds*max(tx,tz), reg_name) # max tx and tz
        synd_circ.add_register(stab_m)
        
        ref = nq + nv_prep[kk]
 
        ancilla = ref
        for i, m in enumerate(z_measurements):
            stab = np.array(np.where(m != 0)[0])
            qubits = list(stab)
            if fault_tolerant:
                measure_flagged(synd_circ, qubits, ancilla, stab_m[clbit_num], z_measurement = True, t = tz)
            else:    
                measure_stab_unflagged(synd_circ, qubits, ancilla, stab_m[clbit_num], z_measurement = True)
            ancilla += 1
            clbit_num += 1
            if visualization_barriers: synd_circ.barrier()
            
        for i, m in enumerate(x_measurements):
            stab = np.array(np.where(m != 0)[0])
            qubits =  list(stab)
            if fault_tolerant:
                measure_flagged(synd_circ, qubits, ancilla, stab_m[clbit_num], z_measurement = False, t = tx)
            else:
                measure_stab_unflagged(synd_circ, qubits, ancilla, stab_m[clbit_num], z_measurement = False)
            ancilla += 1
            clbit_num += 1
            if visualization_barriers: synd_circ.barrier()

        added_qubits = synd_circ.num_qubits - ntotal[kk] 
        ntotal[kk] += added_qubits
        nv_synd.append(added_qubits - (nx + nz))

        synd_circ = remove_meas(synd_circ)
        synd_circ = remove_registries(synd_circ)
        
        synd_circ_all = synd_circ.tensor(synd_circ_all)

        if fault_tolerant: 
            total_circuit = QuantumCircuit(added_qubits).tensor(total_circuit) 
    
    synd_circ_all.barrier()
    
    for _ in range(rounds):
            correction_circuit = synd_circ_all.compose(correction_circuit)

    if visualization_barriers: total_circuit.barrier()

    print("total circuit qubits: ", total_circuit.num_qubits, correction_circuit.num_qubits)
    total_circuit, swap_stab = logic_circuit(total_circuit, logic_qiskit, code, ntotal)
    total_circuit = total_circuit.compose(correction_circuit)
    total_circuit = fix_barriers(total_circuit)
    qasm_file = dumps(total_circuit)

    if file_name is not None:
        with open(file_name, "w") as f:
            f.write(qasm_file)

    return qasm_file, total_circuit, (nv_prep, nv_synd, ntotal, rounds, fault_tolerant)
    
#----------------------- BUILDING THE ERROR DICTIONARY --------------------------
import importlib.util
import sys
import os

def add_error_functions(model_name, error_dict):

    module_path = 'error_models/{}/error_utils.py'.format(model_name)
    module_name = os.path.splitext(os.path.basename(module_path))[0]  # Extract module name from file name

    # Load the module from the file path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    operation_error = error_dict["operations"]
    for error in operation_error.keys():
        for ii in range(len(error_dict["operations"][error])):
            fun_name ="{}_{}".format(error, str(ii))
            fun = getattr(module, fun_name)
            error_dict["operations"][error][ii].append(fun) # chech for multiple errors

    return error_dict