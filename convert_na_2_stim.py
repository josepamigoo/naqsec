import numpy as np
import itertools
import stim

class Convert2STIM:
    def __init__(self, NA_format, cz_zone, dmax, error_model = dict([]), ideal_qubits = [], measure_data = ([],[])):
        self._NA_format = NA_format
        self._cz_zone = cz_zone
        self._dmax = dmax
        self._error_model = error_model
        self._ideal_qubits = ideal_qubits
        self._measure_data = measure_data

        self._is_zeros = []
        self._implemented_gate = []
        self._STIM_circuit = ""

    def convert_native(self, circuit = False):
        """
        Converts .naviz into a set of instruction to create a STIM circuit. 
        ATTENTION !!: This functions and its dependencis are deprecated and no mantainace is assured.
        """

        barriers_ind = self._get_barriers(self._NA_format)
        filtered_text = self._filter_comments(self._NA_format)

        steps = filtered_text.split('@')
        N = len(steps) 

        for ii in range(N):

            instruction, operation, _ = self._get_param(steps[ii])

            if instruction == 'atom':
                id, position = self._list_atoms(operation)
                is_loaded = [ False for ii in range(len(id))]
                self._STIM_circuit += "R " + " ".join(id) + "\n"
                self._get_error(instruction, id)
                atom_id = id

        
            elif instruction == 'load':
                id, load_position = self._list_atoms(operation)
                for kk in id: is_loaded[int(kk)] = True
                self._get_error(instruction, id)
        
            elif instruction == 'store':
                id, store_position = self._list_atoms(operation)
                for kk in id: is_loaded[int(kk)] = False
                self._get_error(instruction, id)

            elif instruction == 'move':
                id_move, move_position = self._list_atoms(operation)
                shuttle_dist = np.sum(np.abs(move_position - position[list(map(int, id_move))]))
                for jj, kk in enumerate(id): position[0:][int(kk)] = move_position[0:][jj]
                self._get_error(instruction, id)

            elif instruction == 'rz':  # is assumed to be only  local
                mygate, id, is_gate = self._get_clifford_gate(instruction, operation, steps, ii, atom_id)
                if is_gate == True: self._STIM_circuit += mygate + " " + " ".join(id) + "\n"
                self._get_error(instruction, id)

            elif instruction == 'ry':  #is assumed to be only global 
                mygate, id, is_gate = self._get_clifford_gate(instruction, operation, steps, ii, atom_id)
                if is_gate == True: self._STIM_circuit += mygate + " " + " ".join(id) + "\n"
                self._get_error(instruction, atom_id)
         
            elif instruction == 'cz':
                id_zone = self._inside_CZ_zone(position) 
                pairs = list(itertools.combinations(id_zone, 2))
                id = []
                for kk in pairs:
                    dist = np.linalg.norm(position[0:][int(kk[0])]-position[0:][int(kk[1])])
                    if dist <= self._dmax: id += kk
                self._STIM_circuit += "CZ " + " ".join(id) + "\n"
                self._get_error(instruction, id)
                # ad error to those in CZ but no pair?

            idling_id = [item for item in atom_id if item not in id]       
            self._get_error('idle', idling_id)
 
        if circuit: return stim.Circuit(self._STIM_circuit)
        else: return self._STIM_circuit    

    def convert_clifford(self, circuit = False):
        """
        Converts .naviz into a set of instruction to create a STIM circuit.
        The .naviz file is manually parsed.
        """
        barriers_ind = self._get_barriers(self._NA_format) # detects if a barrier is present, indicating ancilla reuse
        filtered_text = self._filter_comments(self._NA_format) # filter coments
        

        steps = filtered_text.split('@') 
        N = len(steps) 

        hgate = "u 1.57080 0.00000 3.14159"
        xgate = "u 3.14159 0.00000 3.14159"
        zgate = "u 0.00000 0.00000 3.14159"

        # Iterates over all the instructions. Each instruction is read and each data extracted in order to 
        # create the STIM string and assign the error, if that is the case.

        for ii in range(N):

            instruction, operation, _ = self._get_param(steps[ii]) # extract data from each instruction, time not used so far
            
            if instruction == 'atom': # qubit initialization
                id, position = self._list_atoms(operation)
                is_loaded = [ False for ii in range(len(id))]
                self._STIM_circuit += "R " + " ".join(id) + "\n"
                self._get_error(instruction, id)
                atom_id = id
        
            elif instruction == 'load':
                id, load_position = self._list_atoms(operation)
                for kk in id: is_loaded[int(kk)] = True
                self._get_error(instruction, id, len(atom_id))
                if instruction not in self._error_model["operations"].keys(): id = atom_id  # no idling durin this operation if no error is associated
        
            elif instruction == 'store':
                id, store_position = self._list_atoms(operation)
                for kk in id: is_loaded[int(kk)] = False
                self._get_error(instruction, id, len(atom_id))
                if instruction not in self._error_model["operations"].keys(): id = atom_id # no idling durin this operation if no error is associated 

            elif instruction == 'move':
                id, move_position = self._list_atoms(operation)
                shuttle_dist = np.sum(np.abs(move_position - position[list(map(int, id))]), axis = 1) 
                for jj, kk in enumerate(id): position[0:][int(kk)] = move_position[0:][jj]
                self._get_error(instruction, id,  len(atom_id), shuttle_dist)
                if instruction not in self._error_model["operations"].keys(): id = atom_id # no idling durin this operation if no error is associated 

            elif instruction == ('h' or hgate ): 
                id, hadamard_position = self._list_atoms(operation)
                self._STIM_circuit += "H" + " " + " ".join(id) + "\n"
                self._get_error(instruction, id)      

            elif instruction == ('x' or xgate): 
                id, X_position = self._list_atoms(operation)
                self._STIM_circuit += "X" + " " + " ".join(id) + "\n"
                self._get_error(instruction, id)
            
            elif instruction == ('z' or zgate): 
                id, Z_position = self._list_atoms(operation)
                self._STIM_circuit += "z" + " " + " ".join(id) + "\n"
                self._get_error(instruction, id)
         
            elif instruction == 'cz':
                id_zone = self._inside_CZ_zone(position)  # detect if atoms are in entangling zone
                pairs = list(itertools.combinations(id_zone, 2))
                id = []
                for kk in pairs:
                    dist = np.linalg.norm(position[0:][int(kk[0])]-position[0:][int(kk[1])]) # apply cz gate to those atoms tha are close enough
                    if dist <= self._dmax:  id += kk
                self._STIM_circuit += "CZ " + " ".join(id) + "\n"
                self._get_error(instruction, id)
        
            idling_id = [item for item in atom_id if item not in id]  
            self._get_error('idle', idling_id, instruction) # idling error

            if ii in barriers_ind: # assign measurements when stabilizer round is finnished (only when reusing ancillas)
                qubits = [str(num) for list in self._measure_data for num in list]
                if barriers_ind.index(ii) == len(barriers_ind)-1: 
                    mes_operation = "MZ "
                else: 
                    mes_operation = "MRZ "
                self._get_error("measure", qubits) 
                self._STIM_circuit += mes_operation + " ".join(qubits) + "\n"

                idling_id = [item for item in atom_id if item not in qubits]  
                self._get_error("idle", idling_id, "measure") 

        if circuit: return stim.Circuit(self._STIM_circuit)
        else: return self._STIM_circuit  

    def _filter_comments(self, text):
        """
        Remove pre-defined comments from .navi file
        Comments are defined  as everything tha follows // in the same line or text in between /* and */
        """
        is_comment = False
        ii = 0
        while ii < len(text):
            if text[ii:ii+2] == "//":
                is_comment = True
                ind = ii
            if text[ii] == "\n" and is_comment is True:
                text = text[:ind] + text[ii:]
                is_comment = False
            ii +=1

        is_group = False
        ii = 0
        while ii < len(text):
            if text[ii:ii+2] == "/*":
                is_group = True
                ind2 = ii
            if text[ii: ii+2] == "*/" and is_group is True:
                text = text[:ind2] + text[ii+2:]
                is_group = False
            ii += 1
        return text

    def _get_param(self, step):
        """
        Extracts all the instructions and their corresponding times and operations involved
        """
        if step[0:4] == 'atom':
            id, position = self._list_atoms(step)
            return "atom", step, 0 
        else:
            param = step.split(None, 2)
            time, instruction, operation = param[0], param[1:-1][0], param[-1]
            return instruction, operation, time
        
    def _list_atoms(self, operation):
        """
        Detect the target atoms of a given operation and return its qubit-id and new-position
        """
        id = []
        atom_data, length, nums = self._get_data(operation)

        position = np.zeros((length , 2))
        for ii in nums: 
            id.append(atom_data[ii].split("m")[-1])    
            position[ii-1][:] = self._get_pos(atom_data[ii])[:, 0] # np array --
        return id, position
    
    def _get_data(self, operation):
        """
        Auxiliar function that parses the operations and returns an output to be processed
        """
        atom_data = operation.split('\n')
        while atom_data[-1] == "" or atom_data[-1] == " ": atom_data.pop(-1)
        n =  len(atom_data)
    
        for ii in range(n):
          while atom_data[ii][-1] == " " : atom_data[ii] = atom_data[ii][:-1]
        if atom_data[0][-1] == "[" : length, nums = n - 2, range(1, n-1)
        else : length, nums =  n, range(n)
        return atom_data, length, nums    

    def _get_pos(self, atom_data):
        """
        Extracts positions from the qubits
        """
        aux = atom_data.split('(')
        if len(aux) == 1 : return np.array([[0], [0]])
        aux = aux[1].split(')')
        pos_1, pos_2 = aux[0].split(',')
        return np.array([[float(pos_1)], [float(pos_2)]])
    
    def _get_error(self, instruction, target, *args):
        """
        Reads the assigned errors from dictionary and adds the to STIM circuits when called.
        Given the distinction between gates and operations, two different procedures are used.
        """
        target[:] = [item for item in target if int(item) not in  self._ideal_qubits]

        gate_errors = self._error_model["gates"]
        if instruction in gate_errors.keys(): 
             for ii in range(len(gate_errors[instruction])): 
                probs = [str(p) for p in gate_errors[instruction][ii][1]]
                self._STIM_circuit += gate_errors[instruction][ii][0] + "(" + ", ".join(probs) + ") " + " ".join(target) + "\n"

        op_errors = self._error_model["operations"]

        if instruction in op_errors.keys(): 
            for ii in range(len(op_errors[instruction])): 
                p_val = self._error_model["p_value"]
                args_model = op_errors[instruction][ii][1]
                probs, inds_probs = op_errors[instruction][ii][2](p_val, target, args_model, *args, self._error_model["gate_duration"]) #get error porbabilities for each qubit and its index
                
                for gate_prob, target_ind in zip(np.atleast_2d(probs),np.atleast_2d(inds_probs)):
                    p_err = [str(p) for p in gate_prob.tolist()]
                    new_target = [ target[it] for it in target_ind]
                    self._STIM_circuit += op_errors[instruction][ii][0] + "(" + ", ".join(p_err) + ") " + " ".join(new_target) + "\n"
                

    def _get_clifford_gate(self, instruction, operation, steps, ii, id):
        """
        Reads and stores a sequences of rotations to later assign the associated Clifford Gate 
        DEPRECATED
        """
        rotation_gates = ['rx', 'ry', 'rz']
        target, phase = self._get_phase(operation)
        for jj in range(len(phase)): self._implemented_gate.append((instruction, phase[jj], target[jj]))

        if ii < len(steps)-1: 
            next_instruction, _ , _ = self._get_param(steps[ii+1]) 
            if next_instruction in rotation_gates: 
                return None, [target[-1][-1]], False

        my_gate, target = self._which_gate(id)
        self._implemented_gate = [] 
        self._is_zeros = False 
        return my_gate, target, True
    
    def _get_phase(self, operation):
        """
        Extracts corresponding phase from a list of rotation gates
        """
        id, phase = [], []
        atom_data, _ , nums = self._get_data(operation)

        for ii in nums: 
            aux = atom_data[ii].split(" ")
            while aux[-2] == "" : aux.pop(-2)
            id.append(aux[-1])    
            phase.append(aux[-2])

        return id, phase

    def _which_gate(self, id):
        """
        Given a set of rotations, it identifies its corresponding Clifford Gates
        DEPRECATED
        """
        target = []
        if self._is_Hadamard_1() == True:
            for ii in range(1, len(self._implemented_gate)-1): target.append(self._implemented_gate[ii][2][-1])
            return "H", target
        elif self._is_Hadamard_2() == True:
            # HADAMARD 2 is currently only handed globally
            return "H", id
        elif self._is_Z() == True:
            for ii in range(len(self._implemented_gate)): target.append(self._implemented_gate[ii][2][-1])
            return "Z", target
        elif self._is_X() == True:
            for ii in range(1, len(self._implemented_gate)-1): target.append(self._implemented_gate[ii][2][-1])
            return "X", target 
        elif self._is_iden() == True:
            for ii in range(len(self._implemented_gate)): target.append(self._implemented_gate[ii][2][-1])
            return "I", target
        else:
            raise ValueError("Gate not defined")
        
    def _is_Hadamard_1(self):

        if self._implemented_gate[0] != ('ry', '-0.785398', 'global_ry'): return False
        elif self._implemented_gate[-1] != ('ry', '0.785398', 'global_ry'): return False
        for ii in range(1, len(self._implemented_gate)-1):
            if self._implemented_gate[ii][0:2] != ('rz','3.14159') : return False
            if self._implemented_gate[ii][2][:-1] != 'atom': return False

        return True

    def _is_Hadamard_2(self):

        if self._is_zeros is False: return False
        if self._implemented_gate[0] != ('ry', '1.5708', 'global_ry'): return False
        return True

    def _is_Z(self): #can apply multipke Z gates at the same time (testing pending!!!)

        for ii in range(len(self._implemented_gate)):
            if self._implemented_gate[ii][0:2] != ('rz','3.14159') : return False
            if self._implemented_gate[ii][2][:-1] != 'atom': return False

        return True

    def _is_X(self):

        if self._implemented_gate[0] != ('ry', '-1.5708', 'global_ry'): return False
        elif self._implemented_gate[-1] != ('ry', '1.5708', 'global_ry'): return False
        for ii in range(1, len(self._implemented_gate)-1):
            if self._implemented_gate[ii][0:2] != ('rz','3.14159') : return False
            if self._implemented_gate[ii][2][:-1] != 'atom': return False

        return True

    def _is_iden(self):
        for ii in range(len(self._implemented_gate)):
            if self._implemented_gate[ii][0:2] != ('rz','0.00000') : return False
            if self._implemented_gate[ii][2][:-1] != 'atom': return False

        return True
    
    def _inside_CZ_zone(self, position):
        """
        Detects if atoms are in entangling zone and within the interaction radius. If so, it assigns a CZ gate
        """
        czz_list = []
        # Detect entangling zone???
        for ii in range(len(position[:])) :
            pos = position[ii]
            if all([
                    self._cz_zone[0][0] <= pos[0] <= self._cz_zone[1][0],  # X in range
                    self._cz_zone[0][1] <= pos[1] <= self._cz_zone[1][1]   # Y in range
                    ]):
                czz_list.append(str(ii))

        return czz_list
    
    def _get_barriers(self, text):
        """
        Detects the barriers in the circuit and returns its index after being filtered
        """
        barriers_ind = []
        lines_text = text.splitlines()
        for ii, line in enumerate(lines_text):
            if line.startswith("// barrier"):
                
                op_ind = "\n".join(lines_text[:ii]).count("@")
                barriers_ind.append(op_ind)


        return barriers_ind

