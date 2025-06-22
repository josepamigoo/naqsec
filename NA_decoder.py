import numpy as np
from mqt.qecc import CSSCode
import stim
import matplotlib.pyplot as plt
from mqt.qecc.circuit_synthesis.simulation import LutDecoder, _support
from convert_na_2_stim import Convert2STIM
import os, time

from multiprocessing import Pool


class NAdecoder:

    def __init__(self, code, states, naviz_circuit, circuit_data,  cz_zone, dist_max):
        self._states = states
        self._naviz_circuit = naviz_circuit
        self._cz_zone = cz_zone
        self._dist_max = dist_max

        self._stim_circ = stim.Circuit()
        self._x_measurements, self._z_measurements = [],[]
        self._x_ideal_measurements, self._z_ideal_measurements = [],[]
        self._verification_measurements = []
        self._data_measurements = []
        self._n_measurements = 0
        self._joint_decoding = False

        self._code = code
        self._decoder = LutDecoder(self._code)
        self._nq = self._code.n
        self._nx = len(self._code.Hx)
        self._nz = len(self._code.Hz)
        self._lq = self._code.k
        self._nv_prep,  self._nv_synd, self._ntotal, self._num_rounds, self._reusing = circuit_data

        self._error_model = dict()
        self._ideal_qubits = []


    def benchmark(self, noise_levels, bases, error_model, ideal_qubits = [], nb_shots = 1_000_000, plot_results = True, joint_decoding = False, min_errors = 50, at_least_min_errors = True, parallel_execution = False, num_cores = 4): # correct nb_shots
        """
        Simulates the perfomrance of the logical encoding for several noise levels

        Args:
            noise_levels: list of floats with the noise levels to simulate
            bases: list of bools with the basis to measure each logical qubit
            error_model: dictionary with the error model to use
            ideal_qubits: list of ideal qubits to not apply errors
            nb_shots: number of minimum shots to simulate in abscense of minimum errors
            plot_results: whether to plot the results or not
            joint_decoding: whether to decode all logical qubits together or not
            min_errors: minimum number of errors to sample before stopping
            at_least_min_errors: whether to continue simulating until at least min_errors are found
            parallel_execution: whether to use parallel execution or not
            num_cores: number of cores to use for parallel execution
        """

        self._joint_decoding = joint_decoding
        self._ideal_qubits = ideal_qubits
        self._error_model = error_model

        if self._joint_decoding : self._dim_1 = 1
        else : self._dim_1 = len(self._states)*self._lq

        logic_errors = np.zeros([ self._dim_1, len(noise_levels)])
        accepted_sim = np.zeros([ self._dim_1, len(noise_levels)])

        for jj, p in enumerate(noise_levels):
            
            self._update_error(p)

            if self._num_rounds != 1: measure_data = self.get_ancillas(), self.get_synd_flags()
            else: measure_data = [], []

            Converter = Convert2STIM(self._naviz_circuit, self._cz_zone, self._dist_max, error_model=self._error_model, ideal_qubits = self._ideal_qubits, measure_data = measure_data)
            self._stim_circ = Converter.convert_clifford(circuit = True)

            if self._joint_decoding : decoder_batch = self.decode_batch_joint
            else:  decoder_batch = self.decode_batch
            
            if parallel_execution:
                logic_errors[:,jj],  accepted_sim[:,jj] = self.logical_error_rate_par(decoder_batch, bases, min_errors, nb_shots,  num_cores, at_least_min_errors = at_least_min_errors)
            else:
                logic_errors[:,jj],  accepted_sim[:,jj] = self.logical_error_rate(decoder_batch, bases, min_errors, nb_shots, at_least_min_errors = at_least_min_errors)

        if plot_results:
            self._plot(noise_levels, logic_errors, accepted_sim)

        return logic_errors, accepted_sim
    
    def logical_error_rate(self,batch_decoder, bases, min_errors, nb_shots, shots_per_batch: int = 1_000_000, at_least_min_errors: bool = True) :
        """Estimate the logical error rate of the code.

        Args:
            shots: The number of shots to use.
            shots_per_batch: The number of shots per batch.
            at_least_min_errors: Whether to continue simulating until at least min_errors are found.
            min_errors: The minimum number of errors to find before stopping.
        """
        batch = min(shots_per_batch, nb_shots)
        p_l = np.zeros(self._dim_1)
        p_l_batch = np.zeros(self._dim_1)
        r_a = np.zeros(self._dim_1)

        num_logical_errors = np.zeros(self._dim_1)

        self._detection_circuit(bases)

        i = 1
        while i <= int(np.ceil(nb_shots / batch)) or at_least_min_errors:

            num_logical_errors_batch, discarded_batch = batch_decoder(bases, batch)

            if discarded_batch != batch:            
                p_l_batch = num_logical_errors_batch / (batch - discarded_batch)
                p_l= ((i - 1) * p_l + p_l_batch) / i

            # Update statistics
            num_logical_errors += num_logical_errors_batch

            r_a_batch = 1 - discarded_batch / batch
            r_a = ((i - 1) * r_a + r_a_batch) / i

            print("error rate", p_l, num_logical_errors, discarded_batch, i) # information to keep track of the simulation

            if at_least_min_errors and (num_logical_errors >= min_errors).all():
                break
            i += 1

        return p_l / self._code.k, r_a

    def decode_batch(self, bases, nb_shots):
        """
        Decodes each logical qubits and the errors that may have happened using the LutDecoder
        Returns the failure_num for each logical qubits

        Args:
            bases: list of bools with the basis to measure each logical qubit
            nb_shots: number of shots to simulate
        """

        seed = np.random.randint(1, 10000000000) + os.getpid() #improve?
        
        sampler = self._stim_circ.compile_sampler(seed = seed) 
        measurements = sampler.sample(nb_shots)

        # Sampler of detectors
        failure_num = np.array([])
        acceptance_num = np.array([])

        detector_sampler = self._stim_circ.compile_detector_sampler(seed = seed) 
        detection_events, observable_flips = detector_sampler.sample(nb_shots, separate_observables = True) 
      
        # Filter flag events of state preparation
        flag_index = np.where(np.all(measurements[:, self._verification_measurements] == 0, axis=1))[0]

        # Check all syndromes are the same and filters flags from syndrome measurement
        if self._num_rounds != 1: 
            aux  = np.where(np.all(detection_events == False, axis=1))[0]
            filter_index = list(set(flag_index) & set(aux))
        else: filter_index = flag_index

        filtered_events = measurements[filter_index].astype(np.int8)
        
        if len(filtered_events) == 0:  # All events were discarded
            return np.zeros(len(bases)*self._lq), np.zeros(len(bases)*self._lq) + nb_shots

        for num_logic, z_basis in enumerate(bases):
            if z_basis :
                ancila_data = filtered_events[:,self._z_measurements[0][self._nz*num_logic : self._nz*(num_logic + 1)]]
                estimate = self._decoder.batch_decode_x(ancila_data)
                
                ancila_data_ideal = filtered_events[:,self._z_ideal_measurements[self._nz*num_logic : self._nz*(num_logic + 1)]]
                estimate_ideal = self._decoder.batch_decode_x(ancila_data_ideal)

                observables = self._code.Lz

            else :
                ancila_data = filtered_events[:,self._x_measurements[0][self._nx*num_logic : self._nx*(num_logic + 1)]]
                estimate = self._decoder.batch_decode_z(ancila_data)
                
                ancila_data_ideal = filtered_events[:,self._x_ideal_measurements[self._nx*num_logic : self._nx*(num_logic + 1)]]
                estimate_ideal = self._decoder.batch_decode_z(ancila_data_ideal)

                observables = self._code.Lx

            final_estimate = (estimate + estimate_ideal) % 2

            predictions = (final_estimate @ observables.T % 2).astype(bool)

            num_logical_errors = sum(p != f for p, f in zip(predictions, observable_flips[filter_index, num_logic*self._lq : (num_logic + 1)*self._lq])) 
            num_discarted = measurements.shape[0] - filtered_events.shape[0]
            
            failure_num = np.append(failure_num, num_logical_errors)

        acceptance_num = np.append(acceptance_num, num_discarted) # the accepance rate is the same for the whole circuit
        
        return  failure_num, acceptance_num
    
    
    def decode_batch_joint(self, bases, nb_shots):
        """
        Decodes each logical qubits and the errors that may have happened at any point in the circuit using the LutDecoder 
        Returns the failure_rate of the circuit

        Args:
            bases: list of bools with the basis to measure each logical qubit
            nb_shots: number of shots to simulate
        """
        seed = np.random.randint(1, 1000000000) + os.getpid()#improve?

        sampler = self._stim_circ.compile_sampler(seed = seed) 
        measurements = sampler.sample(nb_shots)

        # sampler of detectors
        failure_num = np.array([])
        acceptance_num= np.array([])
        
        detector_sampler = self._stim_circ.compile_detector_sampler(seed = seed) 
        detection_events, observable_flips = detector_sampler.sample(nb_shots, separate_observables = True) # plug and play for other codes?

        flag_index = np.where(np.all(measurements[:, self._verification_measurements] == 0, axis=1))[0]

        if self._num_rounds != 1: 
            aux  = np.where(np.all(detection_events == False, axis=1))[0]
            filter_index = list(set(flag_index) & set(aux))
        else: filter_index = flag_index

        filtered_events = measurements[filter_index].astype(np.int8)
        
        predictions = np.zeros([filtered_events.shape[0], len(bases)*self._lq])
        
        if len(filtered_events) == 0:  # All events were discarded
            return np.zeros(self._dim_1), np.zeros(self._dim_1) + nb_shots

        for num_logic, z_basis in enumerate(bases):
            if z_basis :
                ancila_data = filtered_events[:,self._z_measurements[0][self._nz*num_logic : self._nz*(num_logic + 1)]]
                estimate = self._decoder.batch_decode_x(ancila_data)
                
                ancila_data_ideal = filtered_events[:,self._z_ideal_measurements[self._nz*num_logic : self._nz*(num_logic + 1)]]
                estimate_ideal = self._decoder.batch_decode_x(ancila_data_ideal)

                observables = self._code.Lz
            else :
                ancila_data = filtered_events[:,self._x_measurements[0][self._nx*num_logic : self._nx*(num_logic + 1)]]
                estimate = self._decoder.batch_decode_z(ancila_data)

                ancila_data_ideal = filtered_events[:,self._x_ideal_measurements[self._nx*num_logic : self._nx*(num_logic + 1)]]
                estimate_ideal = self._decoder.batch_decode_z(ancila_data_ideal)
                
                observables = self._code.Lx
            
            final_estimate = (estimate + estimate_ideal) % 2
            predictions[:, num_logic: (num_logic + self._lq)] = final_estimate @ observables.T % 2 #astype bool

        num_logical_errors = np.sum(predictions != observable_flips[filter_index])
        num_discarted = measurements.shape[0] - filtered_events.shape[0]
             
        failure_num = np.append(failure_num, num_logical_errors) 
        acceptance_num = np.append(acceptance_num, num_discarted)


        return  failure_num, acceptance_num
    
    def _detection_circuit(self, bases):
        """
        Appends the necessary measurements, detectors and observables to properly decode the errors in the selected basis
        """

        nstep = self._nq + self._nx + self._nz
        nstabs = self._nx + self._nz

        self._x_measurements = [[] for _ in range(self._num_rounds)]
        self._z_measurements = [[] for _ in range(self._num_rounds)]
        self._x_ideal_measurements = []
        self._z_ideal_measurements = []
        self._data_measurements, self._verification_measurements = [], []
        self._n_measurements = 0 
        obs_id = 0
            
        if self._reusing : self._compute_reuse_data(bases) # in case of reuse, updates the number of measurements all at once

        for num_logic, z_basis in enumerate(bases):

            qref = sum(self._ntotal[:num_logic])

            if not self._reusing: self._measure_stabilizers(qref, num_logic) #if not resued, all measuremnts are in the end, and its realted data is updated on the fly
            
            v =  self._nv_synd[num_logic] 

            # Appends detectors to check ancillas measurements
            for ii in self._x_measurements[0][:self._nx][::-1]: 
                detector_targets = []
                for round in range(self._num_rounds):
                    if self._reusing: ref_mes = -self._n_measurements  + ii + (v + nstabs)*(len(bases)*round)  + num_logic*nstabs
                    else: ref_mes = -ii -1 -round*nstabs
                    detector_targets.append(stim.target_rec(ref_mes))

                for pair in range(self._num_rounds - 1 ):
                    self._stim_circ.append("DETECTOR", detector_targets[pair : pair + 2])

            for ii in self._z_measurements[0][:self._nz][::-1]: #to check other decoding
                detector_targets = []
                for round in range(self._num_rounds):
                    if self._reusing: ref_mes = -self._n_measurements + ii + (v + nstabs)*(len(bases)*round) + num_logic*nstabs
                    else: ref_mes = -ii -1 -round*nstabs
                    detector_targets.append(stim.target_rec(ref_mes))

                for pair in range(self._num_rounds - 1):
                    self._stim_circ.append("DETECTOR", detector_targets[pair : pair + 2])

            self._measure_verifications(qref, num_logic)

            # Ideal synd extraction
            self._ideal_stabilizers(qref, num_logic)
                
            if z_basis:
                self._measure_z(qref)
            else:
                self._measure_x(qref)

            if z_basis: 
                observable = self._code.Lz
            else: 
                observable = self._code.Lx

            # Appends stim observables
            for logical in observable:
                obs = []
                supp = _support(logical)
                for ind in supp:
                     obs.append(stim.target_rec(self._data_measurements[num_logic][ind] - self._n_measurements))
              
                self._stim_circ.append("OBSERVABLE_INCLUDE", obs, obs_id)
                obs_id += 1
        
        my_circuit = self._stim_circ.diagram('timeline-svg')

        with open('stim_pic.svg', 'w') as f:
                print(my_circuit, file=f)        
        
        with open('stim_circ.stim', 'w') as f:
                f.write(str(self._stim_circ))  

    def _measure_stabilizers(self, qref, num_logic):
        
            """Measure the stabilizers of the code.

            An ancilla is used for each measurement.
            """
            assert self._code.Hx is not None
            assert self._code.Hz is not None

            anc = qref + self._nq + self._nv_prep[num_logic]

            for round in range(self._num_rounds):
                
                for ii in range(self._nz):   
                    self._get_error("measure", [anc])          
                    self._stim_circ.append("MZ", [anc])
                    self._z_measurements[round].append(self._n_measurements)
                    self._n_measurements += 1
                    anc += 1

                for ii in range(self._nx): 
                    self._get_error("measure", [anc])               
                    self._stim_circ.append("MZ", [anc]) #for NA hardware architecture
                    self._x_measurements[round].append(self._n_measurements)
                    self._n_measurements += 1
                    anc += 1


    def _measure_z(self, qref):
        """Measure all data qubits in the Z basis."""

        self._data_measurements.append([self._n_measurements + i for i in range(self._nq)])
        self._n_measurements += self._nq
        #self._get_error("measure", list(range(qref, self._nq + qref)))
        self._stim_circ.append("MZ", list(range(qref, self._nq + qref)))


    def _measure_x(self, qref):
        """Measure all data qubits in the X basis."""

        self._data_measurements.append([self._n_measurements + i for i in range(self._nq)])
        self._n_measurements += self._nq
        #self._get_error("measure", list(range(qref, self._nq + qref)))
        self._stim_circ.append("MX", list(range(qref,self._nq + qref)))

    def _measure_verifications(self, qref, num_logic):
        """ Measure verification checks"""

        vef = qref + self._nq

        for ii in range(self._nv_prep[num_logic]):   
                self._get_error("measure", [vef])        
                self._stim_circ.append("MRZ", [vef])
                self._verification_measurements.append(self._n_measurements) #only 1 type ?
                self._n_measurements += 1
                vef += 1
         
        vef += (self._nx + self._nz)*self._num_rounds 

        if not self._reusing:
            for round in range(self._num_rounds):
                for ii in range(self._nv_synd[num_logic]) :
                    self._get_error("measure", [vef])
                    self._stim_circ.append("MRZ", [vef])
                    self._verification_measurements.append(self._n_measurements) #only 1 type ?
                    self._n_measurements += 1            
                    vef += 1

    def _ideal_stabilizers(self, qref, num_logic):
        """Measure the stabilizers of the code.

        An ancilla is used for each measurement.
        """
        assert self._code.Hx is not None
        assert self._code.Hz is not None

        anc = qref + self._nq + self._nv_prep[num_logic]

        for check in self._code.Hz:
            supp = _support(check)
            for q in supp:
                self._stim_circ.append_operation("CX", [q + qref, anc])
            self._stim_circ.append_operation("MRZ", [anc])
            self._z_ideal_measurements.append(self._n_measurements)
            self._n_measurements += 1
            anc += 1

        for check in self._code.Hx:
            supp = _support(check)
            self._stim_circ.append_operation("H", [anc])
            for q in supp:
                self._stim_circ.append_operation("CX", [anc, q + qref])
            self._stim_circ.append_operation("MRX", [anc])
            self._x_ideal_measurements.append(self._n_measurements)
            self._n_measurements += 1
            anc += 1

    def _update_error(self, p): 
        """
        Updates dictionary for the current p value, for each "gate" operation. 
        Also stores p in the dictionary for NA2STIM to use and deletes attributes 
        of functions in "operations" so they can be updated correctly
        Note: eacch gate can has more than one error channel.

        Args:
           p: changing value of physcial error rate

        """
        gate_errors = self._error_model["gates"]

        for instruction in gate_errors.keys():
            for ii in range(len(gate_errors[instruction])): 
                if gate_errors[instruction][ii][2]:
                    new_prob = [ p*cnt for cnt in gate_errors[instruction][ii][3]]
                    gate_errors[instruction][ii][1] = np.array(new_prob)

        self._error_model["gates"] = gate_errors
        self._error_model["p_value"] = p

        self._delete_attributes()


    def _get_error(self, instruction, target): 
        """
        Reads the assigned errors from dictionary and adds the to STIM circuits when called 
        """
        target[:] = [item for item in target if item not in  self._ideal_qubits]

        if instruction in self._error_model.keys(): # turn into a function
             for ii in range(len(self._error_model[instruction])): 
                self._stim_circ.append(self._error_model[instruction][ii][0], target, self._error_model[instruction][ii][1] )
    
    def _plot(self, noise_levels, logic_errors, accepted_sim):
        """
        Returns a plot in log scale of Physcal error rate of the qubits vs the logical error rate
        """

        fig, axs = plt.subplots(1, 2, figsize=(8, 6)) 
        
        for ii, qubit_errors in enumerate(logic_errors):
            if self._joint_decoding: 
                axs[0].plot(noise_levels, qubit_errors, label = fr"circuit" )                
            else:
                axs[0].plot(noise_levels, qubit_errors, label = fr"$|q_{str(ii)}\rangle_L$" )

        for ii, num_accepted in enumerate(accepted_sim):
            if self._joint_decoding: 
                axs[1].plot(noise_levels, num_accepted, label = fr"circuit")
            else:
                axs[1].plot(noise_levels, num_accepted, label = fr"$|q_{str(ii)}\rangle_L$")

        axs[0].plot(noise_levels, noise_levels, linestyle='--')
        axs[0].plot(noise_levels, 10*noise_levels**2, linestyle='--')
        axs[0].plot(noise_levels, 100*noise_levels**3, linestyle='--')
        axs[0].set_title("Performance of the Decoder")
        axs[1].set_title("Acceptance Rate")
        axs[0].set_ylabel("Logical error rate")
        axs[1].set_ylabel("Acceptance rate")
        for ii in range(2):
            axs[ii].set_xlabel("Physical error rate")
            axs[ii].set_xscale("log")
            axs[ii].get_legend()
        axs[0].set_yscale("log")
        plt.savefig("output_simulation.png")



    def get_ancillas(self):
        """ 
        Returns the list of ancillas used in the circuit,  used  in ft schemes.
        Can also be used to externally shield ancillas from errors.

        """
        ancillas = [] 
        a = self._nx + self._nz

        reps = 1 if self._reusing else self._num_rounds
        
        for ii in range(len(self._states)):  
            qi = sum(self._ntotal[:ii]) + self._nq + self._nv_prep[ii]
            qf = qi + a
            ancillas += (list(range(qi, qf)))
            if self._nv_synd[ii] != 0 and not self._reusing :ancillas += (list(range(qi +  a, qf +  reps*a)))

        return ancillas
    
    def get_synd_flags(self):
        """ 
        Returns the list of flags used in the circuit, used in ft schemes.
        Can also be used to externally shield ancillas from errors.

        """

        flags = [] #
        a = self._nx + self._nz

        reps = 1 if self._reusing else self._num_rounds
        
        for ii in range(len(self._states)):  
            v = self._nv_synd[ii]
            qi = sum(self._ntotal[:ii]) + self._nq + self._nv_prep[ii] + reps*a
            qf = qi + v*reps
            flags += (list(range(qi, qf)))

        return flags

    def _delete_attributes(self):
        """ 
        Deletes attributes of functions in the error model so they can be updated correctly
        """
        ops_errors = self._error_model["operations"].keys()

        for error in ops_errors:
                for ii, element  in enumerate(self._error_model["operations"][error]):
                        print(error, "element", element)
                        fun = element[2]
                        for attr in list(fun.__dict__):
                            delattr(fun, attr)
                        self._error_model["operations"][error][ii][2] = fun 

  
    def _compute_reuse_data(self, bases):
        """
        Appends the necessary measurements, detectors and observables to properly decode the errors in the selected basis
        """

        for round in range(self._num_rounds):
            for num_logic in range(len(bases)):
                for z_mes in range(self._nz): 
                    self._z_measurements[round].append(self._n_measurements)
                    self._n_measurements += 1
                for x_mes in range(self._nx): 
                    self._x_measurements[round].append(self._n_measurements)
                    self._n_measurements += 1
            for num_logic in range(len(bases)):        
                for v_mes in range(self._nv_synd[num_logic]): 
                    self._verification_measurements.append(self._n_measurements)
                    self._n_measurements += 1



    
    def logical_error_rate_par(self,batch_decoder, bases, min_errors, nb_shots, num_cores, shots_per_batch: int = 1_000_000, at_least_min_errors: bool = True) :
        """Estimate the logical error rate of the code in a parallel fashion.

        Args:
            shots: The number of shots to use.
            shots_per_batch: The number of shots per batch.
            at_least_min_errors: Whether to continue simulating until at least min_errors are found.
            min_errors: The minimum number of errors to find before stopping.
        """
        batch = min(shots_per_batch, nb_shots)
        p_l = np.zeros(self._dim_1)
        p_l_batch = np.zeros(self._dim_1)
        r_a = np.zeros(self._dim_1)

        num_logical_errors = np.zeros(self._dim_1)

        self._detection_circuit(bases)
        args_list = [(bases, batch, batch_decoder) for _ in range(num_cores)]

        with Pool(num_cores) as p: #opens pool of workers

            i = num_cores
            print("Pool size:", p._processes)

            while i - num_cores <= int(np.ceil(nb_shots / batch)) or at_least_min_errors:
                    
                    time_init = time.time()
                    
                    results = [p.apply_async(self._worker_batch, args=args) for args in args_list]   
                    all_logic_errors, all_discarded_batches = zip(*[r.get() for r in results]) #results from all workers are gathered

                    num_logical_errors_batch = np.sum(all_logic_errors, axis=0)
                    discarded_batch = np.sum(all_discarded_batches, axis=0)
                                       
                    if discarded_batch != batch*num_cores:            
                        p_l_batch = num_logical_errors_batch / (batch*num_cores - discarded_batch)
                        p_l= ((i - num_cores) * p_l + p_l_batch*num_cores) / i

                    # Update statistics
                    num_logical_errors += num_logical_errors_batch

                    r_a_batch = 1 - discarded_batch / (batch*num_cores)
                    r_a = ((i - num_cores) * r_a + r_a_batch*num_cores) / i

                    print("error rate", p_l, num_logical_errors, discarded_batch, i) # real time tracking
                    print("Time taken for batch:", (time.time() - time_init) / num_cores)

                    if at_least_min_errors and (num_logical_errors >= min_errors).all():
                        break

                    i += num_cores

        return p_l / self._code.k, r_a
    
    def _worker_batch(self, bases, batch, batch_decoder ):
        """
        Function to be executed by each worker in the pool.

        Args is a tuple: (bases, batch, batch_decoder)
        Returns: (num_errors_batch, discarded_batch)
        """
        num_logical_errors_batch, discarded_batch = batch_decoder(bases, batch)

        return num_logical_errors_batch, discarded_batch
    


        


        
        
        
        
            
            
                
