
import numpy as np
from scipy import special

def move_0(p, *args):
        
        """
        Compute the time during shuttling and assign the corresponding idle error
        """
        inds = list(range(len(args[0])))
        jerk, v_lim, t_2, scaling = args[1][0], args[1][1], args[1][2], args[1][3]

        distance = args[3]*scaling

        tfinal = aux_moving_time(distance, jerk, v_lim)
        p_error = 0.5*(1-np.exp(-tfinal/t_2)) 
        
        gate_duration = args[4]
        gate_duration["move"] = tfinal

        return p_error.reshape(-1, 1), np.array(inds).reshape(-1, 1) 

def move_1(p, *args):
        """
        Compute the vivrational increase due to shuttling and compute the atom loss porbability
        with the erf function. When nvib>Nmax, a ideal cooling process is applied
        """

        target = [ int(k) for k in args[0] ]
        inds = list(range(len(target)))
        mass, w0, scaling, nmax, ncooling = args[1][0], args[1][1], args[1][2], args[1][3],  args[1][4]

        qubits = args[2]
        distance = args[3]*scaling
        gate_duration = args[4]

        times = gate_duration["move"]
        

        hbar = 1.0545718e-34
        xpf = (hbar/(2*(mass*1e-25)*w0))**(1/2) #mass given in 1e-25 kg

        mask = times != 0 #for cases when atoms are reused and not moved
        delta_n= np.zeros_like(distance, dtype=float)
        delta_n[mask] = 0.5*(6*distance[mask]/(xpf*w0**2*times[mask]**2))**2


        if not hasattr(move_1,"nvib"):
                 move_1.nvib = np.zeros(qubits) 

        move_1.nvib[target] += delta_n 

        cooling_ind = np.where(move_1.nvib> ncooling)
               
        move_1.nvib[cooling_ind] = 0
       
        p_error = aux_loss_error(nmax, move_1.nvib)
  
        if cooling_ind[0].size > 0: gate_duration["is_cooling"] = True
        else: gate_duration["is_cooling"] = False      

        return  p_error.reshape(-1, 1), np.array(inds).reshape(-1, 1)  

def move_2(p, *args):
        """
        Introduce additional idling error for cooling
        """

        gate_duration = args[4]
        if gate_duration["is_cooling"]:
                target = [ int(k) for k in args[0] ]
                inds = list(range(len(target)))
                t_2 = args[1][0]     
                cooling_time = gate_duration["cooling"]
                p_error = 0.5*(1-np.exp(-cooling_time/t_2))

                return np.array(p_error).reshape(-1, 1), np.array(inds).reshape(-1, 1)
        
        return np.array([]), np.array([])


def idle_0(p, *args):
        """
        Idling error computed with exponential decay formula
        """

        inds = list(range(len(args[0])))
        gate_duration = args[3]
        instruction = args[2]
        if instruction not in gate_duration.keys():
              return np.array([]), np.array([])
        t_op = gate_duration[instruction]
        t_2 = args[1][0]
        p_error = np.max(0.5*(1-np.exp(-t_op/t_2)))

        return np.array(p_error), np.array(inds) 

def store_0(p, *args):
        """
        Additional atom loss porbability while storing
        """
      
        inds = list(range(len(args[0])))
        cnt, t_2 = args[1][0], args[1][0]

        gate_duration = args[3]
        t_op = gate_duration["load"]

        p_error = p*cnt*(1-np.exp(-t_op/t_2))

        return np.array(p_error), np.array(inds)

def load_0(p, *args):
        """ Additional atom loss porbability while loading
        """
      
        inds = list(range(len(args[0])))
        cnt, t_2 = args[1][0], args[1][0]

        gate_duration = args[3]
        t_op = gate_duration["store"]

        p_error = p*cnt*(1-np.exp(-t_op/t_2))

        return np.array(p_error), np.array(inds)



# -------- AUXILIARY FUNCTIONS --------


def aux_moving_time(distance, jerk, v_lim):

    tfinal = (12*distance/jerk)**(1/3)
    v_max = jerk*tfinal**2/8

    if v_max.any() > v_lim:  
          tnew = (8*v_lim/jerk)**(1/2)
          dnew = jerk*tnew**3/12
          tfinal = tnew + (distance - dnew)/v_lim

    return tfinal

def aux_loss_error(nmax, nvib):    

        distr = np.ones_like(nvib, dtype=float)
        mask = nvib != 0
        distr[mask] = special.erf((nmax-nvib[mask])/(np.sqrt(2*nvib[mask])))
        p_loss = 1 - 0.5*(1 + distr)

        return p_loss