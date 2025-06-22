import numpy as np
def idle_0(p, *args):
        """ Fixed idle noise"""
        inds = list(range(len(args[0])))

        return np.array(args[1]), np.array(inds) 