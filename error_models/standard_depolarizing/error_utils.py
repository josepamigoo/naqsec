import numpy as np
def idle_0(p, *args):
        """ Idle error than changes with p
        """

        inds = list(range(len(args[0])))
        list_p = [p*item for item in args[1]] # we are passing a list embedded in a tuple

        return np.array(list_p), np.array(inds) 