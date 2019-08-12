from __future__ import division

import math
import numpy as np


def alwaysUp(data):
    #supposed to be -1 or 1 / 0 or 1
    #supposed to predict everytime 1 (up)
    data=np.asarray(data)
    up=(data == 1).sum()/len(data)
    return up