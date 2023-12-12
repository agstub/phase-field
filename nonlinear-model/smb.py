# Surface mass balance functions (for upper and lower surfaces)
# *Note: the expressions below are written so that they apply to 
#       both UFL objects and numpy arrays 
import numpy as np
from params import L, H
from scipy.special import erf

def smb_s(x,t,m0,stdev):
    m = m0*(np.exp(1)**(-x**2/(stdev**2)))
    return m

def smb_h(x,t,m0,stdev):
    # Surface mass balance functions (at upper surface)
    a = m0*np.sqrt(np.pi)*stdev*erf(L/(2*stdev)) / L 
    return a