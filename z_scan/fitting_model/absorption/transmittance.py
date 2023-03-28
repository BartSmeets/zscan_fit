"""Module containing functions that calculate the transmittance through a sample for the different models"""

from fitting_model.absorption import dI_dz
from .intensity import intensity as I
from scipy.integrate import solve_ivp

#############################################################################

def OPA(z, z0: float, Is1: float, L: float, ALPHA0: float, I0: float, Z_R: float):
    """ returns the normalised transmittance considering 1PA
    
    ## PARAMETERS
    ---
    - z : z-data
    - z0 : z-position of the focal point
    - Is1 : First-order saturation intensity
    - L : Sample thickness
    - ALPHA0 : Linear absorption coefficient
    - I0 : Intensity at the focal point
    - Z_R: Rayleigh length 
    """
    I_in = I(z, z0, I0, Z_R)    # Initial condition
    sol = solve_ivp(dI_dz.OPA, [0, L], I_in, args=(Is1, ALPHA0), t_eval=[L])    # Runge-Kutta 45 method
    I_out = sol.y[:, 0]
    transmittance = (I_out / I_in) / (I_out[0] / I_in[0])
    return transmittance

#############################################################################

def TPA_no_Is1(z, z0: float, Is2: float, beta: float, L: float, ALPHA0: float, I0: float, Z_R: float):
    """ returns the normalised transmittance considering 2PA without Is1
    
    ##P ARAMETERS
    ---
    - z : z-data
    - z0 : z-position of the focal point
    - Is2 : Second-order saturation intensity
    - beta : Nonlinear absorption coefficient
    - L : Sample thickness
    - ALPHA0 : Linear absorption coefficient
    - I0 : Intensity at the focal point
    - Z_R: Rayleigh length 
    """
    I_in = I(z, z0, I0, Z_R)    # Initial condition
    sol = solve_ivp(dI_dz.TPA_no_Is1, [0, L], I_in, args=(Is2, beta, ALPHA0), t_eval=[L])    # Runge-Kutta 45 method
    I_out = sol.y[:, 0]
    transmittance = (I_out / I_in) / (I_out[0] / I_in[0])
    return transmittance

################################################################################

def TPA_no_Is2(z, z0: float, Is1: float, beta: float, L: float, ALPHA0: float, I0: float, Z_R: float):
    """ returns the normalised transmittance considering 2PA without Is2
    
    ## PARAMETERS
    ---
    - z : z-data
    - z0 : z-position of the focal point
    - Is1 : First-order saturation intensity
    - beta : Nonlinear absorption coefficient
    - L : Sample thickness
    - ALPHA0 : Linear absorption coefficient
    - I0 : Intensity at the focal point
    - Z_R: Rayleigh length 
    """
    I_in = I(z, z0, I0, Z_R)    # Initial condition
    sol = solve_ivp(dI_dz.TPA_no_Is2, [0, L], I_in, args=(Is1, beta, ALPHA0), t_eval=[L])    # Runge-Kutta 45 method
    I_out = sol.y[:, 0]
    transmittance = (I_out / I_in) / (I_out[0] / I_in[0])
    return transmittance

##########################################################################################

def TPA_no_sat(z, z0: float, beta: float, L: float, ALPHA0: float, I0: float, Z_R: float):
    """ returns the normalised transmittance considering 2PA without saturation
    
    ## PARAMETERS
    ---
    - z : z-data
    - z0 : z-position of the focal point
    - beta : Nonlinear absorption coefficient
    - L : Sample thickness
    - ALPHA0 : Linear absorption coefficient
    - I0 : Intensity at the focal point
    - Z_R: Rayleigh length 
    """
    I_in = I(z, z0, I0, Z_R)    # Initial condition
    sol = solve_ivp(dI_dz.TPA_no_sat, [0, L], I_in, args=(beta, ALPHA0), t_eval=[L])    # Runge-Kutta 45 method
    I_out = sol.y[:, 0]
    transmittance = (I_out / I_in) / (I_out[0] / I_in[0])
    return transmittance

###########################################################################################

def TPA(z, z0: float, Is1: float, Is2: float, beta: float, L: float, ALPHA0: float, I0: float, Z_R: float):
    """ returns the normalised transmittance considering 2PA
    
    ## PARAMETERS
    - z : z-data
    - z0 : z-position of the focal point
    - Is1 : First-order saturation intensity
    - Is2 : Second-order saturation intensity
    - beta : Nonlinear absorption coefficient
    - L : Sample thickness
    - ALPHA0 : Linear absorption coefficient
    - I0 : Intensity at the focal point
    - Z_R: Rayleigh length 
    """
    I_in = I(z, z0, I0, Z_R)    # Initial condition
    sol = solve_ivp(dI_dz.TPA, [0, L], I_in, args=(Is1, Is2, beta, ALPHA0), t_eval=[L])    # Runge-Kutta 45 method
    I_out = sol.y[:, 0]
    transmittance = (I_out / I_in) / (I_out[0] / I_in[0])
    return transmittance