from zscan.absorption import transmittance
import numpy as np

def TPA(data: np.dtype, z0: float, Is1: float, Is2: float, beta: float, L: float, ALPHA0: float, I0: float, Z_R: float) -> float:
    """Returns the Chi-squared corresponding to the given parameters for the 2PA model
    
    ## PARAMETERS
    ---
    - data : Data structure, of size (N, 3) where N is the number of datapoints.
        [:,0]: z-data - [:, 1]: intensity-data - [:, 2]: uncertainty on intensity
    - z0 : z-position of the focal point
    - Is1 : First-order saturation intensity
    - Is2 : Second-order saturation intensity
    - beta : Non-linear absorption coefficient
    - L : Sample thickness
    - ALPHA0: Linear absorption coefficient
    - I0 : Beam intensity at the focal point
    - Z_R : Rayleigh length
    """
    Z_DATA = data[:, 0]
    I_DATA = data[:, 1]
    
    ## Calculate transmittance
    I_calc = transmittance.TPA(Z_DATA, z0, Is1, Is2, beta, L, ALPHA0, I0, Z_R)
    
    ## Calculate chi2
    if 0 in data[:, 2]:    # Poorly defined uncertainty
        chi2Vector = (I_DATA - I_calc)**2
    else:
        chi2Vector = (I_DATA - I_calc)**2 / np.average(data[:,2])**2

    return np.sum(chi2Vector)

#########################################################################################################

def OPA(data: np.dtype, z0: float, Is1: float, L: float, ALPHA0: float, I0: float, Z_R: float) -> float:
    """Returns the Chi-squared corresponding to the given parameters for the 1PA model
    
    ## PARAMETERS
    ---
    - data : Data structure, of size (N, 3) where N is the number of datapoints.
        [:,0]: z-data - [:, 1]: intensity-data - [:, 2]: uncertainty on intensity
    - z0 : z-position of the focal point
    - Is1 : First-order saturation intensity
    - L : Sample thickness
    - ALPHA0: Linear absorption coefficient
    - I0 : Beam intensity at the focal point
    - Z_R : Rayleigh length
    """
    Z_DATA = data[:, 0]
    I_DATA = data[:, 1]
    
    ## Calculate transmittance
    I_calc = transmittance.OPA(Z_DATA, z0, Is1, L, ALPHA0, I0, Z_R)
    
    ## Calculate chi2
    if 0 in data[:, 2]:    # Poorly defined uncertainty
        chi2Vector = (I_DATA - I_calc)**2
    else:
        chi2Vector = (I_DATA - I_calc)**2 / np.average(data[:,2])**2

    return np.sum(chi2Vector)

#########################################################################################

def TPA_no_Is2(data: np.dtype, z0: float, Is1: float, beta: float, L: float, ALPHA0: float, I0: float, Z_R: float) -> float:
    """Returns the Chi-squared corresponding to the given parameters for the 2PA model without Is2
    
    ## PARAMETERS
    ---
    - data : Data structure, of size (N, 3) where N is the number of datapoints.
        [:,0]: z-data - [:, 1]: intensity-data - [:, 2]: uncertainty on intensity
    - z0 : z-position of the focal point
    - Is1 : First-order saturation intensity
    - beta : Non-linear absorption coefficient
    - L : Sample thickness
    - ALPHA0: Linear absorption coefficient
    - I0 : Beam intensity at the focal point
    - Z_R : Rayleigh length
    """
    Z_DATA = data[:, 0]
    I_DATA = data[:, 1]
    
    ## Calculate transmittance
    I_calc = transmittance.TPA_no_Is2(Z_DATA, z0, Is1, beta, L, ALPHA0, I0, Z_R)
    
    ## Calculate chi2
    if 0 in data[:, 2]:    # Poorly defined uncertainty
        chi2Vector = (I_DATA - I_calc)**2
    else:
        chi2Vector = (I_DATA - I_calc)**2 / np.average(data[:,2])**2

    return np.sum(chi2Vector)

######################################################################################################

def TPA_no_Is1(data: np.dtype, z0: float, Is2: float, beta: float, L: float, ALPHA0: float, I0: float, Z_R: float) -> float:
    """Returns the Chi-squared corresponding to the given parameters for the 2PA model without Is1
    
    ## PARAMETERS
    ---
    - data : Data structure, of size (N, 3) where N is the number of datapoints.
        [:,0]: z-data - [:, 1]: intensity-data - [:, 2]: uncertainty on intensity
    - z0 : z-position of the focal point
    - Is2 : Second-order saturation intensity
    - beta : Non-linear absorption coefficient
    - L : Sample thickness
    - ALPHA0: Linear absorption coefficient
    - I0 : Beam intensity at the focal point
    - Z_R : Rayleigh length
    """
    Z_DATA = data[:, 0]
    I_DATA = data[:, 1]
    
    ## Calculate transmittance
    I_calc = transmittance.TPA_no_Is1(Z_DATA, z0, Is2, beta, L, ALPHA0, I0, Z_R)
    
    ## Calculate chi2
    if 0 in data[:, 2]:    # Poorly defined uncertainty
        chi2Vector = (I_DATA - I_calc)**2
    else:
        chi2Vector = (I_DATA - I_calc)**2 / np.average(data[:,2])**2

    return np.sum(chi2Vector)

#############################################################################################################

def TPA_no_sat(data: np.dtype, z0: float, beta: float, L: float, ALPHA0: float, I0: float, Z_R: float) -> float:
    """Returns the Chi-squared corresponding to the given parameters for the 2PA model without saturation
    
    ## PARAMETERS
    ---
    - data : Data structure, of size (N, 3) where N is the number of datapoints.
        [:,0]: z-data - [:, 1]: intensity-data - [:, 2]: uncertainty on intensity
    - z0 : z-position of the focal point
    - beta : Non-linear absorption coefficient
    - L : Sample thickness
    - ALPHA0: Linear absorption coefficient
    - I0 : Beam intensity at the focal point
    - Z_R : Rayleigh length
    """
    Z_DATA = data[:, 0]
    I_DATA = data[:, 1]
    
    ## Calculate transmittance
    I_calc = transmittance.TPA_no_sat(Z_DATA, z0, beta, L, ALPHA0, I0, Z_R)
    
    ## Calculate chi2
    if 0 in data[:, 2]:    # Poorly defined uncertainty
        chi2Vector = (I_DATA - I_calc)**2
    else:
        chi2Vector = (I_DATA - I_calc)**2 / np.average(data[:,2])**2

    return np.sum(chi2Vector)