""" Define the differential equation that describes the absorption for all different models"""

def OPA(z, I, Is1: float, ALPHA0: float):
    """Returns the differential equation considering only Is1
    
    PARAMETERS
    z : Required parameter for differential equation solver
    I : Intensities that enters the sample
    Is1 : First-order saturation intensity
    ALPHA0 : Linear absorption coefficient
    """
    term1 = - ALPHA0 / (1 + (I/Is1)) * I
    return term1
#############################################################################
def TPA_no_Is1(z, I, Is2: float, beta: float, ALPHA0: float):
    """Returns the differential equation considering 2PA without Is1
    
    PARAMETERS
    z : Required parameter for differential equation solver
    I : Intensity that enters the sample
    Is2 : Second-order saturation intensity
    beta : Nonlinear absorption coefficient
    ALPHA0 : Linear absorption coefficient
    """
    term1 = - ALPHA0 * I
    term2 = - beta / (1 + (I/Is2)) * I**2
    return term1 + term2
#############################################################################
def TPA_no_Is2(z, I, Is1: float, beta: float, ALPHA0: float):
    """Returns the differential equation considering 2PA without Is2
    
    PARAMETERS
    z : Required parameter for differential equation solver
    I : Intensity that enters the sample
    Is1 : First-order saturation intensity
    beta : Nonlinear absorption coefficient
    ALPHA0 : Linear absorption coefficient
    """
    term1 = - ALPHA0 / (1 + (I/Is1)) * I
    term2 = - beta * I**2
    return term1 + term2
##############################################################################
def TPA_no_sat(z, I, beta: float, ALPHA0: float):
    """Returns the differential equation considering 2PA without saturation
    
    PARAMETERS
    z : Required parameter for differential equation solver
    I : Intensity that enters the sample
    beta : Nonlinear absorption coefficient
    ALPHA0 : Linear absorption coefficient
    """
    term1 = - ALPHA0 * I
    term2 = - beta * I**2
    return term1 + term2
#############################################################################
def TPA(z, I, Is1: float, Is2: float, beta: float, ALPHA0: float):
    """Returns the differential equation considering 2PA

    PARAMETERS
    z : Required parameter for differential equation solver
    I : Intensity that enters the sample
    Is1 : First-order saturation intensity
    Is2 : Second-order saturation intensity
    beta : Nonlinear absorption coefficient
    ALPHA0 : Linear absorption coefficient
    """
    term1 = - ALPHA0 / (1 + (I/Is1)) * I
    term2 = - beta / (1 + (I/Is2)) * I**2
    return term1 + term2