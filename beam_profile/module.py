""" # Analyse the beam profile
---
## FUNCTIONS

- gaussian(x, a, b, c)

- width(z, w0, z0, M2, wavelength)
"""

import numpy as np
import numpy.typing as npt

########################################################################################

def gaussian(x: npt.ArrayLike, a: float, b: float, c: float) -> npt.ArrayLike:
    """ Returns the value of a general Gaussian
    
    PARAMETERS

    x: variable
    a, b, c: gaussian parameters
    """
    return a * np.exp(-(x-b)**2 / (2 * c**2))

########################################################################################

def width(z: npt.ArrayLike, w0: float, z0: float, M2: float, wavelength: float) -> npt.ArrayLike:
    """Returns the beam radius (width) of the beam at a given position
    
    PARAMETERS

    z: z-position
    w0: beam radius at the focal point
    z0: z-position of the focal point
    M2: Quality-Factor
    wavelength: wavelength of the beam
    """
    wavelength = wavelength * 1e-3    # nm to um
    z = z * 1e3    # mm to um
    zR = np.pi*(w0**2) / (M2 * wavelength)
    root = 1 + ((z-z0)/zR)**2
    return w0 * np.sqrt(root)    # unit um