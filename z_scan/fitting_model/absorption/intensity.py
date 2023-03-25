def intensity(z, z0: float, I0: float, Z_R: float):
    """ Returns the beam intensity at a given position

    PARAMETERS
    z : Array of z-positions
    z0 : z-position of the focal point
    I0 : Beam intensity at the focal point
    Z_R : Rayleigh length
    """
    return I0 / (1+((z-z0)/Z_R)**2)
