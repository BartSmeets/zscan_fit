"""Module containing:
- best: to generate text from optimal fitting parameters
- individual: to generate text from fitting parameters of individual runs
- output: to generate text from optimal fitting parameters and their errorbars
"""
import numpy as np

###################################################################################

def best(fit_type: int, chi2_best: float, p_best: np.ndarray):
    """Returns string with optimal fitting parameters
    
    ##PARAMETERS
    ---
    - fit_type: integer describing the fitting model
    - chi2_best: optimal chi-squared
    - p_best: optimal fitting parameters
    """
    # 1PA
    if fit_type == 0:
        textstr = '\n'.join((
        r'X2 = %.2e' % (chi2_best, ),
        r'z0 = %.2f mm' % (p_best[0], ),
        r'Is1 = %.2e W/mm2' % (p_best[1],)))

    # 2PA no Is2
    elif fit_type == 1:
        textstr = '\n'.join((
        r'X2 = %.2e' % (chi2_best, ),
        r'z0 = %.2f mm' % (p_best[0], ),
        r'Is1 = %.2e W/mm2' % (p_best[1],),
        r'beta = %.2e mm/W' % (p_best[2],)))

    # 2PA
    elif fit_type == 2:
        textstr = '\n'.join((
        r'X2 = %.2e' % (chi2_best, ),
        r'z0 = %.2f mm' % (p_best[0], ),
        r'Is1 = %.2e W/mm2' % (p_best[1],),
        r'Is2 = %.2e W/mm2' % (p_best[2],),
        r'beta = %.2e mm/W' % (p_best[3],)))
            
    # 2PA no Is1
    elif fit_type == 3:
        textstr = '\n'.join((
        r'X2 = %.2e' % (chi2_best, ),
        r'z0 = %.2f mm' % (p_best[0], ),
        r'Is2 = %.2e W/mm2' % (p_best[1],),
        r'beta = %.2e mm/W' % (p_best[2],)))

    # 2PA no sat
    else:
        textstr = '\n'.join((
        r'X2 = %.2e' % (chi2_best, ),
        r'z0 = %.2f mm' % (p_best[0], ),
        r'beta = %.2e mm/W' % (p_best[1],)))
    return textstr

##############################################################

def individual(fit_type: int, runs: np.ndarray, i: int):
    """Returns string with fitting parameters of individual runs
    
    ##PARAMETERS
    ---
    - fit_type: integer describing the fitting model
    - runs: information of individual runs
    - i: index of individual run
    """

    # 1PA
    if fit_type == 0:
        textstr = '\n'.join((
        r'X2 = %.2e' % (runs[i,2], ),
        r'z0 = %.2f mm' % (runs[i,0], ),
        r'Is1 = %.2e W/mm2' % (runs[i,1],)))

    # 2PA no Is2
    elif fit_type == 1:
        textstr = '\n'.join((
        r'X2 = %.2e' % (runs[i,3], ),
        r'z0 = %.2f mm' % (runs[i,0], ),
        r'Is1 = %.2e W/mm2' % (runs[i,1],),
        r'beta = %.2e mm/W' % (runs[i,2],)))

    # 2PA
    elif fit_type == 2:
        textstr = '\n'.join((
        r'X2 = %.2e' % (runs[i,4], ),
        r'z0 = %.2f mm' % (runs[i,0], ),
        r'Is1 = %.2e W/mm2' % (runs[i,1],),
        r'Is2 = %.2e W/mm2' % (runs[i,2],),
        r'beta = %.2e mm/W' % (runs[i,3],)))
            
    # 2PA no Is1
    elif fit_type == 3:
        textstr = '\n'.join((
        r'X2 = %.2e' % (runs[i,3], ),
        r'z0 = %.2f mm' % (runs[i,0], ),
        r'Is2 = %.2e W/mm2' % (runs[i,1],),
        r'beta = %.2e mm/W' % (runs[i,2],)))

    # 2PA no sat
    else:
        textstr = '\n'.join((
        r'X2 = %.2e' % (runs[i,2], ),
        r'z0 = %.2f mm' % (runs[i,0], ),
        r'beta = %.2e mm/W' % (runs[i,1],)))
    return textstr

###########################################################################

def output(fit_type, chi2_best, p_best, sigma, span):
    """Returns string with optimal fitting parameters and optimal chi-squared
    
    ##PARAMETERS
    ---
    - fit_type: integer describing the fitting model
    - chi2_best: optimal chi-squared
    - p_best: optimal fitting parameters
    - sigma: errorbar of fitting parameter
    - span: volume in chi2-space spanned by the errorbar
    """
    # 1PA
    if fit_type == 0:
        textstr = '\n'.join((
        r'Fitting Parameters',
        r'X2 = %.2e' % (chi2_best, ),
        r'z0 = (%.2f +- %.2f) mm; X2 span: %.2f' % (p_best[0], sigma[0], span[0], ),
        r'Is1 = (%.2e +- %.1e)  W/mm2; X2 span: %.2f' % (p_best[1], sigma[1], span[1],)))

    # 2PA no Is2
    elif fit_type == 1:
        textstr = '\n'.join((
        r'Fitting Parameters',
        r'X2 = %.2e' % (chi2_best, ),
        r'z0 = (%.2f +- %.2f) mm; X2 span: %.2f' % (p_best[0], sigma[0], span[0], ),
        r'Is1 = (%.2e +- %.1e)  W/mm2; X2 span: %.2f' % (p_best[1], sigma[1], span[1], ), 
        r'beta = (%.2e +- %.1e)  mm/W; X2 span: %.2f' % (p_best[2], sigma[2], span[2],)))

    # 2PA
    elif fit_type == 2:
        textstr = '\n'.join((
        r'Fitting Parameters',
        r'X2 = %.2e' % (chi2_best, ),
        r'z0 = (%.2f +- %.2f) mm; X2 span: %.2f' % (p_best[0], sigma[0], span[0], ),
        r'Is1 = (%.2e +- %.1e)  W/mm2; X2 span: %.2f' % (p_best[1], sigma[1], span[1], ),
        r'Is2 = (%.2e +- %.1e)  W/mm2; X2 span: %.2f' % (p_best[2], sigma[2], span[2], ),
        r'beta = (%.2e +- %.1e)  mm/W; X2 span: %.2f' % (p_best[3], sigma[3], span[3],)))
            
    # 2PA no Is1
    elif fit_type == 3:
        textstr = '\n'.join((
        r'Fitting Parameters',
        r'X2 = %.2e' % (chi2_best, ),
        r'z0 = (%.2f +- %.2f) mm; X2 span: %.2f' % (p_best[0], sigma[0], span[0], ),
        r'Is2 = (%.2e +- %.1e)  W/mm2; X2 span: %.2f' % (p_best[1], sigma[1], span[1], ),
        r'beta = (%.2e +- %.1e)  mm/W; X2 span: %.2f' % (p_best[2], sigma[2], span[2],)))

    # 2PA no sat
    else:
        textstr = '\n'.join((
        r'Fitting Parameters',
        r'X2 = %.2e' % (chi2_best, ),
        r'z0 = (%.2f +- %.2f) mm; X2 span: %.2f' % (p_best[0], sigma[0], span[0], ),
        r'beta = (%.2e +- %.1e)  mm/W; X2 span: %.2f' % (p_best[1], sigma[1], span[1],)))
    return textstr

###########################################################################

def parameter_string(parameter_data):
    """Returns string with optimal fitting parameters and optimal chi-squared
    
    ##PARAMETERS
    ---
    - Paramter data set
    """

    [L, ALPHA0, I0, Z_R, W0, E_PULSE] = parameter_data

    PARAMETER_STRING = '\n'.join((
        r'Model Parameters',
        r'alpha0 = %.2f cm-1' % (ALPHA0*10, ),
        r'L = %.1f cm' % (L/10, ),
        r'w0 = %.2f um' % (W0*1e3, ),
        r'zR = %.2f um' % (Z_R*1e3, ),
        r'E_pulse = %.2f uJ' % (E_PULSE*1e6, ),
        r'I0 = %.2e  W/mm2' % (I0, )))
    return PARAMETER_STRING
