import numpy as np
from z_scan.fitting_model.chi2_minimising import chi2

def errorbar(data, fit_type, p_best, chi2_best, param_index, experiment_param):
    # Initialise datastructure
    N_POINTS = len(data[:,0])
    STEP = 0.005
    lBound = [p_best[param_index], 0]
    rBound = [p_best[param_index], 0]
    lTest = np.array(p_best)
    rTest = np.array(p_best)
    if fit_type==0 or fit_type==4:    # If 1PA or 2PA no sat
        n_param = 2
    elif fit_type==1 or fit_type==3:    # If 2PA no Is1 or Is2
        n_param = 3
    else:
        n_param = 4
    
    # Determine chi2 region
    WEIGHT = np.sum(1/np.sqrt(1+((data[:,0]-p_best[0])/experiment_param[3])**2)) / N_POINTS
    #WEIGHT = 1
    CHI2_COMPARISON = (N_POINTS - n_param) * WEIGHT + chi2_best    # Degrees of freedom of model * average value of the weigth function
    
    # Error calculation
    newChi2 = 0
    iteration = 0
    while newChi2 < CHI2_COMPARISON and iteration != 100:
        iteration += 1
        rBound[0] = (1+STEP)*rBound[0]
        lBound[0] = (1-STEP)*lBound[0]
        rTest[param_index] = rBound[0]
        lTest[param_index] = lBound[0]
        
        if fit_type == 0:    # 1PA
            rBound[1] = chi2.OPA(data, rTest[0], rTest[1], *experiment_param)
            lBound[1] = chi2.OPA(data, lTest[0], lTest[1], *experiment_param)
        elif fit_type == 1:    # 2PA no Is2
            rBound[1] = chi2.TPA_no_Is2(data, rTest[0], rTest[1], rTest[2], *experiment_param)
            lBound[1] = chi2.TPA_no_Is2(data, lTest[0], lTest[1], lTest[2], *experiment_param)
        elif fit_type == 2:    # 2PA
            rBound[1] = chi2.TPA(data, rTest[0], rTest[1], rTest[2], rTest[3], *experiment_param)
            lBound[1] = chi2.TPA(data, lTest[0], lTest[1], rTest[2], rTest[3], *experiment_param)
        elif fit_type == 3:    # 2PA no Is1
            rBound[1] = chi2.TPA_no_Is1(data, rTest[0], rTest[1], rTest[2], *experiment_param)
            lBound[1] = chi2.TPA_no_Is1(data, lTest[0], lTest[1], lTest[2], *experiment_param)
        else:
            rBound[1] = chi2.TPA_no_sat(data, rTest[0], rTest[1], *experiment_param)
            lBound[1] = chi2.TPA_no_sat(data, lTest[0], lTest[1], *experiment_param)
        newChi2 = max(lBound[1], rBound[1])
    
    # Choose appropriate error
    if rBound[1] > lBound[1]:
        errorbar = rBound[0]
        varyChi = rBound[1]
    else:
        errorbar = lBound[0]
        varyChi = lBound[1]

    return np.abs(errorbar-p_best[param_index]), varyChi

##################################################################################################

def compute(data, fit_type, p_best, chi2_best, experiment_param):
    if fit_type==0 or fit_type==4:    # If 1PA or 2PA no sat
        SIGMA_P = np.zeros(2)
        CHI2_SPAN = np.zeros(2)
        SIGMA_P[0], CHI2_SPAN[0] = errorbar(data, fit_type, p_best, chi2_best, 0, experiment_param)
        SIGMA_P[1], CHI2_SPAN[1] = errorbar(data, fit_type, p_best, chi2_best, 1, experiment_param)
    elif fit_type==1 or fit_type==3:    # If 2PA no Is1 or Is2
        SIGMA_P = np.zeros(3)
        CHI2_SPAN = np.zeros(3)
        SIGMA_P[0], CHI2_SPAN[0] = errorbar(data, fit_type, p_best, chi2_best, 0, experiment_param)
        SIGMA_P[1], CHI2_SPAN[1] = errorbar(data, fit_type, p_best, chi2_best, 1, experiment_param)
        SIGMA_P[2], CHI2_SPAN[2] = errorbar(data, fit_type, p_best, chi2_best, 2, experiment_param)
    else:
        SIGMA_P = np.zeros(4)
        CHI2_SPAN = np.zeros(4)
        SIGMA_P[0], CHI2_SPAN[0] = errorbar(data, fit_type, p_best, chi2_best, 0, experiment_param)
        SIGMA_P[1], CHI2_SPAN[1] = errorbar(data, fit_type, p_best, chi2_best, 1, experiment_param)
        SIGMA_P[2], CHI2_SPAN[2] = errorbar(data, fit_type, p_best, chi2_best, 2, experiment_param)
        SIGMA_P[3], CHI2_SPAN[3] = errorbar(data, fit_type, p_best, chi2_best, 3, experiment_param)
    return SIGMA_P, CHI2_SPAN