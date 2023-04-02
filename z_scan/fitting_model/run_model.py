import numpy as np
from fitting_model.chi2_minimising import basinhopping
from fitting_model.chi2_minimising import chi2 as X2
import tkinter as tk
from tkinter import ttk
import sys

def run(measurement: np.dtype, fit_type: int, n_runs: int, p0: list, model_param: list, experiment_param: list, root, iter_label, value_label, age_label, pb):
    """Returns the fitting results of all runs, as well as the fitting parameters and chi-squared of the best run
    
    ##PARAMETERS
    ---
    - measurement: z, I and sigma data of the measurement
    - fit_type: integer to specify the fitting model
    - n_runs: number of runs
    - p0: initial guess for the fitting parameters
    - model_param: parameters for the optimisation process
    - experiment_param: list of experiment parameters
    """
    [Z0_0, I_S1_0, I_S2_0, BETA_0] = p0
    BOUNDS = model_param[3]
    [L, ALPHA0, I0, Z_R] = experiment_param

    value_label.config(text= 'Number of models computed: ' + str(0) + '/' + str(n_runs))
    age_label.config(text='Best age: ' + str(0) + '/' + str(model_param[1]))
    iter_label.config(text= 'Iteration : ' + str(0) + '/' + str(model_param[2]))
    pb['value'] = 0
    root.update()

    # Run 1PA Model
    if fit_type == 0:
        ## Prepare model
        del BOUNDS[2:]
        model_param[3] = BOUNDS
        RUNS = np.ndarray((n_runs,3))    # Intialise data structure
    
        ## Run model
        for i in range(n_runs):
            p0 = np.array([Z0_0, I_S1_0])    # Initial guess
            popt, chi2 = basinhopping.OPA(measurement, p0, L, ALPHA0, I0, Z_R, model_param, age_label, iter_label, root)
            ## Store results
            RUNS[i, 0]=popt[0]
            RUNS[i, 1]=popt[1]
            RUNS[i,2] = chi2
            ## Update progress bar
            pb['value'] += 100 / n_runs    
            value_label.config(text= 'Number of models computed: ' + str(i+1) + '/' + str(n_runs))
            root.update()

        ## Find best results
        CHI2_BEST = np.amin(RUNS[:,2])
        index = np.where(RUNS[:,2]==CHI2_BEST)[0][0]
        P_BEST = RUNS[index, 0:2].reshape(-1)


    # Run 2PA no Is2
    elif fit_type == 1:
        ## Prepare model
        del BOUNDS[2]
        model_param[3] = BOUNDS
        RUNS = np.ndarray((n_runs,4))    # Intialise data structure

        ## Run model
        for i in range(n_runs):
            p0 = np.array([Z0_0, I_S1_0, BETA_0])    # Initial guess
            popt, chi2 = basinhopping.TPA_no_Is2(measurement, p0, L, ALPHA0, I0, Z_R, model_param, age_label, iter_label, root)    # Run model
            ## Store results
            RUNS[i, 0]=popt[0]
            RUNS[i, 1]=popt[1]
            RUNS[i, 2]=popt[2]
            RUNS[i,3] = chi2
            ## Update progress bar
            pb['value'] += 100 / n_runs    
            value_label.config(text= 'Number of models computed: ' + str(i+1) + '/' + str(n_runs))
            root.update()

        ## Find best results
        CHI2_BEST = np.amin(RUNS[:,3])
        index = np.where(RUNS[:,3]==CHI2_BEST)[0][0]
        P_BEST = RUNS[index, 0:3].reshape(-1)


    # Run 2PA
    elif fit_type == 2:
        ## Prepare model
        model_param[3] = BOUNDS
        RUNS = np.ndarray((n_runs,5))    # Intialise data structure

        ## Run model
        for i in range(n_runs):
            p0 = np.array([Z0_0, I_S1_0, I_S2_0, BETA_0])    # Initial guess
            popt, chi2 = basinhopping.TPA(measurement, p0, L, ALPHA0, I0, Z_R, model_param, age_label, iter_label, root)    # Run model
            ## Store results
            RUNS[i, 0]=popt[0]
            RUNS[i, 1]=popt[1]
            RUNS[i, 2]=popt[2]
            RUNS[i, 3]=popt[3]
            RUNS[i,4] = chi2
            ## Update progress bar
            pb['value'] += 100 / n_runs    
            value_label.config(text= 'Number of models computed: ' + str(i+1) + '/' + str(n_runs))
            root.update()
        
        ## Find best results
        CHI2_BEST = np.amin(RUNS[:,4])
        index = np.where(RUNS[:,4]==CHI2_BEST)[0][0]
        P_BEST = RUNS[index, 0:4].reshape(-1)


    # Run 2PA no Is1
    elif fit_type == 3:
        ## Prepare model
        del BOUNDS[1]
        model_param[3] = BOUNDS
        RUNS = np.ndarray((n_runs,4))    # Intialise data structure

        ## Run model
        for i in range(n_runs):
            p0 = np.array([Z0_0, I_S2_0, BETA_0])    # Initial guess
            popt, chi2 = basinhopping.TPA_no_Is1(measurement, p0, L, ALPHA0, I0, Z_R, model_param, age_label, iter_label, root)    # Run model
            ## Store results
            RUNS[i, 0]=popt[0]
            RUNS[i, 1]=popt[1]
            RUNS[i, 2]=popt[2]
            RUNS[i,3] = chi2
            ## Update progress bar
            pb['value'] += 100 / n_runs    
            value_label.config(text= 'Number of models computed: ' + str(i+1) + '/' + str(n_runs))
            root.update()

        ## Find best results
        CHI2_BEST = np.amin(RUNS[:,3])
        index = np.where(RUNS[:,3]==CHI2_BEST)[0][0]
        P_BEST = RUNS[index, 0:3].reshape(-1)


    # Run 2PA no sat
    elif fit_type == 4:
        ## Prepare model
        del BOUNDS[1:3]
        model_param[3] = BOUNDS
        RUNS = np.ndarray((n_runs,3))    # Intialise data structure

        ## Run model
        for i in range(n_runs):
            p0 = np.array([Z0_0, BETA_0])    # Initial guess
            popt, chi2 = basinhopping.TPA_no_sat(measurement, p0, L, ALPHA0, I0, Z_R, model_param, age_label, iter_label, root)    # Run model
            ## Store results
            RUNS[i, 0]=popt[0]
            RUNS[i, 1]=popt[1]
            RUNS[i,2] = chi2
            ## Update progress bar
            pb['value'] += 100 / n_runs    
            value_label.config(text= 'Number of models computed: ' + str(i+1) + '/' + str(n_runs))
            root.update()

        ## Find best results
        CHI2_BEST = np.amin(RUNS[:,2])
        index = np.where(RUNS[:,2]==CHI2_BEST)[0][0]
        P_BEST = RUNS[index, 0:2].reshape(-1)

    else:
        sys.exit('No valid fit type')

    # Close progress bar
    return RUNS, P_BEST, CHI2_BEST 