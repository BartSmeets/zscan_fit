import numpy as np
from z_scan.fitting_model.chi2_minimising import basinhopping
from z_scan.fitting_model.chi2_minimising import chi2 as X2
import tkinter as tk
from tkinter import ttk
import sys

def run(MEASUREMENT: np.dtype, FIT_TYPE: int, N_RUNS: int, p0: list, MODEL_PARAMETERS: list, EXPERIMENT_PARAM: list):
    """Returns the fitting results of all runs, as well as the fitting parameters and chi-squared of the best run
    
    PARAMETERS
    """
    [Z0_0, I_S1_0, I_S2_0, BETA_0] = p0
    BOUNDS = MODEL_PARAMETERS[3]
    [L, ALPHA0, I0, Z_R] = EXPERIMENT_PARAM

    # Progress bar
    ## Create window
    root = tk.Tk()
    root.geometry('300x120')
    root.title('Progress bar')
    root.resizable(0,0)
    root.attributes('-topmost', True)
    ## Create and place progress bar
    pb = ttk.Progressbar(root, orient='horizontal', mode='determinate', length='280')
    pb.grid(column=0, row=0, columnspan=2, padx=10, pady=20)
    ## Initialise progress bar
    progress = 'Number of models computed: 0/' + str(N_RUNS) 
    value_label = ttk.Label(root, text=progress)
    value_label.grid(column=0, row=1, columnspan=2)
    root.update()


    # Run 1PA Model
    if FIT_TYPE == 0:
        ## Prepare model
        del BOUNDS[2:]
        MODEL_PARAMETERS[3] = BOUNDS
        RUNS = np.ndarray((N_RUNS,3))    # Intialise data structure
    
        ## Run model
        for i in range(N_RUNS):
            p0 = np.array([Z0_0, I_S1_0])    # Initial guess
            popt, chi2 = basinhopping.OPA(MEASUREMENT, p0, L, ALPHA0, I0, Z_R, MODEL_PARAMETERS)
            ## Store results
            RUNS[i, 0]=popt[0]
            RUNS[i, 1]=popt[1]
            RUNS[i,2] = chi2
            ## Update progress bar
            pb['value'] += 100 / N_RUNS    
            value_label.config(text= 'Number of models computed: ' + str(i+1) + '/' + str(N_RUNS))
            root.update()

        ## Find best results
        CHI2_BEST = np.amin(RUNS[:,2])
        index = np.where(RUNS[:,2]==CHI2_BEST)[0][0]
        P_BEST = RUNS[index, 0:2].reshape(-1)


    # Run 2PA no Is2
    elif FIT_TYPE == 1:
        ## Prepare model
        del BOUNDS[2]
        MODEL_PARAMETERS[3] = BOUNDS
        RUNS = np.ndarray((N_RUNS,4))    # Intialise data structure

        ## Run model
        for i in range(N_RUNS):
            p0 = np.array([Z0_0, I_S1_0, BETA_0])    # Initial guess
            popt, chi2 = basinhopping.TPA_no_Is2(MEASUREMENT, p0, L, ALPHA0, I0, Z_R, MODEL_PARAMETERS)    # Run model
            ## Store results
            RUNS[i, 0]=popt[0]
            RUNS[i, 1]=popt[1]
            RUNS[i, 2]=popt[2]
            RUNS[i,3] = chi2
            ## Update progress bar
            pb['value'] += 100 / N_RUNS    
            value_label.config(text= 'Number of models computed: ' + str(i+1) + '/' + str(N_RUNS))
            root.update()

        ## Find best results
        CHI2_BEST = np.amin(RUNS[:,3])
        index = np.where(RUNS[:,3]==CHI2_BEST)[0][0]
        P_BEST = RUNS[index, 0:3].reshape(-1)


    # Run 2PA
    elif FIT_TYPE == 2:
        ## Prepare model
        MODEL_PARAMETERS[3] = BOUNDS
        RUNS = np.ndarray((N_RUNS,5))    # Intialise data structure

        ## Run model
        for i in range(N_RUNS):
            p0 = np.array([Z0_0, I_S1_0, I_S2_0, BETA_0])    # Initial guess
            popt, chi2 = basinhopping.TPA(MEASUREMENT, p0, L, ALPHA0, I0, Z_R, MODEL_PARAMETERS)    # Run model
            ## Store results
            RUNS[i, 0]=popt[0]
            RUNS[i, 1]=popt[1]
            RUNS[i, 2]=popt[2]
            RUNS[i, 3]=popt[3]
            RUNS[i,4] = chi2
            ## Update progress bar
            pb['value'] += 100 / N_RUNS    
            value_label.config(text= 'Number of models computed: ' + str(i+1) + '/' + str(N_RUNS))
            root.update()
        
        ## Find best results
        CHI2_BEST = np.amin(RUNS[:,4])
        index = np.where(RUNS[:,4]==CHI2_BEST)[0][0]
        P_BEST = RUNS[index, 0:4].reshape(-1)


    # Run 2PA no Is1
    elif FIT_TYPE == 3:
        ## Prepare model
        del BOUNDS[1]
        MODEL_PARAMETERS[3] = BOUNDS
        RUNS = np.ndarray((N_RUNS,4))    # Intialise data structure

        ## Run model
        for i in range(N_RUNS):
            p0 = np.array([Z0_0, I_S2_0, BETA_0])    # Initial guess
            popt, chi2 = basinhopping.TPA_no_Is1(MEASUREMENT, p0, L, ALPHA0, I0, Z_R, MODEL_PARAMETERS)    # Run model
            ## Store results
            RUNS[i, 0]=popt[0]
            RUNS[i, 1]=popt[1]
            RUNS[i, 2]=popt[2]
            RUNS[i,3] = chi2
            ## Update progress bar
            pb['value'] += 100 / N_RUNS    
            value_label.config(text= 'Number of models computed: ' + str(i+1) + '/' + str(N_RUNS))
            root.update()

        ## Find best results
        CHI2_BEST = np.amin(RUNS[:,3])
        index = np.where(RUNS[:,3]==CHI2_BEST)[0][0]
        P_BEST = RUNS[index, 0:3].reshape(-1)


    # Run 2PA no sat
    elif FIT_TYPE == 4:
        ## Prepare model
        del BOUNDS[1:3]
        MODEL_PARAMETERS[3] = BOUNDS
        RUNS = np.ndarray((N_RUNS,3))    # Intialise data structure

        ## Run model
        for i in range(N_RUNS):
            p0 = np.array([Z0_0, BETA_0])    # Initial guess
            popt, chi2 = basinhopping.TPA_no_sat(MEASUREMENT, p0, L, ALPHA0, I0, Z_R, MODEL_PARAMETERS)    # Run model
            ## Store results
            RUNS[i, 0]=popt[0]
            RUNS[i, 1]=popt[1]
            RUNS[i,2] = chi2
            ## Update progress bar
            pb['value'] += 100 / N_RUNS    
            value_label.config(text= 'Number of models computed: ' + str(i+1) + '/' + str(N_RUNS))
            root.update()

        ## Find best results
        CHI2_BEST = np.amin(RUNS[:,2])
        index = np.where(RUNS[:,2]==CHI2_BEST)[0][0]
        P_BEST = RUNS[index, 0:2].reshape(-1)

    else:
        sys.exit('No valid fit type')

    # Close progress bar
    root.destroy()
    return RUNS, P_BEST, CHI2_BEST 