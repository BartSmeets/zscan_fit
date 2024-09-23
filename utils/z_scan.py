import tkinter as tk
from tkinter import filedialog
import os
import toml
import numpy as np
import streamlit as st
import glob
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

class data_structure:
    def __init__(self):
        self.folder = os.environ.get('HOMEPATH')
        self.ui = {'L':None, 'alpha0':None, 'E':None}
        self.type = {'z0': False, 'Is1':False, 'Is2':False, 'beta':False}
        self.p0 = {'z0': np.nan, 'Is1':np.nan, 'Is2':np.nan, 'beta':np.nan}
        self.bounds = {'z0': [None, None], 'Is1':[1e-99, None], 'Is2':[1e-99, None], 'beta':[0, None]}
        self.model_parameters = {'Number Runs': 5, 'Max Perturbation': 2, 'Max Iterations': 500, 'Max Age': 50, 'T':0.8, 'Max Jump':5, 'Max Reject':5}
        self.w0 = np.nan
        self.zR = np.nan
    
    def select(self):
        '''
        Opens a window to select the files you want to load

        ## Generates:
        - self.directory: list with the directories of the selected files
        - self.folder: folder where the files are located
        - self.names: list of the file names
        '''

        # Open Window
        root = tk.Tk()
        root.attributes('-topmost', True)
        root.withdraw()
        st.session_state['data_directory'] = filedialog.askopenfilename(title='Select Data', initialdir=st.session_state['data_directory'], parent=root)
        root.destroy()

        self.raw = np.loadtxt(st.session_state['data_directory'])
        self.z = self.raw[:, 0]
        self.I = self.raw[:, 1]
        self.dI = self.raw[:, 2]

    
    def load_beam(self):
        # Open Window
        root = tk.Tk()
        root.attributes('-topmost', True)
        root.withdraw()
        self.beam_directory = filedialog.askopenfilename(title='Select Config File', initialdir=self.folder, filetypes=[("TOML files", "*.toml")], parent=root)
        root.destroy()
        
        with open(self.beam_directory, 'r') as file:
            config = toml.load(file)
        self.w0 = config['Beam Profile Fitting']['w0'][0]
        self.zR = config['Beam Profile Fitting']['zR'][0]

    def run(self):
        # Calculate I0 from values
        PULSE_WIDTH = 6e-9  # s
        P_laser = self.ui['E'] / PULSE_WIDTH
        self.I0 = 2*P_laser / (np.pi * self.w0**2) * 100  # W/cm2

        self.runs = np.ndarray((self.model_parameters['Number Runs'],2))    # Intialise data structure

        ## Run model
        for i in range(self.model_parameters['Number Runs']):
            self.runs[i, 0], self.runs[i, 1] = bassinhopping(self)
        
        ## Find best results
        index = np.argmin(self.runs[:,1])
        self.pBest = self.runs[index, 0]
        self.chi2Best = self.runs[index, 1]
        st.success(str(self.pBest))
        
    

def bassinhopping(df):
    def dI_dz(z, I, x0, x1, x2, x3):
            term1 = - df.ui['alpha0'] / (1 + (I/x1)) * I
            term2 = - x3 / (1 + (I/x2)) * I**2
            return term1 + term2

    def transmittance(x):
        I_in  = df.I0 / (1 + ((df.z - x[0])/df.zR)**2)  # Initial condition
        sol = solve_ivp(dI_dz, [0, df.ui['L']], I_in, args=x, t_eval=[df.ui['L']])    # Runge-Kutta 45 method
        I_out = sol.y[:, 0]
        transmittance = (I_out / I_in) / (I_out[0] / I_in[0])
        return transmittance

    def chi2(x):
        I_calc = transmittance(x)

        ## Calculate chi2
        if 0 in df.dI:    # Poorly defined uncertainty
            chi2Vector = (df.I- I_calc)**2
        else:
            chi2Vector = (df.I - I_calc)**2 / np.average(df.dI)**2

        return np.sum(chi2Vector)
    
    def fitting_model(x):
        na_option = np.array([0, np.inf, np.inf, 0])
        na_option[list(df.type.values())] = x[:]

        return chi2(na_option)
        
    ## Set initial chi2
    mask = list(df.type.values())
    p0 = np.array(list(df.p0.values()))[mask]
    bounds = np.array(list(df.bounds.values()))[mask]
    popt = minimize(fitting_model, x0=p0, bounds=bounds)
    pBest = pMin = popt.x
    chi2Prev = chi2Best = fitting_model(pMin)

    ## Start minimalisation process
    ### prepare iteration process
    niter = 0
    rejection = 0
    jump = 0
    bestAge = -1

    ### Iteration process
    while niter < df.model_parameters['Max Iterations'] and bestAge < df.model_parameters['Max Age']:
        niter += 1
        bestAge += 1

        #### Monte Carlo Move
        pPerturbation = list(pMin)
        for i, p in enumerate(pMin):
            perturbation = np.random.uniform(1, df.model_parameters['Max Perturbation'])    # Determine perturbation strength
            direction = np.random.choice(['decrease', 'increase'])    # Determine perturbation direction
            if direction == 'decrease':
                pPerturbation[i] = p / perturbation
            else:
                pPerturbation[i] = p * perturbation

        #### Minimise Chi2
        popt = minimize(fitting_model, x0=pPerturbation, bounds=bounds)
        p_newMin = popt.x
        X2 = fitting_model(p_newMin)

        #### Metropolis criterion
        ##### Accept perturbation
        ###### Jumping -> accept perturbation regardless of conditions
        if jump != 0:
            chi2Prev = X2
            pMin = p_newMin
            jump += 1
            ## Check if result is overall best
            if X2 < chi2Best:    
                pBest = p_newMin
                chi2Best = X2
                bestAge = 0
            ## End condition for jumping
            if jump == df.model_parameters['Max Jump']:    
                jump = 0
        ###### Better chi2 than previous chi2
        elif X2 < chi2Prev:    
            chi2Prev = X2
            pMin = p_newMin
            rejection = 0
            ## Check if result is overall best            
            if X2 < chi2Best:    
                pBest = p_newMin
                chi2Best = X2
                bestAge = 0
        ###### Metropolis
        elif np.random.uniform(0,1) < np.exp(-(X2 - chi2Prev)/df.model_parameters['T']):
            chi2Prev = X2
            pMin = p_newMin
            rejection = 0

        ##### Reject perturbation
        else:
            rejection += 1

        ##### Max Rejections achieved? -> Start jumping
        if rejection == df.model_parameters['Max Reject']:
            # Accept perturbation
            chi2Prev = X2
            pMin = p_newMin
            rejection = 0
            jump += 1
        
    return pBest, chi2Best