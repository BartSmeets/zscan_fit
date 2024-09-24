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
        self.ui = {'L':.1, 'alpha0':44.67, 'E':3.55}
        self.type = {'z0': False, 'Is1':False, 'Is2':False, 'beta':False}
        self.p0 = {'z0': 0.0, 'Is1':1e6, 'Is2':1e20, 'beta':0.0}
        self.bounds = {'z0': [-1.797e308, 1.797e308], 'Is1':[1.797e-308, 1.797e308], 'Is2':[1.797e-308, 1.797e308], 'beta':[0, 1.797e308]}
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
        try:
            st.session_state['data_directory'] = filedialog.askopenfilename(title='Select Data', initialdir=st.session_state['data_directory'], parent=root)
        except FileNotFoundError:
            st.session_state['data_directory'] = os.environ.get('HOMEPATH')
        root.destroy()

        self.raw = np.loadtxt(st.session_state['data_directory'])
        self.z = self.raw[:, 0] * 1e-1  # cm
        self.I = self.raw[:, 1]
        self.dI = self.raw[:, 2]

    
    def load_beam(self):
        # Open Window
        root = tk.Tk()
        root.attributes('-topmost', True)
        root.withdraw()
        try:
            self.beam_directory = filedialog.askopenfilename(title='Select Config File', initialdir=self.folder, filetypes=[("TOML files", "*.toml")], parent=root)
        except FileNotFoundError:
            self.beam_directory = os.environ.get('HOMEPATH')
        root.destroy()
        
        with open(self.beam_directory, 'r') as file:
            config = toml.load(file)
        self.w0 = config['Beam Profile Fitting']['w0'][0]
        self.zR = config['Beam Profile Fitting']['zR'][0]

    def run(self):
        # Calculate I0 from values
        PULSE_WIDTH = 6e-9  # s
        P_laser = self.ui['E']*1e-6 / PULSE_WIDTH   # J/s
        self.I0 = 2*P_laser / (np.pi * (self.w0*1e-4)**2)   # W/cm2

        # Initialise
        self.ps = list()
        self.chi2 = list()

        ## Run model
        for run in range(self.model_parameters['Number Runs']):
            self.progress_update(run, 0, 0)
            p, chi2 = bassinhopping(self, progress=run)
            self.ps.append(p)
            self.chi2.append(chi2)
        
        ## Find best results
        index = np.argmin(self.chi2)
        self.pBest = self.ps[index]
        self.chi2Best = self.chi2[index]


    def progress_update(self, N, I, B):
        string = f"""
            Number of models computed: {N}/{self.model_parameters['Number Runs']}
            Iteration: {I}/{self.model_parameters['Max Iterations']}
            Best age: {B}/{self.model_parameters['Max Age']}"""
        self.textbox.text(string)
        percent = float(N/self.model_parameters['Number Runs'])
        self.bar.progress(percent)
    

def bassinhopping(df, progress):
    def chi2(x):
        I_calc = transmittance(df, x)

        ## Calculate chi2
        if 0 in df.dI:    # Poorly defined uncertainty
            chi2Vector = (df.I- I_calc)**2
        else:
            chi2Vector = (df.I - I_calc)**2 / np.average(df.dI)**2

        return np.sum(chi2Vector)
    
    def fitting_model(x):
        na_option = np.array(list(df.p0.values()))
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
        df.progress_update(progress, niter, bestAge)

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

def dI_dz(z, I, df, x0, x1, x2, x3):
            term1 = - df.ui['alpha0'] / (1 + (I/x1)) * I
            term2 = - x3 / (1 + (I/x2)) * I**2
            return term1 + term2

def transmittance(df, x, z=None):
        if z is None:
            I_in  = df.I0 / (1 + ((df.z - x[0])/(df.zR*1e-4))**2)  # Initial condition
        else:
            I_in  = df.I0 / (1 + ((z - x[0])/(df.zR*1e-4))**2)  # Initial condition
        print('start solve')
        sol = solve_ivp(dI_dz, [0, df.ui['L']], I_in, args=(df,*x), t_eval=[df.ui['L']])    # Runge-Kutta 45 method
        print('finished solve')
        try:
            I_out = sol.y[:, 0]
        except TypeError:
            transmittance = 1e9
        else:
            transmittance = (I_out / I_in) / (I_out[0] / I_in[0])
        print(transmittance)
        return transmittance