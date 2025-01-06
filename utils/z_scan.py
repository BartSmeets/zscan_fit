import tkinter as tk
from tkinter import filedialog
import os
import toml
import numpy as np
import streamlit as st
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from datetime import datetime

class data_structure:
    def __init__(self):
        self.folder = os.environ.get('HOMEPATH')
        self.ui = {'L':.1, 'alpha0':44.67, 'E':3.55}
        self.type = {'z0': False, 'Is1':False, 'Is2':False, 'beta':False, 'alpha':False}
        self.p0 = {'z0': 0.0, 'Is1':0.1, 'Is2':1e308, 'beta':0.0, 'alpha':40}
        self.bounds = {'z0': [-1.797e308, 1.797e308], 'Is1':[1.797e-308, 1.797e308], 'Is2':[1.797e-308, 1.797e308], 'beta':[0, 1.797e308], 'alpha':[0, 1.797e308]}
        self.model_parameters = {'Number Runs': 5, 'Max Perturbation': 2, 'Max Iterations': 500, 'Max Age': 50, 'T':0.8, 'Max Jump':5, 'Max Reject':5}
        self.w0 = 10.    # um
        self.zR = 500.   # um
        self.plot_type = 'default'

        # Calculate I0 from values
        PULSE_WIDTH = 6e-9  # s
        P_laser = self.ui['E']*1e-6 / PULSE_WIDTH   # J/s
        self.I0 = 2*P_laser / (np.pi * (self.w0*1e-4)**2)*1e-9   # GW/cm2
        

    def run(self):
        # Initialise
        self.ps = list()
        self.chi2 = list()

        ## Run model
        for run in range(self.model_parameters['Number Runs']):
            self.progress_update(run, 0, 0)
            p, chi2 = self.bassinhopping(progress=run)
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
    

    def bassinhopping(self, progress):
        def chi2(x):
            I_calc = self.transmittance(x)
            print('In chi2: transmittance calculated')

            ## Calculate chi2
            if 0 in self.dI:    # Poorly defined uncertainty
                chi2Vector = (self.I- I_calc)**2
            else:
                chi2Vector = (self.I - I_calc)**2 / np.average(self.dI)**2

            result = np.sum(chi2Vector)
            return result
        
        def fitting_model(x):
            na_option = np.array(list(self.p0.values()))
            na_option[list(self.type.values())] = x[:]

            return chi2(na_option)
            
        ## Set initial chi2
        mask = list(self.type.values())
        p0 = np.array(list(self.p0.values()))[mask]
        bounds = np.array(list(self.bounds.values()))[mask]
        print('0, 0, minimize')
        popt = minimize(fitting_model, x0=p0, bounds=bounds)
        pBest = pMin = popt.x
        print('0, 0, chi2')
        chi2Prev = chi2Best = fitting_model(pMin)

        ## Start minimalisation process
        ### prepare iteration process
        niter = 0
        rejection = 0
        jump = 0
        bestAge = -1

        ### Iteration process
        while niter < self.model_parameters['Max Iterations'] and bestAge < self.model_parameters['Max Age']:
            niter += 1
            bestAge += 1
            self.progress_update(progress, niter, bestAge)

            #### Monte Carlo Move
            pPerturbation = list(pMin)
            for i, p in enumerate(pMin):
                perturbation = np.random.uniform(1, self.model_parameters['Max Perturbation'])    # Determine perturbation strength
                direction = np.random.choice(['decrease', 'increase'])    # Determine perturbation direction
                if direction == 'decrease':
                    pPerturbation[i] = p / perturbation
                else:
                    pPerturbation[i] = p * perturbation

            #### Minimise Chi2
            print(f'{niter}, {bestAge}, minimize')
            popt = minimize(fitting_model, x0=pPerturbation, bounds=bounds)
            p_newMin = popt.x
            print(f'{niter}, {bestAge}, chi2')
            X2 = fitting_model(p_newMin)

            #### Metropolis criterion
            ##### Accept perturbation
            ###### Jumping -> accept perturbation regardless of conditions
            print(f'{niter}, {bestAge}, check for jumping')
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
                if jump == self.model_parameters['Max Jump']:    
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
            elif np.random.uniform(0,1) < np.exp(-(X2 - chi2Prev)/self.model_parameters['T']):
                chi2Prev = X2
                pMin = p_newMin
                rejection = 0

            ##### Reject perturbation
            else:
                rejection += 1

            ##### Max Rejections achieved? -> Start jumping
            if rejection == self.model_parameters['Max Reject']:
                # Accept perturbation
                chi2Prev = X2
                pMin = p_newMin
                rejection = 0
                jump += 1
            
        return pBest, chi2Best

    def transmittance(self, x, z=None):
        if z is None:
            I_in  = self.I0 / (1 + ((self.z - x[0])/(self.zR*1e-4))**2)  # Initial condition
        else:
            I_in  = self.I0 / (1 + ((z - x[0])/(self.zR*1e-4))**2)  # Initial condition
        model = lambda z, I: self.dI_dz(z, I, *x)
        sol = solve_ivp(model, [0, self.ui['L']], I_in, t_eval=[self.ui['L']])    # Runge-Kutta 45 method
        print('In transmittance: differential equation solved')
        try:
            I_out = sol.y[:, 0]
        except TypeError:
            transmittance = 1e9
        else:
            transmittance = (I_out / I_in) / (I_out[0] / I_in[0])
        return transmittance
    
    def dI_dz(self, z, I, x0, x1, x2, x3, x4):
                term1 = - x4 / (1 + (I/x1)) * I
                term2 = - x3 / (1 + (I/x2)) * I**2
                return term1 + term2
    

    def errorbar(self):
        # Initialise datastructure
        N_TOT = 5   # Total amount of parameters (if all in use)
        MAX_ITER = 1e3  # Total number of allowed iterations
        STEP = 0.005    # Perturbation step size

        p = 0.1 # Threshold: 10% deviation from optimal chi2
        CHI2_COMPARISON = (1+p) * self.chi2Best
        self.errorbars = np.zeros(N_TOT)
        self.chi2span = np.zeros(N_TOT)
        na_option = np.array(self.na_option)


        for i in range(N_TOT):
            if not list(self.type.values())[i]:
                continue

            lBound = [na_option[i], 0]  # Index 0: Parameter adjusted up; Index 1: Corresponding chi2
            rBound = [na_option[i], 0]  # Index 0: Parameter adjusted down; Index 1: Corresponding chi2
            lTest = np.array(self.pBest)
            rTest = np.array(self.pBest)

            # Error calculation
            newChi2 = 0
            iteration = 0
            while newChi2 < CHI2_COMPARISON and iteration < MAX_ITER:
                iteration += 1
                
                ## Perturb parameter for testing
                rBound[0] *= (1+STEP)
                lBound[0] *= (1-STEP)
                rTest = rBound[0]
                lTest = lBound[0]

                ## Calculate chi2
                def chi2(test):
                    na_option = np.array(self.na_option)
                    na_option[i] = test
                    I_calc = self.transmittance(na_option)

                    if 0 in self.dI:    # Poorly defined uncertainty
                        chi2 = np.sum((self.I - I_calc)**2)
                    else:
                        chi2 = np.sum((self.I - I_calc)**2 / np.average(self.dI)**2)
                    return chi2

                rBound[1] = chi2(rTest)
                lBound[1] = chi2(lTest)
                newChi2 = max(lBound[1], rBound[1])

            # Save
            if rBound[1] > lBound[1]:
                self.errorbars[i] = np.abs(rBound[0] - na_option[i])
                self.chi2span[i] = rBound[1]
            else:
                self.errorbars[i] = np.abs(lBound[0] - na_option[i])
                self.chi2span[i] = lBound[1]


    def plot(self, na_option):
        figure = plt.figure()

        try:
            z_plot = np.linspace(self.z[0], self.z[-1], 1000)
        except AttributeError:
            pass
        else:
            if self.plot_type == 'default':
                plt.plot(self.z, self.I, '.')
                plt.xlabel('z (cm)')
                plt.ylabel('T')
            else:
                I_in  = self.I0 / (1 + ((self.z - na_option[0])/(self.zR*1e-4))**2)  # Initial condition
                plt.plot(I_in, self.I, '.')
                plt.xlabel('I$_{in}$ (GW/cm$^2$)')
                plt.ylabel('T')
            

        if self.plot_type == 'default':
            if hasattr(self, 'ps'):
                plt.plot(z_plot, self.transmittance(na_option, z_plot))
        
        if self.plot_type == 'intensity':
            if hasattr(self, 'ps'):
                I_in  = self.I0 / (1 + ((z_plot - na_option[0])/(self.zR*1e-4))**2)  # Initial condition
                plt.plot(I_in, self.transmittance(na_option, z_plot))

        return figure

    def plot_all(self):
        try:
            number = len(self.ps)
        except AttributeError:
            return plt.figure()

        fig, ax = plt.subplots(number, 1, figsize=(10,3*number))
        z_plot = np.linspace(self.z[0], self.z[-1], 1000)

        for i, popt in enumerate(self.ps):
            na_option = np.array(list(self.p0.values()))
            na_option[list(self.type.values())] = popt[:]

            string = f"""
                    z$_0$ = {na_option[0]:.3f} cm\\
                    Is$_1$ = {na_option[1]:.3e} GW/cm$^{2}$\\
                    Is$_2$ = {na_option[2]:.3e} GW/cm$^{2}$\\
                    β = {na_option[3]:.3e} cm/GW
                    """

            ax[i].plot(self.z, self.I, '.')
            ax[i].plot(z_plot, self.transmittance(na_option, z_plot))
            ax[i].text(5,1.1, string)
        return fig
    
    def export(self):
        # Create export directory
        timeCode = datetime.now()
        export_folder = "/RESULTS_" + timeCode.strftime("%Y%m%d-%H%M%S")
        export_directory = os.path.dirname(st.session_state['data_directory']) + export_folder
        os.mkdir(export_directory)
        try:
            os.mkdir(export_directory)
        except:
            pass    
        
        # Save images
        temp = self.plot_type
        self.plot_type = 'default'
        figz = self.plot(self.na_option)
        figz.savefig(export_directory + '/RESULT_Z.png', bbox_inches='tight')
        self.plot_type = 'intensity'
        figi = self.plot(self.na_option)
        figi.savefig(export_directory + '/RESULT_I.png', bbox_inches='tight')
        self.plot_type = temp
        
        # Z-Scan
        fitting_results = {
            'Z-Scan Fitting': {
                'z0': [self.na_option[0], self.errorbars[0], self.chi2span[0]],
                'Is1': [self.na_option[1], self.errorbars[1], self.chi2span[1]],
                'Is2': [self.na_option[2], self.errorbars[2], self.chi2span[2]],
                'beta': [self.na_option[3], self.errorbars[3], self.chi2span[3]]
            }
        }

        toml_string = toml.dumps(fitting_results, encoder=toml.TomlNumpyEncoder())
        toml_lines = toml_string.split('\n')
        comments = [toml_lines[0],
                    '# Observable   [Value, Std, Std Span]    Unit',
                    f'{toml_lines[1]}   # cm',
                    f'{toml_lines[2]}   # GW/cm2',
                    f'{toml_lines[3]}   # GW/cm2',
                    f'{toml_lines[4]}   # cm/GW']
        
        with open(export_directory + '/RESULTS_ZSCAN.toml', 'a') as f:
            f.write('\n'.join(comments))

        # Beam Profile
        fitting_results = {
            'Beam Profile Fitting': {
                'w0': self.w0,
                'zR': self.zR
            }
        }

        toml_string = toml.dumps(fitting_results, encoder=toml.TomlNumpyEncoder())
        toml_lines = toml_string.split('\n')
        comments = ['', '', toml_lines[0],
                    '# Observable   Value    Unit',
                    f'{toml_lines[1]}   # um',
                    f'{toml_lines[2]}   # um', '', '']
        
        with open(export_directory + '/RESULTS_ZSCAN.toml', 'a') as f:
            f.write('\n'.join(comments))

        # Everything
        dictionary = dict()
        for attr, value in self.__dict__.items():
            if not isinstance(value, st.delta_generator.DeltaGenerator):
                dictionary[str(attr)] = value

        # Convert the dictionary to a TOML string
        toml_string = toml.dumps({'Everything': dictionary}, encoder=toml.TomlNumpyEncoder())

        # Write the TOML string to the file
        with open(export_directory + '/RESULTS_ZSCAN.toml', 'a') as f:
            f.write(toml_string)

def select(data_structure):
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

    data_structure.raw = np.loadtxt(st.session_state['data_directory'])
    data_structure.z = data_structure.raw[:, 0] * 1e-1  # cm
    data_structure.I = data_structure.raw[:, 1]
    data_structure.dI = data_structure.raw[:, 2]


def load_beam(data_structure):
        # Open Window
        root = tk.Tk()
        root.attributes('-topmost', True)
        root.withdraw()
        data_structure.beam_directory = filedialog.askopenfilename(title='Select Config File', initialdir=data_structure.folder, filetypes=[("TOML files", "*.toml")], parent=root)
        with open(data_structure.beam_directory, 'r') as file:
            config = toml.load(file)
            data_structure.w0 = config['Beam Profile Fitting']['w0'][0]
            data_structure.zR = config['Beam Profile Fitting']['zR'][0]
        root.destroy()

        

    
            
