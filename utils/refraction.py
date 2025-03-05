import numpy as np
import streamlit as st
import tkinter as tk
from tkinter import filedialog
import os
import toml
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from datetime import datetime

def select(data_structure, type):
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
    if type == 'OA':
        try:
            st.session_state['OA'] = filedialog.askopenfilename(title='Select Data', initialdir=st.session_state['OA'], parent=root)
        except FileNotFoundError:
            st.session_state['OA'] = os.environ.get('HOMEPATH')
    else:
        try:
            st.session_state['CA'] = filedialog.askopenfilename(title='Select Data', initialdir=st.session_state['CA'], parent=root)
        except FileNotFoundError:
            st.session_state['CA'] = os.environ.get('HOMEPATH')
    root.destroy()

    if type == 'OA':
        data_structure.raw = np.loadtxt(st.session_state['OA'])
    else:
        data_structure.raw = np.loadtxt(st.session_state['CA'])
    data_structure.z = data_structure.raw[:, 0] * 1e-1  # cm
    data_structure.I = data_structure.raw[:, 1]
    data_structure.dI = data_structure.raw[:, 2]

def load_beam(data_structure):
        # Open Window
        root = tk.Tk()
        root.attributes('-topmost', True)
        root.withdraw()
        beam_directory = filedialog.askopenfilename(title='Select Config File', filetypes=[("TOML files", "*.toml")], parent=root)
        with open(beam_directory, 'r') as file:
            config = toml.load(file)
            data_structure.w0 = config['Beam Profile Fitting']['w0'][0]
            data_structure.zR = config['Beam Profile Fitting']['zR'][0]
        root.destroy()


class data_structure:
    def __init__(self):
        self.ui = {'L':.1, 'E':3.55, 'wavelength':532}
        self.p0 = {'z0': 0.0, 'alpha':40.0, 'n2':0.0}
        self.type = {'z0': False, 'alpha':False, 'n2':False}

        self.w0 = 10.    # um
        self.zR = 500.   # um
        self.plot_type = 'default'
        
        # Calculate I0 from values
        PULSE_WIDTH = 6e-9  # s
        P_laser = self.ui['E']*1e-6 / PULSE_WIDTH   # J/s
        self.I0 = 2*P_laser / (np.pi * (self.w0*1e-4)**2)*1e-9   # GW/cm2

    def update_I0(self):
        # Calculate I0 from values
        PULSE_WIDTH = 6e-9  # s
        P_laser = self.ui['E']*1e-6 / PULSE_WIDTH   # J/s
        self.I0 = 2*P_laser / (np.pi * (self.w0*1e-4)**2)*1e-9   # GW/cm2

def run(OA, CA):
    def model(z, *x):
        def intensity(z):
            return OA.I0 / (1 + ((z)/(OA.zR*1e-4))**2)  # Initial condition

        params = np.array(list(OA.p0.values()))
        params[list(OA.type.values())] = x[:]

        z0 = params[0]
        a0 = params[1]
        n2 = params[2]

        z = z - z0
        zR = OA.zR*1e-4
        k = 1e7/OA.ui['wavelength']   # wavenumber (cm-1)
        
        phi = n2*k*intensity(z)*(1-np.exp(a0*OA.ui['L']))/a0
        T = 1 + (4*phi*(z/zR))/(((z/zR)**2 + 9)*((z/zR)**2 + 1))
        return T
    
    params = np.array(list(OA.p0.values()))
    p0 = params[list(OA.type.values())]
    OA.pBest, OA.pcov = curve_fit(model, OA.z, CA.I / OA.I, p0=p0)

def plot(OA, CA, params):
    def model(z, x):
        def intensity(z):
            return OA.I0 / (1 + ((z)/(OA.zR*1e-4))**2)  # Initial condition

        z0 = x[0]
        a0 = x[1]
        n2 = x[2]

        z = z - z0
        zR = OA.zR*1e-4
        k = 1e7/OA.ui['wavelength']   # wavenumber (cm-1)
        
        phi = n2*k*intensity(z)*(1-np.exp(a0*OA.ui['L']))/a0
        T = 1 + (4*phi*(z/zR))/(((z/zR)**2 + 9)*((z/zR)**2 + 1))
        return T
    
    z_plot = np.linspace(OA.z[0], OA.z[-1], 1000)
    fig = plt.figure()

    plt.plot(OA.z, CA.I/OA.I, '.')
    plt.plot(z_plot, model(z_plot, params))

    plt.xlabel('z (cm)')
    plt.ylabel('T')

    OA.fig = fig

    return fig

def export(OA, params, errorbars):
    # Create export directory
    timeCode = datetime.now()
    export_folder = "/RESULTS_" + timeCode.strftime("%Y%m%d-%H%M%S")
    export_directory = os.path.dirname(st.session_state['OA']) + export_folder
    os.mkdir(export_directory)
    try:
        os.mkdir(export_directory)
    except:
        pass

    # Save images
    figz = OA.fig
    figz.savefig(export_directory + '/CA.png', bbox_inches='tight')

    # Z-Scan
    fitting_results = {
        'Z-Scan Fitting': {
            'z0': [params[0], errorbars[0]],
            'alpha0': [params[1], errorbars[1]],
            'n2': [params[2], errorbars[2]]
        }
    }

    toml_string = toml.dumps(fitting_results, encoder=toml.TomlNumpyEncoder())
    toml_lines = toml_string.split('\n')
    comments = [toml_lines[0],
                '# Observable   [Value, Std, Std Span]    Unit',
                f'{toml_lines[1]}   # cm',
                f'{toml_lines[2]}   # cm-1',
                f'{toml_lines[3]}   # GW-1']
    
    with open(export_directory + '/RESULTS_ZSCAN.toml', 'a') as f:
        f.write('\n'.join(comments))

    # Beam Profile
    fitting_results = {
        'Beam Profile Fitting': {
            'w0': OA.w0,
            'zR': OA.zR
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
    for attr, value in OA.__dict__.items():
        if not isinstance(value, st.delta_generator.DeltaGenerator):
            dictionary[str(attr)] = value

    # Convert the dictionary to a TOML string
    toml_string = toml.dumps({'Everything': dictionary}, encoder=toml.TomlNumpyEncoder())

    # Write the TOML string to the file
    with open(export_directory + '/RESULTS_ZSCAN.toml', 'a') as f:
        f.write(toml_string)
    