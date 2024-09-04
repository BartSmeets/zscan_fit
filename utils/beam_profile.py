import streamlit as st
from scipy.optimize import curve_fit   
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import toml
from datetime import datetime
import os

gaussian = lambda x, a, b, c: a * np.exp(-(x-b)**2 / (2 * c**2))

def width(z, w0: float, z0: float, M2: float, wavelength: float):
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

def gaussian_fit(all_files: list) -> None:
    '''
    Fit a Gaussian beam profile and saves the results in st.session_state:
    - measurements: list containing the data objects
    - w: an array containing the width of the beam at each measurement
    - sigma_w: the std of the width

    ## Argument:
    - all_files: list containing all file names
    '''
    class data:
        def __init__(self, measurement):
            self.x = measurement[:, 0]
            self.Ix = measurement[:, 1] / 100
            self.y = measurement[:, 2]
            self.Iy = measurement[:, 3] / 100

        def fit(self, p0x, p0y):         
            self.x_fit, _ = curve_fit(gaussian, self.x, self.Ix, p0x, bounds=[[0,-np.inf,0],[np.inf,np.inf,np.inf]])
            self.y_fit, _ = curve_fit(gaussian, self.y, self.Iy, p0y, bounds=[[0,-np.inf,0],[np.inf,np.inf,np.inf]])

            self.sigma_x = np.sqrt(self.x_fit[2]/(2*np.sqrt(2*np.pi)*self.x_fit[0]))
            self.sigma_y = np.sqrt(self.y_fit[2]/(2*np.sqrt(2*np.pi)*self.y_fit[0]))
            # Calculate Beam width (e^-2) as average x-width and y-width 
            self.wx = 2* abs(self.x_fit[2])    # e^-2 definition beam width
            self.wy = 2* abs(self.y_fit[2])

    measurement_lst = []
    w = np.zeros(len(all_files))    # First column for beam widths; Second column for error on beam widths
    sigma_w = np.zeros(len(w))    # First column for beam widths; Second column for error on beam widths
    p0x = p0y = [1,0,100]    # First guess for fitting

    for i, file in enumerate(all_files):
        measurement = np.loadtxt(file, skiprows=11)
        loaded_data = data(measurement)
        if i in st.session_state['exclude']:
            measurement_lst.append(loaded_data)
            continue
        
        loaded_data.fit(p0x, p0y)
        
        # Check if Gaussian fit has failed
        ## If Gaussian fit has failed only consider wx or wy
        if loaded_data.x_fit[0]<0.1:
            w[i] = loaded_data.wy
            sigma_w[i] = 2*loaded_data.sigma_y
            p0y[2] = p0x[2] = loaded_data.y_fit[2]
        elif loaded_data.y_fit[0]<0.1:
            w[i] = loaded_data.wx
            sigma_w[i] = 2*loaded_data.sigma_x
            p0x[2] = p0y[2] = loaded_data.x_fit[2]
        ## If Gaussian fit has not failed, map ellipse to circle with equal area           
        else:
            w[i] = np.sqrt(loaded_data.wx*loaded_data.wy)    # Width 
            sigma_w[i] = np.sqrt(loaded_data.wy**2 / (loaded_data.wx*loaded_data.wy) * loaded_data.sigma_x**2 
                            + loaded_data.wx**2 / (loaded_data.wx*loaded_data.wy) * loaded_data.sigma_y**2)    # Std
        measurement_lst.append(loaded_data)

    st.session_state['measurements'] = measurement_lst
    st.session_state['w'] = w
    st.session_state['sigma_w'] = sigma_w
    return
    

def fig_gaussian(all_files):
    '''Generate the figure of the Gaussian fit
    
    ## Argument:
    - all_files: list containing all file names

    ## Returns:
    - fig: matplotlib figure object
    '''
    
    fig, ax = plt.subplots(len(all_files), 2, figsize=(10,3*len(all_files)), tight_layout=True, sharey=True)

    for i, measurement in enumerate(st.session_state['measurements']):
        if i in st.session_state['exclude']:
            continue
        # Plot sub-figure
        x_linspace = np.linspace(measurement.x[0],measurement.x[-1], 1000)
        y_linspace = np.linspace(measurement.y[0],measurement.y[-1], 1000)
        ## Plot x axis
        ax[i, 0].plot(measurement.x, measurement.Ix, 'o', label='Data', color='#008176')
        ax[i, 0].plot(x_linspace, gaussian(x_linspace, *measurement.x_fit), label='Fit', color='#c1272c')
        ax[i, 0].set_xlabel(f'x$_{{{i}}}$ (μm)', fontsize=15)
        ax[i, 0].set_ylabel('Normalised Intensity', fontsize=15)
        ## Plot y axis
        ax[i, 1].plot(measurement.y, measurement.Iy, 'o', label='Data', color='#008176')
        ax[i, 1].plot(y_linspace, gaussian(y_linspace, *measurement.y_fit), label='Fit', color='#c1272c')
        ax[i, 1].set_xlabel(f'y$_{{{i}}}$ (μm)', fontsize=15)

        plt.xticks(fontsize=12.5)
        plt.yticks(fontsize=12.5)
    
    st.session_state['gaussian_fig'] = fig
    return fig

def bp_fit():
    wavelength = st.session_state['wavelength']
    w = st.session_state['w']
    z = st.session_state['step_size'] * np.array(range(len(w)))    # unit mm
    mask = (w!=0)

    w = w[mask]
    z = z[mask]
    sigma_w = st.session_state['sigma_w'][mask]

    w_param, w_var  = curve_fit(lambda x, w0, z0, M2: width(x, w0, z0, M2, wavelength), z, w, [10,10e3,1], 
                     sigma=sigma_w, bounds=([0,-np.inf,1],[np.inf,np.inf,np.inf]), absolute_sigma=True)
    [w0, z0, M2] =  w_param
    zR = np.pi*(w0**2) / (M2*wavelength*1e-3)    # Rayleigh length

    # Error propagation
    sigma_w = np.sqrt(np.diag(w_var)[0])    # Error on beam waist
    sigma_z = np.sqrt(np.diag(w_var)[1])*1e-3    # Error on z-position of focal point
    sigma_M2 = np.sqrt(np.diag(w_var)[2])    # Error on M^2
    dz_dw = (2*np.pi*10) / (M2*wavelength*1e-3)
    dz_dM2 = (np.pi*(10**2)) / (wavelength*1e-3*(M2**2))
    sigma_zR = np.sqrt((dz_dw**2)*(sigma_w**2) + (dz_dM2**2)*(sigma_M2**2))    # Error on Rayleigh length

    st.session_state['w0'] = [w0, sigma_w]
    st.session_state['z0'] = [z0, sigma_z]
    st.session_state['zR'] = [zR, sigma_zR]
    st.session_state['M2'] = [M2, sigma_M2]

def fig_bp():
    fig = plt.figure()
    ax2 = plt.axes()

    w = st.session_state['w']
    z = st.session_state['step_size'] * np.array(range(len(w)))    # unit mm

    mask = (w!=0)
    w = w[mask]
    z = z[mask]

    [z0, _] = st.session_state['z0']
    [w0, _] = st.session_state['w0']
    [M2, _] = st.session_state['M2']
    z_plot = np.linspace(0, z[-1], 500)

    ### Plot
    plt.plot(z-z0*1e-3, w,'.', label='Data', color='#008176')
    plt.plot(z_plot-z0*1e-3, width(z_plot, w0, z0, M2, st.session_state['wavelength']), 
             label='Fit', color='#c1272c')
    #plt.text(0,10,textstr)
    ### Labels

    plt.legend()
    plt.ylabel('Beam width (μm)', fontsize=15)
    plt.xlabel('z - z$_0$ (mm)', fontsize=15)

    plt.xticks(fontsize=12.5)
    plt.yticks(fontsize=12.5)

    ax2.xaxis.set_minor_locator(AutoMinorLocator())
    ax2.yaxis.set_minor_locator(AutoMinorLocator())

    st.session_state['bp_fig'] = fig
    return fig


def export():
    # Create export directory
    timeCode = datetime.now()
    export_directory = st.session_state['profile_directory'] + "/OUTPUT_BEAM_PROFILE_"  + timeCode.strftime("%Y%m%d-%H%M")
    try:
        os.mkdir(export_directory)
    except:
        pass

    fitting_results = {
        'Gaussian Fitting': {
            'w0': st.session_state['w0'],
            'z0': np.array(st.session_state['z0'])*1e-3,
            'zR': st.session_state['zR'],
            'M2': st.session_state['M2']
        }
    }

    toml_string = toml.dumps(fitting_results)
    toml_lines = toml_string.split('\n')
    comments = [toml_lines[0],
                '# Observable   [Value, Std]    Unit',
                f'{toml_lines[1]}   # um',
                f'{toml_lines[2]}   # mm',
                f'{toml_lines[3]}   # um',
                toml_lines[4]]

    with open(export_directory + '/RESULTS_BEAM_PROFILE.toml', 'w') as f:
        f.write('\n'.join(comments))
    st.session_state['gaussian_fig'].savefig(export_directory + "/OUTPUT_WIDTHS.png", bbox_inches='tight')
    st.session_state['bp_fig'].savefig(export_directory + "/OUTPUT_BEAM_PROFILE.png", bbox_inches='tight')