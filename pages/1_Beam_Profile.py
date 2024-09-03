import streamlit as st
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog
import glob

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

if 'wavelength' not in st.session_state:
    st.session_state['wavelength'] = 532.    # nm
if 'step_size' not in st.session_state:
    st.session_state['step_size'] = 1.   # mm
if 'profile_directory' not in st.session_state:
    st.session_state['profile_directory'] = os.environ.get('HOMEPATH')
if 'measurements' not in st.session_state:
    st.session_state['measurements'] = []
if 'w' not in st.session_state:
    st.session_state['w'] = 0
if 'sigma_w' not in st.session_state:
    st.session_state['sigma_w'] = 0

# User inputs
def select_folder():
    root = tk.Tk()
    root.attributes('-topmost', True)
    root.withdraw()
    st.session_state['profile_directory'] = filedialog.askdirectory(title='Select Directory', initialdir=st.session_state['profile_directory'], parent=root)
    root.destroy()

with st.container(border = True):
    st.header('User Inputs')

    # Directory
    st.write('Directory')
    col1, col2 = st.columns([5,1])
    with col1:
        st.session_state['profile_directory'] = st.text_input('Directory', value=st.session_state['profile_directory'], label_visibility='collapsed')
    with col2:
        st.button('Browse', on_click=select_folder)
    all_files = glob.glob(st.session_state['profile_directory'] + '/Data_*.txt') # Load all files having a specific name format within the working directory)
    # Values
    col1, col2 = st.columns(2)
    with col1:
        st.number_input('Wavelength (nm)', min_value=0., key='wavelength')
    with col2:
        st.number_input('Step Size (mm)', min_value=0., key='step_size')

def gaussian_fit():
    with st.spinner():
        class data:
            def __init__(self, measurement):
                self.x = measurement[:, 0]
                self.Ix = measurement[:, 1] / 100
                self.y = measurement[:, 2]
                self.Iy = measurement[:, 3] / 100

            def fit(self, p0x, p0y):
                from scipy.optimize import curve_fit            
                self.x_fit, _ = curve_fit(gaussian, self.x, self.Ix, p0x, bounds=[[0,-np.inf,0],[np.inf,np.inf,np.inf]])
                self.y_fit, _ = curve_fit(gaussian, self.y, self.Iy, p0y, bounds=[[0,-np.inf,0],[np.inf,np.inf,np.inf]])

                self.sigma_x = np.sqrt(self.x_fit[2]/(2*np.sqrt(2*np.pi)*self.x_fit[0]))
                self.sigma_y = np.sqrt(self.y_fit[2]/(2*np.sqrt(2*np.pi)*self.y_fit[0]))
                # Calculate Beam width (e^-2) as average x-width and y-width 
                self.wx = 2* abs(self.x_fit[2])    # e^-2 definition beam width
                self.wy = 2* abs(self.y_fit[2])

        print(all_files)
        measurement_lst = []
        w = np.zeros(len(all_files))    # First column for beam widths; Second column for error on beam widths
        sigma_w = np.zeros(len(all_files))    # First column for beam widths; Second column for error on beam widths
        p0x = p0y = [1,0,100]    # First guess for fitting

        for i, file in enumerate(all_files):
            measurement = np.loadtxt(file, skiprows=11)
            print(measurement)
            loaded_data = data(measurement)
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

def figure(): 
    import matplotlib.pyplot as plt 
    fig, ax = plt.subplots(len(all_files), 2, figsize=(10,3*len(all_files)), tight_layout=True)

    for i, measurement in enumerate(st.session_state['measurements']):
        # Plot sub-figure
        x_linspace = np.linspace(measurement.x[0],measurement.x[-1], 1000)
        y_linspace = np.linspace(measurement.y[0],measurement.y[-1], 1000)
        ## Plot x axis
        ax[i, 0].plot(measurement.x, measurement.Ix, 'o', label='Data', color='#008176')
        ax[i, 0].plot(x_linspace, gaussian(x_linspace, *measurement.x_fit), label='Fit', color='#c1272c')
        ax[i, 0].set_xlabel(r'x [$\mu$m]', fontsize=15)
        ax[i, 0].set_ylabel('Normalised Intensity', fontsize=15)
        ## Plot y axis
        ax[i, 1].plot(measurement.y, measurement.Iy, 'o', label='Data', color='#008176')
        ax[i, 1].plot(y_linspace, gaussian(y_linspace, *measurement.y_fit), label='Fit', color='#c1272c')
        ax[i, 1].set_xlabel(r'y [$\mu$m]', fontsize=15)
        ax[i, 1].set_ylabel('Normalised Intensity', fontsize=15)

        plt.xticks(fontsize=12.5)
        plt.yticks(fontsize=12.5)
    return fig

with st.container(border=True):
    st.header('Gaussian Fits')
    gaussian = lambda x, a, b, c: a * np.exp(-(x-b)**2 / (2 * c**2))
    st.button('Run', on_click=gaussian_fit)
    with st.expander('Show Figure'):
        st.pyplot(figure())


    
    


