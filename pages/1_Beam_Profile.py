import streamlit as st
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog
import glob
import utils.beam_profile as bp

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
    st.session_state['profile_directory'] = filedialog.askdirectory(title='Select Directory', 
                                                                    initialdir=st.session_state['profile_directory'], 
                                                                    parent=root)
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


with st.container(border=True):
    st.header('Gaussian Fits')
    gaussian = lambda x, a, b, c: a * np.exp(-(x-b)**2 / (2 * c**2))
    st.button('Run', on_click=bp.gaussian_fit, args=[all_files])
    with st.expander('Show Figure'):
        try:
            st.pyplot(bp.fig_gaussian(all_files))
        except:
            st.warning('Something went wrong. Have you run the fit first?')


    
    


