import streamlit as st
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog
import glob
import utils.beam_profile as bp

if 'wavelength' not in st.session_state:
    st.session_state['wavelength'] = 532.    # nm
if 'step_size' not in st.session_state:
    st.session_state['step_size'] = 1.   # mm
if 'profile_directory' not in st.session_state:
    st.session_state['profile_directory'] = os.environ.get('HOMEPATH')
if 'exclude' not in st.session_state:
    st.session_state['exclude'] = []
if 'measurements' not in st.session_state:
    st.session_state['measurements'] = []
if 'w' not in st.session_state:
    st.session_state['w'] = []
if 'sigma_w' not in st.session_state:
    st.session_state['sigma_w'] = []
if 'w0' not in st.session_state:
    st.session_state['w0'] = []
if 'z0' not in st.session_state:
    st.session_state['z0'] = []
if 'zR' not in st.session_state:
    st.session_state['zR'] = []
if 'M2' not in st.session_state:
    st.session_state['M2'] = []
if 'gaussian_fig' not in st.session_state:
    st.session_state['gaussian_fig'] = None


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
    st.write('Data Directory')
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


tab1, tab2 = st.tabs(['Gaussian Fit', 'Beam Profile Fit'])

# Gaussian Fit
with tab1:
    st.header('Gaussian Fits')
    gaussian = lambda x, a, b, c: a * np.exp(-(x-b)**2 / (2 * c**2))

    col1, col2 = st.columns([1, 5])
    with col1:
        options = np.arange(0, len(all_files), 1)
        st.multiselect('Exclude Data', options, key='exclude')
        if len(all_files) > 0:
            if st.button('Run', key='gaussian fit'):
                with col2:
                    with st.spinner():
                        bp.gaussian_fit(all_files)
                        st.pyplot(bp.fig_gaussian(all_files))
        else:
            st.button('Run', key='gaussian fit', on_click=bp.gaussian_fit, args=[all_files], disabled=True)
            with col2:
                st.pyplot(st.session_state['gaussian_fig'])

    
with tab2:
    st.header('Beam Profile Fit')

    col1, col2 = st.columns([1, 5])
    with col1:
        if len(st.session_state['w']) > 0:
            st.button('Run', key='bp fit', on_click=bp.bp_fit)
        else:
            st.button('Run', key='bp fit',on_click=bp.bp_fit, disabled=True)
    with col2:
        if len(st.session_state['w0']) > 0:
            st.pyplot(bp.fig_bp())

    
    


