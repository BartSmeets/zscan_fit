import streamlit as st
import numpy as np
import os
from utils.beam_profile import data_structure

st.set_page_config(layout='wide')

if 'wavelength' not in st.session_state:
    st.session_state['wavelength'] = 532.    # nm
if 'step_size' not in st.session_state:
    st.session_state['step_size'] = 1.   # mm
if 'profile_directory' not in st.session_state:
    st.session_state['profile_directory'] = os.environ.get('HOMEPATH')
if 'exclude' not in st.session_state:
    st.session_state['exclude'] = []
if 'data_bp' not in st.session_state:
    st.session_state['data_bp'] = data_structure()
df = st.session_state['data_bp']



st.title('Beam Profile Fit', anchor=False)
st.markdown('''
            This page fits the parameters describing the beam profile
            ''')
st.header('How to Use?', anchor=False)
st.markdown('''
            1. Measure the beam profile at different positions, separated by a constant step size
            2. Enter the user inputs
            3. Fit the beam radii at every position
            4. Fit the beam profile
            ''')

with st.container(border = True):
    st.header('User Inputs', anchor=False)

    # Directory
    st.write('Data Directory')
    col1, col2 = st.columns([5,1])
    with col1:
        st.session_state['profile_directory'] = st.text_input('Directory', value=st.session_state['profile_directory'], label_visibility='collapsed')
    with col2:
        st.button('Browse', on_click=df.select)
    
    # Values
    col1, col2 = st.columns(2)
    with col1:
        st.number_input('Wavelength (nm)', min_value=0., key='wavelength')
    with col2:
        st.number_input('Step Size (mm)', min_value=0., key='step_size')


tab1, tab2 = st.tabs(['Beam Radii Fits', 'Beam Profile Fit'])

# Gaussian Fit
with tab1:
    st.header('Beam Radii Fits', anchor=False)
    st.markdown(''' ''')
    gaussian = lambda x, a, b, c: a * np.exp(-(x-b)**2 / (2 * c**2))
    all_files = df.all_files

    col1, col2 = st.columns([1, 5])
    with col1:
        options = np.arange(0, len(all_files), 1)
        st.multiselect('Exclude Data', options, key='exclude')
        if len(all_files) > 0:
            if st.button('Run', key='gaussian fit'):
                with col2:
                    with st.spinner():
                        df.gaussian_fit()
                        df.fig_gaussian()
                        st.pyplot(df.fig_gaus)
            else:
                with col2:
                    st.pyplot(df.fig_gaus)
        else:
            st.button('Run', key='gaussian fit', disabled=True)
            with col2:
                st.pyplot(df.fig_gaus)

    
with tab2:
    st.header('Beam Profile Fit', anchor=False)
    
    col1, col2 = st.columns([2, 5])
    with col1:
        if len(df.w) > 0:
            if st.button('Run', key='bp fit', on_click=df.bp_fit):
                df.fig_bp()
        else:
            st.button('Run', key='bp fit',on_click=df.bp_fit, disabled=True)

        with st.container(border=True):
            if not np.isnan(df.w0[0]):
                st.write(f'w$_0$ = ({df.w0[0]:.1f} ± {df.w0[1]:.1f}) μm')
                st.write(f'z$_0$ = ({df.z0[0]*1e-3:.2f} ± {df.z0[1]*1e-3:.2f}) mm')
                st.write(f'z$_R$ = ({df.zR[0]:.0f} ± {df.zR[1]:.0f}) μm')
                st.write(f'M$_2$ = ({df.M2[0]:.1f} ± {df.M2[1]:.1f})')
    with col2:
        if not np.isnan(df.w0[0]):
            st.pyplot(df.fig_profile)

    with col1:
        st.button('Export', disabled=np.isnan(df.w0[0]), on_click=df.export)

    
    


