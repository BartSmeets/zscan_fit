import streamlit as st
import numpy as np
import os
from utils.z_scan import data_structure
import os
import matplotlib.pyplot as plt

st.set_page_config(layout='wide')

st.title('Z-Scan Fit', anchor=False)
st.markdown('''
            This page fits the z-scan data
            ''')
st.header('How to Use?', anchor=False)
st.markdown('''
            1. Conduct the z-scan measurement and normalise the data
            2. Enter the user inputs
            3. Fit the desired parameters by checking their checkbox under 'initial guess'
            ''')

if 'data_zf' not in st.session_state:
    st.session_state['data_zf'] = data_structure()
if 'type' not in st.session_state:
    st.session_state['type'] = {'z0': False, 'Is1': False, 'Is2':False, 'beta':False}
if 'data_directory' not in st.session_state:
    st.session_state['data_directory'] = os.environ.get('HOMEPATH')

df = st.session_state['data_zf'] 

col01, col02 = st.columns(2)
with col01:
    # User Inputs
    with st.container(border=True, height=None):
        st.header('User Inputs', anchor=False)
        col1, col2, col3 = st.columns(3)
        with col1:
            df.ui['alpha0'] = st.number_input('α$_0$ (cm$^{-1}$)', value=df.ui['alpha0'])
        with col2:
            df.ui['L'] = st.number_input('L (cm)', value=df.ui['L'])
        with col3:
            df.ui['E'] = st.number_input('E$_{pulse}$ (μJ)', value=df.ui['E'])
        # Directory
        st.write('Data Directory')
        col1, col2 = st.columns([5,1])
        with col1:
            st.session_state['data_directory'] = st.text_input('Directory', value=st.session_state['data_directory'], label_visibility='collapsed')
        with col2:
            st.button('Browse Data', on_click=df.select)


# Load Beam profile
with col02:
    try:
        bp = st.session_state['data_bp']
        df.w0 = bp.w0[0]
        df.zR = bp.zR[0]
    except (KeyError, AttributeError):  
        pass   

    with st.container(border=True, height=None):
        def load_beamProfile():
            df.load_beam()
            
        st.header('Beam Profile', anchor=False)
        col1, col2, col3= st.columns(3)
        with col1:
            df.w0 = st.number_input('w$_0$ (μm)', value=df.w0)
        with col2:
            df.zR = st.number_input('z$_R$ (μm)', value=df.zR)
        with col3:
            st.button('Browse', on_click=load_beamProfile)

# Fit
with st.container(border=True):
    col1, col2, col3 = st.columns(3)

    # Initial Guess
    with col1:
            options = ['z0', 'Is1', 'Is2', 'beta']
            units = {'z0': 'cm', 'Is1': 'GW/cm$^2$', 'Is2':'GW/cm$^2$', 'beta': 'cm/GW'}

            with st.container(border=True):
                st.header('Initial Guess', anchor=False)

                # Iterate over the labels to create the checkboxes and number inputs
                for label in options:
                    col11, col12 = st.columns([1, 19])
                    with col11:
                        df.type[label] = st.checkbox(f'{label} ({units[label]})', value=df.type[label], label_visibility='collapsed')
                    with col12:
                        df.p0[label] = st.number_input(f'{label} ({units[label]})', value=df.p0[label], format='%.3e')

            # Bounds
            with st.container(border=True):
                st.header('Bounds', anchor=False)
                for label in options:
                    col11, col12 = st.columns(2)
                    with col11:
                        df.bounds[label][0] = st.number_input(f'{label} ({units[label]})', value=df.bounds[label][0], key=label+'_min', format='%.3e')
                    with col12:
                        df.bounds[label][1] = st.number_input(f'{label} ({units[label]})' + 'max', value=df.bounds[label][1], label_visibility='hidden', format='%.3e')

    # Model
    with col2:
        with st.container(border=True):
            st.header('Model Parameters', anchor=False)
            model_parameters = ['Number Runs', 'Max Perturbation', 'Max Iterations', 'Max Age', 'T', 'Max Jump', 'Max Reject']
            for parameter in model_parameters:
                df.model_parameters[parameter] = st.number_input(parameter, value=df.model_parameters[parameter])
    
    with col3:
        with st.container(border=True):
            st.header('Run Model', anchor=False)
            st.button('Run', on_click=df.run, use_container_width=True, disabled=not(True in list(df.type.values()) and hasattr(df, 'raw')))
            df.bar = st.progress(0)
            df.textbox = st.empty()
            df.progress_update(0, 0, 0)

        with st.container(border=True):
            st.header('Results', anchor=False)
            na_option = np.array(list(df.p0.values()))
            try:
                na_option[list(df.type.values())] = df.pBest[:]
                df.na_option = na_option
                df.errorbar()
            except AttributeError:
                string = 'Run model to obtain results'
            else:
                string = f"""
                    z$_0$ = {na_option[0]:.3f} ± {df.errorbars[0]:.3f} cm | span: {df.chi2span[0]:.3f}\\
                    Is$_1$ = {na_option[1]:.3e} ± {df.errorbars[1]:.3e} GW/cm$^{2}$ | span: {df.chi2span[1]:.1e}\\
                    Is$_2$ = {na_option[2]:.3e} ± {df.errorbars[2]:.3e} GW/cm$^{2}$ | span: {df.chi2span[2]:.1e}\\
                    β = {na_option[3]:.3e} ± {df.errorbars[3]:.3e} cm/GW | span: {df.chi2span[3]:.1e}\\
                    \\
                    α/β = {df.ui['alpha0']/na_option[3]:.3e} GW/cm$^{2}$
                    """
            finally:
                st.write(string)
                st.warning('The errorbars need checking')

        # Plot
        with st.container(border=True):
            df.plot_type = st.radio('Plot Type', ['default', 'intensity'])
            st.pyplot(df.plot(na_option))

            knop1, knop2 = st.columns(2)
            with knop1:
                with st.popover('All Results'):
                    st.pyplot(df.plot_all())
            with knop2:
                st.button('Export', on_click=df.export, disabled=(not hasattr(df, 'ps')))