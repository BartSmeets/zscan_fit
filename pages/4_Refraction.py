import streamlit as st
from utils.refraction import data_structure, select, load_beam, run, plot, export
import os
import numpy as np

st.set_page_config(layout='wide')

if 'data_OA' not in st.session_state:
    st.session_state['data_OA'] = data_structure()
if 'data_CA' not in st.session_state:
    st.session_state['data_CA'] = data_structure()
if 'OA' not in st.session_state:
    st.session_state['OA'] = os.environ.get('HOMEPATH')
if 'CA' not in st.session_state:
    st.session_state['CA'] = os.environ.get('HOMEPATH')


OA = st.session_state['data_OA'] 
CA = st.session_state['data_CA'] 

col01, col02 = st.columns(2)
with col01:
    # User Inputs
    with st.container(border=True, height=None):
        st.header('User Inputs', anchor=False)
        col1, col2, col3 = st.columns(3)
        with col1:
            OA.ui['L'] = st.number_input('L (cm)', value=0.1)
        with col2:
            OA.ui['E'] = st.number_input('E$_{pulse}$ (μJ)', value=3.5)
            OA.update_I0()
        with col3:
            OA.ui['wavelength'] = st.number_input('wavelength (nm)', value=532)

        # Directory
        st.write('OA Directory')
        col1, col2 = st.columns([5,1])
        with col1:
            st.session_state['OA'] = st.text_input('OA Directory', value=st.session_state['OA'], label_visibility='collapsed')
        with col2:
            st.button('Browse OA', on_click=lambda: select(OA, 'OA'))
        
        st.write('CA Directory')
        col1, col2 = st.columns([5,1])
        with col1:
            st.session_state['CA'] = st.text_input('CA Directory', value=st.session_state['CA'], label_visibility='collapsed')
        with col2:
            st.button('Browse CA', on_click=lambda: select(CA, 'CA'))


# Load Beam profile
with col02:
    try:
        bp = st.session_state['data_bp']
        OA.w0 = bp.w0[0]
        OA.zR = bp.zR[0]
    except (KeyError, AttributeError):  
        pass   

    with st.container(border=True, height=None):           
        st.header('Beam Profile', anchor=False)
        col1, col2, col3= st.columns(3)
        with col1:
            OA.w0 = st.number_input('w$_0$ (μm)', value=OA.w0)
        with col2:
            OA.zR = st.number_input('z$_R$ (μm)', value=OA.zR)
        with col3:
            st.button('Browse', on_click=lambda: load_beam(OA))

# Model
with st.container(border=True):
    col1, col2 = st.columns(2)

    # Initial Guess
    with col1:
        options = ['z0', 'alpha', 'n2'] # All possible fit paramters
        units = {'z0': 'cm', 'alpha':'cm$^{-1}$', 'n2':'GW$^{-1}$'}   # The units of the corresponding parameters

        with st.container(border=True):
            st.header('Initial Guess', anchor=False)

            # Iterate over the labels to create the checkboxes and number inputs
            for label in options:
                col11, col12 = st.columns([1, 19])
                with col11:
                    OA.type[label] = st.checkbox(f'{label} ({units[label]})', value=OA.type[label], label_visibility='collapsed')
                with col12:
                    OA.p0[label] = st.number_input(f'{label} ({units[label]})', value=OA.p0[label], format='%.3e')

    with col2:
        with st.container(border=True):
            st.header('Run Model', anchor=False)
            st.button('Run', on_click=lambda: run(OA, CA), use_container_width=True, disabled=not(True in list(OA.type.values()) and hasattr(OA, 'raw')))

        with st.container(border=True):
            st.header('Results', anchor=False)
            params = np.array(list(OA.p0.values()))
            errorbars = np.zeros(3)
            try:
                params[list(OA.type.values())] = OA.pBest[:]
                errorbars[list(OA.type.values())] = np.sqrt(np.diag(OA.pcov))[:]
            except AttributeError:
                string = 'Run model to obtain results'
            else:
                string = f"""
                    z$_0$ = {params[0]:.3f} ± {errorbars[0]:.3f} cm \\
                    α$_0$ = {params[1]:.3f} ± {errorbars[1]:.3f} cm-1 \\
                    n$_2$ = {params[2]:.3e} ± {errorbars[2]:.3e} GW
                    """
            finally:
                st.write(string)

            # Plot
        with st.container(border=True):
            st.pyplot(plot(OA, CA, params))
            st.button('Export', on_click=lambda: export(OA, params, errorbars))
