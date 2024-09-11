import streamlit as st
from utils.normalise import data_structure

# Initialise User Inputs / Session States
if 'data' not in st.session_state:
    st.session_state['data'] = data_structure()
if 'OA' not in st.session_state:
    st.session_state['OA'] = 'CH1'
if 'CA' not in st.session_state:
    st.session_state['CA'] = 'CH2'
if 'lambda' not in st.session_state:
    st.session_state['lambda'] = 1e6

# Title
st.title('Normalise Data')
st.markdown('''
            This page processes raw z-scan data such that it can be used for further analysis.

            It's functionalities are:
            * Average multiple measurements
            * Calculate errorbars
            * Normalise data
            * Export separate Open Aperture and Closed Aperture data files
            ''')

# User Inputs
with st.container(border = True):
    st.header('User Inputs', anchor=False)

    # Directory Input
    ## Select files
    col1, col2 = st.columns([1,5])
    with col1:
        if st.button('Select Files', on_click=st.session_state['data'].select):
            try:
                st.session_state['data'].load()
            except ValueError:
                st.session_state['data'].names = []    # Reset the loaded names
                error = 'Data sets do not have the same number of data points.\
                         This is currently not supported.'
            else:
                st.session_state['data'].update()   # Normalise data and generate figures
    ## Show files     
    with col2:
        st.write('File Names:')
        for name in st.session_state['data'].names:
            st.write(name)
    
    # Select Channel
    col1, col2 = st.columns(2)
    with col1:
        st.selectbox('Open Aperture', ['CH1', 'CH2'], key='OA')
    with col2:
        st.selectbox('Closed Aperture', ['CH1', 'CH2'], key='CA')

    # Print error if any
    if 'error' in locals():
        st.error(error)

# Plot 
## Raw Data
with st.container(border=True):
    st.header('Raw Data', anchor=False)
    st.pyplot(st.session_state['data'].fig_raw)

## Normalised Data
with st.container(border=True):
    st.header('Normalised Data')
    st.pyplot(st.session_state['data'].fig_norm)
    
    st.select_slider('λ (for baseline correction)', (1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9), 
                        key='lambda', 
                        on_change=st.session_state['data'].update, 
                        disabled=(st.session_state['data'].names == []))
    st.button('Export', on_click=st.session_state['data'].export, 
              disabled=(st.session_state['data'].names == []))
    

