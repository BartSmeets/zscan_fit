import streamlit as st
from utils.normalise import data_structure, select

st.set_page_config(layout='wide')

# Initialise User Inputs / Session States
if 'data_norm' not in st.session_state:
    st.session_state['data_norm'] = data_structure()
if 'OA' not in st.session_state:
    st.session_state['OA'] = 'CH1'
if 'CA' not in st.session_state:
    st.session_state['CA'] = 'CH2'
if 'lambda' not in st.session_state:
    st.session_state['lambda'] = 1e6
if 'z0' not in st.session_state:
    st.session_state['z0'] = {}
df = st.session_state['data_norm']

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
        if st.button('Select Files', on_click=lambda: select(df)):
            try:
                df.load()
            except ValueError as e:
                #st.error(e)
                df.names = []    # Reset the loaded names
                error = 'Data sets do not have the same number of data points.\
                         This is currently not supported.'
            else:
                df.update()   # Normalise data and generate figures
    ## Show files     
    with col2:
        st.write('File Names:')
        for name in df.names:
            st.write(name)
    
    # Select Channel
    col1, col2 = st.columns(2)
    with col1:
        st.selectbox('Open Aperture', ['CH1', 'CH2'], key='OA')
    with col2:
        st.selectbox('Closed Aperture', ['CH1', 'CH2'], key='CA')

    # Select focal point position
    for name in df.names:
        st.session_state['z0'][name] = st.number_input(f'z$_0$ of {name}', value=0.000, format='%.3f')
    df.focal_point()

    # Print error if any
    if 'error' in locals():
        st.error(error)

# Plot
col1, col2 = st.columns(2)
## Raw Data
with col1:
    with st.container(border=True):
        st.header('Raw Data', anchor=False)
        st.pyplot(df.fig_raw)

## Normalised Data
with col2:
    with st.container(border=True):
        st.header('Normalised Data')
        st.pyplot(df.fig_norm)
        
        st.select_slider('λ (for baseline correction)', (1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9), 
                            key='lambda', 
                            on_change=df.update, 
                            disabled=(df.names == []))
        st.button('Export', on_click=df.export, 
                disabled=(df.names == []))
        

