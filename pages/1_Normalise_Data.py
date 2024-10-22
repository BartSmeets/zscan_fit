import streamlit as st
from utils.normalise import data_structure
import tkinter as tk
from tkinter import filedialog

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

def select(data_structure):
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
        data_structure.directory = filedialog.askopenfilenames(title='Select Data Files', initialdir=data_structure.folder, parent=root)
        root.destroy()
        
        # Seperate folder from name
        index = data_structure.directory[0].rfind('/')
        data_structure.folder = data_structure.directory[0][:index]
        data_structure.names = [dir[index+1:] for dir in data_structure.directory]


with st.container(border = True):
    st.header('User Inputs', anchor=False)

    # Directory Input
    ## Select files
    col1, col2 = st.columns([1,5])
    with col1:
        if st.button('Select Files'):
            select(df)
            try:
                df.load()
            except ValueError:
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
        

