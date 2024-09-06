import streamlit as st
import os
import tkinter as tk
from tkinter import filedialog

if 'data_directory' not in st.session_state:
    st.session_state['data_directory'] = os.environ.get('HOMEPATH')

def select_folder():
    root = tk.Tk()
    root.attributes('-topmost', True)
    root.withdraw()
    st.session_state['data_directory'] = filedialog.askdirectory(title='Select Directory', 
                                                                    initialdir=st.session_state['data_directory'], 
                                                                    parent=root)
    root.destroy()

st.title('Normalise Data')

st.markdown('''
            This page processes raw z-scan data such that it can be used for further analysis.

            It's functionalities are:
            * Average multiple measurements
            * Calculate errorbars
            * Normalise data
            * Export separate Open Aperture and Closed Aperture data files
            ''')


with st.container(border = True):
    st.header('User Inputs', anchor=False)
    # Directory
    st.write('Data Directory')
    col1, col2 = st.columns([5,1])
    with col1:
        st.session_state['data_directory'] = st.text_input('Directory', value=st.session_state['data_directory'], label_visibility='collapsed')
    with col2:
        st.button('Browse', on_click=select_folder)