import streamlit as st
import os
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm
from scipy import sparse
from scipy.sparse.linalg import spsolve

class data:
    def __init__(self):
        self.folder = os.environ.get('HOMEPATH')
        self.names = []
        self.fig_raw = plt.figure(figsize=(8,7))
        self.fig_norm = plt.figure(figsize=(8,7))


    def select(self):
        root = tk.Tk()
        root.attributes('-topmost', True)
        root.withdraw()
        self.directory = filedialog.askopenfilenames(title='Select Data Files', 
                                                                    initialdir=self.folder, 
                                                                    parent=root)
        root.destroy()
        
        # Seperate folder from name
        index = self.directory[0].rfind('/')
        self.folder = self.directory[0][:index]
        self.names = [dir[index+1:] for dir in self.directory]

    def load(self):
        # Initialise
        self.OA_raw = []
        self.CA_raw = []

        # Load all data
        for i, file in enumerate(self.directory):
            data = np.loadtxt(file)
            if i == 0:
                N = len(data[:, 0]) # Number of data points
                self.df = np.ndarray((len(self.directory), N, 3))
            
            self.df[i, :, 0] = np.abs(data[:, 0])    # z position
            self.df[i, :, 1] = data[:,1] # Channel 1
            self.df[i, :, 2] = data[:, 2]    # Channel 2

        # Average
        average = np.average(self.df, 0)
        self.z = average[:, 0]

        # Error
        ## Mean absolute error is less sentisitive to outliers
        MAE = np.mean(np.abs(self.df[:, :, 1:] - average[:, 1:]), axis=0)
        MAE = MAE.T

        if st.session_state['OA'] == 'CH1':
            self.OA = [average[:, 1], MAE[0]]
            self.CA = [average[:, 2], MAE[1]]
        else:
            self.CA = [average[:, 1], MAE[0]]
            self.OA = [average[:, 2],MAE[1]]


    def plot_raw(self):
        self.fig_raw, ax  = plt.subplots(2, 2, figsize=(8,7))
        
        # Plot individual measurements
        for i, name in enumerate(self.names):
            if st.session_state['OA'] == 'CH1':
                ax[0, 0].plot(self.df[i, :, 0], self.df[i, :, 1], '.', label=name)
                ax[0, 1].plot(self.df[i, :, 0], self.df[i, :, 2], '.', label=name)
            else:
                ax[0, 1].plot(self.df[i, :, 0], self.df[i, :, 1], '.', label=name)
                ax[0, 0].plot(self.df[i, :, 0], self.df[i, :, 2], '.', label=name)
        # Plot average of measurements
        ax[1, 0].plot(self.z, self.OA[0], '.')
        ax[1, 1].plot(self.z, self.CA[0], '.')

        # Labels
        ax[0,0].legend(bbox_to_anchor=(2.9,1))
        ax[0,0].set_title('Open Aperture')
        ax[1,0].set_title('Open Aperture (average)')
        ax[1,0].set_xlabel('z [mm]')
        ax[0,0].set_ylabel('Transmittance')
        ax[1,0].set_ylabel('Transmittance')
        ax[0,1].set_title('Closed Aperture')
        ax[1,1].set_title('Closed Aperture (average)')
        ax[1,1].set_xlabel('z [mm]')


    def plot_norm(self):
        self.fig_norm, ax = plt.subplots(2, 2, figsize=(8,7))

        # Plot individual measurements
        for i, name in enumerate(self.names):
            if st.session_state['OA'] == 'CH1':
                ax[0, 0].errorbar(self.df_norm[i, :, 0], self.df_norm[i, :, 1], self.OA_norm[1], fmt='.', label=name)
                ax[0, 1].errorbar(self.df_norm[i, :, 0], self.df_norm[i, :, 2], self.CA_norm[1], fmt='.', label=name)
            else:
                ax[0, 1].errorbar(self.df_norm[i, :, 0], self.df_norm[i, :, 1], self.OA_norm[1], fmt='.', label=name)
                ax[0, 0].errorbar(self.df_norm[i, :, 0], self.df_norm[i, :, 2], self.CA_norm[1], fmt='.', label=name)

        # Plot average of measurements
        ax[1, 0].errorbar(self.z, self.OA_norm[0], self.OA_norm[1],fmt= '.')
        ax[1, 1].errorbar(self.z, self.CA_norm[0], self.CA_norm[1], fmt='.')

        # Draw line at 1
        for i, j in np.ndindex(ax.shape):
            ax[i,j].axhline(1, ls=':', color='grey')

        # Label axes
        ax[0,0].legend(bbox_to_anchor=(2.9,1))
        ax[0,0].set_title('Normalised Open Aperture')
        ax[1,0].set_title('Normalised Open Aperture (average)')
        ax[1,0].set_xlabel('z [mm]')
        ax[0,0].set_ylabel('Normalised Transmittance')
        ax[1,0].set_ylabel('Normalised Transmittance')
        ax[0,1].set_title('Normalised Closed Aperture')
        ax[1,1].set_title('Normalised Closed Aperture (average)')
        ax[1,1].set_xlabel('z [mm]')


    def normalise(self):            
        def arPLS(y, ratio=1e-6, niter=10):
            # Initialisation
            ## Matrices
            N = len(y)
            diag = np.ones(N - 2)
            D = sparse.spdiags([diag, -2*diag, diag], [0, -1, -2], N, N-2)
            H = st.session_state['lambda'] * D.dot(D.T)
            w = np.ones(N)
            W = sparse.spdiags(w, 0, N, N)
            ## Loop
            count = 0 

            while True:
                # Update weights
                W.setdiag(w)
                z = spsolve(W + H, W * y)

                # Make d- for logistic function
                d = y - z
                dn = d[d < 0]
                ## mean and std of d-
                m = np.mean(dn)
                s = np.std(dn)
                ## Logistic function
                w_new = 1 / (1 + np.exp(2 * (d - (2*s - m))/s))

                # Check conditions
                count += 1 
                if count > niter:
                    break
                if norm(w - w_new)/norm(w) < ratio:
                    break
                else:
                    w = w_new
            return z

        # Compute Normalisation
        self.OA_norm = self.OA / arPLS(self.OA[0]).reshape(1,-1)
        self.CA_norm = self.CA / arPLS(self.CA[0]).reshape(1,-1)


        self.df_norm = np.ndarray(self.df.shape)
        for i, _ in enumerate(self.df_norm[:, 0, 0]):
            self.df_norm[i, :, 0] = self.df[i, :, 0]
            self.df_norm[i, :, 1] = self.df[i, :, 1] / arPLS(self.df[i, :, 1]).reshape(1,-1)
            self.df_norm[i, :, 2] = self.df[i, :, 2] / arPLS(self.df[i, :, 2]).reshape(1,-1)

    def update(self):
        self.normalise()
        self.plot_norm()
        self.plot_raw()


if 'data' not in st.session_state:
    st.session_state['data'] = data()
if 'OA' not in st.session_state:
    st.session_state['OA'] = 'CH1'
if 'CA' not in st.session_state:
    st.session_state['CA'] = 'CH2'
if 'lambda' not in st.session_state:
    st.session_state['lambda'] = 1e6


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

    # Directory
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
                st.session_state['data'].update()
                
    with col2:
        st.write('File Names:')
        for name in st.session_state['data'].names:
            st.write(name)
        
    col1, col2 = st.columns(2)
    with col1:
        st.selectbox('Open Aperture', ['CH1', 'CH2'], key='OA')
    with col2:
        st.selectbox('Closed Aperture', ['CH1', 'CH2'], key='CA')


    if 'error' in locals():
        st.error(error)

# Raw Data
with st.container(border=True):
    st.header('Raw Data', anchor=False)
    st.pyplot(st.session_state['data'].fig_raw)

# Normalised Data
with st.container(border=True):
    st.header('Normalised Data')
    st.pyplot(st.session_state['data'].fig_norm)
    
    st.select_slider('λ (for baseline correction)', (1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9), 
                     key='lambda', on_change=st.session_state['data'].update)


