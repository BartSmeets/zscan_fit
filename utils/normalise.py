import os
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm
from scipy import sparse
from scipy.sparse.linalg import spsolve
from datetime import datetime
import streamlit as st
import tkinter as tk
from tkinter import filedialog

class data_structure:
    def __init__(self):
        self.folder = os.environ.get('HOMEPATH')
        self.names = []
        self.fig_raw = plt.figure(figsize=(8,7))
        self.fig_norm = plt.figure(figsize=(8,7))


    def load(self):
        '''
        Load the data files and average the data. Also computes the Mean Absolute Error (MAE)

        ## Generates:
        - self.df: raw dataframe containing the unprocessed data
        - self.z: z-positions of the measurements (mm)
        -self.OA(CA): list containing the averaged Open Aperture (Closed Aperture) data and the MAE
        '''    
    
        # Load all data
        for i, file in enumerate(self.directory):
            data = np.loadtxt(file)
            if i == 0:
                N = len(data[:, 0]) # Number of data points
                self.df = np.ndarray((len(self.directory), N, 3))
            
            self.df[i, :, 0] = data[:, 0] - np.average(data[:, 0])    # z position: - average to create identical axes in both measurement directions

            if self.df[i, 0, 0] < 0:
                self.df[i, :, 1] = data[:,1] # Channel 1
                self.df[i, :, 2] = data[:, 2]    # Channel 2
            else:
                self.df[i, :, 0] = np.flip(self.df[i, :, 0])
                self.df[i, :, 1] = np.flip(data[:,1]) # Channel 1
                self.df[i, :, 2] = np.flip(data[:, 2])    # Channel 2

        self.z = self.df[i, :, 0]


    def normalise(self):
        '''
        Perform baseline correction to normalise the data to the far field

        ## Generates:
        - self.df_norm: a normalised version of self.df
        - self.OA(CA)_norm: a normalised version of self.OA(CA)
        '''

        def arPLS(y, ratio=1e-6, niter=10):
            '''
            Baseline Correction 
            From Baek et al. Analyst (2014); DOI:10.1039/C4AN01061B
            '''
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
        ## On individual data
        self.df_norm = np.ndarray(self.df.shape)
        self.baseline = np.empty(np.shape(self.df_norm[:, :, 0]))
        for i, _ in enumerate(self.df_norm[:, 0, 0]):
            self.baseline[i,:] = arPLS(self.df[i, :, 1]).reshape(1,-1)
            self.df_norm[i, :, 0] = self.df[i, :, 0]
            self.df_norm[i, :, 1] = self.df[i, :, 1] / arPLS(self.df[i, :, 1]).reshape(1,-1)
            self.df_norm[i, :, 2] = self.df[i, :, 2] / arPLS(self.df[i, :, 2]).reshape(1,-1)
        
        # Average
        average = np.average(self.df_norm, 0)
        # Error
        ## Mean absolute error is less sentisitive to outliers
        MAE = np.mean(np.abs(self.df_norm[:, :, 1:] - average[:, 1:]), axis=0).T

        # Store
        if st.session_state['OA'] == 'CH1':
            self.OA_norm = [average[:, 1], MAE[0]]
            self.CA_norm = [average[:, 2], MAE[1]]
        else:
            self.CA_norm = [average[:, 1], MAE[0]]
            self.OA_norm = [average[:, 2],MAE[1]]


    def plot_raw(self):
        '''
        Plot the raw unaltered and averaged data

        ## Generates:
        - self.fig_raw: figure object containing the plot
        '''
        self.fig_raw, ax  = plt.subplots(1, 2, figsize=(8,7))
        
        # Plot individual measurements
        for i, name in enumerate(self.names):
            if st.session_state['OA'] == 'CH1':
                ax[0].plot(self.df[i, :, 0], self.df[i, :, 1], '.', label=name)
                ax[0].plot(self.z, self.baseline[i], '.', label='baseline')
                ax[1].plot(self.df[i, :, 0], self.df[i, :, 2], '.', label=name)
            else:
                ax[1].plot(self.df[i, :, 0], self.df[i, :, 1], '.', label=name)
                ax[0].plot(self.df[i, :, 0], self.df[i, :, 2], '.', label=name)
    

        # Labels
        ax[0].legend(bbox_to_anchor=(2.9,1))
        ax[0].set_title('Open Aperture')
        ax[0].set_xlabel('z [mm]')
        ax[0].set_ylabel('Transmittance')
        ax[0].set_ylabel('Transmittance')
        ax[1].set_title('Closed Aperture')

        ax[1].set_xlabel('z [mm]')


    def plot_norm(self):
        '''
        Plot the normalised data

        ## Generates:
        - self.fig_norm: figure object containing the plot
        '''
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

        # Draw baseline at 1
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


    def update(self):
        '''
        Update the normalisation and figures
        '''
        self.normalise()
        self.plot_norm()
        self.plot_raw()


    def export(self):
        '''
        ## Export the data:
        - indivual data
        - averaged data
        - figures

        ## Generates:
        - export_directory: it will create a subfolder in the file directory
        '''
        
        # Create export directory
        timeCode = datetime.now()
        export_folder = "/Normalised_Data_" + timeCode.strftime("%Y%m%d-%H%M%S")
        export_directory = self.folder + export_folder
        try:
            os.mkdir(export_directory)
        except:
            pass
        
        # Save figures
        self.fig_raw.savefig(export_directory + '/OUTPUT_RAW_DATA.png', bbox_inches='tight')
        self.fig_norm.savefig(export_directory + '/OUTPUT_NORMALISED_DATA.png', bbox_inches='tight')
        
        # Intialise Open Aperture Data
        OA_export = np.ndarray((len(self.z), 3))    # 0: z-position; 1: average OA; 2: errorbar
        OA_export[:, 0] = self.z
        OA_export[:, 1] = self.OA_norm[0]
        OA_export[:, 2] = self.OA_norm[1]

        # Intialise Closed Aperture Data
        CA_export = np.ndarray((len(self.z), 3))    # 0: z-position; 1: average OA; 2: errorbar
        CA_export[:, 0] = self.z
        CA_export[:, 1] = self.CA_norm[0]
        CA_export[:, 2] = self.CA_norm[1]

        # Export Files
        np.savetxt(export_directory + '/DATA_OA_AVERAGE.txt', OA_export)
        np.savetxt(export_directory + '/DATA_CA_AVERAGE.txt', CA_export)

        # Export normalised individual measurements
        for i, name in enumerate(self.names):
            ## Initialise data structure
            file_oa_export = np.ndarray((len(self.z), 3))
            file_ca_export = np.ndarray((len(self.z), 3))
            ## Prepare file name
            file_oa_string = export_directory + '/DATA_OA_' + name
            file_ca_string = export_directory + '/DATA_CA_' + name
            ## Store data in data structure
            ### 0: z-position; 1: average OA; 2: errorbar
            file_oa_export[:, 0] = self.df_norm[i, :, 0]
            file_ca_export[:, 0] = self.df_norm[i, :, 0]
            if st.session_state['OA'] == 'CH1':
                file_oa_export[:, 1] = self.df_norm[i, :, 1]
                file_ca_export[:, 1] = self.df_norm[i, :, 2]
            else:
                file_oa_export[:, 1] = self.df_norm[i, :, 2]
                file_ca_export[:, 1] = self.df_norm[i, :, 1]
            file_oa_export[:, 2] = self.OA_norm[1]
            file_ca_export[:, 2] = self.CA_norm[1]
            ## Export data
            np.savetxt(file_oa_string, file_oa_export)
            np.savetxt(file_ca_string, file_ca_export)
        
        # Open output folder
        st.success("Data Exported Succesfully")

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