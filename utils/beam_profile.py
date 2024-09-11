import streamlit as st
from scipy.optimize import curve_fit   
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import toml
from datetime import datetime
import os
import tkinter as tk
from tkinter import filedialog
import glob

class data_structure:  
    def __init__(self):
        self.folder = os.environ.get('HOMEPATH')
        self.all_files = []
        self.fig_gaus = plt.figure()
        self.fig_profile = plt.figure()
        self.w = []
        self.w0 = [np.nan, np.nan]


    def select(self):
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
        st.session_state['profile_directory'] = filedialog.askdirectory(title='Select Directory', initialdir=st.session_state['profile_directory'], parent=root)
        root.destroy()

        self.all_files = glob.glob(st.session_state['profile_directory'] + '/Data_*.txt') # Load all files having a specific name format within the working directory)
    '''
    Fitting models
    '''

    def gaussian_fit(self) :
        '''
        Fit a Gaussian beam profile and saves the results in st.session_state:
        - measurements: list containing the data objects
        - w: an array containing the width of the beam at each measurement
        - sigma_w: the std of the width
        '''

        # Initialise data class for data loading and fitting
        class data: 
            def __init__(self, measurement):
                x = measurement[:, 0]
                Ix = measurement[:, 1] / 100
                y = measurement[:, 2]
                Iy = measurement[:, 3] / 100

                self.x = x[~np.isnan(x)]
                self.Ix = Ix[~np.isnan(Ix)]
                self.y = y[~np.isnan(y)]
                self.Iy = Iy[~np.isnan(Iy)]


            def fit(self, p0x, p0y):
                gaussian = lambda x, a, b, c: a * np.exp(-(x-b)**2 / (2 * c**2))      
                self.x_fit, _ = curve_fit(gaussian, self.x, self.Ix, p0x, 
                                        bounds=[[0,-np.inf,0],[np.inf,np.inf,np.inf]])
                self.y_fit, _ = curve_fit(gaussian, self.y, self.Iy, p0y, 
                                        bounds=[[0,-np.inf,0],[np.inf,np.inf,np.inf]])

                self.sigma_x = np.sqrt(self.x_fit[2]/(2*np.sqrt(2*np.pi)*self.x_fit[0]))
                self.sigma_y = np.sqrt(self.y_fit[2]/(2*np.sqrt(2*np.pi)*self.y_fit[0]))
                # Calculate Beam width (e^-2) as average x-width and y-width 
                self.wx = 2* abs(self.x_fit[2])    # e^-2 definition beam width
                self.wy = 2* abs(self.y_fit[2])

        # Initialise
        self.measurements = []
        w = np.zeros(len(self.all_files))    # First column for beam widths; Second column for error on beam widths
        sigma_w = np.zeros(len(w))    # First column for beam widths; Second column for error on beam widths
        p0x = p0y = [1,0,100]    # First guess for fitting

        # Load (and fit) all files 
        for i, file in enumerate(self.all_files):
            try:
                measurement = np.genfromtxt(file, skip_header=11, delimiter='\t', filling_values=np.nan)
            except Exception as e:
                st.error('Error while loading '+ file)
                raise
            loaded_data = data(measurement)
            if i in st.session_state['exclude']:
                self.measurements.append(loaded_data)
                continue
            
            ## Fit all files
            loaded_data.fit(p0x, p0y)
            
            # Check if Gaussian fit has failed
            ## If Gaussian fit has failed only consider wx or wy
            if loaded_data.x_fit[0]<0.1:
                w[i] = loaded_data.wy
                sigma_w[i] = 2*loaded_data.sigma_y
                p0y[2] = p0x[2] = loaded_data.y_fit[2]
            elif loaded_data.y_fit[0]<0.1:
                w[i] = loaded_data.wx
                sigma_w[i] = 2*loaded_data.sigma_x
                p0x[2] = p0y[2] = loaded_data.x_fit[2]
            ## If Gaussian fit has not failed, map ellipse to circle with equal area           
            else:
                w[i] = np.sqrt(loaded_data.wx*loaded_data.wy)    # Width 
                sigma_w[i] = np.sqrt(loaded_data.wy**2 / (loaded_data.wx*loaded_data.wy) * loaded_data.sigma_x**2 
                                + loaded_data.wx**2 / (loaded_data.wx*loaded_data.wy) * loaded_data.sigma_y**2)    # Std
            self.measurements.append(loaded_data)

        # Store
        self.w = [w, sigma_w]
        return

    def bp_fit(self):
        '''
        Fits the beam profile

        ## Adjusts st.session_state:
        - w0: beam waist and std
        - z0: the focal point and std
        - zR: the Rayleigh length and std
        - M2: the quality factor and std
        '''

        # Initialise
        wavelength = st.session_state['wavelength']
        w = self.w[0]
        sigma_w = self.w[1]
        z = st.session_state['step_size'] * np.array(range(len(w)))    # unit mm
        ## Filter the zeroes
        mask = (w!=0) 
        w = w[mask]
        sigma_w = sigma_w[mask]
        z = z[mask]

        # Fit
        ## Define Model
        def width(z, w0: float, z0: float, M2: float, wavelength: float):
            wavelength = wavelength * 1e-3    # nm to um
            z = z * 1e3    # mm to um
            zR = np.pi*(w0**2) / (M2 * wavelength)
            root = 1 + ((z-z0)/zR)**2
            return w0 * np.sqrt(root)    # unit um
        ## Run Model
        w_param, w_var  = curve_fit(lambda x, w0, z0, M2: width(x, w0, z0, M2, wavelength), z, w, [10,10e3,1], 
                        sigma=sigma_w, bounds=([0,-np.inf,1],[np.inf,np.inf,np.inf]), absolute_sigma=True)
        [w0, z0, M2] =  w_param
        zR = np.pi*(w0**2) / (M2*wavelength*1e-3)    # Rayleigh length

        # Error Propagation
        sigma_w = np.sqrt(np.diag(w_var)[0])    # Error on beam waist
        sigma_z = np.sqrt(np.diag(w_var)[1])*1e-3    # Error on z-position of focal point
        sigma_M2 = np.sqrt(np.diag(w_var)[2])    # Error on M^2
        dz_dw = (2*np.pi*10) / (M2*wavelength*1e-3)
        dz_dM2 = (np.pi*(10**2)) / (wavelength*1e-3*(M2**2))
        sigma_zR = np.sqrt((dz_dw**2)*(sigma_w**2) + (dz_dM2**2)*(sigma_M2**2))    # Error on Rayleigh length

        # Store
        self.w0 = [w0, sigma_w]
        self.z0 = [z0, sigma_z]
        self.zR = [zR, sigma_zR]
        self.M2 = [M2, sigma_M2]
        return

    
    '''
    Generate Figures
    '''

    def fig_gaussian(self):
        '''Generate the figure of the Gaussian fit
        
        ## Argument:
        - all_files: list containing all file names

        ## Returns:
        - fig: matplotlib figure object
        '''
        gaussian = lambda x, a, b, c: a * np.exp(-(x-b)**2 / (2 * c**2))      
        self.fig_gaus, ax = plt.subplots(len(self.all_files), 2, figsize=(10,3*len(self.all_files)), tight_layout=True, sharey=True)

        for i, measurement in enumerate(self.measurements):
            if i in st.session_state['exclude']:
                continue
            # Plot sub-figure
            x_linspace = np.linspace(measurement.x[0],measurement.x[-1], 1000)
            y_linspace = np.linspace(measurement.y[0],measurement.y[-1], 1000)
            ## Plot x axis
            ax[i, 0].plot(measurement.x, measurement.Ix, 'o', label='Data', color='#008176')
            ax[i, 0].plot(x_linspace, gaussian(x_linspace, *measurement.x_fit), label='Fit', color='#c1272c')
            ax[i, 0].set_xlabel(f'x$_{{{i}}}$ (μm)', fontsize=15)
            ax[i, 0].set_ylabel('Normalised Intensity', fontsize=15)
            ## Plot y axis
            ax[i, 1].plot(measurement.y, measurement.Iy, 'o', label='Data', color='#008176')
            ax[i, 1].plot(y_linspace, gaussian(y_linspace, *measurement.y_fit), label='Fit', color='#c1272c')
            ax[i, 1].set_xlabel(f'y$_{{{i}}}$ (μm)', fontsize=15)

            plt.xticks(fontsize=12.5)
            plt.yticks(fontsize=12.5)


    def fig_bp(self):
        '''Generate the figure of the beam profile

        ## Returns:
        - fig: matplotlib figure object
        '''
        
        # Initialise figure
        self.fig_profile = plt.figure()
        ax2 = plt.axes()
        ## Model
        def width(z, w0: float, z0: float, M2: float, wavelength: float):
            wavelength = wavelength * 1e-3    # nm to um
            z = z * 1e3    # mm to um
            zR = np.pi*(w0**2) / (M2 * wavelength)
            root = 1 + ((z-z0)/zR)**2
            return w0 * np.sqrt(root)    # unit um

        # Initialise
        w = self.w[0]
        z = st.session_state['step_size'] * np.array(range(len(w)))    # unit mm
        [z0, _] = self.z0
        [w0, _] = self.w0
        [M2, _] = self.M2
        # Mask all zeroes
        mask = (w!=0)
        w = w[mask]
        z = z[mask]
        ## Generate x-values for plotting
        z_plot = np.linspace(0, z[-1], 500)

        # Plot
        plt.plot(z-z0*1e-3, w,'.', label='Data', color='#008176')
        plt.plot(z_plot-z0*1e-3, width(z_plot, w0, z0, M2, st.session_state['wavelength']), 
                label='Fit', color='#c1272c')
        
        # Labels
        plt.legend()
        plt.ylabel('Beam width (μm)', fontsize=15)
        plt.xlabel('z - z$_0$ (mm)', fontsize=15)
        plt.xticks(fontsize=12.5)
        plt.yticks(fontsize=12.5)
        ax2.xaxis.set_minor_locator(AutoMinorLocator())
        ax2.yaxis.set_minor_locator(AutoMinorLocator())


    def export(self):
        '''
        Export the results in an newly generated output folder
        '''
        # Create export directory
        timeCode = datetime.now()
        export_directory = st.session_state['profile_directory'] + "/OUTPUT_BEAM_PROFILE_"  + timeCode.strftime("%Y%m%d-%H%M")
        try:
            os.mkdir(export_directory)
        except:
            pass

        fitting_results = {
            'Beam Profile Fitting': {
                'w0': self.w0,
                'z0': np.array(self.z0)*1e-3,
                'zR': self.zR,
                'M2': self.M2
            }
        }

        toml_string = toml.dumps(fitting_results)
        toml_lines = toml_string.split('\n')
        comments = [toml_lines[0],
                    '# Observable   [Value, Std]    Unit',
                    f'{toml_lines[1]}   # um',
                    f'{toml_lines[2]}   # mm',
                    f'{toml_lines[3]}   # um',
                    toml_lines[4]]

        with open(export_directory + '/RESULTS_BEAM_PROFILE.toml', 'w') as f:
            f.write('\n'.join(comments))
        self.fig_gaus.savefig(export_directory + "/OUTPUT_WIDTHS.png", bbox_inches='tight')
        self.fig_profile.savefig(export_directory + "/OUTPUT_BEAM_PROFILE.png", bbox_inches='tight')