# Python standard library
from datetime import datetime
import os, sys
import tkinter as tk
from tkinter import filedialog

# Required
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Included in repository
import export_functions
from fitting_model.absorption import intensity


def load(file):
    # Initial path for file selection
    INITIAL_PATH = os.environ.get('HOMEPATH')
    root = tk.Tk()
    root.attributes('-topmost', True)
    root.withdraw()
    FILE_PATH = filedialog.askopenfilename(initialdir=INITIAL_PATH, title='Select data', parent=root)
    file.set(FILE_PATH)
    root.destroy()

    # Change working directory
    ## Directory is the folder where the selected file is located
    index = FILE_PATH.rfind('/')
    DIRECTORY = FILE_PATH[:index]
    FILE_NAME = FILE_PATH[index+1:-4]
    os.chdir(DIRECTORY)

    # Load data
    ## And seperate into physical arrays
    MEASUREMENT = np.loadtxt(FILE_NAME + '.txt')

    # Load parameter data (if existing)
    if os.path.exists('INPUT_PARAMETERS.csv'):
        PARAMETER_DATA = pd.read_csv('INPUT_PARAMETERS.csv', sep=r';', header=0, skiprows=[1], skipinitialspace = True)
    else:
        sys.exit('"INPUT_PARAMETERS.csv" not in directory')

    # Convert parameters to proper units
    PULSE_WIDTH = 6e-9 # 5-8 ns
    ALPHA0 = float(PARAMETER_DATA['alpha']) / 10    # Convert cm-1 to mm-1
    Z_R = float(PARAMETER_DATA['zR']) * 1e-3    # Convert um to mm
    W0 = float(PARAMETER_DATA['W0']) * 1e-3   # Convert um to mm
    E_PULSE = float(PARAMETER_DATA['Pulse Energy']) * 1e-6    # Convert uJ to J
    P_LASER = E_PULSE / PULSE_WIDTH
    I0 = 2*P_LASER / (np.pi * W0**2)
    L = float(PARAMETER_DATA['L']) * 10   # Convert cm to mm

    PARAMETER_DATA = [L, ALPHA0, I0, Z_R, W0, E_PULSE]
    
    return MEASUREMENT, PARAMETER_DATA, FILE_NAME, DIRECTORY

#############################################################################################################

def bounds_convert(bounds): 
    # Initialise data structure
    BOUNDS = []

    # Convert and store data
    for i in [0,1,2,3]:
        for j in [0,1]:
            if bounds[i][j].get() == 'None':
                bounds[i][j] = None
            else:
                bounds[i][j] = float(bounds[i][j].get())
        BOUNDS.append((bounds[i][0], bounds[i][1]))
    return BOUNDS

#############################################################################################################

def fit_type_convert(fit_type):
    fit_type_string = fit_type.get()
    if fit_type_string == '1PA':
        FIT_TYPE = 0
    elif fit_type_string == '2PA without Is2':
        FIT_TYPE = 1
    elif fit_type_string == '2PA':
        FIT_TYPE = 2
    elif fit_type_string == '2PA without Is1':
        FIT_TYPE = 3
    else:
        FIT_TYPE = 4
    return FIT_TYPE

###########################################################################################################

def generate_plots(MEASUREMENT, FILE_NAME, FIT_TYPE, N_RUNS, P_BEST, EXPERIMENT_PARAM, RUNS):
    Z_DATA = MEASUREMENT[:,0]
    I_DATA = MEASUREMENT[:,1]
    Z_PLOT = np.linspace(Z_DATA[0], Z_DATA[-1], 1000)
    SIGMA_PLOT = np.ones(np.size(MEASUREMENT[:,2])) * np.average(MEASUREMENT[:,2])

    # Prepare Title
    FIT_TYPE_STRING = ['1PA', '2PA_no_Is2', '2PA', '2PA_no_Is1', '2PA_no_sat'][FIT_TYPE]
    TITLE = FIT_TYPE_STRING + '_' + FILE_NAME

    # T(z)
    fig2, ax2 = plt.subplots()
    export_functions.plot.Tz(ax2, Z_DATA, I_DATA, Z_PLOT, SIGMA_PLOT, FIT_TYPE, P_BEST, EXPERIMENT_PARAM)
    ## Labels
    ax2.set_xlabel('z [mm]')
    ax2.set_ylabel('Normalised Transmittance')
    ax2.set_title(TITLE)

    # T(z) with I(z)
    fig2_2, ax2_2 = plt.subplots()
    ax2_3 = ax2_2.twinx()  
    export_functions.plot.Tz(ax2_2, Z_DATA, I_DATA, Z_PLOT, SIGMA_PLOT, FIT_TYPE, P_BEST, EXPERIMENT_PARAM)
    ax2_3.plot(Z_PLOT, intensity(Z_PLOT, P_BEST[0], EXPERIMENT_PARAM[2], EXPERIMENT_PARAM[3]), linestyle=':', color='red')
    ax2_2.set_xlabel('z [mm]')
    ax2_2.set_ylabel('Normalised Transmittance')
    ax2_2.set_title(TITLE)
    ax2_3.set_ylabel(r'Intensity [W/mm$^{2}$]', color='red')
    ax2_3.tick_params(axis='y', labelcolor='red')

    # T(I)
    fig2_1, ax2_1 = plt.subplots()
    export_functions.plot.TI(ax2_1, Z_DATA, I_DATA, Z_PLOT, SIGMA_PLOT, FIT_TYPE, P_BEST, EXPERIMENT_PARAM)
    ##Labels
    ax2_1.set_xlabel(r'Intensity [W/mm$^{2}$]')
    ax2_1.set_ylabel('Normalised Transmittance')
    ax2_1.set_title(TITLE)
    ax2_1.legend()

    # Plot all runs
    fig3, ax3 = plt.subplots(N_RUNS, 1, figsize=(10,3*N_RUNS))
    for i in range(N_RUNS):
        textstr = export_functions.text.individual(FIT_TYPE, RUNS, i)
        export_functions.plot.Tz(ax3[i], Z_DATA, I_DATA, Z_PLOT, SIGMA_PLOT, FIT_TYPE, RUNS[i,:-1], EXPERIMENT_PARAM)
        ax3[i].text(5,1.1, textstr)

    return TITLE, fig2, fig2_1, fig2_2, fig3

#####################################################################################################################

def export(DIRECTORY, TITLE, OUTPUT_STRING, fig1, fig2, fig2_1, fig2_2, fig3, PARAMETER_DATA):
    # Create export directory
    timeCode = datetime.now()
    EXPORT_DIRECTORY = DIRECTORY + "/OUTPUT_FIT_" + TITLE + '_' + timeCode.strftime("%Y%m%d-%H%M")
    os.mkdir(EXPORT_DIRECTORY)

    # Save figures
    fig1.savefig(EXPORT_DIRECTORY + '/OUTPUT_' + TITLE + '_RAW_DATA.png', bbox_inches='tight')
    fig2.savefig(EXPORT_DIRECTORY + '/OUTPUT_' + TITLE + '_Tz.png', bbox_inches='tight')
    fig2_1.savefig(EXPORT_DIRECTORY + '/OUTPUT_' + TITLE + '_TI.png', bbox_inches='tight')
    fig2_2.savefig(EXPORT_DIRECTORY + '/OUTPUT_' + TITLE + '_Tz_TI.png', bbox_inches='tight')
    fig3.savefig(EXPORT_DIRECTORY + '/OUTPUT_' + TITLE + '_ALL_RUNS.png', bbox_inches='tight')

    # Export results
    ## Prepare Parameter string for export
    output_file = open(EXPORT_DIRECTORY + '/RESULTS_' + TITLE + '.txt', 'wt')
    output_file.write(export_functions.text.parameter_string(PARAMETER_DATA))
    output_file.write('\n\n')
    output_file.write(OUTPUT_STRING)
    output_file.close()

    # Open output folder
    os.startfile(EXPORT_DIRECTORY)
    return