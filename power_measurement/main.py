## Python standard library
from datetime import datetime
import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
import os, time

## Required
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def main(pb, root, file):
####################################
# LOAD DATA
####################################
    
    pb['value'] = 0
    root.update()
    time.sleep(0.5)

    # Set work directory
    FILE_DIRECTORY = file.get()
    index = FILE_DIRECTORY.rfind('/')
    DIRECTORY  = FILE_DIRECTORY[:index]
    FILE_NAME = FILE_DIRECTORY[index+1:]
    os.chdir(DIRECTORY)

    # Load data
    measurement = pd.read_csv(FILE_DIRECTORY,header=None, skiprows=21,delimiter=',')
    MEASUREMENT = measurement.to_numpy()
    pb['value'] = 25
    root.update()

####################################
# CALCULATE ENERGY AND POWER
####################################

    # Set laser and meter constants
    CONVERSION_RATE = 392.74    # Unit V/J
    FREQUENCY = 10    # Unit Hz
    PULSE_WIDTH = 6e-9 # 5-8 ns

    # Find delta (difference between maximum and baseline)
    ## Find baseline
    indices = np.where(MEASUREMENT[:,0] < 0)    # Baseline indices
    BASELINE = np.average(MEASUREMENT[indices, 1])    # Calculate baseline value
    ## Find maximum
    indices = np.where(MEASUREMENT[:,0] > 0)
    MAXIMUM = np.amax(MEASUREMENT[indices,1])
    ## Calculate difference
    DELTA = MAXIMUM - BASELINE

    # Calculate corresponding energy and power
    ENERGY = DELTA / CONVERSION_RATE
    POWER = ENERGY / PULSE_WIDTH
    pb['value'] = 50
    root.update()

    # Print Results
    OUTPUT_STRING = textstr = '\n'.join((
        r'Delta V = %.2f mV' % (DELTA*1e3, ),
        r'Pulse Energy = %.2f uJ' % (ENERGY*1e6, ),
        r'Peak Power = %.2f W' % (POWER, )))

######################################
# FIGURE
######################################

    fig = plt.figure()

    ## Plot data
    plt.plot(MEASUREMENT[:,0], MEASUREMENT[:,1], 'o', label='Data')    # Data
    plt.axhline(MAXIMUM, ls=':', color='green', label='Maximum')    # Maximum
    plt.axhline(BASELINE, ls=':', color='grey', label='Baseline')    # Baseline
    ## Labels
    plt.ylabel('Signal [V]')
    plt.xlabel('Time [s]')
    plt.title('Oscilloscope Output')
    plt.legend()

    pb['value']=75
    root.update()

######################################
# EXPORT RESULTS
######################################

## Define file names
    timeCode = datetime.now()
    OUTPUT_FILENAME = FILE_NAME[:-4]
    OUTPUT_DIRECTORY = DIRECTORY + '/OUTPUT_POWER_' + OUTPUT_FILENAME + '_' + timeCode.strftime("%Y%m%d-%H%M%S")
    os.mkdir(OUTPUT_DIRECTORY)


## Save figure
    fig.savefig(OUTPUT_DIRECTORY + '/OUTPUT_' + OUTPUT_FILENAME + '.png', bbox_inches='tight')

# Write file
    output_file = open(OUTPUT_DIRECTORY + '/OUTPUT_' + OUTPUT_FILENAME + '.txt', 'wt')
    output_file.write(OUTPUT_STRING)
    output_file.close()

    pb['value']=100
    root.update()

## Open output folder
    os.startfile(OUTPUT_DIRECTORY)
    return