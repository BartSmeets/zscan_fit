# Z-Scan Fit

All code necessary to analyse a z-scan measurement and fit the data according to different nonlinear absorption models.

## Table of contents
* [General information](#general-information)
* [Beam profile](#beam-profile)
* [Power measurement](#power-measurement)
* [Normalise data](#normalise-data)
* [Z-Scan fit](#z-scan-fit)
* [Installation and requirements](#installation-and-requirements)

## General information

This project has been part of my thesis for the degree of Master of Science in Nanoscience, Nanotechnology and Nanoengineering. Read the [thesis report](/docs/thesis.pdf) for methodology of the measurment and documentation of the code.

The code has been split into functional parts:
* [beam profile](#beam-profile)
* [power measurement](#power-measurement)
* [normalise data](#normalise-data)
* [z-scan fit](#z-scan-fit)

## Beam profile

The first step of doing a z-scan measurement is to know the properties of the beam. 

After conducting the measurement, as described in the [report](/documentation/thesis.pdf), the beam properties are exported in a ```.txt``` by running the Jupyter Notebook. A plot of the beam profile and the beam waist is exported as well.

### How to use

1. Simply run: ```./beam_profile/beam_profile.ipynb```

2. If any data point needs to be exclude, enter the index of that data point in the ```Exclude data``` section, i.e.,

```python
# USER INPUT
exclude = []    # Enter index of datapoint to be excluded

# Exclude data
removed = 0
for index in exclude:
    DATA.pop(index-removed)
    Z = np.delete(Z,index-removed,0)
    removed += 1
N_MEASUREMENTS -= removed
```

## Power measurement

Our setup used a oscilloscope to reliably visualise the beam energy. 

The Jupyter Notebook reads the oscilloscope data and exports a ```.txt``` containing the signal voltage, the pulse energy and the peak power. A plot of the signal is exported as well.

### How to use

1. Simply run: ```./power_measurement/power_measurement.ipynb```

## Normalise data

To be compatible with the [final fitting code](#z-scan-fit), the data needs to be normalised first.

The Jupyter Notebook takes the data from one or more measurements and returns normalised data of the data individually, as well as, a normalised average over all measurements. A plot of the normalised data is exported as well.

By running the Notebook, an additional file is generated that is used to provide the experiment parameters to the final fitting code. An empty example of this file can be found in: ```./templates/INPUT_PARAMETERS.csv```.

### How to use

1. Simply run: ```./normalise_data/normalise_data.ipynb```


## Z-Scan fit

Finally, the nonlinear absorption models can be fitted.

The Jupyter Notebook takes the normalised data and a completed input-file that has been generated by the [normalisation code](#normalise-data). If the data was readily normalised and an input-file has not been generated, the template can also be found in: ```./templates/INPUT_PARAMETERS.csv```.

By running the Jupyter Notebook, a ```.txt``` containing the fitted model parameters is exported, as well as, plots of the fitted data.

### How to use

1. Open: ```./z_scan_zscan_fit.ipynb```
2. Run all sections, up to the ```fit``` section
3. Provide the fit type, initial guesses for the fitting parameters, and the basin hopping algorithm in the ```fit``` section, i.e.,

```python
# User Input
## Fit type
### 0: 1PA | 1: 2PA no Is2 | 2: 2PA | 3: 2PA no Is1 | 4: 2PA no sat
FIT_TYPE = 0
## Initial guess
### Only the parameters corresponding to FIT_TYPE are considered. The others are ignored.
Z0_0 = 0
I_S1_0 = 1e5
I_S2_0 = 1e-5
BETA_0 = 1e-5
## Model Parameters
N_RUNS = 3                     # Number of independent model calculations
MAX_PERTURBATION = 10              # Max perturbation size 
BOUNDS = [(None, None), (1e-99, None), (1e-99, None), (0, None)]  # Bounds of fitting-parameters: z0, Is1
MAX_ITER = 500                    # Max number of iterations (stop-condition) (0 = classical algorithm)
MAX_AGE = 50                      # Max age of best Chi2 (stop-condition)
T = 0.8                            # Temperature (related to probability of a non optimal chi2 being taken as step)
MAX_JUMP = 5                       # Number of jumps permitted
MAX_REJECT = 5                     # Local minimum treshold
```

## Installation and requirements

[![Build with](https://img.shields.io/badge/Build%20with-Python%203.8-blue)](https://www.python.org/])

### Required packages
* numpy
* pandas
* matplotlib.pyplot
* scipy.optimize
* sklearn.linear_model

### Installation

1. [Download the latest release](https://github.com/BartSmeets/zscan_fit/releases/latest) or clone the repository:
    
```bash
git clone https://github.com/BartSmeets/zscan_fit.git
```

2. Install the requirements:

```bash
pip install -r requirements.txt
```
