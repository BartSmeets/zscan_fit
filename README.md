# Z-Scan Fit

All code necessary to analyse a z-scan measurement and fit the data according to different nonlinear absorption models.

## Table of contents
* [General information](#general-information)
* [Parts](#parts)
    * [beam profile](#beam-profile)
    * [power measurement](#power-measurement)
    * [normalise data](#normalise-data)
    * [z-scan fit](#z-scan-fit)
* [Requirements](#requirements)

## General information

This project has been part of my thesis for the degree of Master of Science in Nanoscience, Nanotechnology and Nanoengineering. Read the [thesis report](/documentation/thesis.pdf) for methodology of the measurment and documentation of the code.

The code aims to handle all [parts](#parts) of a z-scan measurement and ultimately fit the data according the different nonlinear absorption models. 

[Download the latest release](https://github.com/BartSmeets/zscan_fit/archive/refs/heads/main.zip)

## Parts

The total analysis of a z-scan experiment has been split into parts:
* [beam profile](#beam-profile)
* [power measurement](#power-measurement)
* [normalise data](#normalise-data)
* [z-scan fit](#z-scan-fit)

### Beam profile

```./beam_profile/beam_profile.ipynb```

The first step of doing a z-scan measurement is to know the properties of the beam. 

After conducting the measurement, as described in the [report](/documentation/thesis.pdf), the beam properties are exported in a txt-file by running the Jupyter Notebook. A plot of the beam profile and the beam waist is exported as well.

### Power measurement

```./power_measurement/power_measurement.ipynb```

Our setup used a oscilloscope to reliably visualise the beam energy. 

The Jupyter Notebook reads the oscilloscope data and exports a text-file containing the signal voltage, the pulse energy and the peak power. A plot of the signal is exported as well.

### Normalise data

```./normalise_data/normalise_data.ipynb```

To be compatible with the [final fitting code](#z-scan-fit), the data needs to be normalised first.

The Jupyter Notebook takes the data from one or more measurements and returns normalised data of the data individually, as well as, a normalised average over all measurements. A plot of the normalised data is exported as well.

By running the Notebook, an additional file is generated that is used to provide the experiment parameters to the final fitting code. An empty example of this file can be found in: ```./templates/INPUT_PARAMETERS.csv```.

### Z-Scan fit

```./zscan_fit.ipynb```

Finally, the nonlinear absorption models can be fitted.

The Jupyter Notebook takes the normalised data and a completed input-file that has been generated by the [normalisation code](#normalise-data). If the data was readily normalised and an input-file has not been generated, the template can also be found in: ```./templates/INPUT_PARAMETERS.csv```.

By running the Jupyter Notebook, a text-file containing the fitted model parameters is exported, as well as, plots of the fitted data.

## Requirements

Use the following comment to install all requirements:

```pip install -r requirements.txt```

This project has been created with:

* Python version: 3.8

With packages:
* numpy
* pandas
* matplotlib.pyplot
* scipy.optimize
* sklearn.linear_model




