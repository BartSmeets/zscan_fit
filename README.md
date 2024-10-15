# Z-Scan Fit

[![Streamlit](https://img.shields.io/badge/Powered_by-Streamlit-red?logo=streamlit)](https://streamlit.io/])

All code necessary to analyse a z-scan measurement and fit the data according to different nonlinear absorption models.

To run:
```
$ streamlit run main.py
```

## General information

This project has been part of my thesis for the degree of [Master of Science in Nanoscience, Nanotechnology and Nanoengineering](https://www.kuleuven.be/programmes/master-nanoscience-nanotechnology-nanoengineering) at KU Leuven. Read the [thesis report](/docs/MasterThesis_BartSmeets_final.pdf) for methodology of the measurment and documentation of the code.

The code has been split into four functional parts:
* [beam profile](#beam-profile)
* [normalise data](#normalise-data)
* [z-scan fit](#z-scan-fit)


## Beam profile

The first step of doing a z-scan measurement is to know the properties of the beam. The program generates a ```.toml``` containting the beam properties.


## Normalise data

To be compatible with the [final fitting code](#z-scan-fit), the data needs to be normalised first.

The program takes the data from one or more measurements and returns normalised data of the data individually, as well as, a normalised average over all measurements. A plot of the normalised data is exported as well.


## Z-Scan fit

Finally, the nonlinear absorption models can be fitted.

The program takes the normalised data and a completed input-file that has been generated by the [normalisation code](#normalise-data).

By running the program, a ```.toml``` containing the fitted model parameters is exported, as well as, plots of the fitted data.
