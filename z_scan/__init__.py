""" # Z_scan
---
Contains subpackages and modules to analyse z-scan measurements:
---
## Modules:
- errorbar:
    - errorbar.compute() to calculate errorbars
- export_functions:
    - export_functions.plot.* to plot results
    - export_functions.text.* to generate text
- fitting_model:
    - fitting_model.run() to run model
    - fitting_model.absorption containing modules to describe the absorption
        - absorption.intensity() to calculate the beam intensity at a given position
        - absorption.dI_dz.* to define the differential equation describing the absorption
        - absorption.transmittance.* to calculate the transmittance through a sample
    - fitting_model.chi2_minimising for the optimisatiokn process
        - chi2_minimising.basinhopping.* for the basinhopping algorithm
        - chi2_minimising.chi2.* to define the chi-squared
"""