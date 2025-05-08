# Cost of Capital LCOE Model


The LCOH Ninja is a Python project to simulate the technical potential and levelised cost from wind and solar plants, globally, under the influence of different costs of capital. The wind and solar power profiles are taken from [Renewables.ninja](https://www.renewables.ninja/) (wind) and developed using the [PV Lib module](https://pvlib-python.readthedocs.io/en/stable/) (solar)


## Requirements

[Python](https://www.python.org/) version 3+


Required libraries:
 * numpy
 * xarray
 * matplotlib
 * cartopy
 * pandas
 * scipy
 * joblib
 * datetime
 * os
 * regionmask

## SETUP

### Generate and download renewable power profiles
First, download the necessary input data from NASA's [MERRA-2 reanalysis](https:///gmao.gsfc.nasa.gov/reanalysis/MERRA-2/), including windspeed at 10m and solar irradiance. Using [PV Lib](https://pvlib-python.readthedocs.io/en/stable/) and the [Virtual Wind Farm Model](https://github.com/renewables-ninja/vwf/tree/master), generate renewable profiles for the year and geographies of interest. Renewable profiles can also be developed using the renewable_calculator.py script.

Save the files into the running folder under /DATA and /DATA/ and with the necessary filenames.

### Set up the LCOE model
With the required libraries installed, the model should just require editing of the paths in the wacc_model.py script.

### Process renewables data into electricity production and economics
To run the model, run the wacc_model.py script with the desired regions and input files. This process should yield a set of NetCDF files in the output folder that has been specified, which contain details of the cost, investment requirements and hydrogen output at each MERRA-2 gridcell.

## USAGE INSTRUCTIONS

First, edit the paths in the wacc_model script and ensure that the input folder (DATA) are set up correctly.
Second, ensure that all the required data files are in the DATA folder, with modification of the renewable profiles if desired. Renewable profiles must be translated into a capacity factor average from the selected years.
Third, decide on the model parameters that you wish to use.

Fourth, run the wacc_model.py. 

## POSTPROCESSING

Graph plotting functions and other postprocessing scripts (e.g., cost-supply curve generation, LCOE heatmaps) are included in the postprocessing folder. 


## LICENSE
BSD 3-Clause License
Copyright (C) 2023-2025  Luke Hatton
All rights reserved.

See `LICENSE` for more detail

## Citation

The LCOE code was developed by Luke Hatton, who can be contacted at l.hatton23@imperial.ac.uk

L Hatton, 2025.  A global, levelised cost of electricity (LCOE) model with national costs of capital. Currently under review.

