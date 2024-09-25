# FLAIR


This repository contains the Supporting R/Cpp code for the article "Factor pre-training in Bayesian multivariate logistic models".

The repo is structured as follows:

```
├── application
    ├── helpers_application.R
    ├── insects_application_model_fitting.R
    ├── insects_application_plots.R
    └── plotting_functions.R
├── simulations
    ├── helpers_simulations.R
    ├── numerical_experiments.R
    └── numerical_experiments_supplemental.R
├── FLAIR_wrapper.R
└── helper_functions.cpp
 ```  

The `FLAIR_wrapper.R` script contain an R wrapper for implementing the methodology. `helper_functions.cpp` contains the Cpp subroutines for the method. The `simulations` and `application` folder contain the code to replicate the numerical experiments and the application to Madagascar insects data.

