# GIN-2022

This is a, commented version of the Code used for the experiments described in the paper.

## Figures

  ### Figures2 Proposal
  Contains the proposed figures for the figure2 in the paper 
  - Overall_1
  - Overall_2
  - Main+Subfigures
    
  ### Gaussianity Generated Samples
  Contains the generated samples from smoothened distribution
  - Double Pendulum
  - Single Pendulum
  - Visual Odometry

  ### Empirical Running Times and Parameters
  - Contains emprical running times and parameters table
    

## Code

### double_pendulum image imputation
  Contains simulator for double_pendulum image imputation, see seperate readme
  
### double_pendulum_state_estimation
  Contains simulator for double_pendulum state estimation, see seperate readme
  
### pendulum  image imputation
  Contains simulator for single_pendulum image imputation, see seperate readme
  
### pendulum state estimation
  Contains simulator for single_pendulum state estimation, see seperate readme
  
### Lorenz
  Contains simulator for 
  - "lorenz with known states"(we are aware of the dynamics)
  - "lorenz with known states smoothing" (using smoothing distribution)
  - "lorenz with unknown states"(we lack the dynamics). 
  see seperate readme for each one.

### NCLT data
Contains simulator for 
  - "Known dynamics"(we are aware of the dynamics)
  - "Known dynamics smoothing" (using smoothing distribution)
  - "Unknown Dynamics"(we lack the dynamics). 
 
 see seperate readme for each one.




## Dependencies

  - python 
  - tensorflow 
  - numpy 
  - pillow 
  - torch
  - math
