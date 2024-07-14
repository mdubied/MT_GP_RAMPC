# MT_GP_RAMPC
## Description
This repo contains the code that accompanies the Master Thesis

> Dubied M. "Online Model Learning in Robust MPC via Gaussian Processes"

It is compose of three folders:
- offline_CCM: used to execute the offline algorithm. It computes the CCM and the offline constants. The code is adapted from
  > Erdin AC., Köhler J., Leeman AP. “A Comparison on Robust MPC Methods for Nonlinear Systems”
- online_MPC: used to run the online algorithm, in particular the RMPC and RAMPC scheme. The code uses parts of
  > Lahr, A., Zanelli, A., Carron, A., & Zeilinger, M. N. (2023). Zero-order optimization for Gaussian process-based model predictive control. European Journal of Control, 74, 100862.
  > https://gitlab.ethz.ch/ics/zero-order-gp-mpc
- offline_constants: a folder that contains the offline quantities computed by the offline algorithm. The online algorithm takes the content of this folder as input. 

## Installation and dependencies
After cloning this repo, a few steps are required to run the code.

### Offline part
For the offline part, the following dependencies are required
- Matlab (tested on Matlab 2020b)
- Casadi
- YALMIP
- Mosek
Make sure you add the dependencies to the Matlab path.
### Online part
For the online part, we recommend using a virtual environment, using venv. We recommend using a Python 3.10.12.
In addition you will need to install:
- Acados
- PyTorch
- GPyTorch

## How to run the code
Again, we give separate instruction for the offline and online part

### Offline part

Make sure you add the dependencies to the Matlab path.
### Online part


 
