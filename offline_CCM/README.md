# A Comparison on Robust MPC Methods for Nonlinear Systems

## Description

This repository contains the the MATLAB code that accompanies the semester project:
> Erdin AC., Köhler J., Leeman AP. “A Comparison on Robust MPC Methods for Nonlinear Systems”

## Project Status

- [ ] Test dynamics used in [Robust Adaptive MPC Using CCMs](https://gitlab.ethz.ch/ics/RAMPC-CCM.git) for [Robust Nonlinear Parametric SLS](https://gitlab.ethz.ch/ics/nonlinear-parametric-SLS.git)
- [ ] Test dynamics used in [Robust Nonlinear Parametric SLS](https://gitlab.ethz.ch/ics/nonlinear-parametric-SLS.git) for [Robust Adaptive MPC Using CCMs](https://gitlab.ethz.ch/ics/RAMPC-CCM.git)

## Prerequisites

- MATLAB (tested with version R2023a)
- Casadi

## Installation

### Software

To run this project you need to install the following software.

1. Download and install MATLAB from the [official website](https://www.mathworks.com/products/matlab.html).

2. Install Casadi by following the instructions from the [official Casadi documentation](https://web.casadi.org/get/).

3. Clone this repository or download the code as a ZIP archive and extract it to a folder of your choice.

### Submodules

To get started with this project, you'll need to initialize and update the submodules linked to it.

1. Initialize the submodule (needed after cloning):
   ```bash
   git submodule update --init --recursive
   ```

2. Update the submodule recursively:
   ```bash
   git submodule update --recursive --remote
   ```

## Usage

Open the project `Nonlinear_RMPC_Comparison.prj` and run the `main.m` to execute the algorithms and models discussed in this semester project.

## License

For open source projects, say how it is licensed.
