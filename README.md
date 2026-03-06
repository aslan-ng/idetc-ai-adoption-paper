## Overview

This project develops a computational pipeline that connects agent-based modeling, ODE reduction, and decision-oriented tradeoff analysis.

At the micro level, an Agent-Based Model (ABM) simulates trust and opinion dynamics within an organization.
These dynamics are aggregated into organizational state trajectories (SQLB), reduced into a continuous-time ODE model, and finally transformed into a decision surface showing how AI accuracy and initial sentiment jointly determine adoption outcomes.

```
Agent-Based Model
        ↓
Organizational State Aggregation
        ↓
ODE Reduction (CTMC Generator Fit)
        ↓
Final Adoption Estimation
        ↓
Decision Surface (Tradeoff Analysis)
```

This structure allows complex behavioral dynamics to be translated into interpretable policy and strategy tradeoffs.

## Reproducing the Full Simulation–Reduction–Decision Pipeline

To rerun the complete experimental and analysis workflow from scratch, execute the following steps in order.

This pipeline runs and transforms agent-based simulations into a reduced ODE model and finally into a decision-oriented tradeoff surface.

### Step 0: 

Clone the repository and install the required dependencies:
```
pip install -r requirements.txt
```
This installs all Python packages needed to run the full simulation–reduction–decision pipeline. Python 3.11 or higher is required. It is recommended to use a virtual environment.

### Step 1: Run `config.py`

This script defines the full experimental design grid and generates all parameter combinations. These are saved to `settings.csv` file. Each row represents one simulation configuration.

### Step 2: Run `run.py`

This script reads the configurations from settings.csv, constructs the corresponding models, and runs the simulations (in parallel). The results are saved to the `models/` directory as `<model_name>.json`.

### Step 3: Run `network_visualization.py`

This script creates the network representation of example organizations and saved them to `figures/` directory.

### Step 4: Run `state.py`

This script loads the simulation results from the `models/` directory and computes the organizational SQLB state ratios at each time step. The state at each timestep are saved to the `states/` directory as `<model_name>.csv`.

### Step 5: Run `ode.py`

This script loads the states for each model configuration from the `states/` directory and fits a continuous-time ODE model to estimate transition constants. The fitted ODE parameters are saved to the `odes/` directory as `<model_name>.npz`.

### Step 6: Run `plot.py`

This script compares the Agent-Based Model state trajectories with the fitted ODE results and visualizes them for validation.

### Step 7: Run `adaption.py`

This script loads the fitted ODE results from the `odes/` directory and computes the final steady-state adoption. Final adoption is defined as the sum of quiet and loud adopters. The results are saved to `final_adaption.csv`.

### Step 8: Run `tradeoff.py`

This script loads `final_adaption.csv` and visualizes the tradeoff between `agents_average_initial_opinion` and `technology_success_rate`. The resulting decision surface is saved to the `figures/` directory.