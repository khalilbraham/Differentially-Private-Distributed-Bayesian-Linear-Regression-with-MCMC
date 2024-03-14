# Differentially Private Distributed Bayesian Linear Regression with MCMC

This repository contains code implementation for the paper titled "Differentially Private Distributed Bayesian Linear Regression with MCMC". The project utilizes various Python scripts and a Jupyter notebook for conducting experiments and analyzing results.

## Files Description

- `MCMC_fixedS.py`: This script implements Markov Chain Monte Carlo (MCMC) algorithm for Bayesian linear regression with fixed prior and fixed data. It handles the sampling process for model parameters.
  
- `MCMC_fixedX.py`: Similar to `MCMC_fixedS.py`, this script implements MCMC algorithm for Bayesian linear regression but with fixed prior and varying data. It also handles the sampling process for model parameters.

- `MH_S_update.py`: This script implements Metropolis-Hastings (MH) algorithm for updating model parameters, specifically designed for handling data privacy constraints.

- `bayes_fixedS_fast.py`: This script provides a fast implementation of Bayesian linear regression with fixed prior and fixed data, focusing on computational efficiency.

- `adassp.py`: This script implements the Approximate Differentially Private Algorithm for Sparse Statistical Estimation (ADASSP) method, which ensures differential privacy in distributed settings while estimating sparse parameters.

- `main.ipynb`: This Jupyter notebook serves as the main entry point for conducting experiments. It provides detailed steps for running the code, conducting experiments, and analyzing results.

## Running the Code

To run the code, follow these steps:

1. Install the required dependencies by executing the following command:

```bash
pip install -r requirements.txt
```

This command installs all necessary Python packages specified in the `requirements.txt` file.

2. Open the `main.ipynb` notebook in Jupyter environment.

3. Follow the detailed steps provided in the notebook to conduct experiments. The notebook contains explanations, code snippets, and instructions for running experiments and analyzing results.

## Note

Ensure that you have Python and Jupyter Notebook installed in your environment before proceeding with the execution of the code. Additionally, it's recommended to review the paper associated with this project for a better understanding of the methodologies and algorithms implemented in the code.
