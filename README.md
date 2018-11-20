# MachineLearning
Some Implementation of Gaussian Processes with simulated and real data (Boston Housing Data)

# MachineLearning
# MachineLearning


# BostonGPHousing
Predicts Housing Values in Boston with a GP, taking longitude and latitude as inputs. The Boston Housing Dataset is split in a training and test set. Hyperparameters are optimized by a gradient-based method (conjugate gradients) as proposed by Rasmussen & Williams (2012)
Predictions and test values are plotted against each other. Predictions strongly underestimate high house prices. When more parameters are added to the input, computational difficulties with the Cholesky Decomposition occur (still to be solved...).
