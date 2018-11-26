# MachineLearning
Some Implementation of Gaussian Processes with simulated and real data (Boston Housing Data)

# HparamInLikelihood
Replicates Figure 5.3a and Figure 5.3b from Rasmussen & Williams (2006) in R.
\\
<img src="Rasmussen53a.jpg" width="600">
<img src="Rasmussen53b.jpg" width="600">


# InfluenceOfHyperparameters
Illustrating the effect of different choices of hyperparameters in a Gaussian Process with an SE Kernel. The figure is an extension of figure 15.3 in KPM (2012) and implemented in R based on the Matlab Code gprDemoChangeHparams written by Carl Rasmussen.
\\
<img src="noisyhyper.jpg" width="600">

# BostonGPHousing
Predicts Housing Values in Boston with a GP, taking longitude and latitude as inputs. The Boston Housing Dataset is split in a training and test set. Hyperparameters are optimized by a gradient-based method (conjugate gradients) as proposed by Rasmussen & Williams (2006)
Predictions and test values are plotted against each other. Predictions strongly underestimate high house prices. When more parameters are added to the input, computational difficulties with the Cholesky Decomposition occur (still to be solved...).
\\
<img src="prediction.jpg" width="600">
