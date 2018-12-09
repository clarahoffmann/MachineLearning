
# GP Illustration based on Figure 2.5 Rasmussen & Williams (2006), 
# and Figure 15.3 in KPM (2012) in R

This code illustrates a one-dimensional Gaussian process regression with and without a conjugate-gradient optimizer.
The two code files generate a graphical illustration of the case with noiseless and noisy observations.

Simulated data from a sine function is used as a observed data:

First, a Gaussian kernel function generates the covariance matrix of the Gaussian process.

\
<img src="Rasmussen53a.jpg" align = "center" width="600">
<img src="Rasmussen53b.jpg" width="600">


# InfluenceOfHyperparameters
Illustrating the effect of different choices of hyperparameters in a Gaussian Process with an SE Kernel. The figure is an extension of figure 15.3 in KPM (2012) and implemented in R based on the Matlab Code gprDemoChangeHparams written by Carl Rasmussen.
\
<img src="noisyhyper.jpg" width="600">

# Some different kernel functions
Some sampled functions of different kernels when conditioning on five points of a noisy sine function.
\
<img src="gp_kernels.jpg" width="600">

