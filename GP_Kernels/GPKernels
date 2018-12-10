#################################################

#  Gaussian Process with Different Types of 
#  Kernels Conditioned on Observations from
#  a Noisy Sine function
#################################################

# Structure:
# 0. Load Packages and Generate Data
# 1. Define Kernel Functions
# 2. Compute and Plot Gaussian processes

##################################################

# 0.  Load Packages and Generate Data
##################################################

setwd("/Users/claracharlottehoffmann/Desktop/MachineLearning")

if (!require("pacman")) 
  install.packages("pacman"); library("pacman") 
p_load("MASS", 
       "ggplot2", 
       "reshape2",
       "gridExtra")

# observed values
x <- c(-4,-3, -2, -1 , 1)
y = sin(x)
# x values for prediction 
x.star <- seq(-5,5,len=100) 

##################################################

# 1.  Define Kernel Functions
##################################################

# All kernel functions compute some kernel matrix
# from two or one one-dimensional input vectors
#
# Args:
#   X1:      first/only matrix or vector
#   X2:      second matrix or vector
#   l:       scaling parameter in exponent
#   sigma.f: scaling parameter in base
#
# Returns:
#   Kernel matrix
#
# Types: 
#  Gaussian, Ornstein-Uhlenbeck, Periodic
#  Rational Quadratic, Locally Periodic

# Gaussian kernel
gaussKernel <- function(X1, X2 ,
                        l=1, sigma.f = 1 ) {
  Zero <- matrix( rep( 0, 
                       len=length(X1)*length(X2)), 
                  nrow = length(X1))
  A <- Zero + X1
  B <- t(t(Zero)+X2)
  Sigma <- (sigma.f^2)*exp(-((A-B)^2)/(2*(l^2)))
  return(Sigma)
}
# Ornstein-Uhlenbeck kernel
ouKernel <- function(X1, X2) {
  Zero <- matrix( rep( 0, 
                       len=length(X1)*length(X2)), 
                  nrow = length(X1))
  A <- Zero + X1
  B <- t(t(Zero)+X2)
  Sigma <- exp(-(abs(A-B))/(1))
  return(Sigma)
}
# Periodic kernel
periodicKernel <- function(X1, X2) {
  Zero <- matrix( rep( 0, 
                       len=length(X1)*length(X2)), 
                  nrow = length(X1))
  A <- Zero + X1
  B <- t(t(Zero)+X2)
  Sigma <- exp(-2*((sin(A-B))^2)/(0.3^2))
  return(Sigma)
}
# Rational quadratic kernel
ratKernel <- function(X1, X2) {
  Zero <- matrix( rep( 0, 
                       len=length(X1)*length(X2)), 
                  nrow = length(X1))
  A <- Zero + X1
  B <- t(t(Zero)+X2)
  Sigma <- (1+((A-B)^2)/(5*1^2))^(-5)
  return(Sigma)
}
# Locally periodic kernel
locKernel <- function(X1, X2 ) {
  Zero <- matrix( rep( 0, 
                       len=length(X1)*length(X2)), 
                  nrow = length(X1))
  A <- Zero + X1
  B <- t(t(Zero)+X2)
  Sigma1 <- 1*exp(-((A-B)^2)/(2*(1)))
  Sigma2 <- exp(-2*((sin(A-B))^2)/(1))
  Sigma <- Sigma1*Sigma2
  return(Sigma)
}

# Function: myplot()
# Computes and plot a one-dimensional Gaussian
# process regression from observed x and y 
# values as well as prediction target x values
#
# Args:
#   mykernel: type of kernel used for cov
#             matrix
#   X1:       first/only matrix or vector
#   X2:       second matrix or vector
#   l:        scaling parameter in exponent
#   sigma.f:  scaling parameter in base
#   y:        observed y values
#
# Returns:
#   Plot of the Gaussian process regression
#
# Types: 
#  Gaussian, Ornstein-Uhlenbeck, Periodic
#  Rational Quadratic, Locally Periodic
myplot <- function(mykernel, X = x, 
                   X.star = x.star, Y = y){
  names <- c("gaussian", "ornstein-uhlenbeck", 
             "periodic","rational quadratic", 
             "locally periodic")
  var <- which(names %in% mykernel)
  # Function: kernel()
  # chooses kernel from name vector
  # and computes kernel for two input
  # vectors
  # Args:
  #  var: numeric var that selects
  #       kernel from a name vector
  #   X1: first/only vector
  #   X2: second vector
  #
  kernel <- function(var = var, X1, X2){
    switch(var, 
           return(gaussKernel(X1, X2)),
           return(ouKernel(X1, X2)),
           return(periodicKernel(X1, X2)),
           return(ratKernel(X1, X2)),
           return(locKernel(X1, X2)))
  }  
    # covariance elements
    K <- kernel(var, X, X)
    K.star <- kernel(var, X, X.star) 
    K.star.star <- kernel(var, X.star, X.star) 
    # posterior distribution
    postCov <- K.star.star - 
      t(K.star)%*%solve(K)%*%K.star
    mu.star <- t(K.star)%*%solve(K)%*%y
    # samples from posterior
    sample <- replicate(5, mvrnorm(1, 
                                   mu.star ,
                                   postCov))%>%
      as.data.frame()
    values <- cbind(x = x.star, sample)
    values <- melt(values,id="x")
    # confidence bands
    S2 <- diag(postCov)
    lowerbound <- mu.star +2*sqrt(S2)
    upperbound <- mu.star -2*sqrt(S2)
    # plot Gaussian process
    graph <- ggplot() + 
      geom_ribbon(x = X.star, 
                  aes(ymin= lowerbound, 
                      ymax = upperbound), 
                  fill = "grey70") +
      geom_point( data = NULL,aes( x = X, y=Y)) +
      geom_line(data = values, 
                aes(x = x, y=value, 
                    group=variable, 
                    colour = variable)) +
      theme(legend.position="none") +
      labs(title = paste(names[var])) +
      scale_y_continuous(limits=c(-3, 3)) +
      theme_bw() +
      theme(legend.position="none", 
            panel.grid.major = element_blank(), 
            panel.grid.minor = element_blank())
    return(graph)  
}

##################################################

# 2. Compute and Plot Gaussian processes
##################################################

# plot GP with different kernels
gaussian <- myplot("gaussian")
ornstine <- myplot("ornstein-uhlenbeck")
periodic <- myplot("periodic")
rational <- myplot("rational quadratic")
locped <- myplot("locally periodic")
 
# arrange in grid
figure <- grid.arrange(gaussian, ornstine, 
                       periodic, rational, 
                       locped , nrow = 3)
figure
ggsave("gp_kernels.jpg", plot = figure ) # save
