#################################################

#  Gaussian Process with Simulated data and a
#  Conjugate Gradient Optimization on the Kernel
#  Parameters (Noisy Case)
#  - based on Rasmussen & Williams (2006)
#    and Kevin P. Murphy (2012)
#################################################

# Structure:
# 0. Load Packages and Generate Data
# 1. Defining the Kernel
# 2. GP with Arbitrary Hyperparameters
#   2.1 Covariance and Posterior
#   2.2 Plot Results
# 3. GP with Optimized Hyperparameters
#   3.1 Defining the Log-Likelihood
#   3.2 Optimizing with Conjugate-Gradient
#   3.3 Covariance and Posterior
#   3.4 Plot Results

# set working directory
setwd("/Users/claracharlottehoffmann/Desktop/MachineLearning")

##################################################

# 0.  Load Packages and Generate Data
##################################################

# load packages
if (!require("pacman")) 
  install.packages("pacman"); library("pacman") 
p_load("MASS", 
       "ggplot2", 
       "reshape2")

# generate observed x and y = f(x) values
x <- c(-4,-3, -2, -1 , 1)
noise <- 0.1
y = sin(x) + mvrnorm(1, 0 , noise)
# generate x values at which we need predictions
x.star <- seq(-5,5,len=100) 

##################################################

# 1.  Defining the Kernel
##################################################

# Computes Gaussian kernel for 
# one dimensional input vectors
#
# Args:
#   X1:      first/only matrix or vector
#   X2:      second matrix or vector
#   l:       scaling parameter in exponent
#   sigma.f: scaling parameter in base
#
# Returns:
#   Gaussian kernel matrix
#
getGaussian <- function(X1, X2 ,
                        l=1, sigma.f = 1 ) {
  Zero <- matrix( rep(0, 
                      len=length(X1)*length(X2)), 
                  nrow = length(X1))
  A <- Zero + X1
  B <- t(t(Zero)+X2)
  Sigma <- (sigma.f^2)*exp(-((A-B)^2)/(2*(l^2)))
  return(Sigma)
}

##################################################

# 2. GP with Arbitrary Hyperparameters
##################################################

# 2.1 Covariance and Posterior
# kernel elements
Ky <- getGaussian(x, x) + 
  diag(nrow = length(x))*noise
K.star <- getGaussian(x, x.star) 
K.star.star <- getGaussian(x.star, x.star)
# posterior covariance and mean
postCov <- K.star.star - 
  t(K.star)%*%solve(Ky)%*%K.star
mu.star <- t(K.star)%*%solve(Ky)%*%y

# sample some functions from posterior
sample <- replicate(5, 
                    mvrnorm(1, 
                            mu.star, 
                            postCov))%>%
  as.data.frame()
values <- cbind(x=x.star,sample)
values <- melt(values,id="x")

# confidence bands
S2 <- diag(postCov)
lowerbound <- mu.star +2*sqrt(S2)
upperbound <- mu.star -2*sqrt(S2)

# 2.2 Plot Results
gauss.error <- ggplot() + 
  geom_ribbon(x = x.star, 
              aes(ymin= lowerbound, 
                  ymax = upperbound), 
              fill = "grey70") +
  geom_point( data = NULL,aes( x = x, y=y)) +
  geom_line(data = values, 
            aes(x = x, y = value, 
                group = variable, 
                colour = variable)) +
  theme_bw() +
  theme(panel.grid.major = element_blank(), 
        panel.grid.minor = element_blank()) +
  theme(legend.position="none") +
  ylim(-2.75, 2.75)
gauss.error
ggsave("gperror.jpg", plot = gauss.error ) # save

##################################################

# 3. GP with Optimized Hyperparameters
##################################################

#  3.1 Defining the Log-Likelihood
likelihood <- function(hyper, X = x, Y = y){
  # Computes marg. log-likelihood for a joint 
  # normal distribution of a Gaussian Process
  # with conditions on the kernel parameters
  # (have to be larger or equal zero),
  # i.e. how high is the likelihood given
  # a kernel with this parameter calibration
  # Note: requires function getGaussian() 
  #
  # Args:
  #   hyper:  vector of the transformed 
  #           hyperparameters of the kernel 
  #           matrix of the  observed input 
  #           values x, in the following 
  #           order: 
  #           log(sigma.y^2) , log(l^2) 
  #           log(sigma.f^2)
  #   x:      observed input values
  #   y:      observed output values
  # Returns:
  #   loglik : marginal log-likelihood 
  #            as scalar
  #
  noise <-  (exp(hyper[1]))^0.5
  length <- (exp(hyper[2]))^0.5 
  sigma.f <-(exp(hyper[3]))^0.5
  N <- length(X)
  K <- getGaussian(X, X, 
                   l = length, sigma.f=sigma.f)
  Ky <- K + diag( nrow=ncol(K))*(noise^2)
  Kyi <- chol2inv(chol(Ky))
  loglik <- 0.5*t(Y)%*%Kyi%*%Y + 
    0.5*log(det(Ky)) + 
    N*0.5*log(2*pi)
  return(loglik)
}

# 3.2 Optimizing with Conjugate-Gradient
# optimize gradient parameters with optimize function
hyper.opt <- optim( par =  c(1,1,1), 
                    fn =  likelihood, 
                    method = c("CG")) 
hyper.opt

# extract optimal hyperparameters
noise.opt <- exp(hyper.opt$par[1])^0.5
length.opt <- exp(hyper.opt$par[2])^0.5 
sigma.f.opt <- exp(hyper.opt$par[3])^0.5
#################################################
# compute covariance matrix and posterior means
K.star <- getGaussian(x, x.star,
                      l=length.opt, 
                      sigma.f = sigma.f.opt) 
K.star.star <- getGaussian(x.star, x.star,
                           l=length.opt, 
                           sigma.f = sigma.f.opt) 
Ky <- getGaussian(x, x,l=length.opt, 
                  sigma.f = sigma.f.opt) + 
  diag( nrow=length(x))*noise.opt^2
postCov <- K.star.star - 
  t(K.star)%*%solve(Ky)%*%K.star
mu.star <- t(K.star)%*%solve(Ky)%*%y 

# sample some functions from posterior
sample <- replicate(5, 
                    mvrnorm(1, 
                            mu.star, 
                            postCov))%>%
  as.data.frame()
values <- cbind(x=x.star,sample)
values <- melt(values,id="x")

# confidence bands
S2 <- diag(postCov)
lowerbound <- mu.star +2*sqrt(S2)
upperbound <- mu.star -2*sqrt(S2)

# 3.3 Plot Results
gauss.error <- ggplot() + 
  geom_ribbon(x = x.star, 
              aes(ymin= lowerbound, 
                  ymax = upperbound), 
              fill = "grey70") +
  geom_point( data = NULL,aes( x = x, y=y)) +
  geom_line(data = values, 
            aes(x = x, y = value, 
                group=variable, 
                color=variable)) +
  theme_bw() +
  theme(legend.position="none", 
        panel.grid.major = element_blank(), 
        panel.grid.minor = element_blank()) +
  ylim(-2.75, 2.75)
gauss.error
ggsave("gperror_opt.jpg", 
       plot = gauss.error ) # save
