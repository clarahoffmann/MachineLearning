#################################################

#  Gaussian Process with Varying Hyperparameters
#  Conditioned on Simulated Noisy Data
#  - based on Rasmussen & Williams (2006)
#    and Kevin P. Murphy (2012)
#################################################

# Structure:
# 0. Load Packages and Generate Data
# 1. Defining the Kernel & Plot Function
# 2. Plot Different Parametrizations

##################################################

# 0.  Load Packages and Generate Data
##################################################

if (!require("pacman")) 
  install.packages("pacman"); library("pacman") 
p_load("MASS", 
       "ggplot2", 
       "gridExtra")
setwd("...")

# observed values
x <- sample(-400:400, 20, replace=F)/100
# x values for prediction 
x.star <- seq(-5,5,len=100) 
# true noise
t.noise <- (0.1)^2
# y values
y = sin(x) + mvrnorm(1, 0 , t.noise) 

##################################################

# 1.  Defining the Kernel & Plot Function
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
  Zero <- matrix( rep( 0, 
                       len=length(X1)*length(X2)), 
                  nrow = length(X1))
  A <- Zero + X1
  B <- t(t(Zero)+X2)
  Sigma <- (sigma.f^2)*exp(-((A-B)^2)/(2*(l^2)))
  return(Sigma)
}

# Plot function for 1D Gaussian Process
getGpPlot <- function(x, x.star, y, noise , 
                      l , sigma.f ){
  K.star <- getGaussian(x, x.star, l=l , 
                        sigma.f= sigma.f) 
  K.star.star <- getGaussian(x.star, x.star, l=l , 
                             sigma.f= sigma.f) 
  Ky <- getGaussian(x, x, l=l , sigma.f= sigma.f) + 
    diag( nrow=length(x))*(noise^2)
  Kyi <- chol2inv(chol((Ky)))
  postCov <- K.star.star - 
    t(K.star)%*%Kyi%*%K.star
  mu.star <- t(K.star)%*%Kyi%*%y
  S2 <- diag(postCov)
  lowerbound <- mu.star +2*sqrt(S2)
  upperbound <- mu.star -2*sqrt(S2)
  gaussian.plot <- ggplot() + 
    geom_ribbon(x = x.star, 
                aes(ymin = lowerbound, 
                    ymax = upperbound), 
                fill = "grey70") +
    geom_point( data = NULL,aes( x = x, y = y), 
                shape=3) +
    geom_line(data = NULL, aes(x = x.star, 
                               y = mu.star)) +
    theme(legend.position="none") + 
    scale_x_continuous(name = "x", 
                       limits = c(-5,5)) +
    scale_y_continuous(name = "y", 
                       limits = c(-2,2)) +
    theme_bw() +
    theme(legend.position="none", 
          panel.grid.major = element_blank(), 
          panel.grid.minor = element_blank())
  return(gaussian.plot)
}

##################################################

# 2.  Plot Different Parametrizations
##################################################

# a.) varying noise sigma.y while 
#     keeping the other hyperparameters const.
figurea1 <- getGpPlot(x = x, x.star = x.star, 
                      y = y, noise = (0.025^2),  
                      l = 1, sigma.f = 1)
figurea2 <- getGpPlot(x = x, x.star = x.star, 
                      y = y, noise = (0.5^2),  
                      l = 1, sigma.f = 1)
figurea3 <- getGpPlot(x = x, x.star = x.star, 
                      y = y, noise = (1^2),  
                      l = 1, sigma.f = 1)

figurea <- grid.arrange( figurea1, figurea2, 
                         figurea3, nrow = 1)
figurea

# b.) varying noise sigma.f while 
#     keeping the other hyperparameters const.
figureb1 <- getGpPlot(x = x, x.star = x.star, 
                      y = y, noise = (0.5^2),  
                      l = 1, sigma.f = 0.1) 
figureb2 <- getGpPlot(x = x, x.star = x.star, 
                      y = y, noise = (0.5^2),  
                      l = 1, sigma.f = 0.3)
figureb3 <- getGpPlot(x = x, x.star = x.star, 
                      y = y, noise = (0.5^2),  
                      l = 1, sigma.f = 0.7)

figureb <- grid.arrange( figureb1, figureb2, 
                         figureb3, nrow = 1)
figureb

# c.) varying noise sigma.f while 
#     keeping the other hyperparameters const.
figurec1 <- getGpPlot(x = x, x.star = x.star, 
                      y = y, noise = (0.5^2),  
                      l = 0.1, sigma.f = 0.7)
figurec2 <- getGpPlot(x = x, x.star = x.star, 
                      y = y, noise = (0.5^2),  
                      l = 0.5, sigma.f = 0.7)
figurec3 <- getGpPlot(x = x, x.star = x.star, 
                      y = y, noise = (0.5^2),  
                      l = 1, sigma.f = 0.7)

figurec <- grid.arrange( figurec1, figurec2, 
                         figurec3, nrow = 1)
figurec 

figure <- grid.arrange( figurea, figureb, 
                        figurec, nrow = 3)
ggsave("noisyhyper.jpg", plot = figure ) # save
