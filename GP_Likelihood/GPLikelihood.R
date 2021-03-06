#################################################

#  Influence of the kernel parameters in a 
#  Gaussian Process on the marginal log-
#  likelihood and its respective elements:
#  Replicates Figure 5.3a and Figure 5.3b in
#  Rasmussen & Williams (2006)
#################################################

# Structure:
# 0. Load Packages 
# 1. Data & Functions
# 2. Figure 5.3a
# 3. Figure 5.3b

# set working directory
setwd("...")

##################################################

# 0.  Load Packages and Data
##################################################

# load packages
if (!require("pacman")) 
  install.packages("pacman"); library("pacman") 
p_load("MASS", 
       "ggplot2", 
       "scales",
       "reshape2")

##################################################

# 1.  Data & Functions
##################################################

# generate sample data from a sine function
# with normally distributed noise
x <- sample(-400:400, 7, replace=F)/100
noise <- (0.1)^2
y = sin(x) + mvrnorm(1, 0 , noise) 

getGaussian <- function(X1, X2 = NULL ,l=1, 
                        sigma.f=1 ) {
  # Computes gaussian kernel for matrices/vectors
  # Allows calibration of all kernel parameters
  # Option for two or one matrices/vectors
  #
  # Args:
  #   X1:      first/only matrix or vector
  #   X2:      second matrix or vector
  #   l:       scaling parameter in exponent
  #   sigma.f: scaling parameter in base
  #
  # Returns:
  #   gaussian kernel matrix
  #
  # calculate distance between all elements
  dist <- switch( is.null(X2) + 1, 
                  as.matrix(pdist(X1, X2)),
                  as.matrix(dist(X1)))
  #compute kernel
  Sigma <- (sigma.f^2)*exp(-0.5*(dist^2/(l^2)))
  return(Sigma)
}

likelihood <- function(length, noise=0.1){
  # Computes marg. log-likelihood for a joint 
  # normal distribution of a Gaussian Process
  # and its nonconstant elements
  # Note: requires function getGaussian() 
  #
  # Args:
  #   X:      matrix or vector of observed 
  #           x-values from training set
  #   length: length parameter for Gaussian
  #            kernel
  #   noise: noise parameter for Gaussian
  #          kernel - default = 0.1
  #
  # Returns:
  #  vector of length, marg. log-
  #  likelihood, datafit term, 
  #  complexity term
  #  
  #
  Ky <- getGaussian(x, l=length) + 
    diag(nrow=length(x))*(noise^2)
  Kyi <- chol2inv(chol(Ky))
  prob <- -0.5*y%*%Kyi%*%y - 
    0.5*log(det(Ky)) - 
    0.5*length(x)*log(2*pi)
  datafit <- -0.5*y%*%Kyi%*%y
  complex <- - 0.5*log(det(Ky))
  return(rbind(length, prob, datafit, complex))
}

##################################################

# 2. Figure 5.3a
##################################################

# compute likelihood elements for different 
# lengthscales
sy <- seq(0.5,10, 0.05)
mydata <- as.data.frame(t(sapply(sy, likelihood)))
colnames(mydata)<- c("length", "prob", 
                     "datafit", "minus complexity")
mydata <- melt(mydata,id="length")


Fig53a <- ggplot(data = mydata, 
                 aes(x=length, 
                     y = value, 
                     linetype = variable,
                     color = variable)) + 
  geom_line() +
  scale_x_log10(breaks = trans_breaks(
    "log10", function(x) 10^x),
    labels = trans_format("log10", 
                          math_format(10^.x))) +
  annotation_logticks(sides="b") + 
  scale_y_continuous(name = "log probability", 
                     limits = c(-100,70)) +
  theme_bw() +
  theme(legend.position = c(0.3, 0.3),
        panel.grid.major = element_blank(), 
        panel.grid.minor = element_blank()) +
  theme(legend.title=element_blank())
Fig53a
ggsave("Rasmussen53a.jpg", plot = Fig53a ) # save


##################################################

#  3. Figure 5.3b
##################################################

# generate data
noise <- (0.1)^2
x.small <- sample(-400:400, 8, replace=F)/100
y.small <- sin(x.small) + mvrnorm(1, 0 , noise) 
x.med <- sample(-400:400, 21, replace=F)/100
y.med <- sin(x.med) + mvrnorm(1, 0 , noise) 
x.large <- sample(-400:400, 55, replace=F)/100
y.large <- sin(x.large) + mvrnorm(1, 0 , noise)

# length parameters for which we want to compute 
# the log probability
sy <- seq(0.05,10, 0.05)

# calculate distance between all elements
prob <- function(length, noise=0.1){
  # Computes marg. log-likelihood for a joint 
  # normal distribution of a Gaussian Process
  # and its nonconstant elements
  # Note: requires function getGaussian() 
  #
  # Args:
  #   length: length parameter for Gaussian
  #            kernel
  #   noise: noise parameter for Gaussian
  #          kernel - default = 0.1
  #
  # Returns:
  #  log-likelihood as scalar
  #  
  #
  Ky <- getGaussian(x, l=length) + 
    diag(nrow=length(x))*(noise^2)
  Kyi <- chol2inv(chol(Ky))
  prob <- -0.5*y%*%Kyi%*%y - 
    0.5*log(det(Ky)) - 
    0.5*length(x)*log(2*pi)
  return(prob)
}

# compute log-proability for different sample sizes
x <- x.small
y <- y.small
mydata.small <- as.data.frame(t(sapply(sy, prob)))
x <- x.med
y <- y.med
mydata.med <- as.data.frame(t(sapply(sy, prob)))
x <- x.large
y <- y.large
mydata.large <- as.data.frame(t(sapply(sy, prob)))

samp.size <- as.data.frame(t(rbind(
  x = sy, s =mydata.small, 
  m=mydata.med, l=mydata.large)))
colnames(samp.size)<- c("samp", "n = 8", 
                        "n = 21", "n = 55")
samp.size <- melt(samp.size,id="samp")

Fig53b <- ggplot(data = samp.size , 
                 aes(x = samp, y = value, 
                     linetype = variable,
                     color = variable)) + 
  geom_line() +
  scale_x_log10(name = "length", 
                breaks = trans_breaks(
                  "log10", function(x) 10^x),
                labels = trans_format(
                  "log10", math_format(10^.x))) +
  annotation_logticks(sides="b") + 
  scale_y_continuous(name = "log probability", 
                     limits = c(-100,70)) +
  theme_bw() +
  theme(panel.grid.major = element_blank(), 
        panel.grid.minor = element_blank()) +
  theme(legend.position = c(0.3, 0.3)) +
  theme(legend.title=element_blank())
Fig53b
ggsave("Rasmussen53b.jpg", plot = Fig53b ) # save
