library(expm)
library(gllvm)
library(pROC)
library(truncnorm)

#install.packages("devtools")  # If not installed
#library("devtools")
#devtools::install_github("kidzik/gmf")

library("gmf")

source('FLAIR_wrapper.R')

generate_true_params <- function(p=1000, k=10, q=5, sigma_lambda=1, c_lambda=5, pi_lambda=0.5, 
                                 sigma_beta=1, c_beta=5, seed=NA){
  Lambda_0 <- matrix(rnorm(p*k, 0, sigma_lambda)*rbinom(p*k, 1, pi_lambda), ncol=k)
  Lambda_0[Lambda_0 < -c_lambda] <- -c_lambda
  Lambda_0[Lambda_0 > c_lambda] <- c_lambda
  Beta_0 <- matrix(rnorm(p*q, 0, sigma_beta), ncol=q)
  Beta_0[Beta_0 < -c_beta] <- -c_beta
  Beta_0[Beta_0 > c_beta] <- c_beta
  return(list(Lambda=Lambda_0, Beta=Beta_0))
}

fit_gmf <- function(Y, X, k=2, penaltyV=1, penaltyBeta=1, tol=0.005, method=1){
  if(method==1){
    ptm <- proc.time()
    model.gmf <- gmf(Y = Y, X =X[,-1], p = k, gamma=0.2, maxIter = 1000, family = binomial(),
                     method = "quasi", penaltyV = penaltyV, penaltyU = 1, penaltyBeta=penaltyBeta,
                     intercept = T, tol = tol, init='svd'); 
    time.gmf<- proc.time() - ptm
  }
  else{
    ptm <- proc.time()
    model.gmf = gmf(Y = Y, X =X[,-1], p = k, gamma=0.2, maxIter = 1000, family = binomial(),
                    method = "airwls", penaltyV = penaltyV, penaltyU = 1, penaltyBeta=penaltyBeta,
                    intercept = T, tol = tol, init='svd')
    time.gmf = proc.time() - ptm
  }
  return(list(model=model.gmf, time=time.gmf[3]))
}


run_1_sim_gmf <- function(Lambda_0, Beta_0, n=1000, seed=1, penaltyV=1, penaltyBeta=1, tol=0.005, method=1){
  p <- nrow(Lambda_0); k <- ncol(Lambda_0); q <- ncol(Beta_0)
  set.seed(seed)
  Eta_0 <- matrix(rnorm(n*k), ncol=k)
  X <- cbind(rep(1, n), matrix(rnorm(n*(q-1), 0, 1), ncol=q-1))
  Z_0 <-X %*% t(Beta_0) + Eta_0 %*% t(Lambda_0) 
  P_0 <- 1/(1+exp(-Z_0))
  U <- matrix(runif(n*p), ncol=p)
  Y <- matrix(0, n, p)
  Y[U<P_0] = 1
  gmf_result <- fit_gmf(Y,X, k=k, penaltyV=penaltyV, penaltyBeta=penaltyBeta, tol=tol, method=method)
  return(gmf_result)
  
}

compute_metrics_gmf <- function(Lambda_0_outer, Beta_0, gmf_result){
  
  Lambda.gmf <- gmf_result$model$v
  Lambda.gmf.outer <- Lambda.gmf %*% t(Lambda.gmf)
  p <- nrow(Lambda.gmf); q <- nrow(gmf_result$model$beta)
  B.frobenius <- norm(t(gmf_result$model$beta) - Beta_0, type='F')/norm((Beta_0), type='F')
  B.frobenius.scaled <- norm(t(gmf_result$model$beta) - Beta_0, type='F')/sqrt(p*q)
  Lambda.outer.frobenius <- norm((Lambda_0_outer - Lambda.gmf.outer), type='F') / norm((Lambda_0_outer), type='F')
  computing.time.gmf <- gmf_result$time
  return(list('B_fr'=B.frobenius, 'B_fr_scaled'=B.frobenius.scaled, 'Lambda_outer_fr'=Lambda.outer.frobenius,
              'time'=computing.time.gmf))
}




compute_coverage <- function(Lambda_0_outer_sub, flair_estimate, alpha=0.05){
  flair.cc.qs <- apply(flair_estimate$Lambda_outer_samples_cc, c(1,2), function(x)(quantile(x, probs=c(alpha/2, 1-alpha/2))))
  flair.cc.l <- flair.cc.qs[1,,]
  flair.cc.u <- flair.cc.qs[2,,]
  Lambda_cc_coverage <- (flair.cc.l< Lambda_0_outer_sub & flair.cc.u>Lambda_0_outer_sub)
  Lambda_cc_length <-  flair.cc.u - flair.cc.l
  flair.qs <- apply(flair_estimate$Lambda_outer_samples, c(1,2), function(x)(quantile(x, probs=c(alpha/2, 1-alpha/2))))
  flair.l <- flair.qs[1,,]
  flair.u <- flair.qs[2,,]
  Lambda_coverage <- (flair.l < Lambda_0_outer_sub & flair.u > Lambda_0_outer_sub)
  Lambda_length <-  flair.u - flair.l
  return(list(Lambda_cc_coverage=Lambda_cc_coverage, Lambda_coverage=Lambda_coverage,
              Lambda_cc_length=Lambda_cc_length, Lambda_length=Lambda_length))
}




sample_gllvm <- function(params, vcov_gllvm_fit, n_MC=100){
  p <- nrow(params$theta); k <- ncol(params$theta); q <- ncol(params$Xcoef)+1
  print(c(p, q, k))
  Lambda <- params$theta; Sigma <- params$sigma.lv; Beta <- cbind(params$beta0, params$Xcoef)
  Lambda_outer_samples <- array(NA, dim=c(p, p, n_MC)); Beta_samples <- array(NA, dim=c(p,q, n_MC))
  vcov_lambda <- vcov_gllvm_fit[-c(1:(p*q)), -c(1:(p*q))]
  d <- nrow(vcov_gllvm_fit)
  print(d)
  l <- d - p*q
  e_dec <- eigen(vcov_lambda + + diag(0.000001, l, l))
  diag_m <- sqrt(e_dec$values)
  diag_m[is.na(diag_m)] = 0
  S_vcov_lambda <- e_dec$vectors%*% diag(diag_m)  %*% t(e_dec$vectors)
  print(sum(is.na(S_vcov_lambda )))
  vcov_b <- vcov_gllvm_fit[1:(p*q), 1:(p*q)]
  S_vcov_b =sqrtm(vcov_b) 
  
  print(l)
  for(s in 1:n_MC){
    eps <- S_vcov_lambda %*% matrix(rnorm(l))
    print(Sigma)
    print(eps[1:k])
    Sigma_s <- Sigma + eps[1:k]; Sigma_s[Sigma_s<0.001] <- 0.001
    Lambda_s <- Lambda
    Lambda_s[lower.tri(Lambda_s, diag=F)] <- Lambda_s[lower.tri(Lambda_s, diag=F)] + eps[-c(1:k)]
    Lambda_outer_s <- Lambda_s %*% diag(Sigma_s) %*% t(Lambda_s)
    Lambda_outer_samples[,,s] <- Lambda_outer_s
    Beta_s <- Beta + matrix(S_vcov_b %*% matrix(rnorm(p*q)), ncol=q)
    Beta_samples[,,s] <- Beta_s
  }
  return(list(Beta_samples=Beta_samples, Lambda_outer_samples=Lambda_outer_samples))
}



compute_coverage_gllvm <- function(Lambda_0_sub_outer, Beta_0, samples_gllvm, alpha=0.05){
  gglvm_lambda_qs <- apply(samples_gllvm$Lambda_outer_samples, c(1,2), function(x)(quantile(x, probs=c(alpha/2, 1-alpha/2))))
  gglvm_lambda_l <- gglvm_lambda_qs[1,,]
  gglvm_lambda_u <- gglvm_lambda_qs[2,,]
  Lambda_coverage <- (gglvm_lambda_l< Lambda_0_sub_outer & gglvm_lambda_u>Lambda_0_sub_outer)
  Lambda_length <-  gglvm_lambda_u - gglvm_lambda_l
  
  gglvm_beta_qs <- apply(samples_gllvm$Beta_samples, c(1,2),
                         function(x)(quantile(x, probs=c(alpha/2, 1-alpha/2))))
  gglvm_beta_l  <- gglvm_beta_qs [1,,]
  gglvm_beta_u  <- gglvm_beta_qs [2,,]
  Beta_coverage <- (gglvm_beta_l < Beta_0 & gglvm_beta_u  > Beta_0)
  Beta_length <- gglvm_beta_u - gglvm_beta_l
  return(list(Lambda_coverage=Lambda_coverage, Lambda_length=Lambda_length,
              Beta_coverage=Beta_coverage, Beta_length=Beta_length))
}


