library(expm)
library(gllvm)
library(mirtjml)
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
  #print(c(p, q, k))
  Lambda <- params$theta; Sigma <- params$sigma.lv; Beta <- cbind(params$beta0, params$Xcoef)
  Lambda_outer_samples <- array(NA, dim=c(p, p, n_MC)); Beta_samples <- array(NA, dim=c(p,q, n_MC))
  vcov_lambda <- vcov_gllvm_fit[-c(1:(p*q)), -c(1:(p*q))]
  d <- nrow(vcov_gllvm_fit)
  #print(d)
  l <- d - p*q
  e_dec <- eigen(vcov_lambda + diag(0.000001, l, l))
  diag_m <- sqrt(e_dec$values)
  diag_m[is.na(diag_m)] = 0
  S_vcov_lambda <- e_dec$vectors%*% diag(diag_m)  %*% t(e_dec$vectors)
  #print(sum(is.na(S_vcov_lambda )))
  #vcov_b <- vcov_gllvm_fit[1:(p*q), 1:(p*q)]
  #S_vcov_b =sqrtm(vcov_b) 
  #print(l)
  for(s in 1:n_MC){
    eps <- S_vcov_lambda %*% matrix(rnorm(l))
    #print(Sigma)
    #print(eps[1:k])
    Sigma_s <- Sigma + eps[1:k]; Sigma_s[Sigma_s<0.001] <- 0.001
    Lambda_s <- Lambda
    Lambda_s[lower.tri(Lambda_s, diag=F)] <- Lambda_s[lower.tri(Lambda_s, diag=F)] + eps[-c(1:k)]
    Lambda_outer_s <- Lambda_s %*% diag(Sigma_s) %*% t(Lambda_s)
    Lambda_outer_samples[,,s] <- Lambda_outer_s
    #Beta_s <- Beta + matrix(S_vcov_b %*% matrix(rnorm(p*q)), ncol=q)
    #Beta_samples[,,s] <- Beta_s
  }
  return(list(Lambda_outer_samples=Lambda_outer_samples))
}



compute_coverage_gllvm <- function(Lambda_0_sub_outer, Beta_0, samples_gllvm, alpha=0.05){
  gglvm_lambda_qs <- apply(samples_gllvm$Lambda_outer_samples, c(1,2), function(x)(quantile(x, probs=c(alpha/2, 1-alpha/2))))
  gglvm_lambda_l <- gglvm_lambda_qs[1,,]
  gglvm_lambda_u <- gglvm_lambda_qs[2,,]
  Lambda_coverage <- (gglvm_lambda_l< Lambda_0_sub_outer & gglvm_lambda_u>Lambda_0_sub_outer)
  Lambda_length <-  gglvm_lambda_u - gglvm_lambda_l
  
  #gglvm_beta_qs <- apply(samples_gllvm$Beta_samples, c(1,2),
  #                       function(x)(quantile(x, probs=c(alpha/2, 1-alpha/2))))
  #gglvm_beta_l  <- gglvm_beta_qs [1,,]
  #gglvm_beta_u  <- gglvm_beta_qs [2,,]
  #Beta_coverage <- (gglvm_beta_l < Beta_0 & gglvm_beta_u  > Beta_0)
  #Beta_length <- gglvm_beta_u - gglvm_beta_l
  return(list(Lambda_coverage=Lambda_coverage, Lambda_length=Lambda_length))
              #Beta_coverage=Beta_coverage, Beta_length=Beta_length))
}


compute_performance_gmf <- function(gmf_fit, Beta_0, Lambda_0_outer){
  p <- nrow(Beta_0)
  q <- ncol(Beta_0)
  out <- c()
  Lambda_gmf <- gmf_fit$v
  Lambda_gmf_outer_estimate <- Lambda_gmf %*% t(Lambda_gmf)
  out[1] <- norm(t(gmf_fit$beta) - Beta_0, type='F')/sqrt(p*q)
  out[2] <- norm((Lambda_0_outer - Lambda_gmf_outer_estimate), type='F') / norm((Lambda_0_outer), type='F')
  rmses <- sqrt(rowMeans((t(gmf_fit$beta) - Beta_0)^2))
  out[3] <- median(rmses)
  out[4] <- max(rmses)
  return(out)
}


compute_perfomance_jml <- function(
    jml_fit, X, Beta_0, Lambda_0_outer){
  out <- c()
  p <- nrow(jml_fit$A_hat)
  out[1] <- norm((Beta_0 - jml_fit$d_hat), type='F') / sqrt(p)
  err_beta_j <- sqrt((Beta_0 - jml_fit$d_hat)^2)
  out[11] <- median(err_beta_j)
  out[12] <- max(err_beta_j)
  out[2] <- norm((Lambda_0_outer - jml_fit$A_hat %*% (crossprod(jml_fit$theta_hat)/n)
                  %*% t(jml_fit$A_hat)), type='F') /
    norm((Lambda_0_outer), type='F')
  return(out)
  
}

compute_vars_jml_hat <- function(jml_fit, X, D, R){
  # this function is not optimized for computational efficiency but coded for clarity
  p <- nrow(jml_fit$Gammahat)
  P <- ncol(jml_fit$Gammahat) + ncol(jml_fit$Betahat) + ncol(jml_fit$Ahat)
  n <- nrow(jml_fit$Thetahat)
  Vars_hat <- list()
  SEs_hat <- matrix(NA, p, P)
  Tp <- ncol(R)
  for(j in 1:p){
    gamma_j <- jml_fit$Gammahat[j,]
    beta_j <- jml_fit$Betahat[j,]
    a_j <- jml_fit$Ahat[j,]
    u_j <- c(gamma_j, beta_j, a_j)
    
    Phi_j_hat <- matrix(0, P, P)
    for(i in 1:n){
      theta_i <- jml_fit$Thetahat[i,]
      x_i <- X[i,]
      for(t in 1:Tp){
        e_it <- c(D[t,], x_i, theta_i)
        m_ijt <- sum(u_j * e_it)
        p_ijt <- 1/(1+exp(-m_ijt))
        w_ijt <- p_ijt*(1-p_ijt)
        Phi_j_hat <- Phi_j_hat + R[i,t]*w_ijt* e_it %*% t(e_it)
      }
    }
    Phi_j_hat <- Phi_j_hat/n
    Vars_hat[[j]] <- solve(Phi_j_hat)
    SEs_hat[j,] <- sqrt(diag(Vars_hat[[j]])) / sqrt(n)
  }
  return(list(Vars_hat=Vars_hat, SEs_hat=SEs_hat))
}


sample_Lambda_jml_old <- function(jml_fit, V_hats, n_MC=100, outer=T, subsample_index=1:100){
  Lambdas_hat <- jml_fit$Ahat
  n <- nrow(jml_fit$Thetahat)
  p <- nrow(Lambdas_hat); q <- ncol(Lambdas_hat)
  p_sample <- length(subsample_index)
  Lambdas <- array(NA, dim=c(p,q, n_MC))
  Lambdas_outer <- array(NA, dim=c(p_sample,p_sample, n_MC))
  
  for(i in 1:p){
    V_hat <- V_hats[[i]]
    V_hat_sq <- sqrtm(V_hat)
    if(sum(is.na(V_hat_sq))){
      print('NA')
    }
    Lambda_hat <- Lambdas_hat[i,]
    for(s in 1:n_MC){
      Lambda_s <- Lambda_hat + V_hat_sq %*% rnorm(q) / sqrt(n)
      Lambdas[i,,s] <- Lambda_s
      
    }
    if(outer){
      for(s in 1:n_MC){
        Lambdas_outer[,,s] <- Lambdas[subsample_index,,s] %*% diag(diag(crossprod(jml_fit$Thetahat)/n)) %*% t(Lambdas[subsample_index,,s])
        
      }
    }
  }
  return(list(Lambdas=Lambdas, Lambdas_outer=Lambdas_outer))
}


sample_Lambda_jml <- function(jml_fit, V_hats, n_MC=100, outer=T, subsample_index=1:100){
  Lambdas_hat <- jml_fit$Ahat
  n <- nrow(jml_fit$Thetahat)
  p <- nrow(Lambdas_hat); q <- ncol(Lambdas_hat)
  p_sample <- length(subsample_index)
  Lambdas_outer <- array(NA, dim=c(p_sample,p_sample, n_MC))
  Lambda_sample <- matrix(NA, p_sample, q)
  
  V_hats_sqrt <- list()
  for(i in 1:p_sample){
    i_index <- subsample_index[i]
    V_hat <- V_hats[[i_index]]
    V_hats_sqrt[[i]] <- sqrtm(V_hat)
    if(sum(is.na(V_hats_sqrt[[i]]))){
      print('NA')
    }
  }
  for(s in 1:n_MC){
    for(i in 1:p_sample){
      i_index <- subsample_index[i]
      Lambda_hat <- Lambdas_hat[i_index,]
      Lambda_s <- Lambda_hat +  V_hats_sqrt[[i]] %*% rnorm(q) / sqrt(n)
      Lambda_sample[i,] <- Lambda_s
    }
    Lambdas_outer[,,s] <- 1/n* Lambda_sample %*% crossprod(jml_fit$Thetahat) %*% t(Lambda_sample)
    
  }
  return(list(Lambdas_outer=Lambdas_outer))
}

compute_performance_gmf <- function(gmf_fit, Beta_0, Lambda_0_outer){
  p <- nrow(Beta_0)
  q <- ncol(Beta_0)
  out <- c()
  Lambda_gmf <- gmf_fit$v
  Lambda_gmf_outer_estimate <- Lambda_gmf %*% t(Lambda_gmf)
  out[1] <- norm(t(gmf_fit$beta) - Beta_0, type='F')/sqrt(p*q)
  out[2] <- norm((Lambda_0_outer - Lambda_gmf_outer_estimate), type='F') / norm((Lambda_0_outer), type='F')
  rmses <- sqrt(rowMeans((t(gmf_fit$beta) - Beta_0)^2))
  out[3] <- median(rmses)
  out[4] <- max(rmses)
  return(out)
}

compute_performance_flair <- function(FLAIR_estimate, Beta_0, Lambda_0_outer, Lambda_0_outer_sub, exclude_intercept=F){
  out <- c()
  p <- ncol(Lambda_0_outer)
  if(exclude_intercept){
    Beta_0 <- Beta_0[,-1]
  }
  q <- ncol(Beta_0)
  if(exclude_intercept){
    out[1] <- norm((Beta_0 - FLAIR_estimate$Beta_tilde[,-1]), type='F') / sqrt(p * q) 
    err_beta_j <- sqrt(rowMeans((Beta_0 - FLAIR_estimate$Beta_tilde[,-1])^2))
    out[11] <- median(err_beta_j)
    out[12] <- max(err_beta_j)
  }
  else{
    out[1] <- norm((Beta_0 - FLAIR_estimate$Beta_tilde), type='F') / sqrt(p * q) 
    err_beta_j <- sqrt(rowMeans((Beta_0 - FLAIR_estimate$Beta_tilde)^2))
    out[11] <- median(err_beta_j)
    out[12] <- max(err_beta_j)
  }
  out[2] <- norm((Lambda_0_outer - FLAIR_estimate$Lambda_outer_mean_cc), type='F') /
    norm((Lambda_0_outer), type='F') 
  alpha <- 0.05
  cc <- compute_coverage(Lambda_0_outer_sub, FLAIR_estimate, alpha=alpha)
  out[3] <- mean(cc$Lambda_coverage)
  out[4] <- mean(cc$Lambda_cc_coverage)
  out[5] <- mean(cc$Lambda_length)
  out[6] <- mean(cc$Lambda_cc_length)
  if(exclude_intercept){
    Beta_cc_cis <- compute_ci_Beta(FLAIR_estimate$Beta_tilde[,-1], FLAIR_estimate$Vs[2:ncol(X)], 
                                   FLAIR_estimate$rho_max, alpha=0.05) 
    Beta_cis <- compute_ci_Beta(FLAIR_estimate$Beta_tilde[,-1], FLAIR_estimate$Vs[2:ncol(X)], 
                                1, alpha=0.05) 
  } else {
    Beta_cc_cis <- compute_ci_Beta(FLAIR_estimate$Beta_tilde, FLAIR_estimate$Vs[1:ncol(X)], 
                                   FLAIR_estimate$rho_max, alpha=0.05) 
    Beta_cis <- compute_ci_Beta(FLAIR_estimate$Beta_tilde, FLAIR_estimate$Vs[1:ncol(X)], 
                                1, alpha=0.05) 
  }
  
  
  out[7] <- mean((Beta_cis$ls < Beta_0) & (Beta_cis$us > Beta_0))
  out[8] <- mean((Beta_cc_cis$ls < Beta_0) & (Beta_cc_cis$us > Beta_0))
  out[9] <- mean(Beta_cis$us - Beta_cis$ls)
  out[10] <- mean(Beta_cc_cis$us - Beta_cc_cis$ls)
  return(out)
}

compute_ci_Beta <- function(Beta_hat, Vs_hat, rho=1, alpha=0.05){
  alpha_2 = alpha/2
  q <- qnorm(1-alpha_2)
  qs <- q*rho*sqrt(Vs_hat);
  ls <- Beta_hat - qs; us <- Beta_hat + qs
  return(list(ls=ls, us=us))
}





compute_perfomance_lvhml <- function(
    lvhml_fit, X, Beta_0, Lambda_0_outer, Lambda_0_outer_sub, R, subsample_index=1:100, n_MC=500){
  out <- c()
  q <- ncol(Beta_0); p <- nrow(Beta_0)
  out[1] <- norm((Beta_0 - lvhml_fit$Betahat), type='F') / sqrt(p*q)
  err_beta_j <- sqrt(rowMeans((Beta_0 - lvhml_fit$Betahat)^2))
  out[11] <- median(err_beta_j)
  out[12] <- max(err_beta_j)
  out[2] <- norm((Lambda_0_outer - lvhml_fit$Ahat %*% (crossprod(lvhml_fit$Thetahat)/n)
                  %*% t(lvhml_fit$Ahat)), type='F') /
    norm((Lambda_0_outer), type='F')
  lvhml_beta_cis <- compute_ci_Beta(lvhml_fit$Betahat, lvhml_fit$SE^2)
  out[8] <- mean((lvhml_beta_cis$us > Beta_0) & (lvhml_beta_cis$ls < Beta_0) )
  out[10] <- mean(lvhml_beta_cis$us - lvhml_beta_cis$ls)
  d_i1 <- c(1,0)
  d_i2 <- c(0,1)
  D <- rbind(d_i1, d_i2)
  var_est <- compute_vars_lvhml_hat(lvhml_fit, X, D, R)
  Lambdas_vars <- list()
  index_loading <- 1:(ncol(D)+ q)
  for(j in 1:p){
    Lambdas_vars[[j]] <- var_est$Vars_hat[[j]][-index_loading, -index_loading]
  }
  
  lvhml_lambda_samples <- sample_Lambda_lvhml(lvhml_fit, Lambdas_vars, n_MC, T,
                                              subsample_index=subsample_index)
  lvhml_lambda_qs <- apply(lvhml_lambda_samples$Lambdas_outer, c(1,2), 
                           function(x) (quantile(x, probs=c(0.025, 0.975))))
  out[4] <- mean((lvhml_lambda_qs[1,,]< Lambda_0_outer_sub) & (lvhml_lambda_qs[2,,]>Lambda_0_outer_sub))
  out[6] <- mean(lvhml_lambda_qs[2,,]- lvhml_lambda_qs[1,,])
  return(out)
  
}

compute_vars_lvhml_hat <- function(lvhml_fit, X, D, R){
  # this function is not optimized for computational efficiency but coded for clarity
  p <- nrow(lvhml_fit$Gammahat)
  P <- ncol(lvhml_fit$Gammahat) + ncol(lvhml_fit$Betahat) + ncol(lvhml_fit$Ahat)
  n <- nrow(lvhml_fit$Thetahat)
  Vars_hat <- list()
  SEs_hat <- matrix(NA, p, P)
  Tp <- ncol(R)
  for(j in 1:p){
    gamma_j <- lvhml_fit$Gammahat[j,]
    beta_j <- lvhml_fit$Betahat[j,]
    a_j <- lvhml_fit$Ahat[j,]
    u_j <- c(gamma_j, beta_j, a_j)
    
    Phi_j_hat <- matrix(0, P, P)
    for(i in 1:n){
      theta_i <- lvhml_fit$Thetahat[i,]
      x_i <- X[i,]
      for(t in 1:Tp){
        e_it <- c(D[t,], x_i, theta_i)
        m_ijt <- sum(u_j * e_it)
        p_ijt <- 1/(1+exp(-m_ijt))
        w_ijt <- p_ijt*(1-p_ijt)
        Phi_j_hat <- Phi_j_hat + R[i,t]*w_ijt* e_it %*% t(e_it)
      }
    }
    Phi_j_hat <- Phi_j_hat/n
    Vars_hat[[j]] <- solve(Phi_j_hat)
    SEs_hat[j,] <- sqrt(diag(Vars_hat[[j]])) / sqrt(n)
  }
  return(list(Vars_hat=Vars_hat, SEs_hat=SEs_hat))
}



sample_Lambda_lvhml <- function(lvhml_fit, V_hats, n_MC=100, outer=T, subsample_index=1:100){
  Lambdas_hat <- lvhml_fit$Ahat
  n <- nrow(lvhml_fit$Thetahat)
  p <- nrow(Lambdas_hat); q <- ncol(Lambdas_hat)
  p_sample <- length(subsample_index)
  Lambdas_outer <- array(NA, dim=c(p_sample,p_sample, n_MC))
  Lambda_sample <- matrix(NA, p_sample, q)
  
  V_hats_sqrt <- list()
  for(i in 1:p_sample){
    i_index <- subsample_index[i]
    V_hat <- V_hats[[i_index]]
    V_hats_sqrt[[i]] <- sqrtm(V_hat)
    if(sum(is.na(V_hats_sqrt[[i]]))){
      print('NA')
    }
  }
  for(s in 1:n_MC){
    for(i in 1:p_sample){
      i_index <- subsample_index[i]
      Lambda_hat <- Lambdas_hat[i_index,]
      Lambda_s <- Lambda_hat +  V_hats_sqrt[[i]] %*% rnorm(q) / sqrt(n)
      Lambda_sample[i,] <- Lambda_s
    }
    Lambdas_outer[,,s] <- 1/n* Lambda_sample %*% crossprod(lvhml_fit$Thetahat) %*% t(Lambda_sample)
    
  }
  return(list(Lambdas_outer=Lambdas_outer))
}

compute_performance_gllvm <- function(gllvm_fit, Beta_0, Lambda_0_outer, 
                                      Lambda_0_outer_sub, n_MC=500, coverage=F) {
  gllvm_fit.est <- gllvm_fit$params
  q <- ncol(Beta_0); p <- nrow(Beta_0)
  out <- c()
  out[1] <- norm((Beta_0 - cbind(gllvm_fit.est$beta0, gllvm_fit.est$Xcoef)), type='F') / sqrt(p * q) 
  out[2] <- norm((Lambda_0_outer - gllvm_fit.est$theta %*% diag(gllvm_fit.est$sigma.lv^2) %*% t(gllvm_fit.est$theta)), type='F') / norm(Lambda_0_outer, type='F')
  err_beta_j <- sqrt(rowMeans((Beta_0 - cbind(gllvm_fit.est$beta0, gllvm_fit.est$Xcoef))^2))
  out[11] <- median(err_beta_j)
  out[12] <- max(err_beta_j)
  if(coverage){
    se_fit <- se.gllvm(gllvm_fit)
    gllvm_beta_cis <- compute_ci_Beta(cbind(gllvm_fit.est$beta0, gllvm_fit.est$Xcoef), 
                                      cbind(se_fit$sd$beta0^2, se_fit$sd$Xcoef))
    out[8] <- mean((gllvm_beta_cis$us > Beta_0) & (gllvm_beta_cis$ls < Beta_0) )
    out[10] <- mean(gllvm_beta_cis$us - gllvm_beta_cis$ls)
    
    vcov_gllvm <- vcov.gllvm(gllvm_fit)
    samples_gllvm <- sample_gllvm(gllvm_fit.est, vcov_gllvm, n_MC=n_MC)
    
    cc_gllvm <- compute_coverage_gllvm (Lambda_0_outer, Beta_0, samples_gllvm,
                                        alpha=0.05)
    out[4] <- mean(cc_gllvm$Lambda_coverage)
    out[6] <- mean(cc_gllvm$Lambda_length)
  }

  
  return(out)
}


