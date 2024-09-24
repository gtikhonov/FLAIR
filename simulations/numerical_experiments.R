
setwd("~/simulations")
source('helpers_simulations.R')

# DGM params
k <- 10
q <- 10
p <- 10000
n <- 1000
n_sim <- 50
sigma_lambda=sqrt(0.5); sigma_beta=sqrt(0.5); pi_lambda=0.5; pi_beta=0.5
# generate true params
set.seed(123)
Lambda_0 <- matrix(rtruncnorm(p*k, a=-5, b=5, 0, sigma_lambda)*rbinom(p*k, 1, pi_lambda), ncol=k)
Lambda_0_outer <- Lambda_0 %*% t(Lambda_0)
subsample_index = 1:100
Lambda_0_outer_sub <- Lambda_0_outer[subsample_index, subsample_index]
Beta_0 <- matrix(rtruncnorm(p*q, a=-5, b=5, 0, sigma_beta)*rbinom(p*q, 1, pi_beta), ncol=q)

norm_Beta <- norm((Beta_0), type='F') 
norm_Lambda_outer <- norm((Lambda_0_outer), type='F') 


## - flair ####

# arrays to store results
B_coverage_95 <- array(0, dim=c(p, q, n_sim))
B_cc_coverage_95 <- array(0, dim=c(p, q, n_sim))
B_length_95 <- array(0, dim=c(p, q, n_sim))
B_cc_length_95 <- array(0, dim=c(p, q, n_sim))
Lambda_outer_coverage_95 <- array(0, dim=c(100, 100, n_sim))
Lambda_outer_cc_coverage_95 <- array(0, dim=c(100, 100, n_sim))
Lambda_outer_length_95 <- array(0, dim=c(100, 100, n_sim))
Lambda_outer_cc_length_95 <- array(0, dim=c(100, 100, n_sim))
computing_time_flair <- rep(0, n_sim)
B_tilde_frobenius_scaled <- rep(0, n_sim)
B_tilde_frobenius <- rep(0, n_sim)
Lambda_tilde_outer_frobenius <- rep(0, n_sim)
Sigma_tilde_outer_frobenius <- rep(0, n_sim)

test_flair =T

for(sim in 1:n_sim){
  print(sim)
  # generate data
  set.seed(sim)
  Eta_0 <- matrix(rnorm(n*k), ncol=k)
  X <- cbind(rep(1, n), matrix(rnorm(n*(q-1), 0, 1), ncol=q-1))
  Z_0 <-X %*% t(Beta_0) + Eta_0 %*% t(Lambda_0) 
  P_0 <- 1/(1+exp(-Z_0))
  U <- matrix(runif(n*p), ncol=p)
  Y <- matrix(0, n, p)
  Y[U<P_0] = 1
  if(test_flair){
    # run flair
    ptm <- proc.time()
    flair_estimate <- FLAIR_wrapper(
      Y, X, k_max=20, k=k, method_rho = 'max', eps=0.001, alternate_max=10, max_it=100, tol=0.01, 
      post_process=T, subsample_index = subsample_index, n_MC=300, C_lambda=10,  C_mu=10, C_beta=10, sigma=1.626,
      observed=matrix(1, n, p), randomized_svd = T, loss_tol=0.001)
    time_flair = proc.time() - ptm
    
    # save results
    computing_time_flair[sim] <- time_flair[3]
    B_tilde_frobenius_scaled[sim] <- norm((Beta_0 - flair_estimate$Beta_tilde), type='F') / sqrt(p * q) 
    B_tilde_frobenius[sim] <- norm((Beta_0 - flair_estimate$Beta_tilde), type='F') / norm((Beta_0), type='F') 
    Lambda_tilde_outer_frobenius[sim] <- norm((Lambda_0_outer - flair_estimate$Lambda_tilde %*% t(flair_estimate$Lambda_tilde)), type='F') / norm((Lambda_0_outer), type='F') 
    Sigma_tilde_outer_frobenius[sim] <- norm((Lambda_0_outer - flair_estimate$Lambda_outer_mean_cc), type='F') /  norm((Lambda_0_outer), type='F') 
    alpha <- 0.05
    cc <- compute_coverage(Lambda_0_outer_sub, Beta_0, flair_estimate, alpha=alpha)
    Lambda_outer_coverage_95[,,sim] <- cc$Lambda_coverage
    Lambda_outer_cc_coverage_95[,,sim] <- cc$Lambda_cc_coverage
    Lambda_outer_length_95[,,sim] <- cc$Lambda_length
    Lambda_outer_cc_length_95[,,sim] <- cc$Lambda_cc_length
    
    Beta_cc_cis <- compute_ci_Beta(flair_estimate$Beta_tilde, flair_estimate$Vs[1:ncol(X)], 
                                   flair_estimate$rho_max, alpha=0.05) 
    Beta_cis <- compute_ci_Beta(flair_estimate$Beta_tilde, flair_estimate$Vs[1:ncol(X)], 
                                1, alpha=0.05) 
    B_coverage_95[,,sim] <- (Beta_cis$ls < Beta_0) & (Beta_cis$us > Beta_0)
    B_cc_coverage_95[,,sim] <- (Beta_cc_cis$ls < Beta_0) & (Beta_cc_cis$us > Beta_0)
    B_length_95[,,sim] <- Beta_cis$us - Beta_cis$ls
    B_cc_length_95[,,sim] <- Beta_cc_cis$us - Beta_cc_cis$ls
  }
  print(mean(Sigma_tilde_outer_frobenius[1:sim]))
  print(mean(Lambda_outer_coverage_95[,,sim]))
  print(mean(Lambda_outer_cc_coverage_95[,,sim]))
  print(mean(B_tilde_frobenius_scaled[1:sim]))
  print(mean(B_coverage_95[,,sim]))
  print(mean(B_cc_coverage_95[,,sim]))
  print(mean(computing_time_flair[1:sim]))
  
}


mean(B_tilde_frobenius_scaled); mean(Sigma_tilde_outer_frobenius); mean(computing_time_flair)
mean(Lambda_outer_cc_coverage_95); mean(B_cc_coverage_95); mean(Lambda_outer_coverage_95); mean(B_coverage_95); 

## - GMF ####
library("gmf")

# arrays to store results
B_gmf_newton_frobenius <- rep(0, n_sim)
B_gmf_newton_frobenius_scaled <- rep(0, n_sim)
Lambda_gmf_newton_outer_frobenius <- rep(0, n_sim)
computing_time_gmf_newton <- rep(0, n_sim)

B_gmf_airwls_frobenius <- rep(0, n_sim)
B_gmf_airwls_frobenius_scaled <- rep(0, n_sim)
Lambda_gmf_airwls_outer_frobenius <- rep(0, n_sim)
computing_time_gmf_airwls <- rep(0, n_sim)

# hyperparams
penalties_Beta <- c(0, 0.5, 1, 5, 10)
penalties_Lambda <- c(0, 0.5, 1, 5, 10)
hyperparams <- data.frame(expand.grid(penalties_Beta, penalties_Lambda))

for(sim in 1:n_sim){
  print(sim)
  set.seed(sim)
  # newton
  ## optmize hyperparams
  aucs_newton <- apply(hyperparams, 1, function(x) 
    run_1_sim_gmf_train_test(Lambda_0, Beta_0, n=n, seed=sim, penaltyV=x[2],
                             penaltyBeta=x[1], tol=0.001, method=1))
  best_conf <- which.max(aucs_newton)
  res_sim_newton <- run_1_sim_gmf(Lambda_0, Beta_0, n=n, seed=id, penaltyV=hyperparams[best_conf, 2], 
                                  penaltyBeta=hyperparams[best_conf, 1], tol=0.001, method=1)
  metric_sim_newton <- compute_metrics_gmf(Lambda_0_outer, Beta_0, res_sim_newton)
  res1 = list(seed=id, penaltyV=hyperparams[best_conf, 3],   penaltyBeta=hyperparams[best_conf, 2], method=1)
  res1 <- c(res1, metric_sim_newton)
  B_gmf_newton_frobenius[sim] <- metric_sim_newton$B_fr
  B_gmf_newton_frobenius_scaled[sim] <- metric_sim_newton$B_fr_Scaled
  Lambda_gmf_newton_outer_frobenius[sim] <- metric_sim_newton$Lambda_outer_fr
  computing_time_gmf_newton[sim] <- metric_sim_newton$time
  
  # airwls
  ## optmize hyperparams
  aucs_airwls <- apply(hyperparams, 1, function(x) 
    run_1_sim_gmf_train_test(Lambda_0, Beta_0, n=n, seed=sim, penaltyV=x[2],
                             penaltyBeta=x[1], tol=0.001, method=2))
  best_conf <- which.max(aucs_airwls)
  res_sim_airwls <- run_1_sim_gmf(Lambda_0, Beta_0, n=n, seed=id, penaltyV=hyperparams[best_conf, 2], 
                                  penaltyBeta=hyperparams[best_conf, 1], tol=0.001, method=2)
  metric_sim_airwls <- compute_metrics_gmf(Lambda_0_outer, Beta_0, res_sim_airwls)
  res2 = list(seed=id, penaltyV=hyperparams[best_conf, 3],   penaltyBeta=hyperparams[best_conf, 2], 
              method=2)
  res2 <- c(res2, metric_sim_airwls)
  B_gmf_airwls_frobenius[sim] <- metric_sim_airwls$B_fr
  B_gmf_airwls_frobenius_scaled[sim] <- metric_sim_airwls$B_fr_Scaled
  Lambda_gmf_airwls_outer_frobenius[sim] <- metric_sim_airwls$Lambda_outer_fr
  computing_time_gmf_airwls[sim] <- metric_sim_airwls$time
}

# gmf - newton
mean(B_gmf_newton_frobenius_scaled); mean(Lambda_gmf_newton_outer_frobenius); mean(computing_time_gmf_newton)
# gmf - airwls
mean(B_gmf_airwls_frobenius_scaled); mean(Lambda_gmf_airwls_outer_frobenius); mean(computing_time_gmf_airwls)

## - GLLVM-EVA ####
k <- 2
q <- 2
p <- 100
n <- 100
n_sim <- 50 
sigma_lambda=sqrt(1); sigma_beta=sqrt(1); pi_lambda=1; pi_beta=1
set.seed(123)
Lambda_0 <- matrix(rtruncnorm(p*k, a=-5, b=5, 0, sigma_lambda)*rbinom(p*k, 1, pi_lambda), ncol=k)
Lambda_0_outer <- Lambda_0 %*% t(Lambda_0)
Beta_0 <- matrix(rtruncnorm(p*q, a=-5, b=5, 0, sigma_beta)*rbinom(p*q, 1, pi_beta), ncol=q)

# arrays to store results
Lambda_gllvm_outer_coverage_95 <- array(0, dim=c(p, p, n_sim))
Lambda_gllvm_outer_length_95 <- array(0, dim=c(p, p, n_sim))
B_gllvm_coverage_95 <- array(0, dim=c(p, q, n_sim))
B_gllvm_length_95 <- array(0, dim=c(p, q, n_sim))
B_gllvm_frobenius_scaled <- rep(0, n_sim)
B_gllvm_frobenius <- rep(0, n_sim)
Lambda_gllvm_outer_frobenius <- rep(0, n_sim)
computing_time_gllvm <- rep(0, n_sim)

for(sim in 1:n_sim){
  print(sim)
  # generate data
  set.seed(sim)
  Eta_0 <- matrix(rnorm(n*k), ncol=k)
  X <- cbind(rep(1, n), matrix(rnorm(n*(q-1), 0, 1), ncol=q-1))
  Z_0 <-X %*% t(Beta_0) + Eta_0 %*% t(Lambda_0) 
  P_0 <- 1/(1+exp(-Z_0))
  U <- matrix(runif(n*p), ncol=p)
  Y <- matrix(0, n, p)
  Y[U<P_0] = 1
  ptm <- proc.time()
  if(test_gllvm_eva){
  gllvm.fit.eva <- gllvm(y = Y, X = as.matrix(X[,-1]), num.lv = k, 
                         family = binomial(link = "logit"),  
                         method = "EVA", formula = ~1+., seed = 123) 
  time_gllvm = proc.time() - ptm
  computing_time_gllvm[sim] <- time_gllvm[3]
  gllvm.fit.eva.est <- gllvm.fit.eva$params
  B_gllvm_frobenius_scaled[sim] <- norm((Beta_0 - cbind(gllvm.fit.eva.est$beta0, gllvm.fit.eva.est$Xcoef)), type='F') / sqrt(p * q) 
  B_gllvm_frobenius[sim] <- norm((Beta_0 - cbind(gllvm.fit.eva.est$beta0, gllvm.fit.eva.est$Xcoef)), type='F') / norm(Beta_0, type='F')
  Lambda_gllvm_outer_frobenius[sim] <- norm((Lambda_0_outer - gllvm.fit.eva.est$theta %*% diag(gllvm.fit.eva.est$sigma.lv^2) %*% t(gllvm.fit.eva.est$theta)), type='F') / norm(Lambda_0_outer, type='F')
  samples_gllvm <- sample_gllvm(gllvm.fit.eva.est, gllvm.fit.eva$sd, n_MC=300)
  cc_gllvm <- compute_coverage_gllvm (Lambda_0_outer, Beta_0, samples_gllvm, alpha=0.05)
  Lambda_gllvm_outer_coverage_95[,,sim] <- cc_gllvm$Lambda_coverage
  Lambda_gllvm_outer_length_95[,,sim] <- cc_gllvm$Lambda_length
  B_gllvm_coverage_95[,,sim] <- cc_gllvm$Beta_coverage
  B_gllvm_length_95[,,sim] <- cc_gllvm$Beta_length
  }
  
  f(test_flair){
    # run flair
    ptm <- proc.time()
    flair_estimate <- binary_flair_covariates_wrapper(
      Y, X, k_max=20, k=k, method_rho = 'max', eps=0.001, alternate_max=5, max_it=100, tol=0.001, 
      post_process=T, subsample_index = subsample_index, n_MC=1000, C_lambda=10,  C_mu=10, C_beta=10, sigma=1.626,
      observed=matrix(1, n, p), randomized_svd = T, post_process_1 = F)
    time_flair = proc.time() - ptm
    
    # save results
    computing_time_flair[sim] <- time_flair[3]
    B_bar_frobenius_scaled[sim] <- norm((Beta_0 - flair_estimate$Beta_bar), type='F') / sqrt(p * q) 
    B_bar_frobenius[sim] <- norm((Beta_0 - flair_estimate$Beta_bar), type='F') / norm((Beta_0), type='F') 
    Lambda_bar_outer_frobenius[sim] <- norm((Lambda_0_outer - flair_estimate$Lambda_bar %*% t(flair_estimate$Lambda_bar)), type='F') / norm((Lambda_0_outer), type='F') 
    Sigma_bar_outer_frobenius[sim] <- norm((Lambda_0_outer - flair_estimate$Lambda_outer_mean_cc), type='F') /  norm((Lambda_0_outer), type='F') 
    alpha <- 0.05
    cc <- compute_coverage(Lambda_0_outer_sub, Beta_0, flair_estimate, alpha=alpha)
    Lambda_outer_coverage_95[,,sim] <- cc$Lambda_coverage
    Lambda_outer_cc_coverage_95[,,sim] <- cc$Lambda_cc_coverage
    Lambda_outer_length_95[,,sim] <- cc$Lambda_length
    Lambda_outer_cc_length_95[,,sim] <- cc$Lambda_cc_length
    B_coverage_95[,,sim] <- cc$Beta_coverage
    B_cc_coverage_95[,,sim] <- cc$Beta_cc_coverage
    B_length_95[,,sim] <- cc$Beta_length
    B_cc_length_95[,,sim] <- cc$Beta_cc_length
  }
  
  if(test_gmf){
    model_gmf_airwls = gmf(Y = Y, X = X[,-1], p = k,
                           gamma=0.2, maxIter = 1000,
                           family = binomial(),
                           parallel = 1,
                           penaltyV = 0,
                           penaltyU = 1,
                           penaltyBeta = 0,
                           method = "airwls",
                           tol = 0.001,
                           intercept = T)
    time_gmf_airwls = proc_time() - ptm
    Lambda_gmf_airwls <- model_gmf_airwls$v
    Lambda_gmf_airwls_outer_estimate <- Lambda_gmf_airwls %*% t(Lambda_gmf_airwls)
    B_gmf_airwls_svd_frobenius[sim] <- norm(t(model_gmf_airwls$beta) - Beta_0, type='F')/norm((Beta_0), type='F')
    B_gmf_airwls_svd_frobenius_scaled[sim] <- norm(t(model_gmf_airwls$beta) - Beta_0, type='F')/sqrt(p*q)
    Lambda_gmf_airwls_svd_outer_frobenius[sim] <- norm((Lambda_0_outer - Lambda_gmf_airwls_outer_estimate), type='F') / norm((Lambda_0_outer), type='F')
    computing_time_gmf_airwls_svd[sim] <- time_gmf_airwls[3]
    
    
    model_gmf_newton = gmf(Y = Y, X = X[,-1], p = k,
                           gamma=0.2, maxIter = 1000,
                           family = binomial(),
                           parallel = 1,
                           penaltyV = 0,
                           penaltyU = 1,
                           penaltyBeta = 0,
                           method = "quasi",
                           tol = 0.001,
                           intercept = T, init='svd')
    time_gmf_newton = proc_time() - ptm
    Lambda_gmf_newton <- model_gmf_newton$v
    Lambda_gmf_newton_outer_estimate <- Lambda_gmf_newton %*% t(Lambda_gmf_newton)
    B_gmf_newton_svd_frobenius[sim] <- norm(t(model_gmf_newton$beta) - Beta_0, type='F')/norm((Beta_0), type='F')
    B_gmf_newton_svd_frobenius_scaled[sim] <- norm(t(model_gmf_newton$beta) - Beta_0, type='F')/sqrt(p*q)
    Lambda_gmf_newton_svd_outer_frobenius[sim] <- norm((Lambda_0_outer - Lambda_gmf_newton_outer_estimate), type='F') / norm((Lambda_0_outer), type='F')
    computing_time_gmf_newton_svd[sim] <- time_gmf_newton[3]
  }
  
}

mean(B_gllvm_frobenius_scaled); mean(Lambda_gllvm_outer_frobenius); mean(computing_time_gllvm)
mean(Lambda_gllvm_outer_coverage_95); mean(B_gllvm_coverage_95)
