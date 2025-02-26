
setwd("~/simulations")
#load('.RData')

source('helpers_simulations.R')
source('../FLAIR_wrapper.R')

# DGM params
k <- 2
q <- 2
p <- 50
n <- 100
n_sim <- 50 

# generate true params
sigma_lambda=sqrt(1); sigma_beta=sqrt(1); pi_lambda=1; pi_beta=1
set.seed(123)
Lambda_0 <- matrix(rtruncnorm(p*k, a=-5, b=5, 0, sigma_lambda)*rbinom(p*k, 1, pi_lambda), ncol=k)
Lambda_0_outer <- Lambda_0 %*% t(Lambda_0)
subsample_index <- 1:p
Lambda_0_outer_sub <-  Lambda_0_outer[subsample_index, subsample_index]
Beta_0 <- matrix(rtruncnorm(p*q, a=-5, b=5, 0, sigma_beta)*rbinom(p*q, 1, pi_beta), ncol=q)

# arrays to store results

# FLAIR
B_coverage_95 <- array(0, dim=c(p, q, n_sim))
B_cc_coverage_95 <- array(0, dim=c(p, q, n_sim))
B_length_95 <- array(0, dim=c(p, q, n_sim))
B_cc_length_95 <- array(0, dim=c(p, q, n_sim))
Lambda_outer_coverage_95 <- array(0, dim=c(p, p, n_sim))
Lambda_outer_cc_coverage_95 <- array(0, dim=c(p, p, n_sim))
Lambda_outer_length_95 <- array(0, dim=c(p, p, n_sim))
Lambda_outer_cc_length_95 <- array(0, dim=c(p, p, n_sim))
computing_time_FLAIR <- rep(0, n_sim)
B_tilde_frobenius_scaled <- rep(0, n_sim)
B_tilde_frobenius <- rep(0, n_sim)
Lambda_tilde_outer_frobenius <- rep(0, n_sim)
Sigma_tilde_outer_frobenius <- rep(0, n_sim)

# gmf
B_gmf_newton_frobenius <- rep(0, n_sim)
B_gmf_newton_frobenius_scaled <- rep(0, n_sim)
Lambda_gmf_newton_outer_frobenius <- rep(0, n_sim)
computing_time_gmf_newton <- rep(0, n_sim)
B_gmf_airwls_frobenius <- rep(0, n_sim)
B_gmf_airwls_frobenius_scaled <- rep(0, n_sim)
Lambda_gmf_airwls_outer_frobenius <- rep(0, n_sim)
computing_time_gmf_airwls <- rep(0, n_sim)

# gllvm
Lambda_gllvm_eva_outer_coverage_95 <- array(0, dim=c(p, p, n_sim))
Lambda_gllvm_eva_outer_length_95 <- array(0, dim=c(p, p, n_sim))
B_gllvm_eva_coverage_95 <- array(0, dim=c(p, q, n_sim))
B_gllvm_eva_length_95 <- array(0, dim=c(p, q, n_sim))
B_gllvm_eva_frobenius_scaled <- rep(0, n_sim)
B_gllvm_eva_frobenius <- rep(0, n_sim)
Lambda_gllvm_eva_outer_frobenius <- rep(0, n_sim)
computing_time_gllvm_eva <- rep(0, n_sim)

Lambda_gllvm_la_outer_coverage_95 <- array(0, dim=c(p, p, n_sim))
Lambda_gllvm_la_outer_length_95 <- array(0, dim=c(p, p, n_sim))
B_gllvm_la_coverage_95 <- array(0, dim=c(p, q, n_sim))
B_gllvm_la_length_95 <- array(0, dim=c(p, q, n_sim))
B_gllvm_la_frobenius_scaled <- rep(0, n_sim)
B_gllvm_la_frobenius <- rep(0, n_sim)
Lambda_gllvm_la_outer_frobenius <- rep(0, n_sim)
computing_time_gllvm_la <- rep(0, n_sim)


test_gllvm_eva = F; test_gllvm_la = T; test_FLAIR=T; test_gmf_newton=F; test_gmf_airwls=F


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
  Y[U<P_0] <- 1
  k_hat <- select_k(Y, X, observed=matrix(1, n, p), k_max=20, randomized_svd=T)
  k_hat
  if(test_gllvm_eva){
    ptm <- proc.time()
    gllvm.fit.eva <- gllvm(y = Y, X = as.matrix(X[,-1]), num.lv = k_hat, 
                           family = binomial(link = "logit"),  
                           method = "EVA", formula = ~1+., seed = 123) 
    time_gllvm_eva = proc.time() - ptm
    computing_time_gllvm_eva[sim] <- time_gllvm_eva[3]
    gllvm.fit.eva.est <- gllvm.fit.eva$params
    vcov_gllvm_eva_eva <- vcov.gllvm(gllvm.fit.eva)
    B_gllvm_eva_frobenius_scaled[sim] <- norm((Beta_0 - cbind(gllvm.fit.eva.est$beta0, gllvm.fit.eva.est$Xcoef)), type='F') / sqrt(p * q) 
    B_gllvm_eva_frobenius[sim] <- norm((Beta_0 - cbind(gllvm.fit.eva.est$beta0, gllvm.fit.eva.est$Xcoef)), type='F') / norm(Beta_0, type='F')
    Lambda_gllvm_eva_outer_frobenius[sim] <- norm((Lambda_0_outer - gllvm.fit.eva.est$theta %*% diag(gllvm.fit.eva.est$sigma.lv^2) %*% t(gllvm.fit.eva.est$theta)), type='F') / norm(Lambda_0_outer, type='F')
    samples_gllvm_eva <- sample_gllvm_eva(gllvm.fit.eva.est, vcov_gllvm_eva_eva, n_MC=300)
    cc_gllvm_eva <- compute_coverage_gllvm_eva (Lambda_0_outer, Beta_0, samples_gllvm_eva, alpha=0.05)
    Lambda_gllvm_eva_outer_coverage_95[,,sim] <- cc_gllvm_eva$Lambda_coverage
    Lambda_gllvm_eva_outer_length_95[,,sim] <- cc_gllvm_eva$Lambda_length
    B_gllvm_eva_coverage_95[,,sim] <- cc_gllvm_eva$Beta_coverage
    B_gllvm_eva_length_95[,,sim] <- cc_gllvm_eva$Beta_length
  }
  
  
  if(test_gllvm_la){
    ptm <- proc.time()
    gllvm.fit.la <- gllvm(y = Y, X = as.matrix(X[,-1]), num.lv = k_hat, 
                           family = binomial(link = "logit"),  
                           method = "LA", formula = ~1+., seed = 123) 
    time_gllvm_la = proc.time() - ptm
    computing_time_gllvm[sim] <- time_gllvm_la[3]
    gllvm.fit.la.est <- gllvm.fit.la$params
    vcov_gllvm_la <- vcov.gllvm(gllvm.fit.la)
    
    B_gllvm_la_frobenius_scaled[sim] <- norm((Beta_0 - cbind(gllvm.fit.la.est$beta0, gllvm.fit.la.est$Xcoef)), type='F') / sqrt(p * q) 
    B_gllvm_la_frobenius[sim] <- norm((Beta_0 - cbind(gllvm.fit.la.est$beta0, gllvm.fit.la.est$Xcoef)), type='F') / norm(Beta_0, type='F')
    Lambda_gllvm_la_outer_frobenius[sim] <- norm((Lambda_0_outer - gllvm.fit.la.est$theta %*% diag(gllvm.fit.la.est$sigma.lv^2) %*% t(gllvm.fit.la.est$theta)), type='F') / norm(Lambda_0_outer, type='F')
    samples_gllvm_la <- sample_gllvm_la(gllvm.fit.la.est, vcov_gllvm_la, n_MC=300)
    cc_gllvm_la <- compute_coverage_gllvm_la (Lambda_0_outer, Beta_0, samples_gllvm_la, alpha=0.05)
    Lambda_gllvm_la_outer_coverage_95[,,sim] <- cc_gllvm_la$Lambda_coverage
    Lambda_gllvm_la_outer_length_95[,,sim] <- cc_gllvm_la$Lambda_length
    B_gllvm_la_coverage_95[,,sim] <- cc_gllvm_la$Beta_coverage
    B_gllvm_la_length_95[,,sim] <- cc_gllvm_la$Beta_length
  }
  
  
  if(test_FLAIR){
    # run FLAIR
    ptm <- proc.time()
    FLAIR_estimate <- FLAIR_wrapper(
      Y, X, k_max=20, k=k_hat, method_rho = 'max', eps=0.001, alternate_max=10, max_it=100, tol=0.001, 
      post_process=T, subsample_index = subsample_index, n_MC=1000, C_lambda=10,  C_mu=10, C_beta=10, sigma=1.626,
      observed=matrix(1, n, p), randomized = F, loss_tol=0.001)
    time_FLAIR = proc.time() - ptm
    
    # save results
    computing_time_FLAIR[sim] <- time_FLAIR[3]
    B_tilde_frobenius_scaled[sim] <- norm((Beta_0 - FLAIR_estimate$Beta_tilde), type='F') / sqrt(p * q) 
    B_tilde_frobenius[sim] <- norm((Beta_0 - FLAIR_estimate$Beta_tilde), type='F') / norm((Beta_0), type='F') 
    Lambda_tilde_outer_frobenius[sim] <- norm((Lambda_0_outer - FLAIR_estimate$Lambda_tilde %*% t(FLAIR_estimate$Lambda_tilde)), type='F') / norm((Lambda_0_outer), type='F') 
    Sigma_tilde_outer_frobenius[sim] <- norm((Lambda_0_outer - FLAIR_estimate$Lambda_outer_mean_cc), type='F') /  norm((Lambda_0_outer), type='F') 
    alpha <- 0.05
    cc <- compute_coverage(Lambda_0_outer_sub, Beta_0, FLAIR_estimate, alpha=alpha)
    Lambda_outer_coverage_95[,,sim] <- cc$Lambda_coverage
    Lambda_outer_cc_coverage_95[,,sim] <- cc$Lambda_cc_coverage
    Lambda_outer_length_95[,,sim] <- cc$Lambda_length
    Lambda_outer_cc_length_95[,,sim] <- cc$Lambda_cc_length
    Beta_cc_cis <- compute_ci_Beta(FLAIR_estimate$Beta_tilde, FLAIR_estimate$Vs[1:ncol(X)], 
                                   FLAIR_estimate$rho_max, alpha=0.05) 
    Beta_cis <- compute_ci_Beta(FLAIR_estimate$Beta_tilde, FLAIR_estimate$Vs[1:ncol(X)], 
                                1, alpha=0.05) 
    B_coverage_95[,,sim] <- (Beta_cis$ls < Beta_0) & (Beta_cis$us > Beta_0)
    B_cc_coverage_95[,,sim] <- (Beta_cc_cis$ls < Beta_0) & (Beta_cc_cis$us > Beta_0)
    B_length_95[,,sim] <- Beta_cis$us - Beta_cis$ls
    B_cc_length_95[,,sim] <- Beta_cc_cis$us - Beta_cc_cis$ls
  }
  
  if(test_gmf_airwls){
    ptm <- proc.time()
    model_gmf_airwls = gmf(Y = Y, X = X[,-1], p = k_hat,
                           gamma=0.2, maxIter = 1000,
                           family = binomial(),
                           parallel = 1,
                           penaltyV = 0,
                           penaltyU = 1,
                           penaltyBeta = 0,
                           method = "airwls",
                           tol = 0.001,
                           intercept = T)
    time_gmf_airwls = proc.time() - ptm
    Lambda_gmf_airwls <- model_gmf_airwls$v
    Lambda_gmf_airwls_outer_estimate <- Lambda_gmf_airwls %*% t(Lambda_gmf_airwls)
    B_gmf_airwls_frobenius[sim] <- norm(t(model_gmf_airwls$beta) - Beta_0, type='F')/norm((Beta_0), type='F')
    B_gmf_airwls_frobenius_scaled[sim] <- norm(t(model_gmf_airwls$beta) - Beta_0, type='F')/sqrt(p*q)
    Lambda_gmf_airwls_outer_frobenius[sim] <- norm((Lambda_0_outer - Lambda_gmf_airwls_outer_estimate), type='F') / norm((Lambda_0_outer), type='F')
    computing_time_gmf_airwls[sim] <- time_gmf_airwls[3]
    #print(mean(computing_time_gmf_airwls[1:sim]))
  }
  if(test_gmf_newton){
    ptm <- proc.time()
    model_gmf_newton = gmf(Y = Y, X = X[,-1], p = k_hat,
                           gamma=0.2, maxIter = 1000,
                           family = binomial(),
                           parallel = 1,
                           penaltyV = 0,
                           penaltyU = 1,
                           penaltyBeta = 0,
                           method = "quasi",
                           tol = 0.001,
                           intercept = T, init='svd')
    time_gmf_newton = proc.time() - ptm
    Lambda_gmf_newton <- model_gmf_newton$v
    Lambda_gmf_newton_outer_estimate <- Lambda_gmf_newton %*% t(Lambda_gmf_newton)
    B_gmf_newton_frobenius[sim] <- norm(t(model_gmf_newton$beta) - Beta_0, type='F')/norm((Beta_0), type='F')
    B_gmf_newton_frobenius_scaled[sim] <- norm(t(model_gmf_newton$beta) - Beta_0, type='F')/sqrt(p*q)
    Lambda_gmf_newton_outer_frobenius[sim] <- norm((Lambda_0_outer - Lambda_gmf_newton_outer_estimate), type='F') / norm((Lambda_0_outer), type='F')
    computing_time_gmf_newton[sim] <- time_gmf_newton[3]
  }
  
}


# FLAIR
mean(B_tilde_frobenius_scaled); mean(Sigma_tilde_outer_frobenius); mean(computing_time_FLAIR)
mean(Lambda_outer_cc_coverage_95); mean(B_cc_coverage_95); 
mean(Lambda_outer_coverage_95); mean(B_coverage_95); 

# gmf - newton
mean(B_gmf_newton_frobenius_scaled); mean(Lambda_gmf_newton_outer_frobenius); mean(computing_time_gmf_newton)

# gmf - airwls
mean(B_gmf_airwls_frobenius_scaled); mean(Lambda_gmf_airwls_outer_frobenius); mean(computing_time_gmf_airwls)

# gllvm - eva
mean(B_gllvm_eva_frobenius_scaled); mean(Lambda_gllvm_eva_outer_frobenius); mean(computing_time_gllvm_eva)
mean(Lambda_gllvm_eva_outer_coverage_95); mean(B_gllvm_eva_coverage_95)

# gllvm - la
mean(B_gllvm_la_frobenius_scaled); mean(Lambda_gllvm_la_outer_frobenius); mean(computing_time_gllvm_la)
mean(Lambda_gllvm_la_outer_coverage_95); mean(B_gllvm_la_coverage_95)

