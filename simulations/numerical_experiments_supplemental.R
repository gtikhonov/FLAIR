

source('simulations/helpers_simulations.R')

#devtools::install_version("gllvm", version = "1.4.3")
library(gllvm)

# DGM params
k <- 2
q <- 2
p <- 100
n <- 500
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
df_newton_sup <- data.frame()
df_airwls_sup <- data.frame()
df_eva_sup <- data.frame()
df_la_sup <- data.frame()
df_flair_sup <- data.frame()

test_gllvm_eva=T; test_gllvm_la=T; test_FLAIR=F; test_gmf_newton=F; test_gmf_airwls=F

n_MC <- 500
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
  print(k_hat)
  
  
  if(test_gllvm_eva){
    set.seed(123)
    ptm <- proc.time()
    gllvm.fit.eva <- gllvm(y = Y, X = as.matrix(X[,-1]), num.lv = k, 
                           family = binomial(link = "logit"),  
                           method = "EVA", formula = ~1+., seed = 123) 
    time_gllvm_eva = proc.time() - ptm
    out_eva <- compute_performance_gllvm(gllvm.fit.eva, Beta_0, Lambda_0_outer, 
                                          Lambda_0_outer_sub, n_MC=n_MC, coverage=T)
    out_eva[13] <- time_gllvm_eva[3]
    df_eva_sup <- rbind(df_eva_sup, out_eva)
  }
  
  
  if(test_gllvm_la){
    set.seed(123)
    ptm <- proc.time()
    gllvm.fit.la <- gllvm(y = Y, X = as.matrix(X[,-1]), num.lv = k, 
                           family = binomial(link = "logit"),  
                           method = "LA", formula = ~1+., seed = 123) 
    time_gllvm_la = proc.time() - ptm
    out_la <- compute_performance_gllvm(gllvm.fit.la, Beta_0, Lambda_0_outer, 
                                         Lambda_0_outer_sub, n_MC=n_MC, coverage=T)
    out_la[13] <- time_gllvm_la[3]
    df_la_sup <- rbind(df_la_sup, out_la)
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
    out_flair <- compute_performance_flair(FLAIR_estimate, Beta_0, Lambda_0_outer, Lambda_0_outer_sub, 
                                           exclude_intercept=F)
    out_flair[13] <- time_FLAIR[3]
    print(out_flair)
    # save results
    df_flair_sup <- rbind(df_flair_sup, out_flair) 
  }
  
  if(test_gmf_airwls){
    
    k_hat <- k
    set.seed(123)
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
    out_airwls <- compute_performance_gmf(model_gmf_airwls, Beta_0, Lambda_0_outer)
    out_airwls[5] <- time_gmf_airwls[3]
    print(out_airwls)
    df_airwls_sup <- rbind(df_airwls_sup, out_airwls)
  }
  if(test_gmf_newton){
    
    k_hat = k
    set.seed(123)
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
    out_newton <- compute_performance_gmf(model_gmf_newton, Beta_0, Lambda_0_outer)
    out_newton[5] <- time_gmf_newton[3]
    print(out_newton)
    df_newton_sup <- rbind(df_newton_sup, out_newton)
  }
  
}

names_cols <- c('Beta_fr', 'Lambda_fr', 'Lambda_v_cov', 'Lambda_cc_cov', 'Lambda_v_len', 'Lambda_cc_len',
                'Beta_v_cov', 'Beta_cc_cov', 'Beta_v_len', 'Beta_cc_len', 'median_beta_j', 'max_beta_j',
                'time')

names(df_eva_sup) <- names_cols
names(df_la_sup) <- names_cols

colMeans(df_eva_sup[,])
colSds(as.matrix(df_eva_sup[,]))/sqrt(n_sim)

colMeans(df_la_sup[,])
colSds(as.matrix(df_la_sup[,]))/sqrt(n_sim)

names(df_flair_sup) <- names_cols
colMeans(df_flair_sup[,])
colSds(as.matrix(df_flair_sup[,]))/sqrt(n_sim)


names_gmf <- c('Beta_fr', 'Lambda_fr', 'Median', 'Max', 'time')
names(df_airwls_sup) <- names_gmf
names(df_newton_sup) <- names_gmf

colMeans(df_airwls_sup[,])
colSds(as.matrix(df_airwls_sup[,]))/sqrt(n_sim)

colMeans(df_newton_sup[,])
colSds(as.matrix(df_newton_sup[,]))/sqrt(n_sim)

colMeans(df_flair_sup_2[,])
colSds(as.matrix(df_flair_sup_2[,]))/sqrt(n_sim)


