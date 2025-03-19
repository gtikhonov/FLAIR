
source('simulations/helpers_simulations.R')

# DGM params
k <- 10
q <- 10
p <- 5000
n <- 500
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


df_flair <- data.frame()
df_airwls <- data.frame()
df_newtow <- data.frame()

# hyperparams 4 gmf
penalties_Beta <- c(0, 0.5, 1, 5, 10)
penalties_Lambda <- c(0, 0.5, 1, 5, 10)
hyperparams <- data.frame(expand.grid(penalties_Beta, penalties_Lambda))


test_flair = T; test_airwls = T; test_newton = T

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
  k_hat <- select_k(Y, X, observed=matrix(1, n, p), k_max=20, randomized_svd=T)
  print(k_hat)
  
  if(test_flair){
    # run flair
    ptm <- proc.time()
    flair_estimate <- FLAIR_wrapper(
      Y, X, k_max=20, k=k_hat, method_rho = 'max', eps=0.001, alternate_max=10, max_it=100, tol=0.01, 
      post_process=T, subsample_index = subsample_index, n_MC=300, C_lambda=10,  C_mu=10, C_beta=10, sigma=1.626,
      observed=matrix(1, n, p), randomized_svd = T, loss_tol=0.001)
    time_flair = proc.time() - ptm
    out_flair <- compute_performance_flair(flair_estimate, Beta_0, Lambda_0_outer, Lambda_0_outer_sub, 
                                           exclude_intercept=F)
    out_flair[13] <- time_flair[3]
    print(out_flair)
    # save results
    df_flair <- rbind(df_flair, out_flair) 
  }
  
  if(test_newton){
    ## optmize hyperparams
    aucs_newton <- apply(hyperparams, 1, function(x) 
      run_1_sim_gmf_train_test(Lambda_0, Beta_0, n=n, seed=sim, penaltyV=x[2],
                               penaltyBeta=x[1], tol=0.001, method=1))
    best_conf <- which.max(aucs_newton)
    set.seed(123)
    ptm <- proc.time()
    model_gmf_newton = gmf(Y = Y, X = X[,-1], p = k_hat,
                           gamma=0.2, maxIter = 1000,
                           family = binomial(),
                           parallel = 1,
                           penaltyV = hyperparams[best_conf, 2],
                           penaltyU = 1,
                           penaltyBeta = hyperparams[best_conf, 1],
                           method = "quasi",
                           tol = 0.001,
                           intercept = T, init='svd')
    time_gmf_newton = proc.time() - ptm
    out_newton <- compute_performance_gmf(model_gmf_newton, Beta_0, Lambda_0_outer)
    out_newton[5] <- time_gmf_newton[3]
    print(out_newton)
    df_newton <- rbind(df_newton, out_newton)
    
  }
  
  if(test_airwls){
    # airwls
    ## optmize hyperparams
    aucs_airwls <- apply(hyperparams, 1, function(x) 
      run_1_sim_gmf_train_test(Lambda_0, Beta_0, n=n, seed=sim, penaltyV=x[2],
                               penaltyBeta=x[1], tol=0.001, method=2))
    best_conf <- which.max(aucs_airwls)
    k_hat <- k
    set.seed(123)
    ptm <- proc.time()
    model_gmf_airwls = gmf(Y = Y, X = X[,-1], p = k_hat,
                           gamma=0.2, maxIter = 1000,
                           family = binomial(),
                           parallel = 1,
                           penaltyV = hyperparams[best_conf, 2],
                           penaltyU = 1,
                           penaltyBeta = hyperparams[best_conf, 1],
                           method = "airwls",
                           tol = 0.001,
                           intercept = T)
    time_gmf_airwls = proc.time() - ptm
    out_airwls <- compute_performance_gmf(model_gmf_airwls, Beta_0, Lambda_0_outer)
    out_airwls[5] <- time_gmf_airwls[3]
    print(out_airwls)
    df_airwls <- rbind(df_airwls, out_airwls)
  }
  
}


names_cols <- c('Beta_fr', 'Lambda_fr', 'Lambda_v_cov', 'Lambda_cc_cov', 'Lambda_v_len', 'Lambda_cc_len',
                'Beta_v_cov', 'Beta_cc_cov', 'Beta_v_len', 'Beta_cc_len', 'median_beta_j', 'max_beta_j',
                'time')
names(df_flair) <- names_cols

colMeans(df_flair[,])
colSds(as.matrix(df_flair[,]))/sqrt(n_sim)


names_gmf <- c('Beta_fr', 'Lambda_fr', 'Median', 'Max', 'time')
names(df_airwls) <- names_gmf
names(df_newton) <- names_gmf

colMeans(df_airwls[,])
colSds(as.matrix(df_airwls[,]))/sqrt(n_sim)

colMeans(df_newton[,])
colSds(as.matrix(df_newton[,]))/sqrt(n_sim)
