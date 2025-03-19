
source('simulations/helpers_simulations.R')


# DGM params
k <- 10
q <- 10
p <- 1000
n <- 500
n_sim <- 50 

# generate true params
sigma_lambda=sqrt(0.5); sigma_beta=sqrt(0.5); pi_lambda=0.5; pi_beta=0.5
set.seed(123)
Lambda_0 <- matrix(rtruncnorm(p*k, a=-5, b=5, 0, sigma_lambda)*rbinom(p*k, 1, pi_lambda), ncol=k)
Lambda_0_outer <- Lambda_0 %*% t(Lambda_0)
subsample_index <- 1:100
Lambda_0_outer_sub <-  Lambda_0_outer[subsample_index, subsample_index]
Beta_0 <- matrix(rtruncnorm(p*q, a=-5, b=5, 0, sigma_beta)*rbinom(p*q, 1, pi_beta), ncol=q)

# arrays to store results
df_flair_sup_2 <- data.frame()
df_lvhml_sup_2 <- data.frame()


test_FLAIR=F; test_lvhml=T

#library(remotes) 
#remotes::install_github("Arthurlee51/LVHML")
#library(LVHML)


n_MC <- 500
n_sim <- 50

scenario <- 1

for(sim in 1:n_sim){
  
  print(sim)
  # generate data 
  set.seed(sim)
  Eta_0 <- matrix(rnorm(n*k), ncol=k)
  X <- cbind(rep(1, n), matrix(rnorm(n*(q-1), 0, 1), ncol=q-1))
  Z_0 <-X %*% t(Beta_0) + Eta_0 %*% t(Lambda_0) 
  P_0 <- 1/(1+exp(-Z_0))
  U <- matrix(runif(n*p), ncol=p)
  #Y <- matrix(0, n, p)
  #Y[U<P_0] <- 1
  Tp <- 2
  Y_arr <- array(NA, dim=c(n, p, Tp))
  for(t in 1:Tp){
    U_t <- matrix(runif(n*p), ncol=p)
    Y_t <- matrix(0, n, p)
    Y_t[U_t<P_0] = 1
    Y_arr[,,t] <- Y_t
    if(t == 2 & scenario ==2){
      Y_arr[,,t] <- matrix(NA, n , t)
      Y_arr[1,,t] <- Y_t[1,]
    }
    
  }
  R <- matrix(1, n, Tp)
  Y <- rbind(Y_arr[,,1], Y_arr[,,2])
  X_long <- rbind(X, X)
  if(scenario==2){ 
    R <- cbind(rep(1, n), c(1, rep(0, n-1)) )
    Y <- rbind(Y_arr[,,1], Y_arr[1,,2])
    X_long <- rbind(X, X[1,])
  }
  N <- nrow(Y)
  k_hat <- select_k(Y, X_long, observed=matrix(1, N, p), k_max=20, randomized_svd=T)
  print(k_hat)
  
  if(test_FLAIR){
    # run FLAIR
    set.seed(123)
    ptm <- proc.time()
    FLAIR_estimate <- FLAIR_wrapper(
      Y, X_long, k_max=20, k=k_hat, method_rho = 'max', eps=0.001, alternate_max=10, max_it=100, tol=0.001, 
      post_process=T, subsample_index = subsample_index, n_MC=n_MC, C_lambda=10,  C_mu=10, C_beta=10,
      sigma=1.626, observed=matrix(1, N, p), randomized = T, loss_tol=0.001)
    time_FLAIR = proc.time() - ptm
    # store results
    out_flair <- compute_performance_flair(FLAIR_estimate, Beta_0, Lambda_0_outer, Lambda_0_outer_sub, exclude_intercept=T)
    out_flair[13] <- time_FLAIR[3]
    out_flair
    df_flair_sup_2 <- rbind(df_flair_sup_2, out_flair) 
  }
  
  if(test_lvhml){
    set.seed(123)
    ptm <- proc.time()
    lvhml_fit <- lvhml_est(Y_arr, R, X[,-1],
                           Kset=k_hat, Asymp=T, par=T)
    time_lvhml = proc.time() - ptm
    # store results    
    out_lvhml <- compute_perfomance_lvhml(lvhml_fit, X[,-1], Beta_0[,-1], Lambda_0_outer, 
                                          Lambda_0_outer_sub, R, subsample_index=subsample_index,
                                          n_MC=n_MC)
    out_lvhml[13] <- time_lvhml[3]
    df_lvhml_sup_2 <- rbind(df_lvhml_sup_2, out_lvhml) 
    
  }

}





names_cols <- c('Beta_fr', 'Lambda_fr', 'Lambda_v_cov', 'Lambda_cc_cov', 'Lambda_v_len', 'Lambda_cc_len',
                'Beta_v_cov', 'Beta_cc_cov', 'Beta_v_len', 'Beta_cc_len', 'median_beta_j', 'max_beta_j',
                'time')

names(df_flair_sup_2) <- names_cols
names(df_lvhml_sup_2) <- names_cols

colMeans(df_flair_sup_2[,])
colSds(as.matrix(df_flair_sup_2[,]))/sqrt(n_sim)
colMeans(df_lvhml_sup_2[,])
colSds(as.matrix(df_lvhml_sup_2[,]))/sqrt(n_sim)

