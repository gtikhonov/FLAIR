
source('simulations/helpers_simulations.R')



# DGM params
k <- 10
p <- 1000
n <- 1000
n_sim <- 50 

# generate true params
sigma_lambda=sqrt(0.5); sigma_beta=sqrt(0.5); pi_lambda=0.5; pi_beta=0.5
set.seed(123)
Lambda_0 <- matrix(rtruncnorm(p*k, a=-5, b=5, 0, sigma_lambda)*rbinom(p*k, 1, pi_lambda), ncol=k)
Lambda_0_outer <- Lambda_0 %*% t(Lambda_0)
subsample_index <- 1:100
Lambda_0_outer_sub <-  Lambda_0_outer[subsample_index, subsample_index]
Beta_0 <- matrix(rtruncnorm(p, a=-5, b=5, 0, sigma_beta), ncol=1)

# arrays to store results
df_flair_sup_3 <- data.frame()
df_jml_sup_3 <- data.frame()


test_FLAIR=T; test_jml=T

n_MC <- 500
n_sim <- 50

scenario <- 1

for(sim in 1:n_sim){
  
  print(sim)
  # generate data 
  set.seed(sim)
  Eta_0 <- matrix(rnorm(n*k), ncol=k)
  X <- matrix(1, n, 1)
  Z_0 <-X %*% t(Beta_0) + Eta_0 %*% t(Lambda_0) 
  P_0 <- 1/(1+exp(-Z_0))
  U <- matrix(runif(n*p), ncol=p)
  Y <- matrix(0, n, p)
  Y[U<P_0] <- 1
  k_hat <- select_k(Y, X, observed=matrix(1, n, p), k_max=20, randomized_svd=T)
  print(k_hat)
  
  if(test_FLAIR){
    # run FLAIR
    set.seed(123)
    ptm <- proc.time()
    FLAIR_estimate <- FLAIR_wrapper(
      Y, X, k_max=20, k=k_hat, method_rho = 'max', eps=0.001, alternate_max=10, max_it=100, tol=0.001, 
      post_process=T, subsample_index = subsample_index, n_MC=n_MC, C_lambda=10,  C_mu=10, C_beta=10,
      sigma=1.626, observed=matrix(1, n, p), randomized = F, loss_tol=0.001)
    time_FLAIR = proc.time() - ptm
    # store results
    out_flair <- compute_performance_flair(FLAIR_estimate, Beta_0, Lambda_0_outer, Lambda_0_outer_sub, 
                                           exclude_intercept=F)
    out_flair[13] <- time_FLAIR[3]
    out_flair
    df_flair_sup_3 <- rbind(df_flair_sup_3, out_flair) 
  }
  
  if(test_jml){
    set.seed(124)
    aucs <- c()
    
    test_set <- sample(1:(n*p), as.integer(0.2*n*p))
    Y_train <- Y
    Y_train[test_set] <- NA
    Y_test <- Y[test_set]
    library(pROC)
    
    for(idx in 1:10){
      jml_fit <- mirtjml_expr(Y_train, K=k_hat, cc=idx)
      jml_pred <- rep(1, n) %*% t(jml_fit$d_hat) + jml_fit$theta_hat %*% t(jml_fit$A_hat)
      p_hat_test <- 1/(1+exp(-jml_pred[test_set]))
      roc_test <- roc(c(Y[test_set]), c(p_hat_test))
      aucs[idx] <- auc(roc_test)[1]
      rm(roc_test)
      aucs
      cc <- which.max(aucs)
      print(cc)
    }
    ptm <- proc.time()
    jml_fit <- mirtjml_expr(Y, K=k_hat, cc=cc)
    time_jml = proc.time() - ptm
    # store results    
    out_jml <- compute_perfomance_jml(jml_fit, X[,], Beta_0[], Lambda_0_outer)
    out_jml[13] <- time_jml[3]
    df_jml_sup_3 <- rbind(df_jml_sup_3, out_jml) 
    
  }
  
}





names_cols <- c('Beta_fr', 'Lambda_fr', 'Lambda_v_cov', 'Lambda_cc_cov', 'Lambda_v_len', 'Lambda_cc_len',
                'Beta_v_cov', 'Beta_cc_cov', 'Beta_v_len', 'Beta_cc_len', 'median_beta_j', 'max_beta_j',
                'time')

names(df_flair_sup_3) <- names_cols
names(df_jml_sup_3) <- names_cols


colMeans(df_flair_sup_3[,])
colSds(as.matrix(df_flair_sup_3[,]))/sqrt(n_sim)
colMeans(df_jml_sup_3[,])
colSds(as.matrix(df_jml_sup_3[,]))/sqrt(n_sim)

