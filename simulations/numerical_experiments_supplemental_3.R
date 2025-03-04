install.package('mirtjml')


library(mirtjml)

#load('.RData')


source('simulations/helpers_simulations.R')






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


# DGM params
k <- 10
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



df_jml_sup_2[,]


names_cols <- c('Beta_fr', 'Lambda_fr', 'Lambda_v_cov', 'Lambda_cc_cov', 'Lambda_v_len', 'Lambda_cc_len',
                'Beta_v_cov', 'Beta_cc_cov', 'Beta_v_len', 'Beta_cc_len', 'median_beta_j', 'max_beta_j',
                'time')

names(df_flair_sup_3) <- names_cols
names(df_jml_sup_3) <- names_cols

df_flair_sup_3
df_jml_sup_3



path_res <- paste0("simulations/results/supp_3/",  'p',p, '_n', n, '_N',N, '_df_flair.csv')
write.csv(df_flair_sup_3, path_res)
path_res <- paste0("simulations/results/supp_3/",  'p',p, '_n', n, '_N',N, '_df_jml.csv')
write.csv(df_jml_sup_3, path_res)


