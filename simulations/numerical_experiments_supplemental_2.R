
#load('.RData')


source('simulations/helpers_simulations.R')


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

library(expm)




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


sample_Lambda_lvhml_old <- function(lvhml_fit, V_hats, n_MC=100, outer=T, subsample_index=1:100){
  Lambdas_hat <- lvhml_fit$Ahat
  n <- nrow(lvhml_fit$Thetahat)
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
        Lambdas_outer[,,s] <- Lambdas[subsample_index,,s] %*% diag(diag(crossprod(lvhml_fit$Thetahat)/n)) %*% t(Lambdas[subsample_index,,s])
        
      }
    }
  }
  return(list(Lambdas=Lambdas, Lambdas_outer=Lambdas_outer))
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


# DGM params
k <- 10
q <- 10
p <- 10000
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


test_FLAIR=T; test_lvhml=T

#library(remotes) 
#remotes::install_github("Arthurlee51/LVHML")
#library(LVHML)


n_MC <- 500
n_sim <- 50

scenario <- 2

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
      sigma=1.626, observed=matrix(1, N, p), randomized = F, loss_tol=0.001)
    time_FLAIR = proc.time() - ptm
    # save results
    out_flair <- compute_performance_flair(FLAIR_estimate, Beta_0, Lambda_0_outer, Lambda_0_outer_sub, exclude_intercept=T)
    out_flair[13] <- time_FLAIR[3]
    out_flair
    df_flair_sup_2 <- rbind(df_flair_sup_2, out_flair) 
  }
  
  if(test_lvhml){
    set.seed(124)
    ptm <- proc.time()
    lvhml_fit <- lvhml_est(Y_arr, R, X[,-1],
                           Kset=k_hat, Asymp=T, par=T)
    time_lvhml = proc.time() - ptm
    # save results    
    out_lvhml <- compute_perfomance_lvhml(lvhml_fit, X[,-1], Beta_0[,-1], Lambda_0_outer, 
                                          Lambda_0_outer_sub, R, subsample_index=subsample_index,
                                          n_MC=n_MC)
    out_lvhml[13] <- time_lvhml[3]
    df_lvhml_sup_2 <- rbind(df_lvhml_sup_2, out_lvhml) 
    
  }

}



df_lvhml_sup_2[,]


names_cols <- c('Beta_fr', 'Lambda_fr', 'Lambda_v_cov', 'Lambda_cc_cov', 'Lambda_v_len', 'Lambda_cc_len',
                'Beta_v_cov', 'Beta_cc_cov', 'Beta_v_len', 'Beta_cc_len', 'median_beta_j', 'max_beta_j',
                'time')

names(df_flair_sup_2) <- names_cols
names(df_lvhml_sup_2) <- names_cols

df_flair_sup_2
df_lvhml_sup_2





colMeans(df_flair_sup_2[51:100,])
colSds(as.matrix(df_flair_sup_2[51:100,]))/sqrt(n_sim)
colMeans(df_lvhml_sup_2[51:100,])
colSds(as.matrix(df_lvhml_sup_2[51:100,]))/sqrt(n_sim)



path_res <- paste0("simulations/results/supp_2/",  'p',p, '_n', n, '_N',N, '_df_flair.csv')
write.csv(df_flair_sup_2, path_res)
path_res <- paste0("simulations/results/supp_2/",  'p',p, '_n', n, '_N',N, '_df_lvhml.csv')
write.csv(df_lvhml_sup_2, path_res)


