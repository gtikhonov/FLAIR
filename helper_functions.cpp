#include <RcppArmadillo.h>
#include <Rcpp.h>



// [[Rcpp::depends(RcppArmadillo)]]
using namespace Rcpp;
using namespace arma;


double F_logistic(double x){
  return 1/(1+exp(-x));
}


arma::mat grad_regression_coeffs_logit(
    arma::vec y, arma::vec observed, arma::mat M, arma::vec lambda, double prior_var_beta,
    double prior_var_lambda, int q, int k){
  // returns gradient and hessian of logistic regression log-lik + normal prior
  // p * (p+1) mat where first col is grad and rest is hessian
  int n = y.size();
  int p = M.n_cols;
  
  arma::vec scores(n, arma::fill::zeros);
  arma::vec grad_log_pi(p, arma::fill::zeros);
  arma::mat hessian(p, p, arma::fill::zeros);
  
  for(int i=0; i<n; ++i){
    scores(i) = 0;
    if(observed(i)==1){
      double eta_lambda_i = dot(M.row(i), lambda);
      double F_i = F_logistic(eta_lambda_i);
      scores(i) = y(i) - F_i;
      hessian = hessian + F_i* (1 - F_i)*M(i, arma::span(0, p-1)).t()*M(i, arma::span(0,p-1));
    }
  }
  for(int j=0; j<p; ++j){
    grad_log_pi(j) = sum(scores.t()*M(arma::span(0,n-1), j));
  }
  arma::vec tau_1 = 1/prior_var_beta * arma::ones(q);
  arma::vec tau_2 = 1/prior_var_lambda * arma::ones(k);
  arma::vec prior_var_inv = join_cols(tau_1, tau_2);
  
  grad_log_pi(arma::span(0,q-1)) = grad_log_pi(arma::span(0,q-1)) - lambda(arma::span(0,q-1)) / 
    prior_var_beta;
  grad_log_pi(arma::span(q,k+q-1)) = grad_log_pi(arma::span(q,k+q-1)) - lambda(arma::span(q,k+q-1)) / 
    prior_var_lambda;
  
  arma::mat D = arma::diagmat(join_cols(tau_1, tau_2)); 
  hessian = hessian + D; // negative hessian
  arma::mat result(p, p+1);
  result(arma::span(0, p-1), 0) = grad_log_pi;
  result(arma::span(0, p -1), arma::span(1, p)) = -hessian;
  return result;
}


arma::mat optimize_regression_coeffs_logit(
    arma::vec y, arma::vec observed, arma::mat X, arma::mat M, arma::vec beta_init, arma::vec lambda_init, 
    double prior_var_beta, double prior_var_lambda, int max_it, double epsilon, double C_lambda = 10,
    double C_mu = 10, double C_beta = 10, double step_size = 1){
  // find MAP for regression coeffs and loadings given factors via projected newton descent 
  
  int n = y.size();
  int k = M.n_cols;
  int q = X.n_cols;
  arma::mat M_new = join_rows(X, M);
  arma::vec regression_vector_init = join_cols(beta_init, lambda_init);
  arma::vec regression_vector = regression_vector_init;
  arma::mat H(k+q, k+q, arma::fill::zeros);
  for(int i = 1; i<=max_it; ++i){
    arma::mat res = grad_regression_coeffs_logit(
      y, observed, M_new, regression_vector, prior_var_beta, prior_var_lambda, q, k
    );
    
    arma::vec u = res(arma::span(0, k+q-1), 0);
    H = res(arma::span(0, k+q-1), arma::span(1, k+q));
    arma::vec update = -inv(H)*u;
    double norm_update = norm(update, 2);
    
    if(norm_update<epsilon){
      break;
    }
    
    regression_vector = regression_vector - step_size*inv(H)*u;
    if(regression_vector(0)>C_mu){
      regression_vector(0) = C_mu;
    }
    else if(regression_vector(0)<(-1)*C_mu){
      regression_vector(0) = (-1)*C_mu;
    }
    for(int j=1; j<(q); ++j){
      if(regression_vector(j)>C_beta){
        regression_vector(j) = C_beta;
      }
      else if(regression_vector(j)<(-C_beta)){
        regression_vector(j) = (-1)*C_beta;
      }
    }
    for(int j=q; j<(k+q); ++j){
      if(regression_vector(j)>C_lambda){
        regression_vector(j) = C_lambda;
      }
      else if(regression_vector(j)<(-C_lambda)){
        regression_vector(j) = (-1)*C_lambda;
      }
    }
    
  }
  arma::mat result((k+q), (k+q+1));
  result.col(0) = regression_vector; // MAP
  
  arma::mat I = arma::diagmat(arma::ones(k+q));
  arma::mat H_inv = inv(H); //inverse hessian
  result(arma::span(0, k+q-1), arma::span(1, k+q)) = H_inv;
  
  return result;
}


// [[Rcpp::export]]
arma::mat gen_mvnrnd(arma::vec mu, arma::mat Sigma, int n_MC=100){
  int k = Sigma.n_rows;
  arma::mat samples(n_MC, k);
  for(int i=0; i<n_MC; ++i) {
    samples(i, arma::span(0, k-1)) = arma::mvnrnd(mu, Sigma).t();
  }
  return samples;
}


// [[Rcpp::export]]
List map_regression_coeffs_mvlogit(
    arma::mat Y, arma::mat observed_mat, arma::mat X, arma::mat M, arma::mat Beta_init, arma::mat Lambda_init,
    arma::vec prior_var_beta, arma::vec prior_var_lambda, int n_MC=100, int max_it=200, double epsilon=0.001,
    double C_lambda=5, double C_mu=8, double C_beta=5, double step_size = 1){  
  
  int p = Y.n_cols;
  int n = Y.n_rows;
  int k = M.n_cols;
  int q = X.n_cols;
  
  arma::mat inits = join_rows(Beta_init, Lambda_init);
  arma::cube Lambda_samples(n_MC, k, p);
  arma::cube Beta_samples(n_MC, q, p);
  arma::mat Lambda_hat(p, k);
  arma::mat Beta_hat(p, q);
  arma::mat res_j(k+q, k+q);
  arma::cube V_js(k+q, k+q, p);
  arma::cube Lambda_covs(k+q, k+q, p);
  arma::mat V_j(k+q, k+q);
  arma::vec D(p);
  
  double rho_sq = 1;
  //omp_set_dynamic(1);
  //omp_set_num_threads(4);
  //#pragma omp parallel for 
  for(int j=0; j<p; ++j){
    res_j = optimize_regression_coeffs_logit(
      Y.col(j), observed_mat.col(j), X, M, Beta_init.row(j).t(), Lambda_init.row(j).t(), prior_var_beta(j), 
      prior_var_lambda(j),  max_it, epsilon, C_lambda=C_lambda, C_mu=C_mu, C_beta=C_beta, step_size);
    arma::mat V_j = res_j(arma::span(0, k+q-1), arma::span(1, k+q));
    V_j = -0.5*(V_j + V_j.t());
    D(j) = arma::trace(V_j(arma::span(q, k+q-1), arma::span(q, k+q-1)));
    V_js.slice(j) = V_j;
    
    arma::vec regr_vect_mean = res_j(arma::span(0, k+q-1), 0);
    Lambda_hat.row(j) = regr_vect_mean(arma::span(q, (k+q-1))).t();
    Beta_hat.row(j) = regr_vect_mean(arma::span(0, (q-1))).t();
    arma::mat sample_j = gen_mvnrnd(regr_vect_mean, rho_sq*V_j, n_MC);
    Beta_samples.slice(j) = sample_j(arma::span(0, n_MC-1), arma::span(0, q-1));
    Lambda_samples.slice(j) = sample_j(arma::span(0, n_MC-1), arma::span(q, k+q-1));
  }
  return List::create(Named("Lambda_samples") = Lambda_samples,
                      Named("Beta_samples") = Beta_samples,
                      Named("Lambda_hat") = Lambda_hat,
                      Named("Covariances") = V_js,
                      Named("Beta_hat") = Beta_hat,
                      Named("D") = D);
}

// [[Rcpp::export]]
arma::mat compute_inv_hessian_logit(
    arma::vec y, arma::vec observed, arma::mat X_tilde, arma::vec regr_coeff, 
    double prior_var_beta, double prior_var_lambda, int q){
  
  int n = y.size();
  int q_tilde = X_tilde.n_cols;
  int k = q_tilde - q;
  
  arma::mat hessian(q_tilde, q_tilde, arma::fill::zeros);
  //omp_set_dynamic(1);
  //omp_set_num_threads(4);
  //#pragma omp parallel for 
  for(int i=0; i<n; ++i){
    if(observed(i)==1) {
      double eta_lambda_i = dot(X_tilde.row(i), regr_coeff);
      double F_i = F_logistic(eta_lambda_i);
      hessian = hessian + F_i * (1- F_i) * X_tilde(i, arma::span(0, q_tilde-1)).t() * X_tilde(i, arma::span(0,q_tilde-1));
    }
  }
  arma::vec tau_1 = 1/prior_var_beta * arma::ones(q);
  arma::vec tau_2 = 1/prior_var_lambda * arma::ones(k);
  arma::vec prior_var_inv = join_cols(tau_1, tau_2);
  
  arma::mat D = arma::diagmat(join_cols(tau_1, tau_2)); 
  hessian = hessian + D;
  arma::mat result = inv(-hessian); //inverse hessian
  return result;
  
}


// [[Rcpp::export]]
arma::mat compute_D(
    arma::mat Y, arma::mat observed_mat, arma::mat X, arma::mat M, arma::mat Beta, arma::mat Lambda,
    arma::vec prior_var_beta, arma::vec prior_var_lambda, int n_MC=100){
  
  int p = Y.n_cols;
  int k = M.n_cols;
  int q = X.n_cols;
  
  arma::mat V_j(k+q, k+q);
  arma::vec D(p);
  arma::mat X_tilde = join_rows(X, M);
  arma::mat regr_coeffs = join_rows(Beta, Lambda);
  //omp_set_dynamic(1);
  //omp_set_num_threads(4);
  //#pragma omp parallel for 
  
  for(int j=0; j<p; ++j){
    V_j = compute_inv_hessian_logit(
      Y.col(j), observed_mat.col(j), X_tilde, regr_coeffs.row(j).t(), prior_var_beta(j), 
      prior_var_lambda(j), q);
    V_j = -0.5*(V_j + V_j.t());
    D(j) = arma::trace(V_j(arma::span(q, k+q-1), arma::span(q, k+q-1)));
  }
  return  D;
}


// [[Rcpp::export]]
arma::mat compute_variances(
    arma::mat Y, arma::mat observed_mat, arma::mat X, arma::mat M, arma::mat Beta, arma::mat Lambda,
    arma::vec prior_var_beta, arma::vec prior_var_lambda, int n_MC=100){
  
  int p = Y.n_cols;
  int k = M.n_cols;
  int q = X.n_cols;
  
  arma::mat V_j(k+q, k+q);
  arma::mat D(p, k+q);
  arma::mat X_tilde = join_rows(X, M);
  arma::mat regr_coeffs = join_rows(Beta, Lambda);
  //omp_set_dynamic(1);
  //omp_set_num_threads(4);
  //#pragma omp parallel for 
  
  for(int j=0; j<p; ++j){
    V_j = compute_inv_hessian_logit(
      Y.col(j), observed_mat.col(j), X_tilde, regr_coeffs.row(j).t(), prior_var_beta(j), 
      prior_var_lambda(j), q);
    V_j = -0.5*(V_j + V_j.t());
    D(j, arma::span(0, k+q-1)) = arma::diagvec(V_j).t();
  }
  return D;
}


// [[Rcpp::export]]
List sample_full_conditional_logit(
    arma::mat Y, arma::mat observed_mat, arma::mat X, arma::mat M, arma::mat Beta, arma::mat Lambda,
    arma::vec prior_var_beta, arma::vec prior_var_lambda, int n_MC=100, double rho=1){
  
  
  int p = Y.n_cols;
  int n = Y.n_rows;
  int k = M.n_cols;
  int q = X.n_cols;
  
  arma::mat regr_coeffs = join_rows(Beta, Lambda);
  arma::cube Lambda_samples(n_MC, k, p);
  arma::cube Beta_samples(n_MC, q, p);
  arma::mat Lambda_hat(p, k);
  arma::mat Beta_hat(p, q);
  arma::mat res_j(k+q, k+q);
  arma::cube V_js(k+q, k+q, p);
  arma::cube Lambda_covs(k+q, k+q, p);
  arma::mat V_j(k+q, k+q);
  arma::vec D(p);
  
  double rho_sq = rho*rho;
  
  
  arma::mat X_tilde = join_rows(X, M);
  //omp_set_dynamic(1);
  //omp_set_num_threads(4);
  //#pragma omp parallel for 
  for(int j=0; j<p; ++j){
    
    V_j = compute_inv_hessian_logit(
      Y.col(j), observed_mat.col(j), X_tilde, regr_coeffs.row(j).t(), prior_var_beta(j), 
      prior_var_lambda(j), q);
    Lambda_hat.row(j) = regr_coeffs(j, arma::span(q, (k+q-1)));
    Beta_hat.row(j) = regr_coeffs(j, arma::span(0, (q-1)));
    V_j = -0.5*(V_j + V_j.t());
    D(j) = arma::trace(V_j(arma::span(q, k+q-1), arma::span(q, k+q-1)));
    arma::mat sample_j = gen_mvnrnd(regr_coeffs.row(j).t(), rho_sq*V_j, n_MC);
    V_js.slice(j) = V_j;
    Beta_samples.slice(j) = sample_j(arma::span(0, n_MC-1), arma::span(0, q-1));
    Lambda_samples.slice(j) = sample_j(arma::span(0, n_MC-1), arma::span(q, k+q-1));
    
  }
  return List::create(Named("Lambda_samples") = Lambda_samples,
                      Named("Beta_samples") = Beta_samples,
                      Named("Lambda_hat") = Lambda_hat,
                      Named("Covariances") = V_js,
                      Named("Beta_hat") = Beta_hat,
                      Named("D") = D);
}




arma::mat grad_latent_factors_logit(
    arma::vec y, arma::vec observed, arma::vec x, arma::mat Lambda, arma::mat Beta,
    arma::vec M_i, double prior_var) {
  // returns gradient and hessian of latent logistic regression log-lik + normal prior w.r.t. latent factors
  int p = y.size();
  int k = Lambda.n_cols;
  
  arma::vec scores(p, arma::fill::zeros);
  arma::vec grad_log_pi(k, arma::fill::zeros);
  arma::mat hessian(k, k, arma::fill::zeros);
  
  //omp_set_dynamic(1);
  //omp_set_num_threads(4);
  //#pragma omp parallel for 
  for(int i=0; i<p; ++i){
    scores(i) = 0;
    if(observed(i)==1){
      double eta_lambda_i = dot(Lambda.row(i), M_i);
      double Beta_x_i = dot(Beta.row(i), x);
      double F_i = F_logistic(Beta_x_i + eta_lambda_i);
      scores(i) = y(i) - F_i;
      grad_log_pi = grad_log_pi + scores(i)*Lambda(i, arma::span(0,k-1)).t();
      hessian = hessian + F_i * (1 -F_i) * 
        Lambda(i, arma::span(0, k-1)).t() * Lambda(i, arma::span(0,k-1));
    }
  }
  arma::mat I_k = arma::diagmat(arma::ones(k));
  grad_log_pi = grad_log_pi - M_i/prior_var;
  hessian = hessian + (1/prior_var)*I_k;
  arma::mat result(k, k+1);
  result(arma::span(0, k-1), 0) = grad_log_pi;
  result(arma::span(0, k -1), arma::span(1, k)) = -hessian;
  return result;
}


arma::vec append_to_vec(arma::vec vec, double value) {
  vec.resize(vec.n_elem + 1);
  vec(vec.n_elem - 1) = value;
  return vec;
}

arma::vec optimize_latent_factors_logit(
    arma::vec y, arma::vec observed, arma::vec x, arma::mat Lambda, arma::mat Beta, arma::vec M_i_init, 
    double prior_var, int max_it, double epsilon, int n){
  
  int p = y.size();
  int k = Lambda.n_cols;
  int q = Beta.n_cols;
  
  arma::vec M_i = M_i_init;
  arma::mat H(k, k+1, arma::fill::zeros);
  for(int i = 1; i<=max_it; ++i){
    arma::mat res = grad_latent_factors_logit(
      y, observed, x, Lambda, Beta, M_i, prior_var);
    arma::vec u = res(arma::span(0, k-1), 0);
    H = res(arma::span(0, k-1), arma::span(1, k));
    arma::vec update = -inv(H)*u; 
    double norm_update = norm(update, 2);
    if(norm_update<epsilon){
      break;
    }
    double C_m = 2 * sqrt(log(k*n));
    M_i = M_i - inv(H)*u;
    for(int j=0; j<(k); ++j){
      if(M_i(j)>C_m){
        M_i(j) = C_m;
      }
      else if(M_i(j)<(-C_m)){
        M_i(j) = -C_m;
      }
    }
  }
  
  arma::vec result = M_i;
  return result;
}


// [[Rcpp::export]]
arma::mat map_latent_factors_mvlogit(
    arma::mat Y, arma::mat observed_mat, arma::mat X, arma::mat Lambda, arma::mat Beta, arma::mat M_init, 
    double prior_var=1, int max_it=200, double epsilon=0.001){
  
  
  int p = Y.n_cols;
  int n = Y.n_rows;
  int k = M_init.n_cols;
  int q = X.n_cols;
  
  arma::mat Eta_hat(n, k);
  arma::vec res_i(k);
  
  //omp_set_dynamic(1);
  //omp_set_num_threads(4);
  //#pragma omp parallel for 
  for(int i=0; i<n; ++i){
    res_i = optimize_latent_factors_logit(
      Y.row(i).t(), observed_mat.row(i).t(), X.row(i).t(), Lambda, Beta,  M_init.row(i).t(),
      prior_var,  max_it, epsilon, n);
    Eta_hat(i, arma::span(0, k-1)) = res_i(arma::span(0, k-1)).t();
  }
  return Eta_hat;
}




// [[Rcpp::export]]
arma::mat compute_B_hessian(arma::mat Lambda_hat, arma::vec W_bar, double sigma=1.702){
  int p = Lambda_hat.n_rows;
  arma::mat B_uv(p, p);
  
  double sigma_sq = pow(sigma,2);
  
  for(int u=0; u<p; ++u){
    double sigma_sq_u = sigma_sq + 1/W_bar[u];
    for(int v=u; v<p; ++v){
      double sigma_sq_v = sigma_sq + 1/W_bar[v];
      
      B_uv(u,v) = sqrt(1+ (
        dot( pow(Lambda_hat.row(u),2), pow(Lambda_hat.row(v),2) )  + 
          pow( dot( Lambda_hat.row(u), Lambda_hat.row(v) ), 2 ) 
      ) /  (sigma_sq_v * dot(Lambda_hat.row(u), Lambda_hat.row(u))  + sigma_sq_u * dot( Lambda_hat.row(v), Lambda_hat.row(v))  ) 
      );
      B_uv(v, u) = B_uv(u,v);
    }
    B_uv(u,u) = sqrt(1+ (dot(Lambda_hat.row(u),Lambda_hat.row(u)))/(2*sigma_sq_u));
  }
  double sigma_sq_u = sigma_sq + 1/W_bar[p-1];
  B_uv(p-1,p-1) = sqrt(1+ (dot(Lambda_hat.row(p-1),Lambda_hat.row(p-1)))/(2*sigma_sq_u));
  return B_uv;
}


// [[Rcpp::export]]
List sample_Lambda_outer(arma::cube Lambda_samples, arma::mat Lambda_hat, double rho){
  
  int n_MC = Lambda_samples.n_rows;
  int k = Lambda_samples.n_cols;
  int p = Lambda_samples.n_slices;
  
  arma::mat Lambda(p, k);
  arma::mat Lambda_cc(p, k);
  arma::mat Lambda_outer(p, p);
  arma::mat Lambda_outer_cc(p, p);
  arma::cube Lambda_outer_samples (p, p, n_MC);
  arma::cube Lambda_outer_samples_cc(p, p, n_MC);
  
  for(int s=0; s<n_MC; ++s) {
    Lambda = arma::mat(Lambda_samples.row(s)).t();
    Lambda_cc = Lambda_hat + rho*(Lambda - Lambda_hat);
    Lambda_outer = Lambda * Lambda.t();
    Lambda_outer_cc = Lambda_cc * Lambda_cc.t();
    Lambda_outer_samples.slice(s) = Lambda_outer;
    Lambda_outer_samples_cc.slice(s) = Lambda_outer_cc;
  }
  
  return List::create(Named("Lambda_outer_samples") = Lambda_outer_samples,
                      Named("Lambda_outer_samples_cc") = Lambda_outer_samples_cc);
}

// [[Rcpp::export]]
arma::cube correct_Beta_samples(arma::cube Beta_samples, arma::mat Beta_hat, double rho){
  
  int n_MC = Beta_samples.n_rows;
  int q = Beta_samples.n_cols;
  int p = Beta_samples.n_slices;
  
  arma::mat Beta(p, q);
  arma::mat Beta_cc(p, q);
  arma::cube Beta_samples_cc(p, q, n_MC);
  
  for(int s=0; s<n_MC; ++s) {
    Beta = arma::mat(Beta_samples.row(s)).t();
    Beta_cc = Beta_hat + rho*(Beta - Beta_hat);
    Beta_samples_cc.slice(s) = Beta_cc;
  }
  
  return Beta_samples_cc;
}

