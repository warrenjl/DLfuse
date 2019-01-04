#include "RcppArmadillo.h"
#include "RcppArmadillo.h"
#include "DLfuse.h"
using namespace arma;
using namespace Rcpp;

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]

Rcpp::List SpGPCW(int mcmc_samples,
                  arma::vec y,
                  arma::mat z,
                  arma::vec sample_size,
                  arma::mat spatial_dists,
                  arma::mat neighbors,
                  double metrop_var_A11_trans,
                  double metrop_var_A22_trans,
                  double metrop_var_mu,
                  arma::vec metrop_var_alpha,
                  double metrop_var_phi0_trans,
                  double metrop_var_phi1_trans,
                  Rcpp::Nullable<double> alpha_sigma2_epsilon_prior = R_NilValue,
                  Rcpp::Nullable<double> beta_sigma2_epsilon_prior = R_NilValue,
                  Rcpp::Nullable<double> sigma2_beta_prior = R_NilValue,
                  Rcpp::Nullable<double> sigma2_A_prior = R_NilValue,
                  Rcpp::Nullable<double> sigma2_mu_prior = R_NilValue,
                  Rcpp::Nullable<double> alpha_tau2_prior = R_NilValue,
                  Rcpp::Nullable<double> beta_tau2_prior = R_NilValue,
                  Rcpp::Nullable<double> alpha_phi0_prior = R_NilValue,
                  Rcpp::Nullable<double> beta_phi0_prior = R_NilValue,
                  Rcpp::Nullable<double> alpha_phi1_prior = R_NilValue,
                  Rcpp::Nullable<double> beta_phi1_prior = R_NilValue,
                  Rcpp::Nullable<double> sigma2_epsilon_init = R_NilValue,
                  Rcpp::Nullable<double> beta0_init = R_NilValue,
                  Rcpp::Nullable<double> beta1_init = R_NilValue,
                  Rcpp::Nullable<double> A11_init = R_NilValue,
                  Rcpp::Nullable<double> A22_init = R_NilValue,
                  Rcpp::Nullable<double> A21_init = R_NilValue,
                  Rcpp::Nullable<double> mu_init = R_NilValue,
                  Rcpp::Nullable<Rcpp::NumericVector> alpha_init = R_NilValue,
                  Rcpp::Nullable<double> tau2_init = R_NilValue,
                  Rcpp::Nullable<Rcpp::NumericVector> w0_init = R_NilValue,
                  Rcpp::Nullable<double> phi0_init = R_NilValue,
                  Rcpp::Nullable<Rcpp::NumericVector> w1_init = R_NilValue,
                  Rcpp::Nullable<double> phi1_init = R_NilValue){

//Defining Parameters and Quantities of Interest
int n = y.size();
int m = z.n_rows;
int L = z.n_cols;
arma::mat Dw(m, m); Dw.fill(0);
for(int j = 0; j < m; ++ j){
   Dw(j, j) = sum(neighbors.row(j));
   } 
arma::mat CAR = Dw - 
                neighbors;
int G = 1;  //G Always Equal to One in this Case (One Island because of Inverse Distance Weighting)

arma::vec sigma2_epsilon(mcmc_samples); sigma2_epsilon.fill(0.00);
arma::vec beta0(mcmc_samples); beta0.fill(0.00);
arma::vec beta1(mcmc_samples); beta1.fill(0.00);
arma::vec A11(mcmc_samples); A11.fill(0.00);
arma::vec A22(mcmc_samples); A22.fill(0.00);
arma::vec A21(mcmc_samples); A21.fill(0.00);
arma::vec mu(mcmc_samples); mu.fill(0.00);
arma::mat alpha(m, mcmc_samples); alpha.fill(0.00);
arma::vec tau2(mcmc_samples); tau2.fill(0.00);
arma::mat w0(n, mcmc_samples); w0.fill(0.00);
arma::vec phi0(mcmc_samples); phi0.fill(0.00);
arma::mat w1(n, mcmc_samples); w1.fill(0.00);
arma::vec phi1(mcmc_samples); phi1.fill(0.00);
arma::vec neg_two_loglike(mcmc_samples); neg_two_loglike.fill(0.00);

//Prior Information
double alpha_sigma2_epsilon = 3.00;
if(alpha_sigma2_epsilon_prior.isNotNull()){
  alpha_sigma2_epsilon = Rcpp::as<double>(alpha_sigma2_epsilon_prior);
  }

double beta_sigma2_epsilon = 2.00;
if(beta_sigma2_epsilon_prior.isNotNull()){
  beta_sigma2_epsilon = Rcpp::as<double>(beta_sigma2_epsilon_prior);
  }

double sigma2_beta = 10000.00;
if(sigma2_beta_prior.isNotNull()){
  sigma2_beta = Rcpp::as<double>(sigma2_beta_prior);
  }

double sigma2_A = 1.00;
if(sigma2_A_prior.isNotNull()){
  sigma2_A = Rcpp::as<double>(sigma2_A_prior);
  }

double sigma2_mu = 1.00;
if(sigma2_mu_prior.isNotNull()){
  sigma2_mu = Rcpp::as<double>(sigma2_mu_prior);
  }

double alpha_tau2 = 3.00;
if(alpha_tau2_prior.isNotNull()){
  alpha_tau2 = Rcpp::as<double>(alpha_tau2_prior);
  }

double beta_tau2 = 2.00;
if(beta_tau2_prior.isNotNull()){
  beta_tau2 = Rcpp::as<double>(beta_tau2_prior);
  }

double alpha_phi0 = 1.00;
if(alpha_phi0_prior.isNotNull()){
  alpha_phi0 = Rcpp::as<double>(alpha_phi0_prior);
  }

double beta_phi0 = 1.00;
if(beta_phi0_prior.isNotNull()){
  beta_phi0 = Rcpp::as<double>(beta_phi0_prior);
  }

double alpha_phi1 = 1.00;
if(alpha_phi1_prior.isNotNull()){
  alpha_phi1 = Rcpp::as<double>(alpha_phi1_prior);
  }

double beta_phi1 = 1.00;
if(beta_phi1_prior.isNotNull()){
  beta_phi1 = Rcpp::as<double>(beta_phi1_prior);
  }

//Initial Values
sigma2_epsilon(0) = 1.00;
if(sigma2_epsilon_init.isNotNull()){
  sigma2_epsilon(0) = Rcpp::as<double>(sigma2_epsilon_init);
  }

beta0(0) = 0.00;
if(beta0_init.isNotNull()){
  beta0(0) = Rcpp::as<double>(beta0_init);
  }

beta1(0) = 0.00;
if(beta1_init.isNotNull()){
  beta1(0) = Rcpp::as<double>(beta1_init);
  }

A11(0) = 1.00;
if(A11_init.isNotNull()){
  A11(0) = Rcpp::as<double>(A11_init);
  }

A22(0) = 1.00;
if(A22_init.isNotNull()){
  A22(0) = Rcpp::as<double>(A22_init);
  }

A21(0) = 0.00;
if(A21_init.isNotNull()){
  A21(0) = Rcpp::as<double>(A21_init);
  }

mu(0) = 0.00;
if(mu_init.isNotNull()){
  mu(0) = Rcpp::as<double>(mu_init);
  }

alpha.col(0).fill(0.00);
if(alpha_init.isNotNull()){
  alpha.col(0) = Rcpp::as<arma::vec>(alpha_init);
  }

tau2(0) = 1.00;
if(tau2_init.isNotNull()){
  tau2(0) = Rcpp::as<double>(tau2_init);
  }

w0.col(0).fill(0.00);
if(w0_init.isNotNull()){
  w0.col(0) = Rcpp::as<arma::vec>(w0_init);
  }

phi0(0) = 1.00;
if(phi0_init.isNotNull()){
  phi0(0) = Rcpp::as<double>(phi0_init);
  }

Rcpp::List spatial_corr0_info = spatial_corr_fun(phi0(0), 
                                                 spatial_dists);

w1.col(0).fill(0.00);
if(w1_init.isNotNull()){
  w1.col(0) = Rcpp::as<arma::vec>(w1_init);
  }

phi1(0) = 1.00;
if(phi1_init.isNotNull()){
  phi1(0) = Rcpp::as<double>(phi1_init);
  }

Rcpp::List spatial_corr1_info = spatial_corr_fun(phi1(0), 
                                                 spatial_dists);

Rcpp::List lagged_covars = construct_lagged_covars(z,
                                                   mu(0), 
                                                   alpha.col(0),
                                                   sample_size);
arma::vec lc1 = lagged_covars(0);
arma::vec lc2 = lagged_covars(1);

arma::uvec keep5(5); keep5(0) = 0; keep5(1) = 1; keep5(2) = 2; keep5(3) = 3; keep5(4) = 4;
arma::vec mean_temp = construct_mean(beta0(0), 
                                     beta1(0),
                                     A11(0),
                                     A22(0),
                                     A21(0),
                                     w0.col(0),
                                     w1.col(0),
                                     diagmat(lc1),
                                     keep5,
                                     sample_size);

neg_two_loglike(0) = neg_two_loglike_update(y,
                                            mean_temp,
                                            sigma2_epsilon(0));

//Metropolis Settings
int acctot_A11_trans = 0;
int acctot_A22_trans = 0;
int acctot_mu = 0;
arma::vec acctot_alpha(m); acctot_alpha.fill(0);
int acctot_phi0_trans = 0;
int acctot_phi1_trans = 0;

//Main Sampling Loop
for(int j = 1; j < mcmc_samples; ++j){
  
   //sigma2_epsilon Update
   arma::vec mean_temp = construct_mean(beta0(j-1), 
                                        beta1(j-1),
                                        A11(j-1),
                                        A22(j-1),
                                        A21(j-1),
                                        w0.col(j-1),
                                        w1.col(j-1),
                                        diagmat(lc1),
                                        keep5,
                                        sample_size);
  
   sigma2_epsilon(j) = sigma2_epsilon_update(y,
                                             mean_temp,
                                             sample_size,
                                             alpha_sigma2_epsilon,
                                             beta_sigma2_epsilon);
  
   //beta0 Update
   arma::uvec keep4(4); keep4(0) = 1; keep4(1) = 2; keep4(2) = 3; keep4(3) = 4;
   mean_temp = construct_mean(beta0(j-1), 
                              beta1(j-1),
                              A11(j-1),
                              A22(j-1),
                              A21(j-1),
                              w0.col(j-1),
                              w1.col(j-1),
                              diagmat(lc1),
                              keep4,
                              sample_size);
   
   beta0(j) = beta0_update(y,
                           mean_temp, 
                           sigma2_epsilon(j),
                           sample_size,
                           sigma2_beta);
   
   //beta1 Update
   keep4(0) = 0; keep4(1) = 2; keep4(2) = 3; keep4(3) = 4;
   mean_temp = construct_mean(beta0(j), 
                              beta1(j-1),
                              A11(j-1),
                              A22(j-1),
                              A21(j-1),
                              w0.col(j-1),
                              w1.col(j-1),
                              diagmat(lc1),
                              keep4,
                              sample_size);
   beta1(j) = beta1_update(y,
                           mean_temp,
                           lagged_covars,
                           sigma2_epsilon(j),
                           sample_size,
                           sigma2_beta);
   
   //A11 Update
   Rcpp::List A11_output = A11_update(y,
                                      A11(j-1),
                                      sigma2_epsilon(j),
                                      beta0(j),
                                      beta1(j),
                                      A22(j-1),
                                      A21(j-1),
                                      w0.col(j-1),
                                      w1.col(j-1),
                                      lagged_covars,
                                      keep5,
                                      sample_size,
                                      sigma2_A,
                                      metrop_var_A11_trans,
                                      acctot_A11_trans);
   
   A11(j) = A11_output[0];
   acctot_A11_trans = A11_output[1];
   
   //A22 Update
   Rcpp::List A22_output = A22_update(y,
                                      A22(j-1),
                                      sigma2_epsilon(j),
                                      beta0(j),
                                      beta1(j),
                                      A11(j),
                                      A21(j-1),
                                      w0.col(j-1),
                                      w1.col(j-1),
                                      lagged_covars,
                                      keep5,
                                      sample_size,
                                      sigma2_A,
                                      metrop_var_A11_trans,
                                      acctot_A11_trans);
   
   A22(j) = A22_output[0];
   acctot_A22_trans = A22_output[1];
   
   //A21 Update
   keep4(0) = 0; keep4(1) = 1; keep4(2) = 2; keep4(3) = 5;
   mean_temp = construct_mean(beta0(j), 
                              beta1(j),
                              A11(j),
                              A22(j),
                              A21(j-1),
                              w0.col(j-1),
                              w1.col(j-1),
                              diagmat(lc1),
                              keep4,
                              sample_size);
   
   A21(j) = A21_update(y,
                       mean_temp,
                       lagged_covars,
                       sigma2_epsilon(j),
                       w0.col(j),
                       sigma2_A);
   
   //tau2 Update
   tau2(j) = tau2_update(G,
                         CAR,
                         alpha.col(j),
                         alpha_tau2,
                         beta_tau2);
   
   //w0 Update
   arma::uvec keep3(3); keep3(0) = 0; keep3(1) = 1; keep3(2) = 4;
   mean_temp = construct_mean(beta0(j), 
                              beta1(j),
                              A11(j),
                              A22(j),
                              A21(j),
                              w0.col(j-1),
                              w1.col(j-1),
                              diagmat(lc1),
                              keep3,
                              sample_size);
   
   w0.col(j) = w0_update(y,
                         mean_temp,
                         lagged_covars,
                         sigma2_epsilon(j),
                         A11(j),
                         A21(j),
                         spatial_corr0_info[0]);
  
   //phi0 Update
   Rcpp::List phi0_output = phi_update(phi0(j-1),
                                       spatial_dists,
                                       w0,
                                       spatial_corr0_info,
                                       alpha_phi0,
                                       beta_phi0,
                                       metrop_var_phi0_trans,
                                       acctot_phi0_trans);
  
   phi0(j) = phi0_output[0];
   acctot_phi0_trans = phi0_output[1];
   spatial_corr0_info = phi0_output[2];
   
   //w1 Update
   keep4(0) = 0; keep4(1) = 1; keep4(2) = 2; keep4(3) = 3;
   mean_temp = construct_mean(beta0(j), 
                              beta1(j),
                              A11(j),
                              A22(j),
                              A21(j),
                              w0.col(j),
                              w1.col(j-1),
                              diagmat(lc1),
                              keep4,
                              sample_size);
   
   w1.col(j) = w1_update(y,
                         mean_temp,
                         lagged_covars,
                         sigma2_epsilon(j),
                         A22(j),
                         spatial_corr1_info[0]);
   
   //phi1 Update
   Rcpp::List phi1_output = phi_update(phi1(j-1),
                                       spatial_dists,
                                       w1,
                                       spatial_corr1_info,
                                       alpha_phi1,
                                       beta_phi1,
                                       metrop_var_phi1_trans,
                                       acctot_phi1_trans);
   
   phi1(j) = phi1_output[0];
   acctot_phi1_trans = phi1_output[1];
   spatial_corr1_info = phi1_output[2];
   
   //neg_two_loglike Update
   mean_temp = construct_mean(beta0(j), 
                              beta1(j),
                              A11(j),
                              A22(j),
                              A21(j),
                              w0.col(j),
                              w1.col(j),
                              diagmat(lc1),
                              keep5,
                              sample_size);
   
   neg_two_loglike(j) = neg_two_loglike_update(y,
                                               mean_temp,
                                               sigma2_epsilon(j));
   
   //Progress
   if((j + 1) % 10 == 0){ 
     Rcpp::checkUserInterrupt();
     }
  
   if(((j + 1) % int(round(mcmc_samples*0.10)) == 0)){
     double completion = round(100*((j + 1)/(double)mcmc_samples));
     Rcpp::Rcout << "Progress: " << completion << "%" << std::endl;
     if(rho_zero == 0){
       double accrate_rho_trans = round(100*(acctot_rho_trans/(double)j));
       Rcpp::Rcout << "rho Acceptance: " << accrate_rho_trans << "%" << std::endl;
       }
     double accrate_phi_trans = round(100*(acctot_phi_trans/(double)j));
     Rcpp::Rcout << "phi Acceptance: " << accrate_phi_trans << "%" << std::endl;
     Rcpp::Rcout << "*******************" << std::endl;
     }
  
   }
                                  
return Rcpp::List::create(Rcpp::Named("beta") = beta,
                          Rcpp::Named("theta") = theta,
                          Rcpp::Named("sigma2_theta") = sigma2_theta,
                          Rcpp::Named("eta") = eta,
                          Rcpp::Named("rho") = rho,
                          Rcpp::Named("sigma2_eta") = sigma2_eta,
                          Rcpp::Named("phi") = phi,
                          Rcpp::Named("neg_two_loglike") = neg_two_loglike,
                          Rcpp::Named("acctot_rho_trans") = acctot_rho_trans,
                          Rcpp::Named("acctot_phi_trans") = acctot_phi_trans);

}

