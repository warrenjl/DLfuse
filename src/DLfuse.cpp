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
                  arma::mat spatial_dists,
                  arma::mat neighbors,
                  double metrop_var_A11_trans,
                  double metrop_var_A22_trans,
                  double metrop_var_phi0_trans,
                  double metrop_var_phi1_trans,
                  double metrop_var_mu,
                  arma::vec metrop_var_alpha,
                  Rcpp::Nullable<double> alpha_sigma2_epsilon_prior = R_NilValue,
                  Rcpp::Nullable<double> beta_sigma2_epsilon_prior = R_NilValue,
                  Rcpp::Nullable<double> sigma2_beta_prior = R_NilValue,
                  Rcpp::Nullable<double> sigma2_A_prior = R_NilValue,
                  Rcpp::Nullable<double> alpha_phi0_prior = R_NilValue,
                  Rcpp::Nullable<double> beta_phi0_prior = R_NilValue,
                  Rcpp::Nullable<double> alpha_phi1_prior = R_NilValue,
                  Rcpp::Nullable<double> beta_phi1_prior = R_NilValue
                  Rcpp::Nullable<double> sigma2_mu_prior = R_NilValue,
                  Rcpp::Nullable<double> alpha_tau2_prior = R_NilValue,
                  Rcpp::Nullable<double> beta_tau2_prior = R_NilValue,
                  ){

//Defining Parameters and Quantities of Interest
arma::vec sigma2_epsilon(mcmc_samples); sigma2_epsilon.fill(0.00);
arma::vec beta0(mcmc_samples); beta0.fill(0.00);
arma::vec beta1(mcmc_samples); beta1.fill(0.00);
arma::vec A11(mcmc_samples); A11.fill(0.00);
arma::vec A22(mcmc_samples); A22.fill(0.00);
arma::vec A21(mcmc_samples); A21.fill(0.00);
arma::vec mu(mcmc_samples); mu.fill(0.00);
arma::mat alpha(z.n_rows, mcmc_samples); alpha.fill(0.00);
arma::vec tau2(mcmc_samples); tau2.fill(0.00);
arma::mat w0(y.size(), mcmc_samples); w0.fill(0.00);
arma::vec phi0(mcmc_samples); phi0.fill(0.00);
arma::mat w1(y.size(), mcmc_samples); w1.fill(0.00);
arma::vec phi1(mcmc_samples); phi1.fill(0.00);
arma::vec neg_two_loglike(mcmc_samples); neg_two_loglike.fill(0.00);

//Prior Information
double sigma2_beta = 10000.00;
if(sigma2_beta_prior.isNotNull()){
  sigma2_beta = Rcpp::as<double>(sigma2_beta_prior);
  }

double alpha_sigma2_theta = 3.00;
if(alpha_sigma2_theta_prior.isNotNull()){
  alpha_sigma2_theta = Rcpp::as<double>(alpha_sigma2_theta_prior);
  }
  
double beta_sigma2_theta = 2.00;
if(beta_sigma2_theta_prior.isNotNull()){
  beta_sigma2_theta = Rcpp::as<double>(beta_sigma2_theta_prior);
  }

double a_rho = 0.00;
if(a_rho_prior.isNotNull()){
  a_rho = Rcpp::as<double>(a_rho_prior);
  }

double b_rho = 1.00;
if(b_rho_prior.isNotNull()){
  b_rho = Rcpp::as<double>(b_rho_prior);
  }

double alpha_sigma2_eta = 3.00;
if(alpha_sigma2_eta_prior.isNotNull()){
  alpha_sigma2_eta = Rcpp::as<double>(alpha_sigma2_eta_prior);
  }

double beta_sigma2_eta = 2.00;
if(beta_sigma2_eta_prior.isNotNull()){
  beta_sigma2_eta = Rcpp::as<double>(beta_sigma2_eta_prior);
  }

double a_phi = log(0.9999)/(-(z.n_cols - 1));  
if(a_phi_prior.isNotNull()){
  a_phi = Rcpp::as<double>(a_phi_prior);
  }
  
double b_phi = log(0.0001)/(-1);
if(b_phi_prior.isNotNull()){
  b_phi = Rcpp::as<double>(b_phi_prior);
  }

//Initial Values
beta.col(0).fill(0.00);
if(beta_init.isNotNull()){
  beta.col(0) = Rcpp::as<arma::vec>(beta_init);
  }

theta.col(0).fill(0.00);
if(theta_init.isNotNull()){
  theta.col(0) = Rcpp::as<arma::vec>(theta_init);
  }

sigma2_theta(0) = 1.00;
if(sigma2_theta_init.isNotNull()){
  sigma2_theta(0) = Rcpp::as<double>(sigma2_theta_init);
  }

arma::mat eta_temp(neighbors.n_cols, z.n_cols); eta_temp.fill(0.00);
if(eta_init.isNotNull()){
  eta_temp = Rcpp::as<arma::mat>(eta_init);
  }
eta[0] = eta_temp;

rho(0) = (b_rho - a_rho)*0.50;
if(rho_init.isNotNull()){
  rho(0) = Rcpp::as<double>(rho_init);
  }

sigma2_eta(0) = 1.00;
if(sigma2_eta_init.isNotNull()){
  sigma2_eta(0) = Rcpp::as<double>(sigma2_eta_init);
  }

phi(0) = (b_phi - a_phi)*0.01;
if(phi_init.isNotNull()){
  phi(0) = Rcpp::as<double>(phi_init);
  }

Rcpp::List temporal_corr_info = temporal_corr_fun(z.n_cols, phi(0));
neg_two_loglike(0) = neg_two_loglike_update(y,
                                            x,
                                            z,
                                            site_id,
                                            beta.col(0),
                                            theta.col(0),
                                            eta[0]);

//Non Spatial Option (\rho fixed at 0):
//rho_zero = 0; Spatial
//rho_zero = Any Other Integer (Preferably One); Non Spatial
int rho_zero = 0;
if(rho_zero_indicator.isNotNull()){
  rho_zero = Rcpp::as<int>(rho_zero_indicator);
  }

//Metropolis Settings
int acctot_rho_trans = 0;
int acctot_phi_trans = 0;

//Main Sampling Loop
for(int j = 1; j < mcmc_samples; ++j){
  
   //sigma2_epsilon Update
   sigma2_epsilon(j) = sigma2_epsilon_update(y,
                                             sample_size,
                                             alpha_sigma2_epsilon,
                                             beta_sigma2_epsilon,
                                             mean_temp);
  
   //beta0 Update
   beta0(j) = beta0_update(y,
                           sample_size,
                           sigma2_beta,
                           mean_temp, 
                           sigma2_epsilon(j));
   
   //beta1 Update
   beta1(j) = beta1_update(y,
                           sample_size,
                           sigma2_beta,
                           mean_temp,
                           lagged_covars,
                           sigma2_epsilon(j));
   
   //A21 Update
   A21(j) = A21_update(sigma2_A,
                       mean_temp,
                       lagged_covars,
                       sigma2_epsilon(j),
                       w0.col(j));
   
   //tau2 Update
   tau2(j) = tau2_update(G,
                         CAR,
                         alpha_tau2,
                         beta_tau2,
                         alpha.col(j));
   
   //w0 Update
   w0.col(j) = w0_update(y,
                         mean_temp,
                         lagged_covars,
                         A11(j),
                         A21(j),
                         sigma2_epsilon(j),
                         Sigma0_inv);
  
   //w1 Update
   w1.col(j) = w1_update(y,
                         mean_temp,
                         lagged_covars,
                         A22(j),
                         sigma2_epsilon(j),
                         Sigma1_inv);
   
   //neg_two_loglike Update
   neg_two_loglike(j) = neg_two_loglike_update(y,
                                               mean_temp,
                                               sigma2_epsilon(j));
  
   //beta Update
   beta.col(j) = beta_update(x, 
                             z,
                             site_id,
                             sigma2_beta,
                             w,
                             gamma,
                             theta.col(j-1),
                             eta[j-1]);
   
   //theta Update
   theta.col(j) = theta_update(x, 
                               z,
                               site_id,
                               w,
                               gamma,
                               beta.col(j),
                               eta[j-1],
                               sigma2_theta(j-1),
                               temporal_corr_info(0));
   
   //sigma2_theta Update
   sigma2_theta(j) = sigma2_theta_update(theta.col(j),
                                         temporal_corr_info(0),
                                         alpha_sigma2_theta,
                                         beta_sigma2_theta);
   
   //eta Update
   eta[j] = eta_update(eta[j-1],
                       x, 
                       z,
                       site_id,
                       neighbors,
                       w,
                       gamma,
                       beta.col(j),
                       theta.col(j),
                       rho(j-1),
                       sigma2_eta(j-1),
                       temporal_corr_info[0]);
  
   //rho Update
   //Only if rho_zero = 0
   rho(j) = 0;
   if(rho_zero == 0){
     Rcpp::List rho_output = rho_update(rho(j-1),
                                        neighbors,
                                        eta[j],
                                        sigma2_eta(j-1),
                                        temporal_corr_info[0],
                                        a_rho,
                                        b_rho,
                                        metrop_var_rho_trans,
                                        acctot_rho_trans);
   
     rho(j) = rho_output[0];
     acctot_rho_trans = rho_output[1];
     }
   
   //sigma2_eta Update
   sigma2_eta(j) = sigma2_eta_update(neighbors,
                                     eta[j],
                                     rho(j),
                                     temporal_corr_info(0),
                                     alpha_sigma2_eta,
                                     beta_sigma2_eta);
   
   //phi Update
   Rcpp::List phi_output = phi_update(phi(j-1),
                                      neighbors,
                                      theta.col(j),
                                      sigma2_theta(j),
                                      eta[j],
                                      rho(j),
                                      sigma2_eta(j),
                                      temporal_corr_info,
                                      a_phi,
                                      b_phi,
                                      metrop_var_phi_trans,
                                      acctot_phi_trans);
     
   phi(j) = phi_output[0];
   acctot_phi_trans = phi_output[1];
   temporal_corr_info = phi_output[2];

   //neg_two_loglike Update
   neg_two_loglike(j) = neg_two_loglike_update(y,
                                               x,
                                               z,
                                               site_id,
                                               beta.col(j),
                                               theta.col(j),
                                               eta[j]);
   
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

