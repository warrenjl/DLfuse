#include "RcppArmadillo.h"
#include "DLfuse.h"
using namespace arma;
using namespace Rcpp;

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]

Rcpp::List DLfuse_s(int mcmc_samples,
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
                    Rcpp::Nullable<double> phi1_init = R_NilValue,
                    Rcpp::Nullable<int> model_type_indicator = R_NilValue){
  
//model_type_indicator = 0: Full, Distributed Lag Model
//model_type_indicator = 1: No Distributed Lags, Original Model
//model_type_indicator = 2: Ordinary Kriging Model
//model_type_indicator = 3: Simple Linear Regression
int model_type = 0;
if(model_type_indicator.isNotNull()){
  model_type = Rcpp::as<int>(model_type_indicator);
  }

//Defining Parameters and Quantities of Interest
int n = y.size();
int m = z.n_rows;
arma::mat Dw(m, m); Dw.fill(0.00);
for(int j = 0; j < m; ++ j){
   Dw(j, j) = sum(neighbors.row(j));
   } 
arma::mat CAR = Dw - 
                neighbors;
int G = 1.00;  //G Always Equal to One in this Case (One Island because of Inverse Distance Weighting)
double max_dist = spatial_dists.max();

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
double alpha_sigma2_epsilon = 0.01;
if(alpha_sigma2_epsilon_prior.isNotNull()){
  alpha_sigma2_epsilon = Rcpp::as<double>(alpha_sigma2_epsilon_prior);
  }

double beta_sigma2_epsilon = 0.01;
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

phi0(0) = -log(0.05)/max_dist;  //Effective range equal to largest distance in dataset (strong spatial correlation)
if(phi0_init.isNotNull()){
  phi0(0) = Rcpp::as<double>(phi0_init);
  }

Rcpp::List spatial_corr0_info = spatial_corr_fun(phi0(0), 
                                                 spatial_dists);

w1.col(0).fill(0.00);
if(w1_init.isNotNull()){
  w1.col(0) = Rcpp::as<arma::vec>(w1_init);
  }

phi1(0) = -log(0.05)/max_dist;  //Effective range equal to largest distance in dataset (strong spatial correlation)
if(phi1_init.isNotNull()){
  phi1(0) = Rcpp::as<double>(phi1_init);
  }

Rcpp::List spatial_corr1_info = spatial_corr_fun(phi1(0), 
                                                 spatial_dists);

Rcpp::List lagged_covars = construct_lagged_covars_s(z,
                                                     mu(0), 
                                                     alpha.col(0),
                                                     sample_size);

if(model_type == 1 || model_type == 3){
  double negative_infinity = -std::numeric_limits<double>::infinity();
  lagged_covars = construct_lagged_covars_s(z,
                                            negative_infinity, 
                                            alpha.col(0),
                                            sample_size);
  }
  
arma::vec lc1 = lagged_covars[0];
arma::vec lc2 = lagged_covars[1];

arma::uvec keep5(5); keep5(0) = 0; keep5(1) = 1; keep5(2) = 2; keep5(3) = 3; keep5(4) = 4;
arma::vec mean_temp = construct_mean_s(beta0(0), 
                                       beta1(0),
                                       A11(0),
                                       A22(0),
                                       A21(0),
                                       w0.col(0),
                                       w1.col(0),
                                       diagmat(lc1),
                                       keep5,
                                       sample_size);

neg_two_loglike(0) = neg_two_loglike_update_s(y,
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
for(int j = 1; j < mcmc_samples; ++ j){
  
   //sigma2_epsilon Update
   arma::vec mean_temp = construct_mean_s(beta0(j-1), 
                                          beta1(j-1),
                                          A11(j-1),
                                          A22(j-1),
                                          A21(j-1),
                                          w0.col(j-1),
                                          w1.col(j-1),
                                          diagmat(lc1),
                                          keep5,
                                          sample_size);
  
   sigma2_epsilon(j) = sigma2_epsilon_update_s(y,
                                               mean_temp,
                                               sample_size,
                                               alpha_sigma2_epsilon,
                                               beta_sigma2_epsilon);
  
   //beta0 Update
   arma::uvec keep4(4); keep4(0) = 1; keep4(1) = 2; keep4(2) = 3; keep4(3) = 4;
   mean_temp = construct_mean_s(beta0(j-1), 
                                beta1(j-1),
                                A11(j-1),
                                A22(j-1),
                                A21(j-1),
                                w0.col(j-1),
                                w1.col(j-1),
                                diagmat(lc1),
                                keep4,
                                sample_size);
   
   beta0(j) = beta0_update_s(y,
                             mean_temp, 
                             sigma2_epsilon(j),
                             sample_size,
                             sigma2_beta);
   
   beta1(j) = 0.00;
   if(model_type != 2){

     //beta1 Update
     keep4(0) = 0; keep4(1) = 2; keep4(2) = 3; keep4(3) = 4;
     mean_temp = construct_mean_s(beta0(j), 
                                  beta1(j-1),
                                  A11(j-1),
                                  A22(j-1),
                                  A21(j-1),
                                  w0.col(j-1),
                                  w1.col(j-1),
                                  diagmat(lc1),
                                  keep4,
                                  sample_size);
   
     beta1(j) = beta1_update_s(y,
                               mean_temp,
                               lagged_covars,
                               sigma2_epsilon(j),
                               sample_size,
                               sigma2_beta);
     
     }
   
   A11(j) = 0.00;  
   if(model_type != 3){
  
     //A11 Update
     Rcpp::List A11_output = A11_update_s(y,
                                          A11(j-1),
                                          lagged_covars,
                                          sigma2_epsilon(j),
                                          beta0(j),
                                          beta1(j),
                                          A22(j-1),
                                          A21(j-1),
                                          w0.col(j-1),
                                          w1.col(j-1),
                                          keep5,
                                          sample_size,
                                          sigma2_A,
                                          metrop_var_A11_trans,
                                          acctot_A11_trans);
   
     A11(j) = Rcpp::as<double>(A11_output[0]);
     acctot_A11_trans = A11_output[1];
     }
   
   A22(j) = 0.00;
   A21(j) = 0.00;
   if(model_type == 0 || model_type == 1){

     //A22 Update
     Rcpp::List A22_output = A22_update_s(y,
                                          A22(j-1),
                                          lagged_covars,
                                          sigma2_epsilon(j),
                                          beta0(j),
                                          beta1(j),
                                          A11(j),
                                          A21(j-1),
                                          w0.col(j-1),
                                          w1.col(j-1),
                                          keep5,
                                          sample_size,
                                          sigma2_A,
                                          metrop_var_A22_trans,
                                          acctot_A22_trans);
   
     A22(j) = Rcpp::as<double>(A22_output[0]);
     acctot_A22_trans = A22_output[1];
   
     //A21 Update
     keep4(0) = 0; keep4(1) = 1; keep4(2) = 2; keep4(3) = 4;
     mean_temp = construct_mean_s(beta0(j), 
                                  beta1(j),
                                  A11(j),
                                  A22(j),
                                  A21(j-1),
                                  w0.col(j-1),
                                  w1.col(j-1),
                                  diagmat(lc1),
                                  keep4,
                                  sample_size);
     
     A21(j) = A21_update_s(y,
                           mean_temp,
                           lagged_covars,
                           sigma2_epsilon(j),
                           w0.col(j-1),
                           sigma2_A);
     }
    
   mu(j) = 0.00;
   alpha.col(j).fill(0.00);
   tau2(j) = 0.00;
   if(model_type == 0){
   
     //mu Update
     Rcpp::List mu_output = mu_update_s(y,
                                        z,
                                        mu(j-1),
                                        lagged_covars,
                                        sigma2_epsilon(j),
                                        beta0(j),
                                        beta1(j),
                                        A11(j),
                                        A22(j),
                                        A21(j),
                                        alpha.col(j-1),
                                        w0.col(j-1),
                                        w1.col(j-1),
                                        keep5,
                                        sample_size,
                                        sigma2_mu,
                                        metrop_var_mu,
                                        acctot_mu);
   
     mu(j) = Rcpp::as<double>(mu_output[0]);
     acctot_mu = mu_output[1];
     lagged_covars = mu_output[2];
     lc1 = Rcpp::as<arma::vec>(lagged_covars[0]);
     lc2 = Rcpp::as<arma::vec>(lagged_covars[1]);
   
     //alpha Update
     Rcpp::List alpha_output = alpha_update_s(y,
                                              z,
                                              neighbors,
                                              alpha.col(j-1),
                                              lagged_covars,
                                              sigma2_epsilon(j),
                                              beta0(j),
                                              beta1(j),
                                              A11(j),
                                              A22(j),
                                              A21(j),
                                              mu(j),
                                              tau2(j-1),
                                              w0.col(j-1),
                                              w1.col(j-1),
                                              keep5,
                                              sample_size,
                                              metrop_var_alpha,
                                              acctot_alpha);   
   
     alpha.col(j) = as<arma::vec>(alpha_output[0]);
     acctot_alpha = as<arma::vec>(alpha_output[1]);
     lagged_covars = alpha_output[2];
     lc1 = Rcpp::as<arma::vec>(lagged_covars[0]);
     lc2 = Rcpp::as<arma::vec>(lagged_covars[1]);
   
     //tau2 Update
     tau2(j) = tau2_update(G,
                           CAR,
                           alpha.col(j),
                           alpha_tau2,
                           beta_tau2);
     
     }
     
   w0.col(j).fill(0.00);
   if(model_type != 3){

     //w0 Update
     arma::uvec keep3(3); keep3(0) = 0; keep3(1) = 1; keep3(2) = 4;
     mean_temp = construct_mean_s(beta0(j), 
                                  beta1(j),
                                  A11(j),
                                  A22(j),
                                  A21(j),
                                  w0.col(j-1),
                                  w1.col(j-1),
                                  diagmat(lc1),
                                  keep3,
                                  sample_size);
   
     w0.col(j) = w0_update_s(y,
                             mean_temp,
                             lagged_covars,
                             sigma2_epsilon(j),
                             A11(j),
                             A21(j),
                             spatial_corr0_info[0]);
    
     //phi0 Update
     Rcpp::List phi0_output = phi_update(phi0(j-1),
                                         spatial_dists,
                                         w0.col(j),
                                         spatial_corr0_info,
                                         alpha_phi0,
                                         beta_phi0,
                                         metrop_var_phi0_trans,
                                         acctot_phi0_trans);
  
     phi0(j) = Rcpp::as<double>(phi0_output[0]);
     acctot_phi0_trans = phi0_output[1];
     spatial_corr0_info = phi0_output[2];
     }
   
   w1.col(j).fill(0.00);
   phi1(j) = 0.00;
   if(model_type == 0 || model_type == 1){
     
     //w1 Update
     keep4(0) = 0; keep4(1) = 1; keep4(2) = 2; keep4(3) = 3;
     mean_temp = construct_mean_s(beta0(j), 
                                  beta1(j),
                                  A11(j),
                                  A22(j),
                                  A21(j),
                                  w0.col(j),
                                  w1.col(j-1),
                                  diagmat(lc1),
                                  keep4,
                                  sample_size);
   
     w1.col(j) = w1_update_s(y,
                             mean_temp,
                             lagged_covars,
                             sigma2_epsilon(j),
                             A22(j),
                             spatial_corr1_info[0]);
     
     //phi1 Update
     Rcpp::List phi1_output = phi_update(phi1(j-1),
                                         spatial_dists,
                                         w1.col(j),
                                         spatial_corr1_info,
                                         alpha_phi1,
                                         beta_phi1,
                                         metrop_var_phi1_trans,
                                         acctot_phi1_trans);
     
     phi1(j) = Rcpp::as<double>(phi1_output[0]);
     acctot_phi1_trans = phi1_output[1];
     spatial_corr1_info = phi1_output[2];
     }
   
   //neg_two_loglike Update
   mean_temp = construct_mean_s(beta0(j), 
                                beta1(j),
                                A11(j),
                                A22(j),
                                A21(j),
                                w0.col(j),
                                w1.col(j),
                                diagmat(lc1),
                                keep5,
                                sample_size);
   
   neg_two_loglike(j) = neg_two_loglike_update_s(y,
                                                 mean_temp,
                                                 sigma2_epsilon(j));
   
   //Progress
   if((j + 1) % 10 == 0){ 
     Rcpp::checkUserInterrupt();
     }
  
   if(((j + 1) % int(round(mcmc_samples*0.05)) == 0)){
     
     double completion = round(100*((j + 1)/(double)mcmc_samples));
     Rcpp::Rcout << "Progress: " << completion << "%" << std::endl;
     
     if(model_type != 3){
       double accrate_A11_trans = round(100*(acctot_A11_trans/(double)j));
       Rcpp::Rcout << "A11 Acceptance: " << accrate_A11_trans << "%" << std::endl;
       }
     
     if(model_type == 0 || model_type == 1){
       double accrate_A22_trans = round(100*(acctot_A22_trans/(double)j));
       Rcpp::Rcout << "A22 Acceptance: " << accrate_A22_trans << "%" << std::endl;
       }
     
     if(model_type == 0){
     
       double accrate_mu = round(100*(acctot_mu/(double)j));
       Rcpp::Rcout << "mu Acceptance: " << accrate_mu << "%" << std::endl;
     
       double accrate_alpha_min = round(100*(min(acctot_alpha)/(double)j));
       Rcpp::Rcout << "alpha Acceptance (min): " << accrate_alpha_min << "%" << std::endl;
     
       double accrate_alpha_max = round(100*(max(acctot_alpha)/(double)j));
       Rcpp::Rcout << "alpha Acceptance (max): " << accrate_alpha_max << "%" << std::endl;
     
       }
     
     if(model_type != 3){
       double accrate_phi0_trans = round(100*(acctot_phi0_trans/(double)j));
       Rcpp::Rcout << "phi0 Acceptance: " << accrate_phi0_trans << "%" << std::endl;
       }
     
     if(model_type == 0 || model_type == 1){
       double accrate_phi1_trans = round(100*(acctot_phi1_trans/(double)j));
       Rcpp::Rcout << "phi1 Acceptance: " << accrate_phi1_trans << "%" << std::endl;
       }
     
     if(model_type == 0){
       Rcpp::Rcout << "DLfuse: S" << std::endl;
       Rcpp::Rcout << "***************************" << std::endl;
       }
     
     if(model_type == 1){
       Rcpp::Rcout << "Original: S" << std::endl;
       Rcpp::Rcout << "********************" << std::endl;
       }
     
     if(model_type == 2){
       Rcpp::Rcout << "Ordinary Kriging: S" << std::endl;
       Rcpp::Rcout << "********************" << std::endl;
       }
     
     if(model_type == 3){
       Rcpp::Rcout << "Simple Linear Regression: S" << std::endl;
       Rcpp::Rcout << "***************************" << std::endl;
       }
     
     }
  
   }

Rcpp::List metrop_info = Rcpp::List::create(Rcpp::Named("acctot_A11_trans") = acctot_A11_trans,
                                            Rcpp::Named("acctot_A22_trans") = acctot_A22_trans,
                                            Rcpp::Named("acctot_mu") = acctot_mu,
                                            Rcpp::Named("acctot_alpha") = acctot_alpha,
                                            Rcpp::Named("acctot_phi0_trans") = acctot_phi0_trans,
                                            Rcpp::Named("acctot_phi1_trans") = acctot_phi1_trans);

Rcpp::List lag_info = Rcpp::List::create(Rcpp::Named("mu") = mu,
                                         Rcpp::Named("alpha") = alpha,
                                         Rcpp::Named("tau2") = tau2);
                                  
return Rcpp::List::create(Rcpp::Named("sigma2_epsilon") = sigma2_epsilon,
                          Rcpp::Named("beta0") = beta0,
                          Rcpp::Named("beta1") = beta1,
                          Rcpp::Named("A11") = A11,
                          Rcpp::Named("A22") = A22,
                          Rcpp::Named("A21") = A21,
                          Rcpp::Named("w0") = w0,
                          Rcpp::Named("phi0") = phi0,
                          Rcpp::Named("w1") = w1,
                          Rcpp::Named("phi1") = phi1,
                          Rcpp::Named("lag_info") = lag_info,
                          Rcpp::Named("neg_two_loglike") = neg_two_loglike,
                          Rcpp::Named("metrop_info") = metrop_info);

}

