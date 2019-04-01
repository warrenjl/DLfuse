#include "RcppArmadillo.h"
#include "DLfuse.h"
using namespace arma;
using namespace Rcpp;

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]

Rcpp::List DLfuse_st(int mcmc_samples,
                     Rcpp::List y,
                     Rcpp::List z,
                     Rcpp::List sample_size,
                     Rcpp::List AQS_key,
                     Rcpp::List CMAQ_key,
                     arma::mat spatial_dists,
                     int AQS_unique_total,
                     arma::mat neighbors,
                     int CMAQ_unique_total,
                     double metrop_var_rho1_trans,  //Start of Metropolis Parameters
                     double metrop_var_rho2_trans,
                     double metrop_var_A11_trans,
                     double metrop_var_A22_trans,
                     double metrop_var_mu,
                     arma::vec metrop_var_mut,
                     double metrop_var_rho3_trans,
                     arma::vec metrop_var_alpha,
                     double metrop_var_phi0_trans,
                     double metrop_var_phi1_trans,
                     Rcpp::Nullable<double> alpha_sigma2_epsilon_prior = R_NilValue,  //Start of Priors
                     Rcpp::Nullable<double> beta_sigma2_epsilon_prior = R_NilValue,
                     Rcpp::Nullable<double> sigma2_beta_prior = R_NilValue,
                     Rcpp::Nullable<Rcpp::NumericMatrix> Omega_V_inv_prior = R_NilValue,
                     Rcpp::Nullable<double> nu_V_inv_prior = R_NilValue,
                     Rcpp::Nullable<double> a_rho1_prior = R_NilValue,
                     Rcpp::Nullable<double> b_rho1_prior = R_NilValue,
                     Rcpp::Nullable<double> a_rho2_prior = R_NilValue,
                     Rcpp::Nullable<double> b_rho2_prior = R_NilValue,
                     Rcpp::Nullable<double> sigma2_A_prior = R_NilValue,
                     Rcpp::Nullable<double> alpha_sigma2_delta_prior = R_NilValue,
                     Rcpp::Nullable<double> beta_sigma2_delta_prior = R_NilValue,
                     Rcpp::Nullable<double> a_rho3_prior = R_NilValue,
                     Rcpp::Nullable<double> b_rho3_prior = R_NilValue,
                     Rcpp::Nullable<double> alpha_tau2_prior = R_NilValue,
                     Rcpp::Nullable<double> beta_tau2_prior = R_NilValue,
                     Rcpp::Nullable<double> alpha_phi0_prior = R_NilValue,
                     Rcpp::Nullable<double> beta_phi0_prior = R_NilValue,
                     Rcpp::Nullable<double> alpha_phi1_prior = R_NilValue,
                     Rcpp::Nullable<double> beta_phi1_prior = R_NilValue,
                     Rcpp::Nullable<double> sigma2_epsilon_init = R_NilValue,  //Start of Initial Values
                     Rcpp::Nullable<double> beta0_init = R_NilValue,
                     Rcpp::Nullable<double> beta1_init = R_NilValue,
                     Rcpp::Nullable<Rcpp::NumericMatrix> betat_init = R_NilValue,
                     Rcpp::Nullable<Rcpp::NumericMatrix> V_init = R_NilValue,
                     Rcpp::Nullable<double> rho1_init = R_NilValue,
                     Rcpp::Nullable<double> rho2_init = R_NilValue,
                     Rcpp::Nullable<double> A11_init = R_NilValue,
                     Rcpp::Nullable<double> A22_init = R_NilValue,
                     Rcpp::Nullable<double> A21_init = R_NilValue,
                     Rcpp::Nullable<double> mu_init = R_NilValue,
                     Rcpp::Nullable<Rcpp::NumericVector> mut_init = R_NilValue,
                     Rcpp::Nullable<double> sigma2_delta_init = R_NilValue,
                     Rcpp::Nullable<double> rho3_init = R_NilValue,
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

//Miscellaneous Information
arma::mat Dw(CMAQ_unique_total, CMAQ_unique_total); Dw.fill(0.00);
for(int j = 0; j < CMAQ_unique_total; ++ j){
   Dw(j,j) = sum(neighbors.row(j));
   } 
arma::mat CAR = Dw - 
                neighbors;
int G = 1.00;  //G Always Equal to One in this Case (One Island because of Inverse Distance Weighting)

//Defining Parameters and Quantities of Interest
int d = y.size();
double max_dist = spatial_dists.max();

Rcpp::List AQS_key_mat(d); AQS_key_mat.fill(0.00);
for(int j = 0; j < d; ++ j){
   
   IntegerVector AQS_key_t = AQS_key[j];
   int n_t = AQS_key_t.size();
   arma::mat mat_temp(n_t, AQS_unique_total); mat_temp.fill(0.00);
   
   for(int k = 0; k < n_t; ++ k){
      int ind = (AQS_key_t(k) - 1);
      mat_temp(k,ind) = 1.00;
      }
   
   AQS_key_mat[j] = mat_temp;
   
   }

arma::vec sigma2_epsilon(mcmc_samples); sigma2_epsilon.fill(0.00);
arma::vec beta0(mcmc_samples); beta0.fill(0.00);
arma::vec beta1(mcmc_samples); beta1.fill(0.00);
Rcpp::List betat(mcmc_samples); betat.fill(0.00);
Rcpp::List V(mcmc_samples); V.fill(0.00);
arma::vec rho1(mcmc_samples); rho1.fill(0.00);
arma::vec rho2(mcmc_samples); rho2.fill(0.00);
arma::vec A11(mcmc_samples); A11.fill(0.00);
arma::vec A22(mcmc_samples); A22.fill(0.00);
arma::vec A21(mcmc_samples); A21.fill(0.00);
arma::vec mu(mcmc_samples); mu.fill(0.00);
arma::mat mut(d, mcmc_samples); mut.fill(0.00);
arma::vec sigma2_delta(mcmc_samples); sigma2_delta.fill(0.00);
arma::vec rho3(mcmc_samples); rho3.fill(0.00);
arma::mat alpha(CMAQ_unique_total, mcmc_samples); alpha.fill(0.00);
arma::vec tau2(mcmc_samples); tau2.fill(0.00);
arma::mat w0(AQS_unique_total, mcmc_samples); w0.fill(0.00);
arma::vec phi0(mcmc_samples); phi0.fill(0.00);
arma::mat w1(AQS_unique_total, mcmc_samples); w1.fill(0.00);
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

arma::mat Omega_V_inv(2,2); Omega_V_inv.eye();
if(Omega_V_inv_prior.isNotNull()){
  Omega_V_inv = Rcpp::as<arma::mat>(Omega_V_inv_prior);
  }

double nu_V_inv = 3.00;
if(nu_V_inv_prior.isNotNull()){
  nu_V_inv = Rcpp::as<double>(nu_V_inv_prior);
  }

double a_rho1 = 0.00;
if(a_rho1_prior.isNotNull()){
  a_rho1 = Rcpp::as<double>(a_rho1_prior);
  }

double b_rho1 = 1.00;
if(b_rho1_prior.isNotNull()){
  b_rho1 = Rcpp::as<double>(b_rho1_prior);
  }

double a_rho2 = 0.00;
if(a_rho2_prior.isNotNull()){
  a_rho2 = Rcpp::as<double>(a_rho2_prior);
  }

double b_rho2 = 1.00;
if(b_rho2_prior.isNotNull()){
  b_rho2 = Rcpp::as<double>(b_rho2_prior);
  }

double sigma2_A = 1.00;
if(sigma2_A_prior.isNotNull()){
  sigma2_A = Rcpp::as<double>(sigma2_A_prior);
  }

double alpha_sigma2_delta = 3.00;
if(alpha_sigma2_delta_prior.isNotNull()){
  alpha_sigma2_delta = Rcpp::as<double>(alpha_sigma2_delta_prior);
  }

double beta_sigma2_delta = 2.00;
if(beta_sigma2_delta_prior.isNotNull()){
  beta_sigma2_delta = Rcpp::as<double>(beta_sigma2_delta_prior);
  }

double a_rho3 = 0.00;
if(a_rho3_prior.isNotNull()){
  a_rho3 = Rcpp::as<double>(a_rho3_prior);
  } 

double b_rho3 = 1.00;
if(b_rho3_prior.isNotNull()){
  b_rho3 = Rcpp::as<double>(b_rho3_prior);
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

arma::mat betat_temp(2, d); betat_temp.fill(0.00); 
betat[0] = betat_temp;
if(betat_init.isNotNull()){
  betat[0] = Rcpp::as<arma::mat>(betat_init);
  }

arma::mat V_temp(2,2); V_temp.eye();
V[0] = V_temp;
if(V_init.isNotNull()){
  V[0] = Rcpp::as<arma::mat>(V_init);
  }

rho1(0) = 0.50;
if(rho1_init.isNotNull()){
  rho1(0) = Rcpp::as<double>(rho1_init);
  }

rho2(0) = 0.50;
if(rho2_init.isNotNull()){
  rho2(0) = Rcpp::as<double>(rho2_init);
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

mut.col(0).fill(0.00);
if(mut_init.isNotNull()){
  mut.col(0) = Rcpp::as<arma::vec>(mut_init);
  }

sigma2_delta(0) = 1.00;
if(sigma2_delta_init.isNotNull()){
  sigma2_delta(0) = Rcpp::as<double>(sigma2_delta_init);
  }

rho3(0) = 0.50;
if(rho3_init.isNotNull()){
  rho3(0) = Rcpp::as<double>(rho3_init);
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

Rcpp::List lagged_covars(d); lagged_covars.fill(0.00);
Rcpp::List lc1(d); lc1.fill(0.00);
Rcpp::List lc2(d); lc2.fill(0.00);
for(int j = 0; j < d; ++ j){
  
   Rcpp::List lagged_covars_t = construct_lagged_covars_st(z[j],
                                                           mu(0),
                                                           mut(j,0),
                                                           alpha.col(0),
                                                           sample_size[j],
                                                           CMAQ_key[j]);
   lagged_covars[j] = lagged_covars_t;
   
   arma::vec lc1_t = lagged_covars_t[0];
   arma::vec lc2_t = lagged_covars_t[1];
   lc1[j] = lc1_t;
   lc2[j] = lc2_t;
  
   }

if(model_type == 1 || model_type == 3){
  
  double negative_infinity = -std::numeric_limits<double>::infinity();
  for(int j = 0; j < d; ++ j){

     Rcpp::List lagged_covars_t = construct_lagged_covars_st(z[j],
                                                             negative_infinity,
                                                             negative_infinity,
                                                             alpha.col(0),
                                                             sample_size[j],
                                                             CMAQ_key[j]);
     lagged_covars[j] = lagged_covars_t;
    
     arma::vec lc1_t = lagged_covars_t[0];
     arma::vec lc2_t = lagged_covars_t[1];
     lc1[j] = lc1_t;
     lc2[j] = lc2_t;
    
     }
  
  }
  
arma::uvec keep7(7); keep7(0) = 0; keep7(1) = 1; keep7(2) = 2; keep7(3) = 3; keep7(4) = 4; keep7(5) = 5; keep7(6) = 6;
Rcpp::List mean_temp(d); mean_temp.fill(0.00);
betat_temp = Rcpp::as<arma::mat>(betat[0]);
for(int j = 0; j < d; ++ j){  
   
   arma::vec betat_t = betat_temp.col(j);
   arma::vec lc1_t = lc1[j];
   arma::vec mean_temp_t = construct_mean_st(beta0(0), 
                                             beta1(0),
                                             betat_t,
                                             A11(0),
                                             A22(0),
                                             A21(0),
                                             w0.col(0),
                                             w1.col(0),
                                             diagmat(lc1_t),
                                             keep7,
                                             sample_size[j],
                                             AQS_key_mat[j]);
   mean_temp[j] = mean_temp_t;
  
   }

neg_two_loglike(0) = neg_two_loglike_update_st(y,
                                               mean_temp,
                                               sigma2_epsilon(0));

//Metropolis Settings
int acctot_rho1_trans = 0;
int acctot_rho2_trans = 0;
int acctot_A11_trans = 0;
int acctot_A22_trans = 0;
int acctot_mu = 0;
arma::vec acctot_mut(d); acctot_mut.fill(0);
int acctot_rho3_trans = 0;
arma::vec acctot_alpha(CMAQ_unique_total); acctot_alpha.fill(0);
int acctot_phi0_trans = 0;
int acctot_phi1_trans = 0;

//Main Sampling Loop
for(int j = 1; j < mcmc_samples; ++ j){
  
   //sigma2_epsilon Update
   arma::mat betat_temp = betat[j-1];
   for(int k = 0; k < d; ++ k){
      
      arma::vec betat_t = betat_temp.col(k);
      arma::vec lc1_t = lc1[k];
      arma::vec mean_temp_t = construct_mean_st(beta0(j-1), 
                                                beta1(j-1),
                                                betat_t,
                                                A11(j-1),
                                                A22(j-1),
                                                A21(j-1),
                                                w0.col(j-1),
                                                w1.col(j-1),
                                                diagmat(lc1_t),
                                                keep7,
                                                sample_size[k],
                                                AQS_key_mat[k]);
      mean_temp[k] = mean_temp_t;
      
      }
  
   sigma2_epsilon(j) = sigma2_epsilon_update_st(y,
                                                mean_temp,
                                                alpha_sigma2_epsilon,
                                                beta_sigma2_epsilon);
  
   //beta0 Update
   arma::uvec keep6(6); keep6(0) = 1; keep6(1) = 2; keep6(2) = 3; keep6(3) = 4; keep6(4) = 5; keep6(5) = 6;
   betat_temp = Rcpp::as<arma::mat>(betat[j-1]);
   for(int k = 0; k < d; ++ k){
     
      arma::vec betat_t = betat_temp.col(k);
      arma::vec lc1_t = lc1[k];
      arma::vec mean_temp_t = construct_mean_st(beta0(j-1), 
                                                beta1(j-1),
                                                betat_t,
                                                A11(j-1),
                                                A22(j-1),
                                                A21(j-1),
                                                w0.col(j-1),
                                                w1.col(j-1),
                                                diagmat(lc1_t),
                                                keep6,
                                                sample_size[k],
                                                AQS_key_mat[k]);
      mean_temp[k] = mean_temp_t;
      
      }
   
   beta0(j) = beta0_update_st(y,
                              mean_temp, 
                              sigma2_epsilon(j),
                              sigma2_beta);
   
   beta1(j) = 0.00;
   if(model_type != 2){

     //beta1 Update
     keep6(0) = 0; keep6(1) = 1; keep6(2) = 3; keep6(3) = 4; keep6(4) = 5; keep6(5) = 6;
     arma::mat betat_temp = betat[j-1];
     for(int k = 0; k < d; ++ k){
       
        arma::vec betat_t = betat_temp.col(k);
        arma::vec lc1_t = lc1[k];
        arma::vec mean_temp_t = construct_mean_st(beta0(j), 
                                                  beta1(j-1),
                                                  betat_t,
                                                  A11(j-1),
                                                  A22(j-1),
                                                  A21(j-1),
                                                  w0.col(j-1),
                                                  w1.col(j-1),
                                                  diagmat(lc1_t),
                                                  keep6,
                                                  sample_size[k],
                                                  AQS_key_mat[k]);
        mean_temp[k] = mean_temp_t;
        
        }
   
     beta1(j) = beta1_update_st(y,
                                mean_temp,
                                lagged_covars,
                                sigma2_epsilon(j),
                                sample_size,
                                sigma2_beta);
     
     }
   
   //betat Update
   arma::uvec keep5(5); keep5(0) = 0; keep5(1) = 2; keep5(2) = 4; keep5(3) = 5; keep5(4) = 6;
   betat_temp = Rcpp::as<arma::mat>(betat[j-1]);
   arma::vec betat_t = betat_temp.col(0);
   arma::vec lc1_t = lc1[0];
   mean_temp[0] = construct_mean_st(beta0(j), 
                                    beta1(j),
                                    betat_t,
                                    A11(j-1),
                                    A22(j-1),
                                    A21(j-1),
                                    w0.col(j-1),
                                    w1.col(j-1),
                                    diagmat(lc1_t),
                                    keep5,
                                    sample_size[0],
                                    AQS_key_mat[0]);
   
   arma::vec betat_previous(2); betat_previous.fill(0.00);
   arma::vec betat_next = betat_temp.col(1);
   betat_t = betat_update_st(0,
                             y[0],
                             mean_temp[0],
                             lagged_covars[0],
                             sigma2_epsilon(j),
                             betat_previous,
                             betat_next,
                             V[j-1],
                             rho1(j-1),
                             rho2(j-1));
   betat_temp.col(0) = betat_t;
   
   for(int k = 1; k < d; ++ k){
      
      arma::vec betat_t = betat_temp.col(k);
      arma::vec lc1_t = lc1[k];
      mean_temp[k] = construct_mean_st(beta0(j), 
                                       beta1(j),
                                       betat_t,
                                       A11(j-1),
                                       A22(j-1),
                                       A21(j-1),
                                       w0.col(j-1),
                                       w1.col(j-1),
                                       diagmat(lc1_t),
                                       keep5,
                                       sample_size[k],
                                       AQS_key_mat[k]);
      
      arma::vec betat_previous = betat_temp.col(k-1);
      
      int last_time_ind = 1;
      arma::vec betat_next(2); betat_next.fill(0.00);
      if(k < (d - 1)){
        last_time_ind = 0;
        arma::vec betat_next = betat_temp.col(k+1);
        }
      
       betat_t = betat_update_st(last_time_ind,
                                 y[k],
                                 mean_temp[k],
                                 lagged_covars[k],
                                 sigma2_epsilon(j),
                                 betat_previous,
                                 betat_next,
                                 V[j-1],
                                 rho1(j-1),
                                 rho2(j-1));
      betat_temp.col(k) = betat_t;
      
      }
   
   //Centering for Stability
   betat_temp.row(0) = betat_temp.row(0) -
                       mean(betat_temp.row(0));
   betat_temp.row(1) = betat_temp.row(1) -
                       mean(betat_temp.row(1));
   
   if(model_type == 2){
     betat_temp.row(1).fill(0.00);
     }
   
   betat[j] = betat_temp;
   
   //V Update
   V[j] = V_update_st(betat[j],
                      rho1[j-1],
                      rho2[j-1],
                      Omega_V_inv,
                      nu_V_inv);
   
   //rho1 Update
   Rcpp::List rho1_output = rho1_update_st(rho1(j-1),
                                           betat[j],
                                           V[j],
                                           rho2[j-1],
                                           a_rho1,
                                           b_rho1,
                                           metrop_var_rho1_trans,
                                           acctot_rho1_trans);
   
   rho1(j) = Rcpp::as<double>(rho1_output[0]);
   acctot_rho1_trans = rho1_output[1];
   
   rho2(j) = 0.00;
   if(model_type != 2){
     
     //rho2 Update
     Rcpp::List rho2_output = rho2_update_st(rho2(j-1),
                                             betat[j],
                                             V[j],
                                             rho1[j],
                                             a_rho2,
                                             b_rho2,
                                             metrop_var_rho2_trans,
                                             acctot_rho2_trans);
   
     rho2(j) = Rcpp::as<double>(rho2_output[0]);
     acctot_rho2_trans = rho2_output[1];
     
     }
   
   A11(j) = 0.00;  
   if(model_type != 3){
  
     //A11 Update
     Rcpp::List A11_output = A11_update_st(y,
                                           AQS_key_mat,
                                           A11(j-1),
                                           lagged_covars,
                                           sigma2_epsilon(j),
                                           beta0(j),
                                           beta1(j),
                                           betat[j],
                                           A22(j-1),
                                           A21(j-1),
                                           w0.col(j-1),
                                           w1.col(j-1),
                                           keep7,
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
     Rcpp::List A22_output = A22_update_st(y,
                                           AQS_key_mat,
                                           A22(j-1),
                                           lagged_covars,
                                           sigma2_epsilon(j),
                                           beta0(j),
                                           beta1(j),
                                           betat[j],
                                           A11(j),
                                           A21(j-1),
                                           w0.col(j-1),
                                           w1.col(j-1),
                                           keep7,
                                           sample_size,
                                           sigma2_A,
                                           metrop_var_A22_trans,
                                           acctot_A22_trans);
   
     A22(j) = Rcpp::as<double>(A22_output[0]);
     acctot_A22_trans = A22_output[1];
   
     //A21 Update
     keep6(0) = 0; keep6(1) = 1; keep6(2) = 2; keep6(3) = 3; keep6(4) = 4; keep6(5) = 6;
     arma::mat betat_temp = betat[j];
     for(int k = 0; k < d; ++ k){
       
        arma::vec betat_t = betat_temp.col(k);
        arma::vec lc1_t = lc1[k];
        arma::vec mean_temp_t = construct_mean_st(beta0(j), 
                                                  beta1(j),
                                                  betat_t,
                                                  A11(j),
                                                  A22(j),
                                                  A21(j-1),
                                                  w0.col(j-1),
                                                  w1.col(j-1),
                                                  diagmat(lc1_t),
                                                  keep6,
                                                  sample_size[k],
                                                  AQS_key_mat[k]);
        mean_temp[k] = mean_temp_t;
        
        }
     
     A21(j) = A21_update_st(y,
                            AQS_key_mat,
                            mean_temp,
                            lagged_covars,
                            sigma2_epsilon(j),
                            w0.col(j-1),
                            sigma2_A);
     
     }
    
   mu(j) = 0.00;
   mut.col(j).fill(0.00);
   sigma2_delta(j) = 0.00;
   rho3(j) = 0.00;
   alpha.col(j).fill(0.00);
   if(model_type == 0){
     
     //mu Update
     Rcpp::List mu_output = mu_update_st(y,
                                         z,
                                         mu(j-1),
                                         lagged_covars,
                                         sigma2_epsilon(j),
                                         beta0(j),
                                         beta1(j),
                                         betat[j],
                                         A11(j),
                                         A22(j),
                                         A21(j),
                                         mut.col(j-1),
                                         alpha.col(j-1),
                                         w0.col(j-1),
                                         w1.col(j-1),
                                         keep7,
                                         sample_size,
                                         AQS_key_mat,
                                         CMAQ_key,
                                         metrop_var_mu,
                                         acctot_mu);
     
     mu(j) = Rcpp::as<double>(mu_output[0]);
     acctot_mu = mu_output[1];
     lagged_covars = mu_output[2];
     for(int k = 0; k < d; ++ k){
       
        Rcpp::List lagged_covars_t = lagged_covars[k];
        lc1[k] = Rcpp::as<arma::vec>(lagged_covars_t[0]);
        lc2[k] = Rcpp::as<arma::vec>(lagged_covars_t[1]);
       
        }
     
     //mut Update
     arma::mat betat_temp = betat[j];
     arma::vec betat_t = betat_temp.col(0);
     Rcpp::List mut_output = mut_update_st(0,
                                           y[0],
                                           z[0],
                                           mut(0, (j-1)),
                                           lagged_covars[0],
                                           sigma2_epsilon(j),
                                           beta0(j),
                                           beta1(j),
                                           betat_t,
                                           A11(j),
                                           A22(j),
                                           A21(j),
                                           mu(j),
                                           0.00,
                                           mut(1, (j-1)),
                                           sigma2_delta(j-1),
                                           rho3(j-1),
                                           alpha.col(j-1),
                                           w0.col(j-1),
                                           w1.col(j-1),
                                           keep7,
                                           sample_size[0],
                                           AQS_key_mat[0],
                                           CMAQ_key[0],
                                           metrop_var_mut(0),
                                           acctot_mut(0));
      
     mut(0,j) = Rcpp::as<double>(mut_output[0]);
     acctot_mut(0) = Rcpp::as<int>(mut_output[1]);
     lagged_covars[0] = mut_output[2];
     Rcpp::List lagged_covars_t = lagged_covars[0];
     lc1[0] = Rcpp::as<arma::vec>(lagged_covars_t[0]);
     lc2[0] = Rcpp::as<arma::vec>(lagged_covars_t[1]);
     
     for(int k = 1; k < d; ++ k){
       
        int last_time_ind = 1;
        double mut_next = 0.00;
        if(k < (d - 1)){
          last_time_ind = 0;
          mut_next = mut((k+1), (j-1));
          }
       
        arma::vec betat_t = betat_temp.col(k);
        Rcpp::List mut_output = mut_update_st(last_time_ind,
                                              y[k],
                                              z[k],
                                              mut(k, (j-1)),
                                              lagged_covars[k],
                                              sigma2_epsilon(j),
                                              beta0(j),
                                              beta1(j),
                                              betat_t,
                                              A11(j),
                                              A22(j),
                                              A21(j),
                                              mu(j),
                                              mut((k-1), j),
                                              mut_next,
                                              sigma2_delta(j-1),
                                              rho3(j-1),
                                              alpha.col(j-1),
                                              w0.col(j-1),
                                              w1.col(j-1),
                                              keep7,
                                              sample_size[k],
                                              AQS_key_mat[k],
                                              CMAQ_key[k],
                                              metrop_var_mut(k),
                                              acctot_mut(k));
       
        mut(k,j) = Rcpp::as<double>(mut_output[0]);
        acctot_mut(k) = Rcpp::as<int>(mut_output[1]);
        lagged_covars[k] = mut_output[2];
        Rcpp::List lagged_covars_t = lagged_covars[k];
        lc1[k] = Rcpp::as<arma::vec>(lagged_covars_t[0]);
        lc2[k] = Rcpp::as<arma::vec>(lagged_covars_t[1]);
        
        }
     
     //Centering for Stability + \Phi(.) stabilization
     mut.col(j) = (mut.col(j) - mean(mut.col(j)))/stddev(mut.col(j));
     for(int k = 0; k < d; ++ k){
       
        Rcpp::List lagged_covars_t = construct_lagged_covars_st(z[k],
                                                                mu(j),
                                                                mut(k,j),
                                                                alpha.col(j-1),
                                                                sample_size[k],
                                                                CMAQ_key[k]);
        lagged_covars[k] = lagged_covars_t;
       
        arma::vec lc1_t = lagged_covars_t[0];
        arma::vec lc2_t = lagged_covars_t[1];
        lc1[k] = lc1_t;
        lc2[k] = lc2_t;
       
        }
     
     //sigma2_delta Update
     sigma2_delta(j) = sigma2_delta_update_st(mut.col(j),
                                              rho3(j-1),
                                              alpha_sigma2_delta,
                                              beta_sigma2_delta);
     
     //rho3 Update
     Rcpp::List rho3_output = rho3_update_st(rho3(j-1),
                                             mut.col(j),
                                             sigma2_delta(j),
                                             a_rho3,
                                             b_rho3,
                                             metrop_var_rho3_trans,
                                             acctot_rho3_trans);
     
     rho3(j) = Rcpp::as<double>(rho3_output[0]);
     acctot_rho3_trans = rho3_output[1];
    
     //alpha Update
     Rcpp::List alpha_output = alpha_update_st(y,
                                               z,
                                               neighbors,
                                               alpha.col(j-1),
                                               lagged_covars,
                                               sigma2_epsilon(j),
                                               beta0(j),
                                               beta1(j),
                                               betat[j],
                                               A11(j),
                                               A22(j),
                                               A21(j),
                                               mu(j),
                                               mut.col(j),
                                               tau2(j-1),
                                               w0.col(j-1),
                                               w1.col(j-1),
                                               keep7,
                                               sample_size,
                                               AQS_key_mat,
                                               CMAQ_key,
                                               metrop_var_alpha,
                                               acctot_alpha);   
   
     alpha.col(j) = as<arma::vec>(alpha_output[0]);
     acctot_alpha = as<arma::vec>(alpha_output[1]);
     lagged_covars = alpha_output[2];
     for(int k = 0; k < d; ++ k){
       
        Rcpp::List lagged_covars_t = lagged_covars[k];
        lc1[k] = Rcpp::as<arma::vec>(lagged_covars_t[0]);
        lc2[k] = Rcpp::as<arma::vec>(lagged_covars_t[1]);
       
        }
     
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
     arma::uvec keep5(5); keep5(0) = 0; keep5(1) = 1; keep5(2) = 2; keep5(3) = 3; keep5(4) = 6;
     betat_temp = Rcpp::as<arma::mat>(betat[j]);
     for(int k = 0; k < d; ++ k){
       
        arma::vec betat_t = betat_temp.col(k);
        arma::vec lc1_t = lc1[k];
        arma::vec mean_temp_t = construct_mean_st(beta0(j), 
                                                  beta1(j),
                                                  betat_t,
                                                  A11(j),
                                                  A22(j),
                                                  A21(j),
                                                  w0.col(j-1),
                                                  w1.col(j-1),
                                                  diagmat(lc1_t),
                                                  keep5,
                                                  sample_size[k],
                                                  AQS_key_mat[k]);
        mean_temp[k] = mean_temp_t;
       
        }
   
     w0.col(j) = w0_update_st(y,
                              AQS_key_mat,
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
     keep6(0) = 0; keep6(1) = 1; keep6(2) = 2; keep6(3) = 3;; keep6(4) = 4; keep6(5) = 5;
     betat_temp = Rcpp::as<arma::mat>(betat[j]);
     for(int k = 0; k < d; ++ k){
       
        arma::vec betat_t = betat_temp.col(k);
        arma::vec lc1_t = lc1[k];
        arma::vec mean_temp_t = construct_mean_st(beta0(j), 
                                                  beta1(j),
                                                  betat_t,
                                                  A11(j),
                                                  A22(j),
                                                  A21(j),
                                                  w0.col(j),
                                                  w1.col(j-1),
                                                  diagmat(lc1_t),
                                                  keep6,
                                                  sample_size[k],
                                                  AQS_key_mat[k]);
        mean_temp[k] = mean_temp_t;
       
        }
   
     w1.col(j) = w1_update_st(y,
                              AQS_key_mat,
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
   betat_temp = Rcpp::as<arma::mat>(betat[j]);
   for(int k = 0; k < d; ++ k){
     
     arma::vec betat_t = betat_temp.col(k);
     arma::vec lc1_t = lc1[k];
     arma::vec mean_temp_t = construct_mean_st(beta0(j), 
                                               beta1(j),
                                               betat_t,
                                               A11(j),
                                               A22(j),
                                               A21(j),
                                               w0.col(j),
                                               w1.col(j),
                                               diagmat(lc1_t),
                                               keep7,
                                               sample_size[k],
                                               AQS_key_mat[k]);
     mean_temp[k] = mean_temp_t;
     
     }
   
   neg_two_loglike(j) = neg_two_loglike_update_st(y,
                                                  mean_temp,
                                                  sigma2_epsilon(j));
   
   //Progress
   if((j + 1) % 10 == 0){ 
     Rcpp::checkUserInterrupt();
     }
  
   if(((j + 1) % int(round(mcmc_samples*0.05)) == 0)){
     
     double completion = round(100*((j + 1)/(double)mcmc_samples));
     Rcpp::Rcout << "Progress: " << completion << "%" << std::endl;
     
     double accrate_rho1_trans = round(100*(acctot_rho1_trans/(double)j));
     Rcpp::Rcout << "rho1 Acceptance: " << accrate_rho1_trans << "%" << std::endl;
       
     if(model_type != 2){
       double accrate_rho2_trans = round(100*(acctot_rho2_trans/(double)j));
       Rcpp::Rcout << "rho2 Acceptance: " << accrate_rho2_trans << "%" << std::endl;
       }
     
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
     
       double accrate_mut_min = round(100*(min(acctot_mut)/(double)j));
       Rcpp::Rcout << "mut Acceptance (min): " << accrate_mut_min << "%" << std::endl;
       
       double accrate_mut_max = round(100*(max(acctot_mut)/(double)j));
       Rcpp::Rcout << "mut Acceptance (max): " << accrate_mut_max << "%" << std::endl;
       
       double accrate_rho3_trans = round(100*(acctot_rho3_trans/(double)j));
       Rcpp::Rcout << "rho3 Acceptance: " << accrate_rho3_trans << "%" << std::endl;
     
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
       Rcpp::Rcout << "DLfuse: ST" << std::endl;
       Rcpp::Rcout << "***************************" << std::endl;
       }
     
     if(model_type == 1){
       Rcpp::Rcout << "Original: ST" << std::endl;
       Rcpp::Rcout << "********************" << std::endl;
       }
     
     if(model_type == 2){
       Rcpp::Rcout << "Ordinary Kriging: ST" << std::endl;
       Rcpp::Rcout << "********************" << std::endl;
       }
     
     if(model_type == 3){
       Rcpp::Rcout << "Simple Linear Regression: ST" << std::endl;
       Rcpp::Rcout << "****************************" << std::endl;
       }
     
     }
  
   }

Rcpp::List metrop_info = Rcpp::List::create(Rcpp::Named("acctot_rho1_trans") = acctot_rho1_trans,
                                            Rcpp::Named("acctot_rho2_trans") = acctot_rho2_trans,
                                            Rcpp::Named("acctot_A11_trans") = acctot_A11_trans,
                                            Rcpp::Named("acctot_A22_trans") = acctot_A22_trans,
                                            Rcpp::Named("acctot_mu") = acctot_mu,
                                            Rcpp::Named("acctot_mut") = acctot_mut,
                                            Rcpp::Named("acctot_rho3_trans") = acctot_rho3_trans,
                                            Rcpp::Named("acctot_alpha") = acctot_alpha,
                                            Rcpp::Named("acctot_phi0_trans") = acctot_phi0_trans,
                                            Rcpp::Named("acctot_phi1_trans") = acctot_phi1_trans);

Rcpp::List lag_info = Rcpp::List::create(Rcpp::Named("mu") = mu,
                                         Rcpp::Named("mut") = mut,
                                         Rcpp::Named("sigma2_delta") = sigma2_delta,
                                         Rcpp::Named("rho3") = rho3,
                                         Rcpp::Named("alpha") = alpha,
                                         Rcpp::Named("tau2") = tau2);
                                  
return Rcpp::List::create(Rcpp::Named("sigma2_epsilon") = sigma2_epsilon,
                          Rcpp::Named("beta0") = beta0,
                          Rcpp::Named("beta1") = beta1,
                          Rcpp::Named("betat") = betat,
                          Rcpp::Named("V") = V,
                          Rcpp::Named("rho1") = rho1,
                          Rcpp::Named("rho2") = rho2,
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

