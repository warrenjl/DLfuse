#include "RcppArmadillo.h"
#include "DLfuse.h"
using namespace arma;
using namespace Rcpp;

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]

Rcpp::List mu_update_st(Rcpp::List y,
                        Rcpp::List z,
                        double mu_old,
                        Rcpp::List lagged_covars,
                        double sigma2_epsilon,
                        double beta0,
                        double beta1,
                        arma::mat betat,
                        double A11,
                        double A22,
                        double A21,
                        arma::vec mut_old,
                        arma::vec alpha_old,
                        arma::vec w0_old,
                        arma::vec w1_old,
                        arma::uvec keep7,
                        Rcpp::List sample_size,
                        Rcpp::List AQS_key_mat,
                        Rcpp::List CMAQ_key,
                        double metrop_var_mu,
                        int acctot_mu,
                        int weights_definition){
  
int d = y.size();

/*Second*/
Rcpp::List lagged_covars_old = lagged_covars;
double second = 0.00;
for(int j = 0; j < d; ++ j){
  
   Rcpp::List lagged_covars_old_t = lagged_covars_old[j];
   arma::vec lc1_old_t = lagged_covars_old_t[0];
   arma::vec betat_t = betat.col(j);
   
   arma::vec mean_temp_old_t = construct_mean_st(beta0, 
                                                 beta1,
                                                 betat_t,
                                                 A11,
                                                 A22,
                                                 A21,
                                                 w0_old,
                                                 w1_old,
                                                 diagmat(lc1_old_t),
                                                 keep7,
                                                 sample_size[j],
                                                 AQS_key_mat[j]);
  
   arma::vec y_t = y[j];
   int ss_t = y_t.size();
   for(int k = 0; k < ss_t; ++ k){
      second = second + 
               R::dnorm(y_t(k),
                        mean_temp_old_t(k),
                        sqrt(sigma2_epsilon),
                        1);
     
      } 
  
   }

if(weights_definition == 0){
  second = second +
           R::dnorm(mu_old,
                    0.00,
                    sqrt(1.00),
                    1);
  }

if(weights_definition == 1){
  second = second +
           R::dnorm(mu_old,
                    0.00,
                    sqrt(1.00),
                    1);
  }

/*First*/
double mu = R::rnorm(mu_old, 
                     sqrt(metrop_var_mu));

double first = 0.00;
for(int j = 0; j < d; ++ j){
  
   lagged_covars[j] = construct_lagged_covars_st(z[j],
                                                 mu,
                                                 mut_old(j),
                                                 alpha_old,
                                                 sample_size[j],
                                                 CMAQ_key[j],
                                                 weights_definition);
   Rcpp::List lagged_covars_t = lagged_covars[j];
   arma::vec lc1_t = lagged_covars_t[0];
   arma::vec betat_t = betat.col(j);
   
   arma::vec mean_temp_t = construct_mean_st(beta0, 
                                             beta1,
                                             betat_t,
                                             A11,
                                             A22,
                                             A21,
                                             w0_old,
                                             w1_old,
                                             diagmat(lc1_t),
                                             keep7,
                                             sample_size[j],
                                             AQS_key_mat[j]);
   
   arma::vec y_t = y[j];
   int ss_t = y_t.size();
   for(int k = 0; k < ss_t; ++ k){
      first = first + 
              R::dnorm(y_t(k),
                       mean_temp_t(k),
                       sqrt(sigma2_epsilon),
                       1);
     
      } 
  
   }

if(weights_definition == 0){
  first = first +
          R::dnorm(mu,
                   0.00,
                   sqrt(1.00),
                   1);
  }

if(weights_definition == 1){
  first = first +
          R::dnorm(mu,
                   0.00,
                   sqrt(1.00),
                   1);
  }

/*Decision*/
double ratio = exp(first - second);   
int acc = 1;
if(ratio < R::runif(0.00, 1.00)){
   mu = mu_old;
   lagged_covars = lagged_covars_old;
   acc = 0;
   }
acctot_mu = acctot_mu + 
            acc;

return Rcpp::List::create(Rcpp::Named("mu") = mu,
                          Rcpp::Named("acctot_mu") = acctot_mu,
                          Rcpp::Named("lagged_covars") = lagged_covars);

}

