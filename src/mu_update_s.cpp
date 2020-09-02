#include "RcppArmadillo.h"
#include "DLfuse.h"
using namespace arma;
using namespace Rcpp;

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]

Rcpp::List mu_update_s(arma::vec y,
                       arma::mat z,
                       double mu_old,
                       Rcpp::List lagged_covars,
                       double sigma2_epsilon,
                       double beta0,
                       double beta1,
                       double A11,
                       double A22,
                       double A21,
                       arma::vec alpha_old,
                       arma::vec w0_old,
                       arma::vec w1_old,
                       arma::uvec keep5,
                       arma::vec sample_size,
                       double metrop_var_mu,
                       int acctot_mu,
                       int weights_definition){
  
int n = y.size();

/*Second*/
Rcpp::List lagged_covars_old = lagged_covars;
arma::vec lc1_old = lagged_covars_old[0];

arma::vec mean_temp_old = construct_mean_s(beta0, 
                                           beta1,
                                           A11,
                                           A22,
                                           A21,
                                           w0_old,
                                           w1_old,
                                           diagmat(lc1_old),
                                           keep5,
                                           sample_size);

double second = 0.00;
for(int j = 0; j < n; ++ j){
   second = second + 
            R::dnorm(y(j),
                     mean_temp_old(j),
                     sqrt(sigma2_epsilon),
                     1);
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
lagged_covars = construct_lagged_covars_s(z,
                                          mu, 
                                          alpha_old,
                                          sample_size,
                                          weights_definition);
arma::vec lc1 = lagged_covars(0);

arma::vec mean_temp = construct_mean_s(beta0, 
                                       beta1,
                                       A11,
                                       A22,
                                       A21,
                                       w0_old,
                                       w1_old,
                                       diagmat(lc1),
                                       keep5,
                                       sample_size);

double first = 0.00;
for(int j = 0; j < n; ++ j){
   first = first + 
           R::dnorm(y(j),
                    mean_temp(j),
                    sqrt(sigma2_epsilon),
                    1);
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
