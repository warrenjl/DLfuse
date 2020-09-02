#include "RcppArmadillo.h"
#include "DLfuse.h"
using namespace arma;
using namespace Rcpp;

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]

Rcpp::List mut_update_st(int last_time_ind,
                         arma::vec y_t,
                         arma::mat z_t,
                         double mut_t_old,
                         Rcpp::List lagged_covars_t,
                         double sigma2_epsilon,
                         double beta0,
                         double beta1,
                         arma::vec betat_t,
                         double A11,
                         double A22,
                         double A21,
                         double mu,
                         double mut_previous,
                         double mut_next,
                         double sigma2_delta_old,
                         double rho3_old,
                         arma::vec alpha_old,
                         arma::vec w0_old,
                         arma::vec w1_old,
                         arma::uvec keep7,
                         arma::vec sample_size_t,
                         arma::mat AQS_key_mat_t,
                         arma::vec CMAQ_key_t,
                         double metrop_var_mut_t,
                         int acctot_mut_t,
                         int weights_definition){
 
int ss_t = y_t.size();
  
/*Second*/
Rcpp::List lagged_covars_t_old = lagged_covars_t;

arma::vec lc1_t_old = lagged_covars_t_old[0];
arma::vec mean_temp_old_t = construct_mean_st(beta0, 
                                              beta1,
                                              betat_t,
                                              A11,
                                              A22,
                                              A21,
                                              w0_old,
                                              w1_old,
                                              diagmat(lc1_t_old),
                                              keep7,
                                              sample_size_t,
                                              AQS_key_mat_t);
  
double second = 0.00;
for(int j = 0; j < ss_t; ++ j){
   second = second + 
            R::dnorm(y_t(j),
                     mean_temp_old_t(j),
                     sqrt(sigma2_epsilon),
                     1);
   } 
second = second +
         R::dnorm(mut_t_old,
                  (rho3_old*mut_previous),
                  sqrt(sigma2_delta_old),
                  1);
if(last_time_ind == 0){
  
  second = second +
           R::dnorm(mut_next,
                    (rho3_old*mut_t_old),
                    sqrt(sigma2_delta_old),
                    1);
  
  }

/*First*/
double mut_t = R::rnorm(mut_t_old, 
                        sqrt(metrop_var_mut_t));

lagged_covars_t = construct_lagged_covars_st(z_t,
                                             mu,
                                             mut_t,
                                             alpha_old,
                                             sample_size_t,
                                             CMAQ_key_t,
                                             weights_definition);
arma::vec lc1_t = lagged_covars_t[0];
  
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
                                          sample_size_t,
                                          AQS_key_mat_t);

double first = 0.00;
for(int j = 0; j < ss_t; ++ j){
   first = first + 
            R::dnorm(y_t(j),
                     mean_temp_t(j),
                     sqrt(sigma2_epsilon),
                     1);
   } 
first = first +
        R::dnorm(mut_t,
                 (rho3_old*mut_previous),
                 sqrt(sigma2_delta_old),
                 1);
if(last_time_ind == 0){
  
  first = first +
          R::dnorm(mut_next,
                   (rho3_old*mut_t),
                   sqrt(sigma2_delta_old),
                   1);
  
  }

/*Decision*/
double ratio = exp(first - second);   
int acc = 1;
if(ratio < R::runif(0.00, 1.00)){
   mut_t = mut_t_old;
   lagged_covars_t = lagged_covars_t_old;
   acc = 0;
   }
acctot_mut_t = acctot_mut_t + 
               acc;

return Rcpp::List::create(Rcpp::Named("mut_t") = mut_t,
                          Rcpp::Named("acctot_mut_t") = acctot_mut_t,
                          Rcpp::Named("lagged_covars_t") = lagged_covars_t);

}

