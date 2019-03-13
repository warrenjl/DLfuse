#include "RcppArmadillo.h"
#include "DLfuse.h"
using namespace arma;
using namespace Rcpp;

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]

Rcpp::List A11_update_st(Rcpp::List y,
                         Rcpp::List AQS_key_mat,
                         double A11_old,
                         Rcpp::List lagged_covars,
                         double sigma2_epsilon,
                         double beta0,
                         double beta1,
                         arma::mat betat,
                         double A22_old,
                         double A21_old,
                         arma::vec w0_old,
                         arma::vec w1_old,
                         arma::uvec keep7,
                         Rcpp::List sample_size,
                         double sigma2_A,
                         double metrop_var_A11_trans,
                         int acctot_A11_trans){
  
int d = y.size();
  
/*Second*/
double A11_trans_old = log(A11_old);
double second = 0.00;
for(int j = 0; j < d; ++ j){
  
   Rcpp::List lagged_covars_t = lagged_covars[j];
   arma::vec lc1_t = lagged_covars_t[0];
   arma::vec betat_t = betat.col(j);
   
   arma::vec mean_temp_old_t = construct_mean_st(beta0, 
                                                 beta1,
                                                 betat_t,
                                                 A11_old,
                                                 A22_old,
                                                 A21_old,
                                                 w0_old,
                                                 w1_old,
                                                 diagmat(lc1_t),
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
second = second +
         R::dnorm(A11_trans_old,
                  0.00,
                  sqrt(sigma2_A),
                  1);

/*First*/
double A11_trans = R::rnorm(A11_trans_old, 
                            sqrt(metrop_var_A11_trans));
double A11 = exp(A11_trans);

double first = 0.00;
for(int j = 0; j < d; ++ j){
  
   Rcpp::List lagged_covars_t = lagged_covars[j];
   arma::vec lc1_t = lagged_covars_t[0];
   arma::vec betat_t = betat.col(j);
   
   arma::vec mean_temp_t = construct_mean_st(beta0, 
                                             beta1,
                                             betat_t,
                                             A11,
                                             A22_old,
                                             A21_old,
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
first = first +
        R::dnorm(A11_trans,
                 0.00,
                 sqrt(sigma2_A),
                 1);

/*Decision*/
double ratio = exp(first - second);   
int acc = 1;
if(ratio < R::runif(0.00, 1.00)){
  A11 = A11_old;
  acc = 0;
  }
acctot_A11_trans = acctot_A11_trans + 
                   acc;

return Rcpp::List::create(Rcpp::Named("A11") = A11,
                          Rcpp::Named("acctot_A11_trans") = acctot_A11_trans);

}
                 