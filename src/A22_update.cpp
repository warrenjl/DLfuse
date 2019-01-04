#include "RcppArmadillo.h"
#include "DLfuse.h"
using namespace arma;
using namespace Rcpp;

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]

Rcpp::List A22_update(arma::vec y,
                      double A22_old,
                      double sigma2_epsilon,
                      double beta0,
                      double beta1,
                      double A11,
                      double A21_old,
                      arma::vec w0_old,
                      arma::vec w1_old,
                      Rcpp::List lagged_covars,
                      arma::uvec keep5,
                      arma::vec sample_size,
                      double sigma2_A,
                      double metrop_var_A22_trans,
                      int acctot_A22_trans){
  
arma::vec lc1 = lagged_covars[0];
arma::vec lc2 = lagged_covars[1];
int n = y.size();

/*Second*/
double A22_trans_old = log(A22_old);

arma::vec mean_temp_old = construct_mean(beta0, 
                                         beta1,
                                         A11,
                                         A22_old,
                                         A21_old,
                                         w0_old,
                                         w1_old,
                                         diagmat(lc1),
                                         keep5,
                                         sample_size);

double second = 0;
for(int j = 0; j < n; ++ j){
   second = second + 
            R::dnorm(y(j),
                     mean_temp_old(j),
                     sqrt(sigma2_epsilon),
                     1);
   }
second = second +
         R::dnorm(A22_trans_old,
                  0,
                  sqrt(sigma2_A),
                  1);

/*First*/
double A22_trans = R::rnorm(A22_trans_old, 
                            sqrt(metrop_var_A22_trans));
double A22 = exp(A22_trans);

arma::vec mean_temp = construct_mean(beta0, 
                                     beta1,
                                     A11,
                                     A22,
                                     A21_old,
                                     w0_old,
                                     w1_old,
                                     diagmat(lc1),
                                     keep5,
                                     sample_size);

double first = 0;
for(int j = 0; j < n; ++ j){
   first = first + 
           R::dnorm(y(j),
                    mean_temp(j),
                    sqrt(sigma2_epsilon),
                    1);
   }
first = first +
        R::dnorm(A22_trans,
                 0,
                 sqrt(sigma2_A),
                 1);

/*Decision*/
double ratio = exp(first - second);   
int acc = 1;
if(ratio < R::runif(0.00, 1.00)){
  A22 = A22_old;
  acc = 0;
  }
acctot_A22_trans = acctot_A22_trans + 
                   acc;

return Rcpp::List::create(Rcpp::Named("A22") = A22,
                          Rcpp::Named("acctot_A22_trans") = acctot_A22_trans);

}

                 