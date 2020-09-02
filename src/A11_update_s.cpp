#include "RcppArmadillo.h"55
#include "DLfuse.h"
using namespace arma;
using namespace Rcpp;

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
Rcpp::List A11_update_s(arma::vec y,
                        double A11_old,
                        Rcpp::List lagged_covars,
                        double sigma2_epsilon,
                        double beta0,
                        double beta1,
                        double A22_old,
                        double A21_old,
                        arma::vec w0_old,
                        arma::vec w1_old,
                        arma::uvec keep5,
                        arma::vec sample_size,
                        double sigma2_A,
                        double metrop_var_A11_trans,
                        int acctot_A11_trans){
  
arma::vec lc1 = lagged_covars[0];
arma::vec lc2 = lagged_covars[1];
int n = y.size();

/*Second*/
double A11_trans_old = log(A11_old);

arma::vec mean_temp_old = construct_mean_s(beta0, 
                                           beta1,
                                           A11_old,
                                           A22_old,
                                           A21_old,
                                           w0_old,
                                           w1_old,
                                           diagmat(lc1),
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
second = second +
         R::dnorm(A11_trans_old,
                  0.00,
                  sqrt(sigma2_A),
                  1);

/*First*/
double A11_trans = R::rnorm(A11_trans_old, 
                            sqrt(metrop_var_A11_trans));
double A11 = exp(A11_trans);

arma::vec mean_temp = construct_mean_s(beta0, 
                                       beta1,
                                       A11,
                                       A22_old,
                                       A21_old,
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
                 