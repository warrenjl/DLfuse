#include "RcppArmadillo.h"
#include "DLfuse.h"
using namespace arma;
using namespace Rcpp;

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]

double A21_update(arma::vec y,
                  arma::vec mean_temp,
                  Rcpp::List lagged_covars,
                  double sigma2_epsilon,
                  arma::vec w0_old,
                  double sigma2_A){

arma::vec lc1 = lagged_covars[0];
arma::vec lc2 = lagged_covars[1];

double A21_var = (sigma2_A*sigma2_epsilon)/(sigma2_A*sum((w0_old%w0_old)%(lc1%lc1)) + sigma2_epsilon);
  
double A21_mean = sigma2_A*dot(w0_old, (lc1%(y - mean_temp)))/(sigma2_A*sum((w0_old%w0_old)%(lc1%lc1)) + sigma2_epsilon);
  
double A21 = R::rnorm(A21_mean,
                      sqrt(A21_var));
  
return(A21);

}


