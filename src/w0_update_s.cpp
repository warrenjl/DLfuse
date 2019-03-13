#include "RcppArmadillo.h"
#include "DLfuse.h"
using namespace arma;
using namespace Rcpp;

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]

arma::vec w0_update_s(arma::vec y,
                      arma::vec mean_temp,
                      Rcpp::List lagged_covars,
                      double sigma2_epsilon,
                      double A11,
                      double A21,
                      arma::mat Sigma0_inv){
  
int n = y.size();
  
arma::vec lc1 = lagged_covars[0];
  
arma::mat cov_w0 = inv_sympd(diagmat((A11 + A21*lc1)%(A11 + A21*lc1))/sigma2_epsilon + Sigma0_inv);

arma::vec mean_w0 = cov_w0*((A11 + A21*lc1)%(y - mean_temp))/sigma2_epsilon;
  
arma::mat ind_norms = arma::randn(1, n);
arma::vec w0 = mean_w0 + 
               trans(ind_norms*arma::chol(cov_w0));

//Centering for Stability
w0 = w0 -
     mean(w0);

return(w0);

}



  



