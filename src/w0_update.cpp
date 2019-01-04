#include "RcppArmadillo.h"
#include "DLfuse.h"
using namespace arma;
using namespace Rcpp;

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]

arma::vec w0_update(arma::vec y,
                    arma::vec mean_temp,
                    Rcpp::List lagged_covars,
                    double sigma2_epsilon,
                    double A11,
                    double A21,
                    arma::mat Sigma0_inv){
  
arma::vec lc2 = lagged_covars[1];
  
arma::mat cov_w0 = inv_sympd(diagmat((A11 + A21*lc2)%(A11 + A21*lc2))/sigma2_epsilon + Sigma0_inv);

arma::vec mean_w0 = cov_w0*((A11 + A21*lc2)%(y - mean_temp))/sigma2_epsilon;
  
arma::mat ind_norms = arma::randn(1, y.size());
arma::vec w0 = mean_w0 + 
               trans(ind_norms*arma::chol(cov_w0));

return(w0);

}



  



