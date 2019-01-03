#include "RcppArmadillo.h"
#include "DLfuse.h"
using namespace arma;
using namespace Rcpp;

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]

arma::vec w1_update(arma::vec y,
                    arma::vec mean_temp,
                    Rcpp::List lagged_covars,
                    double A22,
                    double sigma2_epsilon,
                    arma::mat Sigma1_inv){
  
arma::vec lc2 = lagged_covars[1];
  
arma::mat cov_w1 = inv_sympd((A22*A22)*diagmat(lc2%lc2)/sigma2_epsilon + Sigma1_inv);

arma::vec mean_w1 = cov_w1*(A22*(lc2%(y - mean_temp)))/sigma2_epsilon;
  
arma::mat ind_norms = arma::randn(1, y.size());
arma::vec w1 = mean_w1 + 
               trans(ind_norms*arma::chol(cov_w1));

return(w1);

}

  
  

  



