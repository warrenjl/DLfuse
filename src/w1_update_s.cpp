#include "RcppArmadillo.h"
#include "DLfuse.h"
using namespace arma;
using namespace Rcpp;

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]

arma::vec w1_update_s(arma::vec y,
                      arma::vec mean_temp,
                      Rcpp::List lagged_covars,
                      double sigma2_epsilon,
                      double A22,
                      arma::mat Sigma1_inv){
  
int n = y.size();
  
arma::vec lc1 = lagged_covars[0];
  
arma::mat cov_w1 = inv_sympd((A22*A22)*diagmat(lc1%lc1)/sigma2_epsilon + Sigma1_inv);

arma::vec mean_w1 = cov_w1*(A22*(lc1%(y - mean_temp)))/sigma2_epsilon;
  
arma::mat ind_norms = arma::randn(1,n);
arma::vec w1 = mean_w1 + 
               trans(ind_norms*arma::chol(cov_w1));

//Centering for Stability
w1 = w1 -
     mean(w1);

return(w1);

}

  
  

  



