#include "RcppArmadillo.h"
#include "DLfuse.h"
using namespace arma;
using namespace Rcpp;

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]

double neg_two_loglike_update(arma::vec y,
                              arma::vec mean_temp,
                              double sigma2_epsilon){

int n = y.size();
arma::vec dens(n); dens.fill(0.00);
for(int j = 0; j < n; ++ j){
   dens(j) = R::dnorm(y(j),
                      mean_temp(j),
                      sqrt(sigma2_epsilon),
                      TRUE);
   }

double neg_two_loglike = -2.00*sum(dens);

return neg_two_loglike;

}





















































