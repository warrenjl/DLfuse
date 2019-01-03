#include "RcppArmadillo.h"
#include "DLfuse.h"
using namespace arma;
using namespace Rcpp;

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]

double tau2_update(int G,
                   arma::mat CAR,
                   double alpha_tau2,
                   double beta_tau2,
                   arma::vec alpha){

double alpha_tau2_update = 0.50*(alpha.size() - G) + 
                           alpha_tau2;

double beta_tau2_update = 0.50*dot(alpha, (CAR*alpha)) + 
                          beta_tau2;

double tau2 = 1.00/R::rgamma(alpha_tau2_update,
                             (1.00/beta_tau2_update));

return(tau2);

}




