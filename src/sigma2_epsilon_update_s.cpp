#include "RcppArmadillo.h"
#include "DLfuse.h"
using namespace arma;
using namespace Rcpp;

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]

double sigma2_epsilon_update_s(arma::vec y,
                               arma::vec mean_temp,
                               arma::vec sample_size,
                               double alpha_sigma2_epsilon,
                               double beta_sigma2_epsilon){

double alpha_sigma2_epsilon_update = 0.50*sum(sample_size) + 
                                     alpha_sigma2_epsilon;

double beta_sigma2_epsilon_update = 0.50*dot((y - mean_temp), (y - mean_temp)) + 
                                    beta_sigma2_epsilon;

double sigma2_epsilon = 1.00/R::rgamma(alpha_sigma2_epsilon_update,
                                       (1.00/beta_sigma2_epsilon_update));

return(sigma2_epsilon);

}





