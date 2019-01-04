#include "RcppArmadillo.h"
#include "DLfuse.h"
using namespace arma;
using namespace Rcpp;

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]

double beta1_update(arma::vec y,
                    arma::vec mean_temp,
                    Rcpp::List lagged_covars,
                    double sigma2_epsilon,
                    arma::vec sample_size,
                    double sigma2_beta){

arma::vec lc1 = lagged_covars[0];
arma::vec lc2 = lagged_covars[1];
  
double beta1_var = (sigma2_beta*sigma2_epsilon)/(sigma2_beta*sum(sample_size%((lc2%lc2))) + sigma2_epsilon);
  
double beta1_mean = sigma2_beta*(dot(lc1, (y - mean_temp)))/(sigma2_beta*sum(sample_size%((lc2%lc2))) + sigma2_epsilon);
  
double beta1 = R::rnorm(beta1_mean,
                        sqrt(beta1_var));
  
return(beta1);

}



