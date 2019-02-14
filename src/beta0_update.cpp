#include "RcppArmadillo.h"
#include "DLfuse.h"
using namespace arma;
using namespace Rcpp;

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]

double beta0_update(arma::vec y,
                    arma::vec mean_temp, 
                    double sigma2_epsilon,
                    arma::vec sample_size,
                    double sigma2_beta){
  
double beta0_var = (sigma2_beta*sigma2_epsilon)/(sum(sample_size)*sigma2_beta + sigma2_epsilon);
  
double beta0_mean = sigma2_beta*sum(y - mean_temp)/(sum(sample_size)*sigma2_beta + sigma2_epsilon);

double beta0 = R::rnorm(beta0_mean,
                        sqrt(beta0_var));

return(beta0);

}



