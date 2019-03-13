#include "RcppArmadillo.h"
#include "DLfuse.h"
using namespace arma;
using namespace Rcpp;

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]

double beta1_update_st(Rcpp::List y,
                       Rcpp::List mean_temp,
                       Rcpp::List lagged_covars,
                       double sigma2_epsilon,
                       Rcpp::List sample_size,
                       double sigma2_beta){
  
int d = y.size();
double mean_piece1 = 0.00;
double mean_piece2 = 0.00;
  
for(int j = 0; j < d; ++ j){
  
   Rcpp::List lagged_covars_t = lagged_covars[j];
   arma::vec lc1_t = lagged_covars_t[0];
   arma::vec lc2_t = lagged_covars_t[1];

   arma::vec sample_size_t = sample_size[j];
   mean_piece1 = mean_piece1 + 
                 dot(sample_size_t, ((lc2_t%lc2_t)));
   
   arma::vec y_t = y[j];
   arma::vec mean_temp_t = mean_temp[j];
   mean_piece2 = mean_piece2 +
                 dot(lc1_t, (y_t - mean_temp_t));
   
   }
     
double beta1_var = (sigma2_beta*sigma2_epsilon)/(sigma2_beta*mean_piece1 + sigma2_epsilon);
  
double beta1_mean = sigma2_beta*mean_piece2/(sigma2_beta*mean_piece1 + sigma2_epsilon);
  
double beta1 = R::rnorm(beta1_mean,
                        sqrt(beta1_var));
  
return(beta1);

}



