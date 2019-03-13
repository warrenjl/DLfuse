#include "RcppArmadillo.h"
#include "DLfuse.h"
using namespace arma;
using namespace Rcpp;

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]

double A21_update_st(Rcpp::List y,
                     Rcpp::List AQS_key_mat,
                     Rcpp::List mean_temp,
                     Rcpp::List lagged_covars,
                     double sigma2_epsilon,
                     arma::vec w0_old,
                     double sigma2_A){
  
int d = y.size();
double mean_piece1 = 0.00;
double mean_piece2 = 0.00;
  
for(int j = 0; j < d; ++ j){
  
   Rcpp::List lagged_covars_t = lagged_covars[j];
   arma::vec lc1_t = lagged_covars_t[0];
    
   arma::mat AQS_key_mat_t = AQS_key_mat[j];
   mean_piece1 = mean_piece1 + 
                 dot(((AQS_key_mat_t*w0_old)%(AQS_key_mat_t*w0_old)), (lc1_t%lc1_t));
   
   arma::vec y_t = y[j];
   arma::vec mean_temp_t = mean_temp[j];
   mean_piece2 = mean_piece2 +
                 dot((AQS_key_mat_t*w0_old), (lc1_t%(y_t - mean_temp_t)));
    
   }
  
double A21_var = (sigma2_A*sigma2_epsilon)/(sigma2_A*mean_piece1 + sigma2_epsilon);
  
double A21_mean = sigma2_A*mean_piece2/(sigma2_A*mean_piece1 + sigma2_epsilon);
  
double A21 = R::rnorm(A21_mean,
                      sqrt(A21_var));
  
return(A21);

}


