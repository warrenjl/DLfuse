#include "RcppArmadillo.h"
#include "DLfuse.h"
using namespace arma;
using namespace Rcpp;

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]

double beta0_update_st(Rcpp::List y,
                       Rcpp::List mean_temp, 
                       double sigma2_epsilon,
                       double sigma2_beta){
  
int d = y.size();
int total_ss = 0;
double mean_piece = 0.00;

for(int j = 0; j < d; ++ j){
   
   arma::vec y_t = y[j];
   total_ss = total_ss + 
              y_t.size();
   
   arma::vec mean_temp_t = mean_temp[j];
   mean_piece = mean_piece + 
                sum(y_t - mean_temp_t);
   
   }
  
double beta0_var = (sigma2_beta*sigma2_epsilon)/(total_ss*sigma2_beta + sigma2_epsilon);
  
double beta0_mean = sigma2_beta*mean_piece/(total_ss*sigma2_beta + sigma2_epsilon);

double beta0 = R::rnorm(beta0_mean,
                        sqrt(beta0_var));

return(beta0);

}



