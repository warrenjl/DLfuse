#include "RcppArmadillo.h"
#include "DLfuse.h"
using namespace arma;
using namespace Rcpp;

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]

double sigma2_epsilon_update_st(Rcpp::List y,
                                Rcpp::List mean_temp,
                                double alpha_sigma2_epsilon,
                                double beta_sigma2_epsilon){
  
int d = y.size();
int total_ss = 0;
double mean_piece = 0.00;
for(int j = 0; j < d; ++ j){
    
   arma::vec y_t = y[j];
   total_ss = total_ss + 
              y_t.size();
    
   arma::vec mean_temp_t = mean_temp[j];
   mean_piece = mean_piece + 
                dot((y_t - mean_temp_t), (y_t - mean_temp_t));
    
   }

double alpha_sigma2_epsilon_update = 0.50*total_ss + 
                                     alpha_sigma2_epsilon;

double beta_sigma2_epsilon_update = 0.50*mean_piece + 
                                    beta_sigma2_epsilon;

double sigma2_epsilon = 1.00/R::rgamma(alpha_sigma2_epsilon_update,
                                       (1.00/beta_sigma2_epsilon_update));

return(sigma2_epsilon);

}





