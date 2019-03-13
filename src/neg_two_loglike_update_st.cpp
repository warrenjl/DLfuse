#include "RcppArmadillo.h"
#include "DLfuse.h"
using namespace arma;
using namespace Rcpp;

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]

double neg_two_loglike_update_st(Rcpp::List y,
                                 Rcpp::List mean_temp,
                                 double sigma2_epsilon){

int d = y.size();
double dens = 0;  
for(int j = 0; j < d; ++ j){
  
   arma::vec y_t = y[j];
   int n_t = y_t.size();
   arma::vec mean_temp_t = mean_temp[j];
   for(int k = 0; k < n_t; ++ k){
     
      dens = dens +
             R::dnorm(y_t(k),
                      mean_temp_t(k),
                      sqrt(sigma2_epsilon),
                      TRUE);
     
      }
   
   }

double neg_two_loglike = -2.00*sum(dens);

return neg_two_loglike;

}





















































