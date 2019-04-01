#include "RcppArmadillo.h"
#include "DLfuse.h"
using namespace arma;
using namespace Rcpp;

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]

double sigma2_delta_update_st(arma::vec mut,
                              double rho3_old,
                              double alpha_sigma2_delta,
                              double beta_sigma2_delta){
  
int d = mut.size();
double mean_piece = mut[0]*mut[0];
  
for(int j = 1; j < d; ++ j){
  
   mean_piece = mean_piece + 
                (mut[j] - rho3_old*mut[j-1])*(mut[j] - rho3_old*mut[j-1]);
  
   }

double alpha_sigma2_delta_update = 0.50*d + 
                                   alpha_sigma2_delta;

double beta_sigma2_delta_update = 0.50*mean_piece + 
                                  beta_sigma2_delta;

double sigma2_delta = 1.00/R::rgamma(alpha_sigma2_delta_update,
                                     (1.00/beta_sigma2_delta_update));

return(sigma2_delta);

}





