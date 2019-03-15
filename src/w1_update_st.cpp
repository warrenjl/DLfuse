#include "RcppArmadillo.h"
#include "DLfuse.h"
using namespace arma;
using namespace Rcpp;

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]

arma::vec w1_update_st(Rcpp::List y,
                       Rcpp::List AQS_key_mat,
                       Rcpp::List mean_temp,
                       Rcpp::List lagged_covars,
                       double sigma2_epsilon,
                       double A22,
                       arma::mat Sigma1_inv){
  
int d = y.size();
int n = Sigma1_inv.n_rows;
arma::mat mat_piece(n,n); mat_piece.fill(0.00);
arma::vec mean_piece(n); mean_piece.fill(0.00);
  
for(int j = 0; j < d; ++ j){
    
   Rcpp::List lagged_covars_t = lagged_covars[j];
   arma::vec lc1_t = lagged_covars_t[0];
   arma::mat AQS_key_mat_t = AQS_key_mat[j];
   mat_piece = mat_piece +
               trans(AQS_key_mat_t)*diagmat(lc1_t%lc1_t)*AQS_key_mat_t;
    
   arma::vec y_t = y[j];
   arma::vec mean_temp_t = mean_temp[j];
   mean_piece = mean_piece + 
                trans(AQS_key_mat_t)*(lc1_t%(y_t - mean_temp_t));
    
   }
  
arma::mat cov_w1 = inv_sympd((A22*A22)*mat_piece/sigma2_epsilon + Sigma1_inv);

arma::vec mean_w1 = cov_w1*(A22*mean_piece)/sigma2_epsilon;
  
arma::mat ind_norms = arma::randn(1,n);
arma::vec w1 = mean_w1 + 
               trans(ind_norms*arma::chol(cov_w1));

//Centering for Stability
w1 = w1 -
     mean(w1);

return(w1);

}

  
  

  



