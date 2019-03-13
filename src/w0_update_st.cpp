#include "RcppArmadillo.h"
#include "DLfuse.h"
using namespace arma;
using namespace Rcpp;

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]

arma::vec w0_update_st(Rcpp::List y,
                       Rcpp::List AQS_key_mat,
                       Rcpp::List mean_temp,
                       Rcpp::List lagged_covars,
                       double sigma2_epsilon,
                       double A11,
                       double A21,
                       arma::mat Sigma0_inv){
  
int d = y.size();
int n = Sigma0_inv.n_rows;
arma::mat mat_piece(n,n); mat_piece.fill(0.00);
arma::vec mean_piece(n); mean_piece.fill(0.00);

for(int j = 0; j < d; ++ j){
  
   Rcpp::List lagged_covars_t = lagged_covars[j];
   arma::vec lc1_t = lagged_covars_t[0];
   arma::mat AQS_key_mat_t = AQS_key_mat[j];
   mat_piece = mat_piece +
               trans(AQS_key_mat_t)*diagmat((A11 + A21*lc1_t)%(A11 + A21*lc1_t))*AQS_key_mat_t;
   
   arma::vec y_t = y[j];
   arma::vec mean_temp_t = mean_temp[j];
   mean_piece = mean_piece + 
                trans(AQS_key_mat_t)*((A11 + A21*lc1_t)%(y_t - mean_temp_t));
   
   }

arma::mat cov_w0 = inv_sympd(mat_piece/sigma2_epsilon + Sigma0_inv);

arma::vec mean_w0 = cov_w0*mean_piece/sigma2_epsilon;
  
arma::mat ind_norms = arma::randn(1, n);
arma::vec w0 = mean_w0 + 
               trans(ind_norms*arma::chol(cov_w0));

//Centering for Stability
w0 = w0 -
     mean(w0);

return(w0);

}



  



