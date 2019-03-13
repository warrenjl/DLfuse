#include "RcppArmadillo.h"
#include "DLfuse.h"
using namespace arma;
using namespace Rcpp;

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]

Rcpp::List construct_lagged_covars_s(arma::mat z,
                                     double mu, 
                                     arma::vec alpha,
                                     arma::vec sample_size){
  
int L = z.n_cols;
int m = z.n_rows;
int n = sum(sample_size);
arma::mat regression_weights(m, L); regression_weights.fill(0.00);
for(int j = 0; j < m; ++ j){
   for(int k = 0; k < L; ++ k){
      regression_weights(j,k) = pow((R::pnorm(mu + alpha(j), 0.00, 1.00, 1, 0)), k);
      }
   regression_weights.row(j) = regression_weights.row(j)/sum(regression_weights.row(j));
   }

arma::vec lagged_covars_reduced(m); lagged_covars_reduced.fill(0.00);
arma::mat temp_mat = z%regression_weights;
for(int j = 0; j < m; ++ j){
   lagged_covars_reduced(j) = sum(temp_mat.row(j));
   }

arma::vec lagged_covars_full(n); lagged_covars_full.fill(0.00);
int counter = 0;
for(int j = 0; j < m; ++ j){
   int ss = sample_size(j);
   for(int k = 0; k < ss; ++ k){
      lagged_covars_full(counter) = lagged_covars_reduced(j);
      ++ counter;
      }
   }
    
return Rcpp::List::create(Rcpp::Named("lagged_covars_full") = lagged_covars_full,
                          Rcpp::Named("lagged_covars_reduced") = lagged_covars_reduced);

}



