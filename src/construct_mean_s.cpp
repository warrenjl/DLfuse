#include "RcppArmadillo.h"
#include "DLfuse.h"
using namespace arma;
using namespace Rcpp;

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]

arma::vec construct_mean_s(double beta0, 
                           double beta1,
                           double A11,
                           double A22,
                           double A21,
                           arma::vec w0,
                           arma::vec w1,
                           arma::mat Omega,
                           arma::uvec keep,
                           arma::vec sample_size){
  
int n = sum(sample_size);
arma::mat mean_mat(n, 5.00); mean_mat.fill(0.00);
arma::vec ones(n); ones.fill(1.00);

mean_mat.col(0) = beta0*ones;
mean_mat.col(1) = beta1*(Omega*ones);
mean_mat.col(2) = A11*w0;
mean_mat.col(3) = A21*(Omega*w0);
mean_mat.col(4) = A22*(Omega*w1);
  
arma::vec mean_vec(n); mean_vec.fill(0);
for(int j = 0; j < n; ++ j){
  arma::vec temp_vec = trans(mean_mat.row(j));
  mean_vec(j) = sum(temp_vec.elem(keep));
  }
  
return(mean_vec);

}



