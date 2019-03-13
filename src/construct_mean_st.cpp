#include "RcppArmadillo.h"
#include "DLfuse.h"
using namespace arma;
using namespace Rcpp;

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]

arma::vec construct_mean_st(double beta0, 
                            double beta1,
                            arma::vec betat_t,
                            double A11,
                            double A22,
                            double A21,
                            arma::vec w0,
                            arma::vec w1,
                            arma::mat Omega_t,
                            arma::uvec keep,
                            arma::vec sample_size_t,
                            arma::mat AQS_key_mat_t){
  
int n_t = sum(sample_size_t);
arma::mat mean_mat(n_t, 7); mean_mat.fill(0.00);
arma::vec ones(n_t); ones.fill(1.00);

mean_mat.col(0) = beta0*ones;
mean_mat.col(1) = betat_t(0)*ones;
mean_mat.col(2) = beta1*(Omega_t*ones);
mean_mat.col(3) = betat_t(1)*(Omega_t*ones);
mean_mat.col(4) = A11*AQS_key_mat_t*w0;
mean_mat.col(5) = A21*(Omega_t*(AQS_key_mat_t*w0));
mean_mat.col(6) = A22*(Omega_t*(AQS_key_mat_t*w1));

arma::vec mean_vec(n_t); mean_vec.fill(0);
for(int j = 0; j < n_t; ++ j){
  
  arma::vec temp_vec = trans(mean_mat.row(j));
  mean_vec(j) = sum(temp_vec.elem(keep));
  
  }
  
return(mean_vec);

}



