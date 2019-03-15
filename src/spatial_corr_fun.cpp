#include "RcppArmadillo.h"
#include "DLfuse.h"
using namespace arma;
using namespace Rcpp;

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]

Rcpp::List spatial_corr_fun(double phi,
                            arma::mat spatial_dists){

double log_deter = 0.00; 
double sign = 0.00;     

arma::mat spatial_corr = exp(-phi*spatial_dists);
arma::mat spatial_corr_inv = inv_sympd(spatial_corr);
log_det(log_deter, sign, spatial_corr);

return Rcpp::List::create(Rcpp::Named("spatial_corr_inv") = spatial_corr_inv,
                          Rcpp::Named("log_deter") = log_deter);

}
