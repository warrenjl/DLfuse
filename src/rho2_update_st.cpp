#include "RcppArmadillo.h"
#include "DLfuse.h"
using namespace arma;
using namespace Rcpp;

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]

Rcpp::List rho2_update_st(double rho2_old,
                          arma::mat betat,
                          arma::mat V,
                          double rho1,
                          double a_rho2,
                          double b_rho2,
                          double metrop_var_rho2_trans,
                          int acctot_rho2_trans){
  
  int d = betat.n_cols;
  arma::mat V_inv = inv_sympd(V);
  
  /*Second*/
  double rho2_trans_old = log((rho2_old - a_rho2)/(b_rho2 - rho2_old));
  arma::vec second(1); second.fill(0.00);
  arma::mat Omega_old(2,2); Omega_old.fill(0.00);
  Omega_old(0,0) = rho1;
  Omega_old(1,1) = rho2_old;
  
  for(int j = 1; j < d; ++ j){
    
     arma::mat betat_t = betat.col(j);
     arma::mat betat_previous = betat.col(j-1);
     second = second +
              -0.50*dot((betat_t - Omega_old*betat_previous), (V_inv*(betat_t - Omega_old*betat_previous)));
      
     }
  second = second + 
           rho2_trans_old -
           2.00*log(1.00 + exp(rho2_trans_old));
  
  /*First*/
  double rho2_trans = R::rnorm(rho2_trans_old, 
                               sqrt(metrop_var_rho2_trans));
  double rho2 = (b_rho2*exp(rho2_trans) + a_rho2)/(exp(rho2_trans) + 1.00);
  arma::vec first(1); first.fill(0.00);
  arma::mat Omega(2,2); Omega.fill(0.00);
  Omega(0,0) = rho1;
  Omega(1,1) = rho2;
  
  for(int j = 1; j < d; ++ j){
    
     arma::mat betat_t = betat.col(j);
     arma::mat betat_previous = betat.col(j-1);
     first = first +
             -0.50*dot((betat_t - Omega*betat_previous), (V_inv*(betat_t - Omega*betat_previous)));
      
     }
  first = first + 
          rho2_trans -
          2.00*log(1.00 + exp(rho2_trans));
  
  /*Decision*/
  arma::vec ratio = exp(first - second);   
  double acc = 1;
  arma::vec uni_draw(1);
  uni_draw(0) = R::runif(0.00, 1.00);
  if(ratio(0) < uni_draw(0)){
    rho2 = rho2_old;
    acc = 0;
    }
  acctot_rho2_trans = acctot_rho2_trans + 
                      acc;
  
  return Rcpp::List::create(Rcpp::Named("rho2") = rho2,
                            Rcpp::Named("acctot_rho2_trans") = acctot_rho2_trans);
  
  }



