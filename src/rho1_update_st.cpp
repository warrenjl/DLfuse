#include "RcppArmadillo.h"
#include "DLfuse.h"
using namespace arma;
using namespace Rcpp;

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]

Rcpp::List rho1_update_st(double rho1_old,
                          arma::mat betat,
                          arma::mat V,
                          double rho2_old,
                          double a_rho1,
                          double b_rho1,
                          double metrop_var_rho1_trans,
                          int acctot_rho1_trans){
  
int d = betat.n_cols;
arma::mat V_inv = inv_sympd(V);

/*Second*/
double rho1_trans_old = log((rho1_old - a_rho1)/(b_rho1 - rho1_old));
arma::vec second(1); second.fill(0.00);
arma::mat Omega_old(2,2); Omega_old.fill(0.00);
Omega_old(0,0) = rho1_old;
Omega_old(1,1) = rho2_old;

for(int j = 1; j < d; ++ j){
  
   arma::mat betat_t = betat.col(j);
   arma::mat betat_previous = betat.col(j-1);
   second = second +
            -0.50*dot((betat_t - Omega_old*betat_previous), (V_inv*(betat_t - Omega_old*betat_previous)));
  
   }
second = second + 
         rho1_trans_old -
         2.00*log(1.00 + exp(rho1_trans_old));

/*First*/
double rho1_trans = R::rnorm(rho1_trans_old, 
                             sqrt(metrop_var_rho1_trans));
double rho1 = (b_rho1*exp(rho1_trans) + a_rho1)/(exp(rho1_trans) + 1.00);
arma::vec first(1); first.fill(0.00);
arma::mat Omega(2,2); Omega.fill(0.00);
Omega(0,0) = rho1;
Omega(1,1) = rho2_old;

for(int j = 1; j < d; ++ j){
  
   arma::mat betat_t = betat.col(j);
   arma::mat betat_previous = betat.col(j-1);
   first = first +
           -0.50*dot((betat_t - Omega*betat_previous), (V_inv*(betat_t - Omega*betat_previous)));
    
   }
first = first + 
        rho1_trans -
        2.00*log(1.00 + exp(rho1_trans));

/*Decision*/
arma::vec ratio = exp(first - second);   
double acc = 1;
arma::vec uni_draw(1);
uni_draw(0) = R::runif(0.00, 1.00);
if(ratio(0) < uni_draw(0)){
  rho1 = rho1_old;
  acc = 0;
  }
acctot_rho1_trans = acctot_rho1_trans + 
                    acc;

return Rcpp::List::create(Rcpp::Named("rho1") = rho1,
                          Rcpp::Named("acctot_rho1_trans") = acctot_rho1_trans);

}



