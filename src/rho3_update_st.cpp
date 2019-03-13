#include "RcppArmadillo.h"
#include "DLfuse.h"
using namespace arma;
using namespace Rcpp;

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]

Rcpp::List rho3_update_st(double rho3_old,
                          arma::vec mut,
                          double sigma2_delta,
                          double a_rho3,
                          double b_rho3,
                          double metrop_var_rho3_trans,
                          int acctot_rho3_trans){
  
int d = mut.size();

/*Second*/
double rho3_trans_old = log((rho3_old - a_rho3)/(b_rho3 - rho3_old));
double second = 0;

for(int j = 1; j < d; ++ j){
  
   double mean_temp_t = rho3_old*mut(j-1);
  
   second = second + 
            R::dnorm(mut(j),
                     mean_temp_t,
                     sqrt(sigma2_delta),
                     1);
   
   }
second = second + 
         rho3_trans_old -
         2.00*log(1.00 + exp(rho3_trans_old));

/*First*/
double rho3_trans = R::rnorm(rho3_trans_old, 
                             sqrt(metrop_var_rho3_trans));
double rho3 = (b_rho3*exp(rho3_trans) + a_rho3)/(exp(rho3_trans) + 1.00);
double first = 0.00;

for(int j = 1; j < d; ++ j){
  
   double mean_temp_t = rho3*mut(j - 1);
  
   second = second + 
   R::dnorm(mut(j),
            mean_temp_t,
            sqrt(sigma2_delta),
            1);
  
   }
first = first + 
        rho3_trans -
        2.00*log(1.00 + exp(rho3_trans));

/*Decision*/
double ratio = exp(first - second);   
double acc = 1;
if(ratio < R::runif(0.00, 1.00)){
  rho3 = rho3_old;
  acc = 0;
  }
acctot_rho3_trans = acctot_rho3_trans + 
                    acc;

return Rcpp::List::create(Rcpp::Named("rho3") = rho3,
                          Rcpp::Named("acctot_rho3_trans") = acctot_rho3_trans);

}



