#include "RcppArmadillo.h"
#include "DLfuse.h"
using namespace arma;
using namespace Rcpp;

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]

arma::mat V_update_st(arma::mat betat,
                      double rho1_old,
                      double rho2_old,
                      arma::mat Omega_V_inv,
                      double nu_v_inv){

arma::mat Omega(2,2); Omega.fill(0.00);
Omega(0,0) = rho1_old;
Omega(1,1) = rho2_old;

int d = betat.n_cols;
arma::mat mean_piece(2,2); mean_piece.fill(0.00);
arma::vec betat_t = betat.col(0);
mean_piece = (betat_t)*trans(betat_t);

for(int j = 1; j < d; ++ j){
   arma::vec betat_t = betat.col(j);
   arma::vec betat_previous = betat.col(j-1);
   mean_piece = mean_piece +
                (betat_t - Omega*betat_previous)*trans(betat_t - Omega*betat_previous);
   }
 
arma::mat mean = inv_sympd(mean_piece + Omega_V_inv);

double df = d + 
            nu_v_inv;

//Bartlett Decomposition
arma::mat L = arma::chol(mean,
                         "lower");
arma::mat A(2,2); A.fill(0.00);
A(1,0) = R::rnorm(0.00,
                  sqrt(1.00));
A(0,0) = sqrt(R::rchisq(df));
A(1,1) = sqrt(R::rchisq(df - 1.00));

arma::mat V_inv = L*A*trans(A)*trans(L);
arma::mat V = inv_sympd(V_inv);

return(V);

}

  
  

  



