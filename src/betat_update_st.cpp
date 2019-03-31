#include "RcppArmadillo.h"
#include "DLfuse.h"
using namespace arma;
using namespace Rcpp;

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]

arma::vec betat_update_st(int last_time_ind,
                          arma::vec y_t,
                          arma::vec mean_temp_t,
                          Rcpp::List lagged_covars_t,
                          double sigma2_epsilon,
                          arma::vec betat_previous,
                          arma::vec betat_next,
                          arma::mat V_old,
                          double rho1_old,
                          double rho2_old){

arma::mat Omega(2,2); Omega.fill(0.00);
Omega(0,0) = rho1_old;
Omega(1,1) = rho2_old;

arma::mat V_inv_old = inv_sympd(V_old);

int n_t = y_t.size();
arma::mat gamma_t(n_t,2); gamma_t.fill(0.00);
arma::vec ones(n_t); ones.fill(1.00);
arma::vec lc1_t = lagged_covars_t[0];
gamma_t.submat(0, 0, (n_t - 1), 0) = ones;
gamma_t.submat(0, 1, (n_t - 1), 1) = lc1_t;

arma::mat cov_betat(2,2); cov_betat.fill(0.00);
arma::vec mean_betat(2); mean_betat.fill(0.00);
if(last_time_ind == 0){
  
  cov_betat = inv_sympd((trans(gamma_t)*gamma_t)/sigma2_epsilon + V_inv_old + Omega*(V_inv_old*Omega));

  mean_betat = cov_betat*((trans(gamma_t)*(y_t - mean_temp_t))/sigma2_epsilon + V_inv_old*(Omega*betat_previous) + Omega*(V_inv_old*betat_next));
  
  }

if(last_time_ind == 1){
  
  cov_betat = inv_sympd((trans(gamma_t)*gamma_t)/sigma2_epsilon + V_inv_old);
  
  mean_betat = cov_betat*((trans(gamma_t)*(y_t - mean_temp_t))/sigma2_epsilon + V_inv_old*(Omega*betat_previous));
  
  }

arma::mat ind_norms = arma::randn(1,2);
arma::vec betat_t = mean_betat + 
                    trans(ind_norms*arma::chol(cov_betat));

return(betat_t);

}

  
  

  



