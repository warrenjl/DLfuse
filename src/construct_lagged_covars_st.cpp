#include "RcppArmadillo.h"
#include "DLfuse.h"
using namespace arma;
using namespace Rcpp;

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]

Rcpp::List construct_lagged_covars_st(arma::mat z_t,
                                      double mu,
                                      double mut_t,
                                      arma::vec alpha,
                                      arma::vec sample_size_t,
                                      arma::vec CMAQ_key,
                                      int weights_definition){
  
int L = z_t.n_cols;
int m_t = z_t.n_rows;
int n_t = sum(sample_size_t);
arma::mat regression_weights(m_t, L); regression_weights.fill(0.00);

if(weights_definition == 0){
  for(int j = 0; j < m_t; ++ j){
  
     for(int k = 0; k < L; ++ k){
      regression_weights(j,k) = pow((R::pnorm(mu + mut_t + alpha(CMAQ_key(j) - 1), 0.00, 1.00, 1, 0)), k);
      }
  
     regression_weights.row(j) = regression_weights.row(j)/sum(regression_weights.row(j));
  
     }
  }

if(weights_definition == 1){
  for(int j = 0; j < m_t; ++ j){
    
     for(int k = 0; k < L; ++ k){
        
        double phi_temp = exp(mu + mut_t + alpha(CMAQ_key(j) - 1));
        regression_weights(j,k) = (1.00 - 1.50*(k/phi_temp) + 0.50*pow((k/phi_temp), 3.00))*(phi_temp > k);
       
        }
    
     regression_weights.row(j) = regression_weights.row(j)/sum(regression_weights.row(j));
    
     }
  }

arma::vec lagged_covars_reduced(m_t); lagged_covars_reduced.fill(0.00);
arma::mat temp_mat = z_t%regression_weights;
for(int j = 0; j < m_t; ++ j){
   
   lagged_covars_reduced(j) = sum(temp_mat.row(j));
   
   }

arma::vec lagged_covars_full(n_t); lagged_covars_full.fill(0.00);
int counter = 0;
for(int j = 0; j < m_t; ++ j){
  
   int ss = sample_size_t(j);
  
   for(int k = 0; k < ss; ++ k){
  
      lagged_covars_full(counter) = lagged_covars_reduced(j);
      ++ counter;
  
      }
  
   }
    
return Rcpp::List::create(Rcpp::Named("lagged_covars_full") = lagged_covars_full,
                          Rcpp::Named("lagged_covars_reduced") = lagged_covars_reduced);

}



