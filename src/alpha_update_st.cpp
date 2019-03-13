#include "RcppArmadillo.h"
#include "DLfuse.h"
using namespace arma;
using namespace Rcpp;

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]

Rcpp::List alpha_update_st(Rcpp::List y,
                           Rcpp::List z,
                           arma::mat neighbors,
                           arma::vec alpha_old,
                           Rcpp::List lagged_covars,
                           double sigma2_epsilon,
                           double beta0,
                           double beta1,
                           arma::mat betat,
                           double A11,
                           double A22,
                           double A21,
                           double mu,
                           arma::vec mut,
                           double tau2_old,
                           arma::vec w0_old,
                           arma::vec w1_old,
                           arma::uvec keep7,
                           Rcpp::List sample_size,
                           Rcpp::List AQS_key,
                           Rcpp::List CMAQ_key,
                           arma::vec metrop_var_alpha,
                           arma::vec acctot_alpha){

int d = y.size();  
int m = alpha_old.size();
arma::vec alpha = alpha_old;
for(int j = 0; j < m; ++ j){
  
   /*Second*/
   Rcpp::List lagged_covars_old = lagged_covars;
   double second = 0.00;
   
   for(int k = 0; k < d; ++ k){
     
      Rcpp::List lagged_covars_old_t = lagged_covars_old[k];
      arma::vec lc1_old_t = lagged_covars_old_t[0];
      arma::vec betat_t = betat.col(k);
      
      arma::vec mean_temp_old_t = construct_mean_st(beta0, 
                                                    beta1,
                                                    betat_t,
                                                    A11,
                                                    A22,
                                                    A21,
                                                    w0_old,
                                                    w1_old,
                                                    diagmat(lc1_old_t),
                                                    keep7,
                                                    sample_size[k],
                                                    AQS_key[k]);

      arma::vec y_t = y[k];
      int ss_t = y_t.size();
      for(int l = 0; l < ss_t; ++ l){
        
         second = second + 
                  R::dnorm(y_t(l),
                           mean_temp_old_t(l),
                           sqrt(sigma2_epsilon),
                           1);
        
         }
      second = second +
               R::dnorm(alpha_old(j),
                        (dot(neighbors.row(j), alpha)/sum(neighbors.row(j))),
                        sqrt(tau2_old/sum(neighbors.row(j))),
                        1);
      
      }
   
   /*First*/
   alpha(j) = R::rnorm(alpha_old(j), 
                       sqrt(metrop_var_alpha(j)));
   double first = 0.00;
   
   for(int k = 0; k < d; ++ k){
   
      lagged_covars[k] = construct_lagged_covars_st(z[k],
                                                    mu,
                                                    mut(k),
                                                    alpha,
                                                    sample_size[k],
                                                    CMAQ_key[k]);
      
      Rcpp::List lagged_covars_t = lagged_covars[k];
      arma::vec lc1_t = lagged_covars_t[0];
      arma::vec betat_t = betat.col(k);
      
      arma::vec mean_temp_t = construct_mean_st(beta0, 
                                                beta1,
                                                betat_t,
                                                A11,
                                                A22,
                                                A21,
                                                w0_old,
                                                w1_old,
                                                diagmat(lc1_t),
                                                keep7,
                                                sample_size[k],
                                                AQS_key[k]);
      
      arma::vec y_t = y[k];
      int ss_t = y_t.size();
      for(int l = 0; l < ss_t; ++ l){
        
         first = first + 
                 R::dnorm(y_t(l),
                          mean_temp_t(l),
                          sqrt(sigma2_epsilon),
                          1);
        
         }
     first = first +
             R::dnorm(alpha(j),
                      (dot(neighbors.row(j), alpha)/sum(neighbors.row(j))),
                      sqrt(tau2_old/sum(neighbors.row(j))),
                      1);
      
     }

   /*Decision*/
   double ratio = exp(first - second);   
   int acc = 1;
   if(ratio < R::runif(0.00, 1.00)){
     
     alpha(j) = alpha_old(j);
     lagged_covars = lagged_covars_old;
     acc = 0;
     
     }
   
  acctot_alpha(j) = acctot_alpha(j) + 
                    acc;
   
  } 

alpha = alpha -
        mean(alpha);  //Centering-on-the-Fly (ICAR)

for(int j = 0; j < d; ++ j){
  
   lagged_covars[j] = construct_lagged_covars_st(z[j],
                                                 mu,
                                                 mut(j),
                                                 alpha,
                                                 sample_size[j],
                                                 CMAQ_key[j]);
   
   }

return Rcpp::List::create(Rcpp::Named("alpha") = alpha,
                          Rcpp::Named("acctot_alpha") = acctot_alpha,
                          Rcpp::Named("lagged_covars") = lagged_covars);

}
  
