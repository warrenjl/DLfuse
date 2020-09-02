#include "RcppArmadillo.h"
#include "DLfuse.h"
using namespace arma;
using namespace Rcpp;

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]

Rcpp::List alpha_update_s(arma::vec y,
                          arma::mat z,
                          arma::mat neighbors,
                          arma::vec alpha_old,
                          Rcpp::List lagged_covars,
                          double sigma2_epsilon,
                          double beta0,
                          double beta1,
                          double A11,
                          double A22,
                          double A21,
                          double mu,
                          double tau2_old,
                          arma::vec w0_old,
                          arma::vec w1_old,
                          arma::uvec keep5,
                          arma::vec sample_size,
                          arma::vec metrop_var_alpha,
                          arma::vec acctot_alpha,
                          int weights_definition){

int n = y.size();  
int m = alpha_old.size();
arma::vec alpha = alpha_old;
for(int j = 0; j < m; ++ j){
  
   /*Second*/
   Rcpp::List lagged_covars_old = lagged_covars;
   arma::vec lc1_old = lagged_covars_old[0];
   
   arma::vec mean_temp_old = construct_mean_s(beta0, 
                                              beta1,
                                              A11,
                                              A22,
                                              A21,
                                              w0_old,
                                              w1_old,
                                              diagmat(lc1_old),
                                              keep5,
                                              sample_size);

   double second = 0.00;
   for(int k = 0; k < n; ++ k){
      second = second + 
               R::dnorm(y(k),
                        mean_temp_old(k),
                        sqrt(sigma2_epsilon),
                        1);
      }
   second = second +
            R::dnorm(alpha_old(j),
                     (dot(neighbors.row(j), alpha)/sum(neighbors.row(j))),
                     sqrt(tau2_old/sum(neighbors.row(j))),
                     1);

   /*First*/
   alpha(j) = R::rnorm(alpha_old(j), 
                       sqrt(metrop_var_alpha(j)));
   lagged_covars = construct_lagged_covars_s(z,
                                             mu, 
                                             alpha,
                                             sample_size,
                                             weights_definition);
   arma::vec lc1 = lagged_covars(0);
   
   arma::vec mean_temp = construct_mean_s(beta0, 
                                          beta1,
                                          A11,
                                          A22,
                                          A21,
                                          w0_old,
                                          w1_old,
                                          diagmat(lc1),
                                          keep5,
                                          sample_size);

  double first = 0.00;
  for(int k = 0; k < n; ++ k){
     first = first + 
             R::dnorm(y(k),
                      mean_temp(k),
                      sqrt(sigma2_epsilon),
                      1);
     }
  first = first +
          R::dnorm(alpha(j),
                   (dot(neighbors.row(j), alpha)/sum(neighbors.row(j))),
                   sqrt(tau2_old/sum(neighbors.row(j))),
                   1);

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

alpha = (alpha - mean(alpha))/stddev(alpha);  //Centering-on-the-Fly (ICAR) + \Phi(.) stabilization

lagged_covars = construct_lagged_covars_s(z,
                                          mu, 
                                          alpha,
                                          sample_size,
                                          weights_definition);

return Rcpp::List::create(Rcpp::Named("alpha") = alpha,
                          Rcpp::Named("acctot_alpha") = acctot_alpha,
                          Rcpp::Named("lagged_covars") = lagged_covars);

}
  
