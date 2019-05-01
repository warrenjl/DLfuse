#include "RcppArmadillo.h"
#include "DLfuse.h"
using namespace arma;
using namespace Rcpp;

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]

Rcpp::List ppd_st(Rcpp::List modeling_output,
                  int n_pred,
                  int m_pred,
                  arma::mat z_pred,
                  arma::vec sample_size_pred,
                  arma::mat spatial_dists_full,
                  arma::mat neighbors_full,
                  arma::vec inference_set,
                  Rcpp::Nullable<int> params_only_indicator = R_NilValue,
                  Rcpp::Nullable<int> model_type_indicator = R_NilValue){
  
//params_only_indicator = 0: Predictions of the Outcome and Parameters are Provided
//params_only_indicator = 1: Only Predictions of the Parameters are Provided
int params_only = 0;
if(params_only_indicator.isNotNull()){
   params_only = Rcpp::as<int>(params_only_indicator);
   }
  
//model_type_indicator = 0: Full, Distributed Lag Model
//model_type_indicator = 1: No Distributed Lags, Original Model
//model_type_indicator = 2: Ordinary Kriging Model
//model_type_indicator = 3: Simple Linear Regression
int model_type = 0;
if(model_type_indicator.isNotNull()){
  model_type = Rcpp::as<int>(model_type_indicator);
  }
  
arma::vec sigma2_epsilon = modeling_output[0];
arma::vec beta0 = modeling_output[1];
arma::vec beta1 = modeling_output[2];
Rcpp::List betat = modeling_output[3];
Rcpp::List V = modeling_output[4];
arma::vec rho1 = modeling_output[5];
arma::vec rho2 = modeling_output[6];
arma::vec A11 = modeling_output[7];
arma::vec A22 = modeling_output[8];
arma::vec A21 = modeling_output[9];
arma::mat w0 = modeling_output[10];
arma::vec phi0 = modeling_output[11];
arma::mat w1 = modeling_output[12];
arma::vec phi1 = modeling_output[13];
Rcpp::List lag_info = modeling_output[14];
arma::vec mu = lag_info[0];
arma::mat mut = lag_info[1];
arma::vec sigma2_delta = lag_info[2];
arma::vec rho3 = lag_info[3];
arma::mat alpha = lag_info[4];
arma::vec tau2 = lag_info[5];

int inference_samples = inference_set.size();
int d = mut.n_rows;
arma::mat y_pred(n_pred, inference_samples); y_pred.fill(0.00);
arma::mat intercepts_pred(n_pred, inference_samples); intercepts_pred.fill(0.00);
arma::mat slopes_pred(n_pred, inference_samples); slopes_pred.fill(0.00);
arma::mat lags_pred(m_pred, inference_samples); lags_pred.fill(0.00);

arma::uvec keep7(7); keep7(0) = 0; keep7(1) = 1; keep7(2) = 2; keep7(3) = 3; keep7(4) = 4; keep7(5) = 5; keep7(6) = 6;
arma::mat neighbors_full_12 = neighbors_full.submat(0, m_pred, (m_pred - 1), (neighbors_full.n_cols - 1)); 
int n_model = w0.n_rows;
int m_model = (neighbors_full.n_cols - m_pred);
double negative_infinity = -std::numeric_limits<double>::infinity();

arma::mat AQS_key_mat_pred(n_pred, n_pred); AQS_key_mat_pred.eye();
arma::vec CMAQ_key_pred = regspace(1, 1, m_pred);

for(int i = 0; i < inference_samples;  ++ i){
  
   arma::vec w0_pred(n_pred); w0_pred.fill(0.00);
   if(model_type != 3){
     
     //w0  
     arma::mat Sigma0_full = exp(-phi0(inference_set(i) - 1)*spatial_dists_full);
     arma::mat Sigma0_11 = Sigma0_full.submat(0, 0, (n_pred - 1), (n_pred - 1));
     arma::mat Sigma0_22 = Sigma0_full.submat(n_pred, n_pred, (n_model + n_pred - 1), (n_model + n_pred - 1));
     arma::mat Sigma0_22_inv = inv_sympd(Sigma0_22);
     arma::mat Sigma0_12 = Sigma0_full.submat(0, n_pred, (n_pred - 1), (n_model + n_pred - 1));
      
     arma::vec w0_pred_mean = Sigma0_12*(Sigma0_22_inv*w0.col(inference_set(i) - 1));
     arma::mat w0_pred_cov = Sigma0_11 - 
                             Sigma0_12*(Sigma0_22_inv*trans(Sigma0_12));

     for(int j = 0; j < n_pred; ++ j){
       
        if(sum(spatial_dists_full.row(j) == 0.00) == 1){
          w0_pred(j) = R::rnorm(w0_pred_mean(j),
                                sqrt(w0_pred_cov(j,j)));
          }
       
        if(sum(spatial_dists_full.row(j) == 0.00) > 1){
          w0_pred(j) = w0_pred_mean(j);
          }
       
        }
     
     }
   
   arma::vec w1_pred(n_pred); w1_pred.fill(0.00);
   if(model_type == 0 || model_type == 1){

     //w1      
     arma::mat Sigma1_full = exp(-phi1(inference_set(i) - 1)*spatial_dists_full);
     arma::mat Sigma1_11 = Sigma1_full.submat(0, 0, (n_pred - 1), (n_pred - 1));
     arma::mat Sigma1_22 = Sigma1_full.submat(n_pred, n_pred, (n_model + n_pred - 1), (n_model + n_pred - 1));
     arma::mat Sigma1_22_inv = inv_sympd(Sigma1_22);
     arma::mat Sigma1_12 = Sigma1_full.submat(0, n_pred, (n_pred - 1), (n_model + n_pred - 1));

     arma::vec w1_pred_mean = Sigma1_12*(Sigma1_22_inv*w1.col(inference_set(i) - 1));
     arma::mat w1_pred_cov = Sigma1_11 - 
                             Sigma1_12*(Sigma1_22_inv*trans(Sigma1_12));

     for(int j = 0; j < n_pred; ++ j){
       
        if(sum(spatial_dists_full.row(j) == 0.00) == 1){
          w1_pred(j) = R::rnorm(w1_pred_mean(j),
                                sqrt(w1_pred_cov(j,j)));
          }
       
        if(sum(spatial_dists_full.row(j) == 0.00) > 1){
          w1_pred(j) = w1_pred_mean(j);
          }
       
        }
     
     }
   
   //betat
   arma::mat betat_iter = betat[inference_set(i) - 1];
   
   arma::mat Omega(2,2); Omega.fill(0.00);
   Omega(0,0) = rho1(inference_set(i) - 1);
   Omega(1,1) = rho2(inference_set(i) - 1);
   arma::vec betat_pred_mean = Omega*betat_iter.col(d-1);
   arma::mat betat_pred_cov = V[inference_set(i) - 1];
   arma::mat ind_norms = arma::randn(1,2);
   arma::vec betat_pred = betat_pred_mean + 
                          trans(ind_norms*arma::chol(betat_pred_cov));
   betat_pred = betat_iter.col(2);
   
   intercepts_pred.col(i).fill(beta0(inference_set(i) - 1) + betat_pred(0));
     
   if(model_type != 3){
     intercepts_pred.col(i) = intercepts_pred.col(i) +
                              A11(inference_set(i) - 1)*w0_pred;
     }
   
   if(model_type == 3){
     slopes_pred.col(i).fill(beta1(inference_set(i) - 1) + betat_pred(1));
     }
   
   if(model_type == 0 || model_type == 1){
     
     slopes_pred.col(i).fill(beta1(inference_set(i) - 1) + betat_pred(1));
     slopes_pred.col(i) = slopes_pred.col(i) +
                          A21(inference_set(i) - 1)*w0_pred +
                          A22(inference_set(i) - 1)*w1_pred;
   
     }
     
   double mut_pred = negative_infinity;
   arma::vec alpha_pred(m_pred); alpha_pred.fill(0.00);
   if(model_type == 0){
     
     //mut, alpha
     double mut_pred_mean = rho3(inference_set(i) - 1)*mut((d-1), (inference_set(i) - 1));
     double mut_pred = R::rnorm(mut_pred_mean,
                                sqrt(sigma2_delta(inference_set(i) - 1)));
     
     for(int j = 0; j < m_pred; ++ j){
              
        if(arma::is_finite(sum(neighbors_full.row(j))) == 0){ 
          
          for(int k = 0; k < m_model; ++ k){
          
             if(arma::is_finite(neighbors_full_12(j,k)) == 0){
               alpha_pred(j) = alpha(k, (inference_set(i) - 1));  
               }
             
             }
          
          }
              
        if(arma::is_finite(sum(neighbors_full.row(j))) == 1){
          
          double alpha_pred_mean = dot(neighbors_full_12.row(j), alpha.col(inference_set(i) - 1))/sum(neighbors_full_12.row(j));
          double alpha_pred_var = tau2(inference_set(i) - 1)/sum(neighbors_full_12.row(j));
                
          alpha_pred(j) = R::rnorm(alpha_pred_mean,
                                   sqrt(alpha_pred_var));
          
          }
        
        }
     
     lags_pred.col(i) = mu(inference_set(i) - 1) + 
                        mut_pred + 
                        alpha_pred;
     
     }

   if(params_only == 0){
     
     Rcpp::List lagged_covars = construct_lagged_covars_st(z_pred,
                                                           mu(inference_set(i) - 1),
                                                           mut_pred,
                                                           alpha_pred,
                                                           sample_size_pred,
                                                           CMAQ_key_pred);
     arma::vec lc1 = lagged_covars[0];
      
     //Predictions
     arma::vec mean_temp = construct_mean_st(beta0(inference_set(i) - 1), 
                                             beta1(inference_set(i) - 1),
                                             betat_pred,
                                             A11(inference_set(i) - 1),
                                             A22(inference_set(i) - 1),
                                             A21(inference_set(i) - 1),
                                             w0_pred,
                                             w1_pred,
                                             diagmat(lc1),
                                             keep7,
                                             sample_size_pred,
                                             AQS_key_mat_pred);
                
     for(int j = 0; j < n_pred; ++ j){
        y_pred(j,i) = R::rnorm(mean_temp(j),
                               sqrt(sigma2_epsilon(inference_set(i) - 1)));
        }   
     
     }
   
   //Progress
   if((i + 1) % 10 == 0){ 
     Rcpp::checkUserInterrupt();
     }
   
   if(((i + 1) % int(round(inference_samples*0.05)) == 0)){
     double completion = round(100*((i + 1)/(double)inference_samples));
     Rcpp::Rcout << "Progress: " << completion << "%" << std::endl;
     Rcpp::Rcout << "**************" << std::endl;
     }
    
   }
    
return Rcpp::List::create(Rcpp::Named("y_pred") = y_pred,
                          Rcpp::Named("intercepts_pred") = intercepts_pred,
                          Rcpp::Named("slopes_pred") = slopes_pred,
                          Rcpp::Named("lags_pred") = lags_pred);

}