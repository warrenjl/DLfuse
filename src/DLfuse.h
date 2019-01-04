#ifndef __DLfuse__
#define __DLfuse__

Rcpp::List spatial_corr_fun(double phi,
                            arma::mat spatial_dists);

Rcpp::List construct_lagged_covars(arma::mat z,
                                   double mu, 
                                   arma::vec alpha,
                                   arma::vec sample_size);

arma::vec construct_mean(double beta0, 
                         double beta1,
                         double A11,
                         double A22,
                         double A21,
                         arma::vec w0,
                         arma::vec w1,
                         arma::mat Omega,
                         arma::uvec keep,
                         arma::vec sample_size);

double sigma2_epsilon_update(arma::vec y,
                             arma::vec mean_temp,
                             arma::vec sample_size,
                             double alpha_sigma2_epsilon,
                             double beta_sigma2_epsilon);

double beta0_update(arma::vec y,
                    arma::vec mean_temp, 
                    double sigma2_epsilon,
                    arma::vec sample_size,
                    double sigma2_beta);

double beta1_update(arma::vec y,
                    arma::vec mean_temp,
                    Rcpp::List lagged_covars,
                    double sigma2_epsilon,
                    arma::vec sample_size,
                    double sigma2_beta);

Rcpp::List A11_update(arma::vec y,
                      double A11_old,
                      double sigma2_epsilon,
                      double beta0,
                      double beta1,
                      double A22_old,
                      double A21_old,
                      arma::vec w0_old,
                      arma::vec w1_old,
                      Rcpp::List lagged_covars,
                      arma::uvec keep5,
                      arma::vec sample_size,
                      double sigma2_A,
                      double metrop_var_A11_trans,
                      int acctot_A11_trans);

Rcpp::List A22_update(arma::vec y,
                      double A22_old,
                      double sigma2_epsilon,
                      double beta0,
                      double beta1,
                      double A11,
                      double A21_old,
                      arma::vec w0_old,
                      arma::vec w1_old,
                      Rcpp::List lagged_covars,
                      arma::uvec keep5,
                      arma::vec sample_size,
                      double sigma2_A,
                      double metrop_var_A22_trans,
                      int acctot_A22_trans);

double A21_update(arma::vec y,
                  arma::vec mean_temp,
                  Rcpp::List lagged_covars,
                  double sigma2_epsilon,
                  arma::vec w0_old,
                  double sigma2_A);

double tau2_update(int G,
                   arma::mat CAR,
                   arma::vec alpha,
                   double alpha_tau2,
                   double beta_tau2);

arma::vec w0_update(arma::vec y,
                    arma::vec mean_temp,
                    Rcpp::List lagged_covars,
                    double sigma2_epsilon,
                    double A11,
                    double A21,
                    arma::mat Sigma0_inv);

arma::vec w1_update(arma::vec y,
                    arma::vec mean_temp,
                    Rcpp::List lagged_covars,
                    double sigma2_epsilon,
                    double A22,
                    arma::mat Sigma1_inv);

Rcpp::List phi_update(double phi_old,
                      arma::mat spatial_dists,
                      arma::vec w,
                      Rcpp::List spatial_corr_info,
                      double alpha_phi,
                      double beta_phi,
                      double metrop_var_phi_trans,
                      int acctot_phi_trans);

double neg_two_loglike_update(arma::vec y,
                              arma::vec mean_temp,
                              double sigma2_epsilon);

arma::vec theta_update(arma::mat x, 
                       arma::mat z,
                       arma::vec site_id,
                       arma::vec w,
                       arma::vec gamma,
                       arma::vec beta,
                       arma::mat eta_old,
                       double sigma2_theta_old,
                       arma::mat corr_inv);

double sigma2_theta_update(arma::vec theta,
                           arma::mat corr_inv,
                           double alpha_sigma2_theta,
                           double beta_sigma2_theta);

arma::mat eta_update(arma::mat eta_old,
                     arma::mat x, 
                     arma::mat z,
                     arma::vec site_id,
                     arma::mat neighbors,
                     arma::vec w,
                     arma::vec gamma,
                     arma::vec beta,
                     arma::vec theta,
                     double rho_old,
                     double sigma2_eta_old,
                     arma::mat corr_inv);

Rcpp::List rho_update(double rho_old,
                      arma::mat neighbors,
                      arma::mat eta,
                      double sigma2_eta_old,
                      arma::mat corr_inv,
                      double a_rho,
                      double b_rho,
                      double metrop_var_rho_trans,
                      int acctot_rho_trans);

double sigma2_eta_update(arma::mat neighbors,
                         arma::mat eta,
                         double rho,
                         arma::mat corr_inv,
                         double alpha_sigma2_eta,
                         double beta_sigma2_eta);

Rcpp::List phi_update(double phi_old,
                      arma::mat neighbors,
                      arma::vec theta,
                      double sigma2_theta,
                      arma::mat eta,
                      double rho,
                      double sigma2_eta,
                      Rcpp::List temporal_corr_info,
                      double a_phi,
                      double b_phi,
                      double metrop_var_phi_trans,
                      int acctot_phi_trans);

double neg_two_loglike_update(arma::vec y,
                              arma::mat x,
                              arma::mat z,
                              arma::vec site_id,
                              arma::vec beta,
                              arma::vec theta,
                              arma::mat eta);

Rcpp::List DLfuse(int mcmc_samples,
                  arma::vec y,
                  arma::mat x,
                  arma::mat z,
                  arma::vec site_id,
                  arma::mat neighbors,
                  double metrop_var_rho_trans,
                  double metrop_var_phi_trans,
                  Rcpp::Nullable<double> sigma2_beta_prior,
                  Rcpp::Nullable<double> alpha_sigma2_theta_prior,
                  Rcpp::Nullable<double> beta_sigma2_theta_prior,
                  Rcpp::Nullable<double> a_rho_prior,
                  Rcpp::Nullable<double> b_rho_prior,
                  Rcpp::Nullable<double> alpha_sigma2_eta_prior,
                  Rcpp::Nullable<double> beta_sigma2_eta_prior,
                  Rcpp::Nullable<double> a_phi_prior,
                  Rcpp::Nullable<double> b_phi_prior,
                  Rcpp::Nullable<Rcpp::NumericVector> beta_init,
                  Rcpp::Nullable<Rcpp::NumericVector> theta_init,
                  Rcpp::Nullable<double> sigma2_theta_init,
                  Rcpp::Nullable<Rcpp::NumericMatrix> eta_init,
                  Rcpp::Nullable<double> rho_init,
                  Rcpp::Nullable<double> sigma2_eta_init,
                  Rcpp::Nullable<double> phi_init,
                  Rcpp::Nullable<int> rho_zero_indicator); 

#endif // __DFfuse__
