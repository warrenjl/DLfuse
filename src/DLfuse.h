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
                      Rcpp::List lagged_covars,
                      double sigma2_epsilon,
                      double beta0,
                      double beta1,
                      double A22_old,
                      double A21_old,
                      arma::vec w0_old,
                      arma::vec w1_old,
                      arma::uvec keep5,
                      arma::vec sample_size,
                      double sigma2_A,
                      double metrop_var_A11_trans,
                      int acctot_A11_trans);

Rcpp::List A22_update(arma::vec y,
                      double A22_old,
                      Rcpp::List lagged_covars,
                      double sigma2_epsilon,
                      double beta0,
                      double beta1,
                      double A11,
                      double A21_old,
                      arma::vec w0_old,
                      arma::vec w1_old,
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

Rcpp::List mu_update(arma::vec y,
                     arma::mat z,
                     double mu_old,
                     Rcpp::List lagged_covars,
                     double sigma2_epsilon,
                     double beta0,
                     double beta1,
                     double A11,
                     double A22,
                     double A21,
                     arma::vec alpha_old,
                     arma::vec w0_old,
                     arma::vec w1_old,
                     arma::uvec keep5,
                     arma::vec sample_size,
                     double sigma2_mu,
                     double metrop_var_mu,
                     int acctot_mu);

Rcpp::List alpha_update(arma::vec y,
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
                        arma::vec acctot_alpha);

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

Rcpp::List DLfuse(int mcmc_samples,
                  arma::vec y,
                  arma::mat z,
                  arma::vec sample_size,
                  arma::mat spatial_dists,
                  arma::mat neighbors,
                  double metrop_var_A11_trans,
                  double metrop_var_A22_trans,
                  double metrop_var_mu,
                  arma::vec metrop_var_alpha,
                  double metrop_var_phi0_trans,
                  double metrop_var_phi1_trans,
                  Rcpp::Nullable<double> alpha_sigma2_epsilon_prior,
                  Rcpp::Nullable<double> beta_sigma2_epsilon_prior,
                  Rcpp::Nullable<double> sigma2_beta_prior,
                  Rcpp::Nullable<double> sigma2_A_prior,
                  Rcpp::Nullable<double> sigma2_mu_prior,
                  Rcpp::Nullable<double> alpha_tau2_prior,
                  Rcpp::Nullable<double> beta_tau2_prior,
                  Rcpp::Nullable<double> alpha_phi0_prior,
                  Rcpp::Nullable<double> beta_phi0_prior,
                  Rcpp::Nullable<double> alpha_phi1_prior,
                  Rcpp::Nullable<double> beta_phi1_prior,
                  Rcpp::Nullable<double> sigma2_epsilon_init,
                  Rcpp::Nullable<double> beta0_init,
                  Rcpp::Nullable<double> beta1_init,
                  Rcpp::Nullable<double> A11_init,
                  Rcpp::Nullable<double> A22_init,
                  Rcpp::Nullable<double> A21_init,
                  Rcpp::Nullable<double> mu_init,
                  Rcpp::Nullable<Rcpp::NumericVector> alpha_init,
                  Rcpp::Nullable<double> tau2_init,
                  Rcpp::Nullable<Rcpp::NumericVector> w0_init,
                  Rcpp::Nullable<double> phi0_init,
                  Rcpp::Nullable<Rcpp::NumericVector> w1_init,
                  Rcpp::Nullable<double> phi1_init,
                  Rcpp::Nullable<int> collapse_indicator); 

#endif // __DLfuse__
