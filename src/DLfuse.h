#ifndef __DLfuse__
#define __DLfuse__

//Functions Common to the Spatial and Spatiotemporal Models
Rcpp::List spatial_corr_fun(double phi,
                            arma::mat spatial_dists);

double tau2_update(int G,
                   arma::mat CAR,
                   arma::vec alpha,
                   double alpha_tau2,
                   double beta_tau2);

Rcpp::List phi_update(double phi_old,
                      arma::mat spatial_dists,
                      arma::vec w,
                      Rcpp::List spatial_corr_info,
                      double alpha_phi,
                      double beta_phi,
                      double metrop_var_phi_trans,
                      int acctot_phi_trans);

//Functions Specific to the Spatial Model
Rcpp::List construct_lagged_covars_s(arma::mat z,
                                     double mu, 
                                     arma::vec alpha,
                                     arma::vec sample_size,
                                     int weights_indicator);

arma::vec construct_mean_s(double beta0, 
                           double beta1,
                           double A11,
                           double A22,
                           double A21,
                           arma::vec w0,
                           arma::vec w1,
                           arma::mat Omega,
                           arma::uvec keep,
                           arma::vec sample_size);

double sigma2_epsilon_update_s(arma::vec y,
                               arma::vec mean_temp,
                               arma::vec sample_size,
                               double alpha_sigma2_epsilon,
                               double beta_sigma2_epsilon);

double beta0_update_s(arma::vec y,
                      arma::vec mean_temp, 
                      double sigma2_epsilon,
                      arma::vec sample_size,
                      double sigma2_beta);

double beta1_update_s(arma::vec y,
                      arma::vec mean_temp,
                      Rcpp::List lagged_covars,
                      double sigma2_epsilon,
                      arma::vec sample_size,
                      double sigma2_beta);

Rcpp::List A11_update_s(arma::vec y,
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

Rcpp::List A22_update_s(arma::vec y,
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

double A21_update_s(arma::vec y,
                    arma::vec mean_temp,
                    Rcpp::List lagged_covars,
                    double sigma2_epsilon,
                    arma::vec w0_old,
                    double sigma2_A);

Rcpp::List mu_update_s(arma::vec y,
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
                       double metrop_var_mu,
                       int acctot_mu,
                       int weights_definition);

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
                          int weights_definition);

arma::vec w0_update_s(arma::vec y,
                      arma::vec mean_temp,
                      Rcpp::List lagged_covars,
                      double sigma2_epsilon,
                      double A11,
                      double A21,
                      arma::mat Sigma0_inv);

arma::vec w1_update_s(arma::vec y,
                      arma::vec mean_temp,
                      Rcpp::List lagged_covars,
                      double sigma2_epsilon,
                      double A22,
                      arma::mat Sigma1_inv);

double neg_two_loglike_update_s(arma::vec y,
                                arma::vec mean_temp,
                                double sigma2_epsilon);

Rcpp::List DLfuse_s(int mcmc_samples,
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
                    Rcpp::Nullable<int> weights_definition_indicator,
                    Rcpp::Nullable<int> model_type_indicator); 

Rcpp::List ppd_s(Rcpp::List modeling_output,
                 int n_pred,
                 int m_pred,
                 arma::mat z_pred,
                 arma::vec sample_size_pred,
                 arma::mat spatial_dists_full,
                 arma::mat neighbors_full,
                 arma::vec inference_set,
                 Rcpp::Nullable<int> params_only_indicator,
                 Rcpp::Nullable<int> weights_definition_indicator,
                 Rcpp::Nullable<int> model_type_indicator);

//Functions Specific to the Spatiotemporal Model
Rcpp::List construct_lagged_covars_st(arma::mat z_t,
                                      double mu,
                                      double mut_t,
                                      arma::vec alpha,
                                      arma::vec sample_size_t,
                                      arma::vec CMAQ_key_t,
                                      int weights_definition);

arma::vec construct_mean_st(double beta0, 
                            double beta1,
                            arma::vec betat_t,
                            double A11,
                            double A22,
                            double A21,
                            arma::vec w0,
                            arma::vec w1,
                            arma::mat Omega_t,
                            arma::uvec keep,
                            arma::vec sample_size_t,
                            arma::mat AQS_key_mat_t);

double sigma2_epsilon_update_st(Rcpp::List y,
                                Rcpp::List mean_temp,
                                double alpha_sigma2_epsilon,
                                double beta_sigma2_epsilon);

double beta0_update_st(Rcpp::List y,
                       Rcpp::List mean_temp, 
                       double sigma2_epsilon,
                       double sigma2_beta);

double beta1_update_st(Rcpp::List y,
                       Rcpp::List mean_temp,
                       Rcpp::List lagged_covars,
                       double sigma2_epsilon,
                       Rcpp::List sample_size,
                       double sigma2_beta);

arma::vec betat_update_st(int last_time_ind,
                          arma::vec y_t,
                          arma::vec mean_temp_t,
                          Rcpp::List lagged_covars_t,
                          double sigma2_epsilon,
                          arma::vec betat_previous,
                          arma::vec betat_next,
                          arma::mat V_old,
                          double rho1_old,
                          double rho2_old);

arma::mat V_update_st(arma::mat betat,
                      double rho1_old,
                      double rho2_old,
                      arma::mat Omega_V_inv,
                      double nu_v_inv);

Rcpp::List rho1_update_st(double rho1_old,
                          arma::mat betat,
                          arma::mat V,
                          double rho2_old,
                          double a_rho1,
                          double b_rho1,
                          double metrop_var_rho1_trans,
                          int acctot_rho1_trans);

Rcpp::List rho2_update_st(double rho2_old,
                          arma::mat betat,
                          arma::mat V,
                          double rho1,
                          double a_rho2,
                          double b_rho2,
                          double metrop_var_rho2_trans,
                          int acctot_rho2_trans);

Rcpp::List A11_update_st(Rcpp::List y,
                         Rcpp::List AQS_key_mat,
                         double A11_old,
                         Rcpp::List lagged_covars,
                         double sigma2_epsilon,
                         double beta0,
                         double beta1,
                         arma::mat betat,
                         double A22_old,
                         double A21_old,
                         arma::vec w0_old,
                         arma::vec w1_old,
                         arma::uvec keep7,
                         Rcpp::List sample_size,
                         double sigma2_A,
                         double metrop_var_A11_trans,
                         int acctot_A11_trans);

Rcpp::List A22_update_st(Rcpp::List y,
                         Rcpp::List AQS_key_mat,
                         double A22_old,
                         Rcpp::List lagged_covars,
                         double sigma2_epsilon,
                         double beta0,
                         double beta1,
                         arma::mat betat,
                         double A11,
                         double A21_old,
                         arma::vec w0_old,
                         arma::vec w1_old,
                         arma::uvec keep7,
                         Rcpp::List sample_size,
                         double sigma2_A,
                         double metrop_var_A22_trans,
                         int acctot_A22_trans);

double A21_update_st(Rcpp::List y,
                     Rcpp::List AQS_key_mat,
                     Rcpp::List mean_temp,
                     Rcpp::List lagged_covars,
                     double sigma2_epsilon,
                     arma::vec w0_old,
                     double sigma2_A);

Rcpp::List mu_update_st(Rcpp::List y,
                        Rcpp::List z,
                        double mu_old,
                        Rcpp::List lagged_covars,
                        double sigma2_epsilon,
                        double beta0,
                        double beta1,
                        arma::mat betat,
                        double A11,
                        double A22,
                        double A21,
                        arma::vec mut_old,
                        arma::vec alpha_old,
                        arma::vec w0_old,
                        arma::vec w1_old,
                        arma::uvec keep7,
                        Rcpp::List sample_size,
                        Rcpp::List AQS_key_mat,
                        Rcpp::List CMAQ_key,
                        double metrop_var_mu,
                        int acctot_mu,
                        int weights_definition);

Rcpp::List mut_update_st(int last_time_ind,
                         arma::vec y_t,
                         arma::mat z_t,
                         double mut_t_old,
                         Rcpp::List lagged_covars_t,
                         double sigma2_epsilon,
                         double beta0,
                         double beta1,
                         arma::vec betat_t,
                         double A11,
                         double A22,
                         double A21,
                         double mu,
                         double mut_previous,
                         double mut_next,
                         double sigma2_delta_old,
                         double rho3_old,
                         arma::vec alpha_old,
                         arma::vec w0_old,
                         arma::vec w1_old,
                         arma::uvec keep7,
                         arma::vec sample_size_t,
                         arma::mat AQS_key_mat_t,
                         arma::vec CMAQ_key_t,
                         double metrop_var_mut_t,
                         int acctot_mut_t,
                         int weights_definition);

double sigma2_delta_update_st(arma::vec mut,
                              double rho3_old,
                              double alpha_sigma2_delta,
                              double beta_sigma2_delta);

Rcpp::List rho3_update_st(double rho3_old,
                          arma::vec mut,
                          double sigma2_delta,
                          double a_rho3,
                          double b_rho3,
                          double metrop_var_rho3_trans,
                          int acctot_rho3_trans);

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
                           Rcpp::List AQS_key_mat,
                           Rcpp::List CMAQ_key,
                           arma::vec metrop_var_alpha,
                           arma::vec acctot_alpha,
                           int weights_definition);

arma::vec w0_update_st(Rcpp::List y,
                       Rcpp::List AQS_key_mat,
                       Rcpp::List mean_temp,
                       Rcpp::List lagged_covars,
                       double sigma2_epsilon,
                       double A11,
                       double A21,
                       arma::mat Sigma0_inv);

arma::vec w1_update_st(Rcpp::List y,
                       Rcpp::List AQS_key_mat,
                       Rcpp::List mean_temp,
                       Rcpp::List lagged_covars,
                       double sigma2_epsilon,
                       double A22,
                       arma::mat Sigma1_inv);

double neg_two_loglike_update_st(Rcpp::List y,
                                 Rcpp::List mean_temp,
                                 double sigma2_epsilon);

Rcpp::List DLfuse_st(int mcmc_samples,
                     Rcpp::List y,
                     Rcpp::List z,
                     Rcpp::List sample_size,
                     Rcpp::List AQS_key,
                     Rcpp::List CMAQ_key,
                     arma::mat spatial_dists,
                     int AQS_unique_total,
                     arma::mat neighbors,
                     int CMAQ_unique_total,
                     double metrop_var_rho1_trans,  //Start of Metropolis Parameters
                     double metrop_var_rho2_trans,
                     double metrop_var_A11_trans,
                     double metrop_var_A22_trans,
                     double metrop_var_mu,
                     arma::vec metrop_var_mut,
                     double metrop_var_rho3_trans,
                     arma::vec metrop_var_alpha,
                     double metrop_var_phi0_trans,
                     double metrop_var_phi1_trans,
                     Rcpp::Nullable<double> alpha_sigma2_epsilon_prior,  //Start of Priors
                     Rcpp::Nullable<double> beta_sigma2_epsilon_prior,
                     Rcpp::Nullable<double> sigma2_beta_prior,
                     Rcpp::Nullable<Rcpp::NumericMatrix> Omega_V_inv_prior,
                     Rcpp::Nullable<double> nu_V_inv_prior,
                     Rcpp::Nullable<double> a_rho1_prior,
                     Rcpp::Nullable<double> b_rho1_prior,
                     Rcpp::Nullable<double> a_rho2_prior,
                     Rcpp::Nullable<double> b_rho2_prior,
                     Rcpp::Nullable<double> sigma2_A_prior,
                     Rcpp::Nullable<double> alpha_sigma2_delta_prior,
                     Rcpp::Nullable<double> beta_sigma2_delta_prior,
                     Rcpp::Nullable<double> a_rho3_prior,
                     Rcpp::Nullable<double> b_rho3_prior,
                     Rcpp::Nullable<double> alpha_tau2_prior,
                     Rcpp::Nullable<double> beta_tau2_prior,
                     Rcpp::Nullable<double> alpha_phi0_prior,
                     Rcpp::Nullable<double> beta_phi0_prior,
                     Rcpp::Nullable<double> alpha_phi1_prior,
                     Rcpp::Nullable<double> beta_phi1_prior,
                     Rcpp::Nullable<double> sigma2_epsilon_init,  //Start of Initial Values
                     Rcpp::Nullable<double> beta0_init,
                     Rcpp::Nullable<double> beta1_init,
                     Rcpp::Nullable<Rcpp::NumericMatrix> betat_init,
                     Rcpp::Nullable<Rcpp::NumericMatrix> V_init,
                     Rcpp::Nullable<double> rho1_init,
                     Rcpp::Nullable<double> rho2_init,
                     Rcpp::Nullable<double> A11_init,
                     Rcpp::Nullable<double> A22_init,
                     Rcpp::Nullable<double> A21_init,
                     Rcpp::Nullable<double> mu_init,
                     Rcpp::Nullable<Rcpp::NumericVector> mut_init,
                     Rcpp::Nullable<double> sigma2_delta_init,
                     Rcpp::Nullable<double> rho3_init,
                     Rcpp::Nullable<Rcpp::NumericVector> alpha_init,
                     Rcpp::Nullable<double> tau2_init,
                     Rcpp::Nullable<Rcpp::NumericVector> w0_init,
                     Rcpp::Nullable<double> phi0_init,
                     Rcpp::Nullable<Rcpp::NumericVector> w1_init,
                     Rcpp::Nullable<double> phi1_init,
                     Rcpp::Nullable<int> weights_definition_indicator,
                     Rcpp::Nullable<int> model_type_indicator);

Rcpp::List ppd_st(Rcpp::List modeling_output,
                  int n_pred,
                  int m_pred,
                  arma::mat z_pred,
                  arma::vec sample_size_pred,
                  arma::vec AQS_key_pred,
                  arma::vec CMAQ_key_pred,
                  arma::mat spatial_dists_full,
                  arma::mat neighbors_full,
                  arma::vec inference_set,
                  Rcpp::Nullable<int> params_only_indicator,
                  Rcpp::Nullable<int> weights_definition_indicator,
                  Rcpp::Nullable<int> model_type_indicator);

#endif // __DLfuse__