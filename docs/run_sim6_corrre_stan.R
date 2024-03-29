rm(list = ls())

library(lme4)
library(tidyverse)
library(cmdstanr)

set_cmdstan_path('/Users/nico/GROUNDMOTION/SOFTWARE/cmdstan-2.33.1')

dir_stan <- '/Users/nico/GROUNDMOTION/PROJECTS/RESID_VAR/Git/MixedModels_Biases/stan'
dir_res <- '/Users/nico/GROUNDMOTION/PROJECTS/RESID_VAR/STAN/RESULTS'
mod <- cmdstan_model(file.path(dir_stan, 'gmm_partition_corrfull.stan'))

data_reg <- read.csv(file.path('../data','data_cb.csv'))

eq <- data_reg$eq
stat <- data_reg$stat

n_rec <- nrow(data_reg)
n_eq <- max(eq)
n_stat <- max(stat)

# --------------------------------------------------
# correlaton
tau_sim1 <- 0.4
phi_s2s_sim1 <- 0.43
phi_ss_sim1 <- 0.5

tau_sim2 <- 0.45
phi_s2s_sim2 <- 0.4
phi_ss_sim2 <- 0.55

sigma_tot1 <- sqrt(tau_sim1^2 + phi_s2s_sim1^2 + phi_ss_sim1^2)
sigma_tot2 <- sqrt(tau_sim2^2 + phi_s2s_sim2^2 + phi_ss_sim2^2)

n_sam <- 5
for(cor_name in c('high','low')) {
  print(cor_name)
  if(cor_name == 'high') {
    rho_tau <- 0.95
    rho_ss <- 0.9
    rho_s2s <- 0.85
  } else {
    rho_tau <- 0.45
    rho_ss <- 0.5
    rho_s2s <- 0.55
  }
  
  rho_total <- (rho_tau * tau_sim1 * tau_sim2 + rho_s2s * phi_s2s_sim1 * phi_s2s_sim2 + rho_ss * phi_ss_sim1 * phi_ss_sim2) /
    (sigma_tot1 * sigma_tot2)
  
  cov_tau <- matrix(c(tau_sim1^2, rho_tau * tau_sim1 * tau_sim2,
                      rho_tau * tau_sim1 * tau_sim2, tau_sim2^2), ncol = 2)
  cov_s2s <- matrix(c(phi_s2s_sim1^2, rho_s2s * phi_s2s_sim1 * phi_s2s_sim2,
                      rho_s2s * phi_s2s_sim1 * phi_s2s_sim2, phi_s2s_sim2^2), ncol = 2)
  cov_ss <- matrix(c(phi_ss_sim1^2, rho_ss * phi_ss_sim1 * phi_ss_sim2,
                     rho_ss * phi_ss_sim1 * phi_ss_sim2, phi_ss_sim2^2), ncol = 2)
  

  set.seed(5618)
  for(i in 1:n_sam) {
    eqt2 <- mvtnorm::rmvnorm(n_eq, sigma = cov_tau)
    statt2 <- mvtnorm::rmvnorm(n_stat, sigma = cov_s2s)
    rect2 <- mvtnorm::rmvnorm(n_rec, sigma = cov_ss)
    
    data_reg$y_sim1 <- eqt2[eq,1] + statt2[stat,1] + rect2[,1]
    data_reg$y_sim2 <- eqt2[eq,2] + statt2[stat,2] + rect2[,2]
    
    data_list_cor <- list(
      N = n_rec,
      NEQ = n_eq,
      NSTAT = n_stat,
      NP = 2,
      Y = cbind(data_reg$y_sim1, data_reg$y_sim2),
      eq = eq,
      stat = stat
    )
    
    fit <- mod$sample(
      data = data_list_cor,
      seed = 8472,
      chains = 3,
      iter_sampling = 200,
      iter_warmup = 200,
      refresh = 100,
      max_treedepth = 10,
      adapt_delta = 0.8,
      parallel_chains = 3,
      show_exceptions = FALSE
    )
    fit$save_object(file.path(dir_res, sprintf('fit_corrfull_%s_%d.RDS', cor_name, i)))
  }
}
