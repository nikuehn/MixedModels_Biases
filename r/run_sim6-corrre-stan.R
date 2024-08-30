rm(list = ls())

library(cmdstanr)
library(posterior)
library(tidyverse)

set_cmdstan_path('/Users/nico/GROUNDMOTION/SOFTWARE/cmdstan-2.35.0')

data_reg <- read.csv(file.path('../data','data_cb.csv'))
mod <- cmdstan_model(file.path('../stan/gmm_partition_corrre_cond.stan'))

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
  
  n_sam <- 50
  mat_cor_stan <- matrix(nrow = n_sam, ncol =  6)
  set.seed(5618)
  for(i in 1:n_sam) {
    print(paste0('i = ',i))
    eqt2 <- mvtnorm::rmvnorm(n_eq, sigma = cov_tau)
    statt2 <- mvtnorm::rmvnorm(n_stat, sigma = cov_s2s)
    rect2 <- mvtnorm::rmvnorm(n_rec, sigma = cov_ss)
    
    data_reg$y_sim1 <- eqt2[eq,1] + statt2[stat,1] + rect2[,1]
    data_reg$y_sim2 <- eqt2[eq,2] + statt2[stat,2] + rect2[,2]
    
    data_list <- list(
      N = n_rec,
      NEQ = n_eq,
      NSTAT = n_stat,
      Y = data_reg[,c('y_sim1','y_sim2')],
      eq = eq,
      stat = stat
    )
    
    fit <- mod$sample(
      data = data_list,
      seed = 1701,
      chains = 2,
      parallel_chains = 2,
      show_exceptions = FALSE,
      iter_sampling = 200,
      iter_warmup = 200,
    )
    draws <- fit$draws()
    rv <- as_draws_rvars(draws)
    
    mat_cor_stan[i,] <- c(mean(rv$rho_stat), mean(rv$rho_eq), mean(rv$rho_rec),
                          sum(quantile(rv$rho_stat, 0.05) < rho_s2s & quantile(rv$rho_stat, 0.95) > rho_s2s),
                          sum(quantile(rv$rho_eq, 0.05) < rho_tau & quantile(rv$rho_eq, 0.95) > rho_tau),
                          sum(quantile(rv$rho_rec, 0.05) < rho_ss & quantile(rv$rho_rec, 0.95) > rho_ss))
    print(mat_cor_stan[i,])
    
  }
  save(mat_cor_stan,
       file = sprintf('../results/res_corrre_stan_CB14_%s.Rdata', cor_name))
}
