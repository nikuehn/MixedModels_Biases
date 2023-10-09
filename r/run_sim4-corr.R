rm(list = ls())

library(lme4)
library(cmdstanr)
library(tidyverse)
library(matrixStats)

data_reg <- read.csv(file.path('../data','data_cb.csv'))

eq <- data_reg$eq
stat <- data_reg$stat

n_rec <- nrow(data_reg)
n_eq <- max(eq)
n_stat <- max(stat)

# -----------------------------------------
# correlation (e.g. stress drop)

tau_sim <- 0.4
phi_s2s_sim <- 0.43
phi_ss_sim <- 0.5

rho <- 0.7
tau2 <- 1

cov_tau <- matrix(c(tau_sim^2, rho * tau_sim * tau2, rho * tau_sim * tau2, tau2^2), ncol = 2)

mod <- cmdstan_model(file.path('../stan', 'gmm_partition_wvar_corr.stan'))

n_sam <- 50
res_cor <- matrix(nrow = n_sam, ncol = 6)
set.seed(8472)
for(i in 1:n_sam) {
  print(paste0('i = ',i))
  rect <- rnorm(n_rec, sd = phi_ss_sim)
  statt <- rnorm(n_stat, sd = phi_s2s_sim)
  eqt2 <- mvtnorm::rmvnorm(n_eq, sigma = cov_tau)
  
  y_sim <- eqt2[eq,1] + statt[stat] + rect
  data_reg$y_sim <- y_sim
  
  fit_lmer_sim <- lmer(y_sim ~ (1 | eq) + (1 | stat), data_reg)
  deltaB_sim <- ranef(fit_lmer_sim)$eq$`(Intercept)`
  
  data_list_cor <- list(
    N = n_rec,
    NEQ = n_eq,
    NSTAT = n_stat,
    Y = data_reg$y_sim,
    E = eqt2[,2],
    eq = eq,
    stat = stat,
    alpha = c(1,1,1)
  )
  
  fit <- mod$sample(
    data = data_list_cor,
    seed = 8472,
    chains = 4,
    iter_sampling = 500,
    iter_warmup = 500,
    refresh = 50,
    max_treedepth = 10,
    adapt_delta = 0.8,
    parallel_chains = 2,
    show_exceptions = FALSE
  )
  print(fit$diagnostic_summary())
  draws_corr <- fit$draws() 
  
  res_cor[i,] <- c(cor(eqt2[,1], eqt2[,2]), cor(deltaB_sim, eqt2[,2]),
                   cor(eqt2[,2], colMeans(subset(as_draws_matrix(draws_corr), variable = 'eqterm', regex = TRUE))),
                   colMeans(subset(as_draws_matrix(draws_corr), variable = 'rho')),
                   colQuantiles(subset(as_draws_matrix(draws_corr), variable = 'rho'), probs = c(0.05,0.95)))
  print(res_cor[i,])
}
df_res_cor <- set_names(data.frame(res_cor), c('cor_sim','cor_lme','cor_stan','cor_mean','cor_q05','cor_q95'))
write.csv(df_res_cor, file.path('../results', sprintf('res_sim_cor_CB_N%d.csv', n_sam)),
          row.names = FALSE)
