rm(list = ls())

library(lme4)
library(cmdstanr)
library(tidyverse)
library(matrixStats)
library(INLA)

set_cmdstan_path('/Users/nico/GROUNDMOTION/SOFTWARE/cmdstan-2.33.1')

data_reg <- read.csv(file.path('../data','data_cb.csv'))

eq <- data_reg$eq
stat <- data_reg$stat

n_rec <- nrow(data_reg)
n_eq <- max(eq)
n_stat <- max(stat)

mageq <- unique(data_reg[,c('eq','M')])$M
magstat <- unique(data_reg[,c('stat','M_stat')])$M_stat # station-specific magnitude

# set standard deviations
tau_sim <- 0.4
phi_s2s_sim <- 0.43
phi_ss_sim <- 0.5

# calculate precisions
prec_s2s <- 1/phi_s2s_sim^2
prec_tau <- 1/tau_sim^2
prec_ss <- 1/phi_ss_sim^2

prior_prec_tau    <- list(prec = list(prior = 'pc.prec', param = c(0.8, 0.01)))
prior_prec_phiS2S    <- list(prec = list(prior = 'pc.prec', param = c(0.8, 0.01))) 
prior_prec_phiSS    <- list(prec = list(prior = 'pc.prec', param = c(0.8, 0.01))) 

# function to calculate mean of hyperparameters from posterior of internal representation
func_sd <- function(par, fit_inla) {
  inla_tmarg <- inla.tmarginal(function(x) sqrt(exp(-x)),
                               fit_inla$internal.marginals.hyperpar[[par]])
  return(inla.emarginal(function(x) x, inla_tmarg))
}

mod <- cmdstan_model(file.path('../stan', 'gmm_partition_wvar2.stan'),
                     force_recompile = TRUE)

n_sam <- 200
res_val <- matrix(nrow = n_sam, ncol = 9)
res_sd <- matrix(nrow = n_sam, ncol = 6)
res_ci <- matrix(nrow = n_sam, ncol = 6)
res_ci_diff <- matrix(nrow = n_sam, ncol = 6)
res_bpt <- matrix(nrow = n_sam, ncol = 6)

res_val_inla <- matrix(nrow = n_sam, ncol = 6)
res_sd_inla <- matrix(nrow = n_sam, ncol = 3)
res_ci_inla <- matrix(nrow = n_sam, ncol = 3)

set.seed(5618)
for(i in 1:n_sam) {
  print(paste0('i = ',i))
  rect <- rnorm(n_rec, sd = phi_ss_sim)
  statt <- rnorm(n_stat, sd = phi_s2s_sim)
  eqtt <- rnorm(n_eq, sd = tau_sim)
  
  data_reg$y_sim <- eqtt[eq] + statt[stat] + rect
  
  fit_sim <- lmer(y_sim ~ (1 | eq) + (1 | stat), data_reg)
  ci_sim <- confint(fit_sim, level = 0.9)
  
  data_list <- list(
    N = n_rec,
    NEQ = n_eq,
    NSTAT = n_stat,
    Y = data_reg$y_sim,
    eq = eq,
    stat = stat,
    alpha = c(1,1,1) # prior for dirichlet distribution on variance partitions
  )
  
  fit <- mod$sample(
    data = data_list,
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
  draws <- fit$draws()
  ci_stan <- colQuantiles(subset(as_draws_matrix(draws), variable = c('phi_S2S','tau_0', 'phi_0')),
                          probs = c(0.05,0.95))
  
  res_val[i,] <- c(as.data.frame(VarCorr(fit_sim))$sdcor,
                   colMeans(subset(as_draws_matrix(draws), variable = c('phi_S2S','tau_0', 'phi_0'))),
                   colMedians(subset(as_draws_matrix(draws), variable = c('phi_S2S','tau_0', 'phi_0'))))
  
  res_ci[i,] <- c(sum(phi_s2s_sim > ci_sim[1,1] & phi_s2s_sim <= ci_sim[1,2]),
                  sum(tau_sim > ci_sim[2,1] & tau_sim <= ci_sim[2,2]),
                  sum(phi_ss_sim > ci_sim[3,1] & phi_ss_sim <= ci_sim[3,2]),
                  sum(phi_s2s_sim > ci_stan[1,1] & phi_s2s_sim <= ci_stan[1,2]),
                  sum(tau_sim > ci_stan[2,1] & tau_sim <= ci_stan[2,2]),
                  sum(phi_ss_sim > ci_stan[3,1] & phi_ss_sim <= ci_stan[3,2]))
  
  res_ci_diff[i,] <- c(rowDiffs(ci_sim[c(1,2,3),]),rowDiffs(ci_stan))
  
  dWS <- data_reg$y_sim - predict(fit_sim)
  dS <- ranef(fit_sim)$stat$`(Intercept)`
  dB <- ranef(fit_sim)$eq$`(Intercept)`
  
  res_sd[i,] <- c(sd(dS),
                  sd(dB),
                  sd(dWS),
                  sd(colMeans(subset(as_draws_matrix(draws), variable = 'statterm', regex = TRUE))),
                  sd(colMeans(subset(as_draws_matrix(draws), variable = 'eqterm', regex = TRUE))),
                  sd(colMeans(subset(as_draws_matrix(draws), variable = 'resid', regex = TRUE))))
  
  res_bpt[i,] <- c(lmtest::bptest(dws ~ M, data = data.frame(M = data_reg$M, dws = dWS))$p.value,
                   lmtest::bptest(dB ~ M, data = data.frame(M = mageq, dB = dB))$p.value,
                   lmtest::bptest(dS ~ M, data = data.frame(M = magstat, dS = dS))$p.value,
                   lmtest::bptest(dws ~ M, data = data.frame(M = data_reg$M, dws = rect))$p.value,
                   lmtest::bptest(dB ~ M, data = data.frame(M = mageq, dB = eqtt))$p.value,
                   lmtest::bptest(dS ~ M, data = data.frame(M = magstat, dS = statt))$p.value
  )
  
  
  
  # Inla
  form <- y_sim ~ f(eq, model = "iid", hyper = prior_prec_tau) + 
    f(stat, model = "iid",hyper = prior_prec_phiS2S)
  
  fit_inla <- inla(form, 
                   data = data_reg,
                   family="gaussian",
                   control.family = list(hyper = prior_prec_phiSS),
                   quantiles = c(0.05, 0.5, 0.95)
  )
  
  
  res_val_inla[i,] <- c(
    func_sd("Log precision for stat", fit_inla),
    func_sd("Log precision for eq", fit_inla),
    func_sd("Log precision for the Gaussian observations", fit_inla),
    sqrt(exp(-fit_inla$internal.summary.hyperpar[['0.5quant']][c(3,2,1)]))
  )
  
  res_ci_inla[i,] <- c(sum(prec_s2s > fit_inla$summary.hyperpar['Precision for stat','0.05quant'] 
                           & prec_s2s <= fit_inla$summary.hyperpar['Precision for stat','0.95quant']),
                       sum(prec_tau > fit_inla$summary.hyperpar['Precision for eq','0.05quant'] 
                           & prec_tau <= fit_inla$summary.hyperpar['Precision for eq','0.95quant']),
                       sum(prec_ss > fit_inla$summary.hyperpar['Precision for the Gaussian observations','0.05quant'] 
                           & prec_ss <= fit_inla$summary.hyperpar['Precision for the Gaussian observations','0.95quant'])
  )
  
  dWS <- data_reg$y_sim - fit_inla$summary.fitted.values[,'mean']
  dS <- fit_inla$summary.random$stat[,'mean']
  dB <- fit_inla$summary.random$eq[,'mean']
  
  res_sd_inla[i,] <- c(sd(dS), sd(dB), sd(dWS))
  
  rm(fit_inla)
}
save(res_val, res_sd, res_ci, res_ci_diff, res_bpt, res_val_inla, res_ci_inla,
     file = file.path('../results', 'results_sim1_CB_rev.Rdata'))
