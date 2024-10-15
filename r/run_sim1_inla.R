rm(list = ls())

library(INLA)
library(tidyverse)

data_reg <- read.csv(file.path('../data','data_cb.csv'))

eq <- data_reg$eq
stat <- data_reg$stat

n_rec <- nrow(data_reg)
n_eq <- max(eq)
n_stat <- max(stat)

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

n_sam <- 200
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
save(res_val_inla, res_sd_inla, res_ci_inla,
     file = file.path('../results', 'results_sim1_inla_CB.Rdata'))

