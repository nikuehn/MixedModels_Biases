rm(list = ls())

library(TMB)
library(matrixStats)
library(tidyverse)

calc_cormat <- function(cor_unconstrained) {
  tmp <- diag(rep(1,2))
  tmp[1,2] <- 1/cor_unconstrained
  tmp[2,2] <- abs(cov2cor(t(tmp) %*% tmp)[1,2])
  tmp[1,2] <- cor_unconstrained * tmp[2,2]
  return(t(tmp) %*% tmp)
}
scale <- qnorm(0.95)

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


rho_tau <- 0.95
rho_ss <- 0.54
rho_s2s <- 0.77

rho_total <- (rho_tau * tau_sim1 * tau_sim2 + rho_s2s * phi_s2s_sim1 * phi_s2s_sim2 + rho_ss * phi_ss_sim1 * phi_ss_sim2) /
  (sigma_tot1 * sigma_tot2)

cov_tau <- matrix(c(tau_sim1^2, rho_tau * tau_sim1 * tau_sim2,
                    rho_tau * tau_sim1 * tau_sim2, tau_sim2^2), ncol = 2)
cov_s2s <- matrix(c(phi_s2s_sim1^2, rho_s2s * phi_s2s_sim1 * phi_s2s_sim2,
                    rho_s2s * phi_s2s_sim1 * phi_s2s_sim2, phi_s2s_sim2^2), ncol = 2)
cov_ss <- matrix(c(phi_ss_sim1^2, rho_ss * phi_ss_sim1 * phi_ss_sim2,
                   rho_ss * phi_ss_sim1 * phi_ss_sim2, phi_ss_sim2^2), ncol = 2)


compile(file.path('../tmb', "mv_mixed_model.cpp"))
dyn.load(dynlib(file.path('../tmb', "mv_mixed_model")))


init_values <- list(
  u_eq = matrix(0, nrow = n_eq, ncol = 2),    # Random effects for eq
  u_stat = matrix(0, nrow = n_stat, ncol = 2),# Random effects for stat
  beta = rep(0,2),                                # Intercept
  log_sigma_rec = rep(0,2),                       # Log residual standard deviation
  log_sigma_eq = rep(0,2),                        # Log eq standard deviation
  log_sigma_stat = rep(0,2),                      # Log stat standard deviation
  rho_eq = 0,
  rho_stat=0,
  rho_rec = 0
)

n_sam <- 100
mat_cor_tmb <- matrix(nrow = n_sam, ncol =  8)
set.seed(5618)
for(i in 1:n_sam) {
  eqt2 <- mvtnorm::rmvnorm(n_eq, sigma = cov_tau)
  statt2 <- mvtnorm::rmvnorm(n_stat, sigma = cov_s2s)
  rect2 <- mvtnorm::rmvnorm(n_rec, sigma = cov_ss)
  
  data_reg$y_sim1 <- eqt2[eq,1] + statt2[stat,1] + rect2[,1]
  data_reg$y_sim2 <- eqt2[eq,2] + statt2[stat,2] + rect2[,2]
  
  data_list <- list(Y = cbind(data_reg$y_sim1, data_reg$y_sim2), eq= eq - 1, stat = stat - 1)
  
  obj <- MakeADFun(data = data_list, parameters = init_values, random = c("u_eq", "u_stat"), 
                   DLL = "mv_mixed_model")
  fit <- nlminb(obj$par, obj$fn, obj$gr)
  report <- sdreport(obj)
  
  fit_pars <- report$par.fixed
  sd_par <- sqrt(diag(report$cov.fixed))
  
  sum(calc_cormat(fit_pars[names(fit_pars) == 'rho_eq'] - scale * sd_par[names(fit_pars) == 'rho_eq'])[1,2] < rho_tau
      & calc_cormat(fit_pars[names(fit_pars) == 'rho_eq'] + scale * sd_par[names(fit_pars) == 'rho_eq'])[1,2] > rho_tau)
  
  mat_cor_tmb[i,] <- c(obj$report()$Cor_stat[1,2],
                       obj$report()$Cor_eq[1,2],
                       obj$report()$Cor_rec[1,2],
                       NA,
                       sum(calc_cormat(fit_pars[names(fit_pars) == 'rho_stat'] - scale * sd_par[names(fit_pars) == 'rho_stat'])[1,2] < rho_s2s
                           & calc_cormat(fit_pars[names(fit_pars) == 'rho_stat'] + scale * sd_par[names(fit_pars) == 'rho_stat'])[1,2] > rho_s2s),
                       sum(calc_cormat(fit_pars[names(fit_pars) == 'rho_eq'] - scale * sd_par[names(fit_pars) == 'rho_eq'])[1,2] < rho_tau
                           & calc_cormat(fit_pars[names(fit_pars) == 'rho_eq'] + scale * sd_par[names(fit_pars) == 'rho_eq'])[1,2] > rho_tau),
                       sum(calc_cormat(fit_pars[names(fit_pars) == 'rho_rec'] - scale * sd_par[names(fit_pars) == 'rho_rec'])[1,2] < rho_ss
                           & calc_cormat(fit_pars[names(fit_pars) == 'rho_rec'] + scale * sd_par[names(fit_pars) == 'rho_rec'])[1,2] > rho_ss),
                       NA)
  
}
save(mat_cor_tmb, 
     file = sprintf('../results/res_corrre_tmb_CB14_%s.Rdata', 'eas'))

