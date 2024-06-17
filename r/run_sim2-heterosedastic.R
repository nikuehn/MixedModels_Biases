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

mageq <- unique(data_reg[,c('eq','M')])$M
magstat <- unique(data_reg[,c('stat','M_stat')])$M_stat # station-specific magnitude

coeffs <- c(3.421046409, 0.193954090, -0.021982777, 0.287149291, -1.405635476, -0.002911264, -0.394575970)
names_coeffs <- c("intercept", "M1", "M2", "MlogR", "logR", "R", "logVS")

# Set linear predictors
mh = 5.5
mref = 5.324
h = 6.924
data_reg$M1 <- (data_reg$M-mh)*(data_reg$M<=mh)
data_reg$M2 <- (data_reg$M-mh)*(data_reg$M>mh)
data_reg$MlogR <- (data_reg$M-mref)*log10(sqrt(data_reg$Rjb^2+h^2))
data_reg$logR <- log10(sqrt(data_reg$Rjb^2+h^2))
data_reg$R <- sqrt(data_reg$Rjb^2+h^2)
data_reg$logVS <- log10(data_reg$VS_gmean/800)*(data_reg$VS_gmean<=1500)+log10(1500/800)*(data_reg$VS_gmean>1500)

X_mat <- data_reg[,c("M1", "M2", "MlogR", "logR", "R", "logVS")]

# --------------------------------------------------
# magitude dependent tau and phi

phi_s2s_sim <- 0.43
tau_sim_val <- c(0.4,0.25)
phi_ss_sim_val <- c(0.55,0.4)
mb_tau <- c(5,6)
mb_phi <- c(4.5,5.5)

# define tau for each event
# define linear predictors
m1_eq <- 1 * (mageq < mb_tau[2]) - (mageq - mb_tau[1]) / (mb_tau[2] - mb_tau[1]) * (mageq > mb_tau[1] & mageq < mb_tau[2])
m2_eq <- 1 * (mageq >= mb_tau[2]) + (mageq - mb_tau[1]) / (mb_tau[2] - mb_tau[1]) * (mageq > mb_tau[1] & mageq < mb_tau[2])

tau_sim <- m1_eq * tau_sim_val[1] + m2_eq * tau_sim_val[2]

# define phi_ss for each record
# define linear predictors
m1_rec <- 1 * (data_reg$M < mb_phi[2]) - (data_reg$M - mb_phi[1]) / (mb_phi[2] - mb_phi[1]) * (data_reg$M > mb_phi[1] & data_reg$M < mb_phi[2])
m2_rec <- 1 * (data_reg$M >= mb_phi[2]) + (data_reg$M - mb_phi[1]) / (mb_phi[2] - mb_phi[1]) * (data_reg$M > mb_phi[1] & data_reg$M < mb_phi[2])

phi_ss_sim <- m1_rec * phi_ss_sim_val[1] + m2_rec * phi_ss_sim_val[2]



n_sam <- 100

res_coeffs <-matrix(nrow = n_sam, ncol = length(coeffs))
res_coeffs_ci <-matrix(nrow = n_sam, ncol = length(coeffs))

res_tau <- matrix(nrow = n_sam, ncol = 4)
res_phi <- matrix(nrow = n_sam, ncol = 4)
res_phis2s <- matrix(nrow = n_sam, ncol = 4)

res_tau_stan <- matrix(nrow = n_sam, ncol = 4)
res_phi_stan <- matrix(nrow = n_sam, ncol = 4)

res_coeffs_stan2 <-matrix(nrow = n_sam, ncol = length(coeffs))
res_coeffs_stan2_ci <-matrix(nrow = n_sam, ncol = length(coeffs))

res_phis2s_stan2 <- matrix(nrow = n_sam, ncol = 2)
res_tau_stan2 <- matrix(nrow = n_sam, ncol = 4)
res_phi_stan2 <- matrix(nrow = n_sam, ncol = 4)

set.seed(1701)
for(i in 1:n_sam) {
  print(paste0('i = ',i))
  eqt <- rnorm(n_eq, sd = tau_sim)
  rect <- rnorm(n_rec, sd = phi_ss_sim)
  statt <- rnorm(n_stat, sd = phi_s2s_sim)
  
  data_reg$y_sim <- as.matrix(data_reg[,names_coeffs]) %*% coeffs + eqt[eq] + statt[stat] + rect
  fit_sim <- lmer(y_sim ~ M1 + M2 + MlogR + logR + R + logVS + (1|eq) + (1|stat), data_reg)
  ci_sim <- confint(fit_sim, level = 0.9)
  tmp <- as.data.frame(VarCorr(fit_sim))$sdcor
  
  res_coeffs[i,] <- fixef(fit_sim)
  
  for(k in 1:length(coeffs)) {res_coeffs_ci[i,k] <- sum(ci_sim[k+3,1] < coeffs[k] & ci_sim[k+3,2] > coeffs[k])}
  print(res_coeffs_ci[i,])
  
  deltaB_sim <- ranef(fit_sim)$eq$`(Intercept)`
  sd_deltaB_sim <- as.numeric(arm::se.ranef(fit_sim)$eq)
  deltaWS_sim <- data_reg$y_sim - predict(fit_sim)
  sd_deltaS_sim <- as.numeric(arm::se.ranef(fit_sim)$stat)
  sd_deltaWS_sim <- sqrt(sd_deltaB_sim[eq]^2 + sd_deltaS_sim[stat]^2)
  
  
  df_eq_sim <-data.frame(M= mageq, dB = deltaB_sim, sd_dB = sd_deltaB_sim)
  df_rec_sim <- data.frame(M = data_reg$M, dWS = deltaWS_sim, sd_dWS = sd_deltaWS_sim)
  
  
  res_tau[i,] <- c(sd(df_eq_sim[df_eq_sim$M <= mb_tau[1],'dB']),
                   sqrt(sum(df_eq_sim[mageq <= mb_tau[1],'dB']^2)/sum(mageq <= mb_tau[1]) +
                          sum(df_eq_sim[mageq <= mb_tau[1],'sd_dB']^2)/sum(mageq <= mb_tau[1])),
                   sd(df_eq_sim[df_eq_sim$M >= mb_tau[2],'dB']),
                   sqrt(sum(df_eq_sim[mageq >= mb_tau[2],'dB']^2)/sum(mageq >= mb_tau[2]) +
                          sum(df_eq_sim[mageq >= mb_tau[2],'sd_dB']^2)/sum(mageq >= mb_tau[2])))
  
  
  res_phi[i,] <- c(sd(df_rec_sim[df_rec_sim$M <= mb_phi[1],'dWS']),
                   sqrt(sum(df_rec_sim[df_rec_sim$M <= mb_phi[1],'dWS']^2)/sum(df_rec_sim$M <= mb_phi[1]) +
                          sum(df_rec_sim[df_rec_sim$M <= mb_phi[1],'sd_dWS']^2)/sum(df_rec_sim$M <= mb_phi[1])),
                   sd(df_rec_sim[df_rec_sim$M >= mb_phi[2],'dWS']),
                   sqrt(sum(df_rec_sim[df_rec_sim$M >= mb_phi[2],'dWS']^2)/sum(df_rec_sim$M >= mb_phi[2]) +
                          sum(df_rec_sim[df_rec_sim$M >= mb_phi[2],'sd_dWS']^2)/sum(df_rec_sim$M >= mb_phi[2])))
  
  # Stan model from total residuals
  resid_sim <- data_reg$y_sim - predict(fit_sim, re.form=NA)
  
  data_list <- list(
    N = n_rec,
    NEQ = n_eq,
    NSTAT = n_stat,
    Y = as.numeric(resid_sim),
    eq = eq,
    stat = stat,
    MEQ = mageq,
    tau_mb = mb_tau,
    phi_mb = mb_phi,
    M1_eq = m1_eq,
    M2_eq = m2_eq,
    M1_rec = m1_rec,
    M2_rec = m2_rec
  )
  
  mod <- cmdstan_model(file.path('../stan', 'gmm_partition_tauM_phiM.stan'))
  
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
  
  ci_stan <- colQuantiles(subset(as_draws_matrix(draws), variable = c('phi_s2s'), regex = TRUE), probs = c(0.05,0.95))
  res_phis2s[i,] <- c(tmp[1], sum(phi_s2s_sim > ci_sim[1,1] & phi_s2s_sim <= ci_sim[1,2]),
                      colMeans(subset(as_draws_matrix(draws), variable = c('phi_s2s'), regex = TRUE)),
                      sum(phi_s2s_sim > ci_stan[1] & phi_s2s_sim <= ci_stan[2]))
  
  
  ci_stan <- colQuantiles(subset(as_draws_matrix(draws), variable = c('tau_'), regex = TRUE), probs = c(0.05,0.95))
  res_tau_stan[i,] <- c(colMeans(subset(as_draws_matrix(draws), variable = c('tau_'), regex = TRUE)),
                        sum(tau_sim_val[1] > ci_stan[1,1] & tau_sim_val[1] <= ci_stan[1,2]),
                        sum(tau_sim_val[2] > ci_stan[2,1] & tau_sim_val[2] <= ci_stan[2,2]))
  
  ci_stan <- colQuantiles(subset(as_draws_matrix(draws), variable = c('phi_ss_'), regex = TRUE), probs = c(0.05,0.95))
  res_phi_stan[i,] <- c(colMeans(subset(as_draws_matrix(draws), variable = c('phi_ss_'), regex = TRUE)),
                        sum(phi_sim_val[1] > ci_stan[1,1] & phi_sim_val[1] <= ci_stan[1,2]),
                        sum(phi_sim_val[2] > ci_stan[2,1] & phi_sim_val[2] <= ci_stan[2,2]))
  
  
  # full Stan model
  mod <- cmdstan_model(file.path('../stan', 'gmm_full_qr_tauM_phiM.stan'))
  
  data_list <- list(
    N = n_rec,
    NEQ = n_eq,
    NSTAT = n_stat,
    K = length(coeffs),
    X = X_mat,
    Y = as.numeric(data_reg$y_sim),
    eq = eq,
    stat = stat,
    MEQ = mageq,
    tau_mb = mb_tau,
    phi_mb = mb_phi,
    M1_eq = m1_eq,
    M2_eq = m2_eq,
    M1_rec = m1_rec,
    M2_rec = m2_rec
  )
  
  fit <- mod$sample(
    data = data_list,
    seed = 8472,
    chains = 4,
    iter_sampling = 200,
    iter_warmup = 200,
    refresh = 50,
    max_treedepth = 10,
    adapt_delta = 0.8,
    parallel_chains = 2,
    show_exceptions = FALSE
  )
  print(fit$diagnostic_summary())
  draws <- fit$draws()
  
  res_coeffs_stan2[i,] <- colMeans(subset(as_draws_matrix(draws), variable = c('c\\['), regex = TRUE))
  ci_stan <- colQuantiles(subset(as_draws_matrix(draws), variable = c('c\\['), regex = TRUE), probs = c(0.05,0.95))
  
  for(k in 1:length(coeffs)) {res_coeffs_stan2_ci[i,k] <- sum(ci_stan[k,1] < coeffs[k] & ci_stan[k,2] > coeffs[k])}
  print(res_coeffs_stan2_ci[i,])
  
  
  ci_stan <- colQuantiles(subset(as_draws_matrix(draws), variable = c('phi_s2s'), regex = TRUE), probs = c(0.05,0.95))
  res_phis2s_stan2[i,] <- c(colMeans(subset(as_draws_matrix(draws), variable = c('phi_s2s'), regex = TRUE)),
                            sum(phi_s2s_sim > ci_stan[1] & phi_s2s_sim <= ci_stan[2]))
  
  
  ci_stan <- colQuantiles(subset(as_draws_matrix(draws), variable = c('tau_'), regex = TRUE), probs = c(0.05,0.95))
  res_tau_stan2[i,] <- c(colMeans(subset(as_draws_matrix(draws), variable = c('tau_'), regex = TRUE)),
                         sum(tau_sim_val[1] > ci_stan[1,1] & tau_sim_val[1] <= ci_stan[1,2]),
                         sum(tau_sim_val[2] > ci_stan[2,1] & tau_sim_val[2] <= ci_stan[2,2]))
  
  ci_stan <- colQuantiles(subset(as_draws_matrix(draws), variable = c('phi_ss_'), regex = TRUE), probs = c(0.05,0.95))
  res_phi_stan2[i,] <- c(colMeans(subset(as_draws_matrix(draws), variable = c('phi_ss_'), regex = TRUE)),
                         sum(phi_sim_val[1] > ci_stan[1,1] & phi_sim_val[1] <= ci_stan[1,2]),
                         sum(phi_sim_val[2] > ci_stan[2,1] & phi_sim_val[2] <= ci_stan[2,2]))
}

save(res_tau, res_tau_stan, res_phi, res_phi_stan, res_phis2s, res_coeffs_ci, res_coeffs,
     file = file.path('../results','results_sim2_heteroscedastic_coeff_CB.Rdata'))
save(res_tau_stan2, res_phi_stan2, res_phis2s_stan2, res_coeffs_stan2, res_coeffs_stan2_ci,
     file = file.path('../results', 'results_sim2_heteroscedastic_coeff_stan2_CB.Rdata'))
