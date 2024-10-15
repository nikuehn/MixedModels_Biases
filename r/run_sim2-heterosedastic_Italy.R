rm(list = ls())

library(lme4)
library(cmdstanr)
library(tidyverse)
library(matrixStats)

data_reg <- read.csv(file.path('../data/','italian_data_pga_id_utm_stat.csv'))

eq <- data_reg$EQID
stat <- data_reg$STATID

n_rec <- nrow(data_reg)
n_eq <- max(eq)
n_stat <- max(stat)

mageq <- unique(data_reg[,c('EQID','mag')])$mag

coeffs <- c(3.421046409, 0.193954090, -0.021982777, 0.287149291, -1.405635476, -0.002911264, -0.394575970)
names_coeffs <- c("intercept", "M1", "M2", "MlogR", "logR", "R", "logVS")

# Set linear predictors
mh = 5.5
mref = 5.324
h = 6.924
data_reg$intercept <- 1
data_reg$M1 <- (data_reg$mag-mh)*(data_reg$mag<=mh)
data_reg$M2 <- (data_reg$mag-mh)*(data_reg$mag>mh)
data_reg$MlogR <- (data_reg$mag-mref)*log10(sqrt(data_reg$JB_complete^2+h^2))
data_reg$logR <- log10(sqrt(data_reg$JB_complete^2+h^2))
data_reg$R <- sqrt(data_reg$JB_complete^2+h^2)
data_reg$logVS <- log10(data_reg$vs30/800)*(data_reg$vs30<=1500)+log10(1500/800)*(data_reg$vs30>1500)

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
m1_rec <- 1 * (data_reg$mag < mb_phi[2]) - (data_reg$mag - mb_phi[1]) / (mb_phi[2] - mb_phi[1]) * (data_reg$mag > mb_phi[1] & data_reg$mag < mb_phi[2])
m2_rec <- 1 * (data_reg$mag >= mb_phi[2]) + (data_reg$mag - mb_phi[1]) / (mb_phi[2] - mb_phi[1]) * (data_reg$mag > mb_phi[1] & data_reg$mag < mb_phi[2])

phi_ss_sim <- m1_rec * phi_ss_sim_val[1] + m2_rec * phi_ss_sim_val[2]



n_sam <- 100

res_coeffs <-matrix(nrow = n_sam, ncol = length(coeffs))

res_tau <- matrix(nrow = n_sam, ncol = 4)
res_phi <- matrix(nrow = n_sam, ncol = 4)
res_sd_lmer <- matrix(nrow = n_sam, ncol = 3)

set.seed(1701)
for(i in 1:n_sam) {
  print(paste0('i = ',i))
  eqt <- rnorm(n_eq, sd = tau_sim)
  rect <- rnorm(n_rec, sd = phi_ss_sim)
  statt <- rnorm(n_stat, sd = phi_s2s_sim)
  
  data_reg$y_sim <- as.matrix(data_reg[,names_coeffs]) %*% coeffs + eqt[eq] + statt[stat] + rect
  fit_sim <- lmer(y_sim ~ M1 + M2 + MlogR + logR + R + logVS + (1|eq) + (1|stat), data_reg)
  tmp <- as.data.frame(VarCorr(fit_sim))$sdcor
  
  res_coeffs[i,] <- fixef(fit_sim)
  
  deltaB_sim <- ranef(fit_sim)$eq$`(Intercept)`
  sd_deltaB_sim <- as.numeric(arm::se.ranef(fit_sim)$eq)
  deltaWS_sim <- data_reg$y_sim - predict(fit_sim)
  sd_deltaS_sim <- as.numeric(arm::se.ranef(fit_sim)$stat)
  sd_deltaWS_sim <- sqrt(sd_deltaB_sim[eq]^2 + sd_deltaS_sim[stat]^2)
  
  
  df_eq_sim <-data.frame(M= mageq, dB = deltaB_sim, sd_dB = sd_deltaB_sim)
  df_rec_sim <- data.frame(M = data_reg$mag, dWS = deltaWS_sim, sd_dWS = sd_deltaWS_sim)
  
  
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
  
  res_sd_lmer[i,] <- tmp
}


res_tau_it <- res_tau
res_phi_it <- res_phi
res_sd_lmer_it <- res_sd_lmer

save(res_tau_it, res_phi_it, res_sd_lmer_it,
     file = file.path('../results','results_sim2_heteroscedastic_coeff_Italy.Rdata'))


