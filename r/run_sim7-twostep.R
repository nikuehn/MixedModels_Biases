rm(list = ls())

library(lme4)
library(tidyverse)
library(matrixStats)

data_reg <- read.csv(file.path('../data','data_cb.csv'))

eq <- data_reg$eq
stat <- data_reg$stat

n_rec <- nrow(data_reg)
n_eq <- max(eq)
n_stat <- max(stat)

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


# set parameters
tau_sim <- 0.17
phi_s2s_sim <- 0.23
phi_ss_sim <- 0.2
sds_sim <- c(phi_s2s_sim, tau_sim, phi_ss_sim)

n_sam <- 100
mat_fix <- matrix(ncol = length(coeffs), nrow = n_sam)
mat_fix2 <- matrix(ncol = length(coeffs), nrow = n_sam)
mat_ci <- matrix(nrow = n_sam, ncol = length(coeffs))
mat_ci2 <- matrix(nrow = n_sam, ncol = length(coeffs))
mat_ci_sd <- matrix(nrow = n_sam, ncol = 3)
mat_ci_sd2 <- matrix(nrow = n_sam, ncol = 3)
mat_sd <- matrix(nrow = n_sam, ncol = 3)
mat_sd2 <- matrix(nrow = n_sam, ncol = 3)
set.seed(8472)
for(i in 1:n_sam) {
  print(paste('i = ',i))
  dWS_sim <- rnorm(n_rec, sd = phi_ss_sim)
  dS_sim <- rnorm(n_stat, sd = phi_s2s_sim)
  dB_sim <- rnorm(n_eq, sd = tau_sim)
  
  data_reg$y_sim <- as.matrix(data_reg[,names_coeffs]) %*% coeffs + dB_sim[eq] + dS_sim[stat] + dWS_sim
  fit_sim <- lmer(y_sim ~ M1 + M2 + MlogR + logR + R + logVS  + (1|eq) + (1|stat), data_reg)
  fit_sim2 <- lmer(y_sim ~ M1 + M2 + MlogR + logR + R + logVS  + (1|eq), data_reg)
  
  ci_sim <- confint(fit_sim, level = 0.9)
  ci_sim2 <- confint(fit_sim2, level = 0.9)
  
  for(k in 1:length(coeffs)) {mat_ci[i,k] <- sum(coeffs[k] > ci_sim[k+3,1] & coeffs[k] <= ci_sim[k+3,2])}
  for(k in 1:length(coeffs)) {mat_ci2[i,k] <- sum(coeffs[k] > ci_sim2[k+2,1] & coeffs[k] <= ci_sim2[k+2,2])}
  
  for(k in 1:length(sds_sim)) {mat_ci_sd[i,k] <- sum(sds_sim[k] > ci_sim[k,1] & sds_sim[k] <= ci_sim[k,2])}
  
  mat_fix[i,] <- fixef(fit_sim)
  mat_fix2[i,] <- fixef(fit_sim2)
  
  mat_sd[i,] <- as.data.frame(VarCorr(fit_sim))$sdcor
  
  data_reg$dR2 <- resid(fit_sim2)
  fit_sim2a <- lmer(dR2 ~ (1 | stat), data_reg)
  tmp <- as.data.frame(VarCorr(fit_sim2a))$sdcor
  mat_sd2[i,] <- c(tmp[1], as.data.frame(VarCorr(fit_sim2))$sdcor[1], tmp[2])
  ci_sim2a <- confint(fit_sim2a, level = 0.9)
  
  mat_ci_sd2[i,] <- c(sum(phi_s2s_sim > ci_sim2a[1,1] & phi_s2s_sim <= ci_sim2a[1,2]),
                     sum(tau_sim > ci_sim2[1,1] & tau_sim <= ci_sim2[1,2]),
                     sum(phi_ss_sim > ci_sim2a[2,1] & phi_ss_sim <= ci_sim2a[2,2]))
}

save(mat_fix, mat_fix2, mat_sd, mat_sd2, mat_ci_sd, mat_ci_sd2, mat_ci, mat_ci2,
      file = file.path('../results', 'res_twostep_ita18_CB.Rdata'))
