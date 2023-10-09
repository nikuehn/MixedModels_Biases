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

idx <- as.numeric(which(table(stat) >= 10))


# set parameters
tau_sim <- 0.17
phi_s2s_sim <- 0.23
phi_sim <- 0.2

n_sam <- 200
res_val <- matrix(ncol = 18, nrow = n_sam)
res_ci <- matrix(nrow = n_sam, ncol = 8)
res_ci_diff <- matrix(nrow = n_sam, ncol = 4)

set.seed(1701)
for(i in 1:n_sam) {
  print(paste0('i = ',i))
  eqt <- rnorm(n_eq, mean =0, sd = tau_sim)
  statt <- rnorm(n_stat, mean =0, sd =phi_s2s_sim)
  rect <- rnorm(n_rec, mean = 0, sd = phi_sim)
  
  data_reg$y_sim <- as.numeric(rowSums(t(t(data_reg[,names_coeffs]) * coeffs)) + eqt[eq] + statt[stat] + rect)
  fit_sim <- lmer(y_sim ~ M1 + M2 + MlogR + logR + R + logVS + (1|eq) + (1|stat), data_reg)
  ci <- confint(fit_sim, level = 0.9)
  
  
  fit_sim2 <- lmer(y_sim ~ M1 + M2 + MlogR + logR + R + (1|eq) +(1|stat), data_reg)
  deltaS_sim <- ranef(fit_sim2)$stat$`(Intercept)`
  deltaB_sim <- ranef(fit_sim2)$eq$`(Intercept)`
  tmp <- data.frame(unique(data_reg[,c('stat','logVS')]), deltaS_sim = deltaS_sim)
  
  ### all station terms
  fit_sim2a <- lm(deltaS_sim ~ logVS, tmp)
  ci2 <- confint(fit_sim2a, level = 0.9)
  
  data_reg$y_sim2 <- data_reg$y_sim - (predict(fit_sim2, re.form = NA) + predict(fit_sim2a, tmp)[stat])
  fit_sim2b <- lmer(y_sim2 ~ (1|eq) +(1|stat), data_reg)
  ci2b <- confint(fit_sim2b, level = 0.9)
  
  
  ### subset station terms
  fit_sim3a <- lm(deltaS_sim ~ logVS, tmp[idx,])
  ci3 <- confint(fit_sim3a, level = 0.9)
  
  data_reg$y_sim3 <- data_reg$y_sim - (predict(fit_sim2, re.form = NA) + predict(fit_sim3a, tmp)[stat])
  fit_sim3b <- lmer(y_sim3 ~ (1|eq) +(1|stat), data_reg)
  ci3b <- confint(fit_sim3b, level = 0.9)
  
  #### lmer on total residuals
  data_reg$y_sim4 <- data_reg$y_sim - predict(fit_sim2, re.form = NA)
  fit_sim4 <- lmer(y_sim4 ~ logVS + (1|eq) + (1|stat), data_reg)
  ci4 <- confint(fit_sim4, level = 0.9)
  
  res_val[i,] <- c(fixef(fit_sim)[7], as.data.frame(VarCorr(fit_sim))$sdcor,
                   summary(fit_sim2a$coefficients)[1], summary(fit_sim2b)$sigma,
                   as.data.frame(VarCorr(fit_sim2b))$sdcor,
                   summary(fit_sim3a$coefficients)[1], summary(fit_sim3b)$sigma,
                   as.data.frame(VarCorr(fit_sim3b))$sdcor,
                   fixef(fit_sim4)[2], as.data.frame(VarCorr(fit_sim4))$sdcor)
  
  res_ci[i,] <- c(sum(phi_s2s_sim > ci[1,1] & phi_s2s_sim <= ci[1,2]),
                  sum(coeffs[7] >= ci[10,1] & coeffs[7] <= ci[10,2]),
                  sum(phi_s2s_sim > ci2b[1,1] & phi_s2s_sim <= ci2b[1,2]),
                  sum(coeffs[7] >= ci2[2,1] & coeffs[7] <= ci2[2,2]),
                  sum(phi_s2s_sim > ci3b[1,1] & phi_s2s_sim <= ci3b[1,2]),
                  sum(coeffs[7] >= ci3[2,1] & coeffs[7] <= ci3[2,2]),
                  sum(phi_s2s_sim > ci4[1,1] & phi_s2s_sim <= ci4[1,2]),
                  sum(coeffs[7] >= ci4[5,1] & coeffs[7] <= ci4[5,2]))
  
  res_ci_diff[i,] <- c(diff(ci[10,]), diff(ci2[2,]), diff(ci3[2,]), diff(ci4[5,]))
}

# save(res_ci, res_ci_diff, res_val,
#      file = file.path('../results', 'res_vs_ita18_CB.Rdata'))