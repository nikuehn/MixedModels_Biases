rm(list = ls())

library(lme4)
library(tidyverse)

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
  
  n_sam <- 200
  mat_cor <- matrix(nrow = n_sam, ncol = 9)
  mat_cor_sample <- matrix(nrow = n_sam, ncol = 4)
  set.seed(5618)
  for(i in 1:n_sam) {
    eqt2 <- mvtnorm::rmvnorm(n_eq, sigma = cov_tau)
    statt2 <- mvtnorm::rmvnorm(n_stat, sigma = cov_s2s)
    rect2 <- mvtnorm::rmvnorm(n_rec, sigma = cov_ss)
    
    data_reg$y_sim1 <- eqt2[eq,1] + statt2[stat,1] + rect2[,1]
    data_reg$y_sim2 <- eqt2[eq,2] + statt2[stat,2] + rect2[,2]
    
    mat_cor_sample[i,] <- c(cor(statt2)[1,2],cor(eqt2)[1,2],cor(rect2)[1,2],cor(data_reg$y_sim1, data_reg$y_sim2))
    
    fit_sim1 <- lmer(y_sim1 ~ (1 | eq) + (1 | stat), data_reg)
    fit_sim2 <- lmer(y_sim2 ~ (1 | eq) + (1 | stat), data_reg)
    
    dB1 <- ranef(fit_sim1)$eq$`(Intercept)`
    dS1 <- ranef(fit_sim1)$stat$`(Intercept)`
    dWS1 <- resid(fit_sim1)
    dR1 <- data_reg$y_sim1 - predict(fit_sim1, re.form=NA)
    
    dB2 <- ranef(fit_sim2)$eq$`(Intercept)`
    dS2 <- ranef(fit_sim2)$stat$`(Intercept)`
    dWS2 <- resid(fit_sim2)
    dR2 <- data_reg$y_sim2 - predict(fit_sim2, re.form=NA)
    
    sds1 <- as.data.frame(VarCorr(fit_sim1))$sdcor
    sds2 <- as.data.frame(VarCorr(fit_sim2))$sdcor
    
    sds1a <- c(sd(dS1), sd(dB1), sd(dWS1))
    sds2a <- c(sd(dS2), sd(dB2), sd(dWS2))
    
    mat_cor[i,] <- c(cor(dS1, dS2), cor(dB1, dB2), cor(dWS1, dWS2), cor(dR1, dR2),
                     (sds1[1] * sds2[1] * cor(dS1, dS2) + sds1[2] * sds2[2] * cor(dB1, dB2) + sds1[3] * sds2[3] * cor(dWS1, dWS2)) /
                       (sqrt(sum(sds1^2)) * sqrt(sum(sds2^2))),
                     cov(dS1,dS2)/(sds1[1] * sds2[1]), 
                     cov(dB1,dB2)/(sds1[2] * sds2[2]),
                     cov(dWS1,dWS2)/(sds1[3] * sds2[3]),
                     (sds1a[1] * sds2a[1] * cor(dS1, dS2) + sds1a[2] * sds2a[2] * cor(dB1, dB2) + sds1a[3] * sds2a[3] * cor(dWS1, dWS2)) /
                       (sqrt(sum(sds1a^2)) * sqrt(sum(sds2a^2)))
    )
  }
  save(mat_cor, mat_cor_sample,
       file = sprintf('../results/res_corrre_CB14_%s.Rdata', cor_name))
}
