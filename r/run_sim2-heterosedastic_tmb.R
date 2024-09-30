rm(list = ls())

library(TMB)
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



compile(file.path('../tmb', "full_linear_mixed_model_tauM_phiM.cpp"))
dyn.load(dynlib(file.path('../tmb', "full_linear_mixed_model_tauM_phiM")))

init_values <- list(
  u_eq = rep(0, n_eq),    # Random effects for eq
  u_stat = rep(0, n_stat),# Random effects for stat
  beta = rep(0, length(coeffs)),
  log_phi_ss_sm = 0,
  log_phi_ss_lm = 0,
  log_tau_sm = 0,
  log_tau_lm = 0,
  log_phi_s2s = 0
)

n_sam <- 100

scale <- qnorm(0.95)


res_coeffs_tmb <-matrix(nrow = n_sam, ncol = length(coeffs))
res_coeffs_tmb_ci <-matrix(nrow = n_sam, ncol = length(coeffs))

res_phis2s_tmb <- matrix(nrow = n_sam, ncol = 2)
res_tau_tmb <- matrix(nrow = n_sam, ncol = 4)
res_phiss_tmb <- matrix(nrow = n_sam, ncol = 4)

set.seed(1701)
for(i in 1:n_sam) {
  print(paste0('i = ',i))
  eqt <- rnorm(n_eq, sd = tau_sim)
  rect <- rnorm(n_rec, sd = phi_ss_sim)
  statt <- rnorm(n_stat, sd = phi_s2s_sim)
  
  data_reg$y_sim <- as.matrix(data_reg[,names_coeffs]) %*% coeffs + eqt[eq] + statt[stat] + rect
  
  data_list <- list(Y = data_reg$y_sim, X = as.matrix(data_reg[,names_coeffs]),
                    eq= eq - 1, stat = stat - 1,
                    M1_eq=m1_eq, M2_eq =m2_eq, M1_rec = m1_rec, M2_rec = m2_rec)

  # Create TMB object
  obj <- MakeADFun(data = data_list, parameters = init_values, random = c("u_eq", "u_stat"), 
                   DLL = "full_linear_mixed_model_tauM_phiM")
  fit <- nlminb(obj$par, obj$fn, obj$gr)
  report <- sdreport(obj)
  
  fit_pars <- report$par.fixed
  res_coeffs_tmb[i,] <- fit_pars[1:length(coeffs)]
  sd_par <- sqrt(diag(report$cov.fixed))
  for(k in 1:length(coeffs)) {
    res_coeffs_tmb_ci[i,k] <- sum((fit_pars[k] - scale * sd_par[k]) < coeffs[k] 
                                  & (fit_pars[k] + scale * sd_par[k]) > coeffs[k])
  }
  
  ref <- 'log_phi_s2s'
  res_phis2s_tmb[i,1] <- exp(fit_pars[names(fit_pars) == ref])
  res_phis2s_tmb[i,2] <- sum(fit_pars[names(fit_pars) == ref] - scale * sd_par[names(fit_pars) == ref] < log(phi_s2s_sim)
      & fit_pars[names(fit_pars) == ref] + scale * sd_par[names(fit_pars) == ref] > log(phi_s2s_sim))
  
  ref <- 'log_phi_ss_sm'
  res_phiss_tmb[i,1] <- exp(fit_pars[names(fit_pars) == ref])
  res_phiss_tmb[i,3] <- sum(fit_pars[names(fit_pars) == ref] - scale * sd_par[names(fit_pars) == ref] < log(phi_ss_sim_val[1])
                             & fit_pars[names(fit_pars) == ref] + scale * sd_par[names(fit_pars) == ref] > log(phi_ss_sim_val[1]))
  
  ref <- 'log_phi_ss_lm'
  res_phiss_tmb[i,2] <- exp(fit_pars[names(fit_pars) == ref])
  res_phiss_tmb[i,4] <- sum(fit_pars[names(fit_pars) == ref] - scale * sd_par[names(fit_pars) == ref] < log(phi_ss_sim_val[2])
                            & fit_pars[names(fit_pars) == ref] + scale * sd_par[names(fit_pars) == ref] > log(phi_ss_sim_val[2]))
  
  ref <- 'log_tau_sm'
  res_tau_tmb[i,1] <- exp(fit_pars[names(fit_pars) == ref])
  res_tau_tmb[i,3] <- sum(fit_pars[names(fit_pars) == ref] - scale * sd_par[names(fit_pars) == ref] < log(tau_sim_val)[1]
                            & fit_pars[names(fit_pars) == ref] + scale * sd_par[names(fit_pars) == ref] > log(tau_sim_val[1]))
  
  ref <- 'log_tau_lm'
  res_tau_tmb[i,2] <- exp(fit_pars[names(fit_pars) == ref])
  res_tau_tmb[i,4] <- sum(fit_pars[names(fit_pars) == ref] - scale * sd_par[names(fit_pars) == ref] < log(tau_sim_val[2])
                            & fit_pars[names(fit_pars) == ref] + scale * sd_par[names(fit_pars) == ref] > log(tau_sim_val[2]))
  
}

save(res_coeffs_tmb, res_coeffs_tmb_ci, res_phis2s_tmb, res_phiss_tmb, res_tau_tmb,
     file = file.path('../results', 'results_sim2_heteroscedastic_coeff_tmb_CB.Rdata'))
