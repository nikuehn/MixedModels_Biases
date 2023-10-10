rm(list = ls())

library(lme4)
library(tidyverse)
library(matrixStats)
library(INLA)


### read data
data <- read.csv(file.path('../data','italian_data_pga_id_utm_stat.csv'))

# Set linear predictors
mh = 5.5
mref = 5.324
h = 6.924
attach(data)
b1 = (mag-mh)*(mag<=mh)
b2 = (mag-mh)*(mag>mh)
c1 = (mag-mref)*log10(sqrt(JB_complete^2+h^2))
c2 = log10(sqrt(JB_complete^2+h^2))
c3 = sqrt(JB_complete^2+h^2)
f1 = as.numeric(fm_type_code == "SS")
f2 = as.numeric(fm_type_code == "TF")
k = log10(vs30/800)*(vs30<=1500)+log10(1500/800)*(vs30>1500)
y = log10(rotD50_pga)
detach(data)

eq <- data$EQID
stat <- data$STATID

data_reg <- data.frame(Y = y,
                       M1 = b1,
                       M2 = b2,
                       MlogR = c1,
                       logR = c2,
                       R = c3,
                       Fss = f1,
                       Frv = f2,
                       logVS = k,
                       eq = eq,
                       stat = stat,
                       intercept = 1
)

# priors for standard deviation paramters
prior_prec_tau    <- list(prec = list(prior = 'pc.prec', param = c(0.3, 0.01)))
prior_prec_phiS2S    <- list(prec = list(prior = 'pc.prec', param = c(0.3, 0.01))) 
prior_prec_phiSS    <- list(prec = list(prior = 'pc.prec', param = c(0.3, 0.01))) 

n_eq <- max(eq)
n_stat <-  max(stat)
n_rec <- length(eq)

# correlation function
cMatern <- function(h, nu, kappa) {
  ifelse(h > 0, besselK(h * kappa, nu) * (h * kappa)^nu / 
           (gamma(nu) * 2^(nu - 1)), 1)
}

# Function to sample from zero mean multivariate normal
rmvnorm0 <- function(n, cov, R = NULL) { 
  if (is.null(R))
    R <- chol(cov)
  
  return(crossprod(R, matrix(rnorm(n * ncol(R)), ncol(R))))
}

co_stat_utm <- unique(data[,c("STATID", "X_stat","Y_stat")])[,c(2,3)]

# set spatial range for simulation
range <- 30
nu <- 1
kappa <- sqrt(8*nu)/range

coeffs <- c(3.421046409, 0.193954090, -0.021982777, 0.287149291, -1.405635476,
            -0.002911264, 0.085983743, 0.010500239, -0.394575970)
names_coeffs <- c("intercept", "M1", "M2", "MlogR", "logR", "R", "Fss", "Frv", "logVS")

# set standard deviatons for simulation
wvar <- 0.65 # variance ratio for phi_s2s
tau_sim <- 0.17
phi_s2s_sim <- 0.23
phi_s2s_0 <- sqrt((1 - wvar) * phi_s2s_sim^2)
phi_s2s_c <- sqrt(wvar * phi_s2s_sim^2)
phi_ss_sim <- 0.2

cov <- phi_s2s_c^2 * cMatern(as.matrix(dist(co_stat_utm)), nu, kappa) + diag(10^-9, n_stat)

# define mesh
max.edge2    <- 5 #0.04
bound.outer2 <- 40 #0.3
mesh = inla.mesh.2d(loc=co_stat_utm,
                    max.edge = c(1,5)*max.edge2,
                    # - use 5 times max.edge in the outer extension/offset/boundary
                    cutoff = max.edge2,#0.029,
                    offset = c(5 * max.edge2, bound.outer2))
print(mesh$n)

# SPDE prior for spatial models
spde_stat <- inla.spde2.pcmatern(
  # Mesh and smoothness parameter
  mesh = mesh, alpha = 2,
  prior.range = c(100, 0.9),
  prior.sigma = c(.3, 0.01))

# projector matrices and station index
A_stat <- inla.spde.make.A(mesh, loc = as.matrix(co_stat_utm[stat,]))
A_stat_unique   <- inla.spde.make.A(mesh, loc = as.matrix(co_stat_utm))
idx_stat   <- inla.spde.make.index("idx_stat",spde_stat$n.spde)

# full model formula
form_spatial_stat <- y ~ 0 + intercept + M1 + M2 + logR + MlogR + R + Fss + Frv + logVS +
  f(eq, model = "iid", hyper = prior_prec_tau) + 
  f(stat, model = "iid",hyper = prior_prec_phiS2S) +
  f(idx_stat, model = spde_stat)

# formula for model from site terms
form_spatial_stat_u <- y ~ f(idx_stat, model = spde_stat)

# fomula for total residuals
form_spatial_total <- y ~ 0 + intercept + 
  f(eq, model = "iid", hyper = prior_prec_tau) + 
  f(stat, model = "iid",hyper = prior_prec_phiS2S) +
  f(idx_stat, model = spde_stat)


n_sam <- 50
res_spatial <- matrix(nrow = n_sam, ncol = 8)
res_spatial_q05 <- matrix(nrow = n_sam, ncol = 8)
res_spatial_q50 <- matrix(nrow = n_sam, ncol = 8)
res_spatial_q95 <- matrix(nrow = n_sam, ncol = 8)

res_spatial_tot <- matrix(nrow = n_sam, ncol = 5)
res_spatial_tot_q05 <- matrix(nrow = n_sam, ncol = 5)
res_spatial_tot_q50 <- matrix(nrow = n_sam, ncol = 5)
res_spatial_tot_q95 <- matrix(nrow = n_sam, ncol = 5)

seed <- 1701
set.seed(seed)
for(i in 1:n_sam) {
  print(paste0('i = ',i))
  eqt <- rnorm(n_eq, mean =0, sd = tau_sim)
  statt <- rnorm(n_stat, mean =0, sd = phi_s2s_0)
  rect <- rnorm(n_rec, mean = 0, sd = phi_ss_sim)
  
  data_reg$y_sim <- as.numeric(rowSums(t(t(data_reg[,names_coeffs]) * coeffs)) +
                                 eqt[eq] + statt[stat] + rect + rmvnorm0(1, cov)[stat])
  
  ##### full model
  # create the stack
  stk_spatial_stat <- inla.stack(
    data = list(y = data_reg$y_sim), 
    A = list(A_stat, 1), 
    effects = list(idx_stat = idx_stat,
                   data_reg
    ))
  
  fit_inla_spatial_stat <- inla(form_spatial_stat,
                                data = inla.stack.data(stk_spatial_stat),
                                control.predictor = list(A = inla.stack.A(stk_spatial_stat)),
                                family="gaussian",
                                control.family = list(hyper = prior_prec_phiSS),
                                control.inla = list(int.strategy = "eb", strategy = "gaussian"),
                                quantiles = c(0.05, 0.5, 0.95)
  )
  
  ##### lme fit wthout spatial terms
  fit_lme_sim <- lmer(y_sim ~ M1 + M2 + logR + MlogR + R + Fss + Frv + logVS + (1|eq) + (1|stat), data_reg)
  
  ##### spatial model from site terms
  data_reg_stat <- data.frame(
    dS = ranef(fit_lme_sim)$stat$`(Intercept)`,
    intercept = 1
  )
  
  # create the stack
  stk_spatial_stat_u <- inla.stack(
    data = list(y = data_reg_stat$dS),
    #A = list(A_eq, A_eq, A_eq, A_stat, A_stat, 1), 
    A = list(A_stat_unique, 1), 
    effects = list(idx_stat = idx_stat,
                   data_reg_stat
    ))
  
  fit_inla_spatial_stat_u <- inla(form_spatial_stat_u,
                                  data = inla.stack.data(stk_spatial_stat_u),
                                  control.predictor = list(A = inla.stack.A(stk_spatial_stat_u)),
                                  family="gaussian",
                                  control.family = list(hyper = prior_prec_phiS2S),
                                  control.inla = list(int.strategy = "eb", strategy = "gaussian"),
                                  quantiles = c(0.05, 0.5, 0.95)
  )
  
  res_spatial[i,] <- c(fit_inla_spatial_stat$summary.hyperpar$mean,
                       fit_inla_spatial_stat_u$summary.hyperpar$mean)
  res_spatial_q05[i,] <- c(fit_inla_spatial_stat$summary.hyperpar[,'0.05quant'],
                           fit_inla_spatial_stat_u$summary.hyperpar[,'0.05quant'])
  res_spatial_q50[i,] <- c(fit_inla_spatial_stat$summary.hyperpar[,'0.5quant'],
                           fit_inla_spatial_stat_u$summary.hyperpar[,'0.5quant'])
  res_spatial_q95[i,] <- c(fit_inla_spatial_stat$summary.hyperpar[,'0.95quant'],
                           fit_inla_spatial_stat_u$summary.hyperpar[,'0.95quant'])
  
  print(res_spatial[i,])
  
  rm('fit_inla_spatial_stat','fit_inla_spatial_stat_u')
  
  ###### total residuals from lme fit
  data_reg$deltaR <- data_reg$y_sim - 
    (predict(fit_lme_sim,random.only=FALSE) - predict(fit_lme_sim,random.only=TRUE))
  
  # create the stack
  stk_spatial_total <- inla.stack(
    data = list(y = data_reg$deltaR),
    #A = list(A_eq, A_eq, A_eq, A_stat, A_stat, 1), 
    A = list(A_stat, 1), 
    effects = list(idx_stat = idx_stat,
                   data_reg
    ))
  
  fit_inla_spatial_total <- inla(form_spatial_total,
                                 data = inla.stack.data(stk_spatial_total),
                                 control.predictor = list(A = inla.stack.A(stk_spatial_total)),
                                 family="gaussian",
                                 control.family = list(hyper = prior_prec_phiSS),
                                 control.inla = list(int.strategy = "eb", strategy = "gaussian"),
                                 quantiles = c(0.05, 0.5, 0.95)
  )
  
  res_spatial_tot[i,] <- c(fit_inla_spatial_total$summary.hyperpar$mean)
  res_spatial_tot_q05[i,] <- c(fit_inla_spatial_total$summary.hyperpar[,'0.05quant'])
  res_spatial_tot_q50[i,] <- c(fit_inla_spatial_total$summary.hyperpar[,'0.5quant'])
  res_spatial_tot_q95[i,] <- c(fit_inla_spatial_total$summary.hyperpar[,'0.95quant'])
  
  print(res_spatial_tot[i,])
  
  rm('fit_inla_spatial_total')
}
save(res_spatial_tot, res_spatial_tot_q05, res_spatial_tot_q95, res_spatial_tot_q50,
     file = file.path('../results', sprintf('res_spatial_ita18b_italy_seed%d.Rdata', seed)))
save(res_spatial, res_spatial_q05, res_spatial_q95, res_spatial_q50,
     file = file.path('../results', sprintf('res_spatial_ita18_italy_seed%d.Rdata', seed)))


