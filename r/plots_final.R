rm(list = ls())

library(ggplot2)
library(posterior)
library(bayesplot)
library(tidyverse)
library(matrixStats)
library(latex2exp)
library(patchwork)
library(INLA)
library(lme4)
set1 <- RColorBrewer::brewer.pal(7, "Set1")

cols <- c("darkblue", "dodgerblue1", "cadetblue2", "white")

source('/Users/nico/GROUNDMOTION/PROJECTS/NONERGODIC/SCENARIO_MAPS/COREG/R_functions/functions_factor.R')
`%notin%` <- Negate(`%in%`)

lw <- 1.5
sp <- 4
wid <- 8
asp <- 0.8

size_title <- 30
size_st <- 20
theme_set(theme_bw() + theme(#panel.grid.minor = element_blank(),
  axis.title = element_text(size = size_title),
  axis.text = element_text(size = size_st),
  plot.title = element_text(size = size_title),
  plot.subtitle = element_text(size = size_st),
  legend.text = element_text(size = size_st),
  legend.title = element_text(size = size_st),
  plot.tag = element_text(size = size_title),
  legend.key.width = unit(1, "cm"),
  legend.box.background = element_rect(colour = "black"),
  panel.grid = element_line(color = "gray",linewidth = 0.75),
  legend.spacing.y = unit(0, "pt")
))

breaks <- 10^(-10:10)
minor_breaks <- rep(1:9, 21)*(10^rep(-10:10, each=9))


setwd('/Users/nico/GROUNDMOTION/PROJECTS/RESID_VAR/Git/MixedModels_Biases/r/')

path_data <- file.path('../data/')
path_plot <- file.path('../plots')
path_res <- file.path('../results/')




################################################################################
######### Figure 1 and 2
#### data Italy
model_name <- 'Italy'

# ------------------------------------------
# standard errors of random effects

data <- read.csv(file.path(path_data,'italian_data_pga_id_utm_stat.csv'))
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

n_rec <- length(b1)
eq <- data$EQID
stat <- data$STATID
n_eq <- max(eq)
n_stat <- max(stat)
n_rec <- nrow(data)

mageq <- unique(data[,c('EQID','mag')])[,2]

data_reg <- data.frame(Y = y,
                       M1 = b1,
                       M2 = b2,
                       MlnR = c1,
                       lnR = c2,
                       R = c3,
                       Fss = f1,
                       Frv = f2,
                       lnVS = k,
                       eq = eq,
                       stat = stat,
                       intercept = 1,
                       M = data$mag
)

# priors for standard deviation paramters
prior_prec_tau    <- list(prec = list(prior = 'pc.prec', param = c(0.3, 0.01)))
prior_prec_phiS2S    <- list(prec = list(prior = 'pc.prec', param = c(0.3, 0.01))) 
prior_prec_phiSS    <- list(prec = list(prior = 'pc.prec', param = c(0.3, 0.01))) 

form <- Y ~ M1 + M2 + lnR + MlnR + R + Fss + Frv + lnVS +
  f(eq, model = "iid", hyper = prior_prec_tau) + 
  f(stat, model = "iid",hyper = prior_prec_phiS2S)

fit_inla <- inla(form, 
                 data = data_reg,
                 family="gaussian",
                 control.family = list(hyper = prior_prec_phiSS),
                 control.compute = list(dic = TRUE, cpo = TRUE, waic = TRUE)
)

sd_deltaS_inla <-fit_inla$summary.random$stat$sd
sd_deltaB_inla <-fit_inla$summary.random$eq$sd
sd_deltaWS_inla <- fit_inla$summary.fitted.values$sd

fit_lme <- lmer(Y ~ M1 + M2 + lnR + MlnR + R + Fss + Frv + lnVS + (1|eq) + (1|stat), data_reg)

tmp <- as.data.frame(VarCorr(fit_lme))$sdcor
phi_s2s_lme <- tmp[1]
tau_lme <- tmp[2]
phi_ss_lme <- tmp[3]

deltaB <- ranef(fit_lme)$eq$`(Intercept)`
deltaS <- ranef(fit_lme)$stat$`(Intercept)`
sd_deltaB <- as.numeric(arm::se.ranef(fit_lme)$eq)
sd_deltaS <- as.numeric(arm::se.ranef(fit_lme)$stat)
deltaWS <- data_reg$Y - predict(fit_lme)
sd_deltaWS <- sqrt(sd_deltaB[eq]^2 + sd_deltaS[stat]^2)

df_eq <- data.frame(unique(data[,c('EQID','mag')]),
                    dB = deltaB, sd_dB_lmer = sd_deltaB,
                    sd_dB_inla = sd_deltaB_inla,
                    nrec = as.numeric(table(eq)))
df_stat <- data.frame(unique(data[,c('STATID','vs30')]),
                      logvs = unique(data_reg[,c('stat','lnVS')])[,2],
                      dS = deltaS, sd_dS_lmer = sd_deltaS,
                      sd_dS_inla = sd_deltaS_inla,
                      nrec = as.numeric(table(stat)))

p1 <- df_eq %>% pivot_longer(c(sd_dB_lmer, sd_dB_inla)) %>%
  ggplot() +
  geom_point(aes(x = nrec, y = value, color = name, size = mag, shape = name)) +
  scale_x_log10(breaks = breaks, minor_breaks = minor_breaks) +
  ylim(c(0,0.125)) +
  labs(x = 'number of records per event', y = TeX("$\\psi(\\widehat{\\delta B})$")) +
  guides(color = guide_legend(title=NULL, override.aes = list(size = sp)),
         size = guide_legend(title = 'M'),
         shape = guide_legend(title=NULL)) +
  scale_color_manual(values=c("blue", 'red'),
                     labels=c("inla", "lmer")) +
  scale_shape_manual(values=c(16,17),
                     labels=c("inla", "lmer")) +
  theme(legend.position = c(0.85,0.75)) +
  labs(tag = '(a)')

p2 <- df_stat %>% pivot_longer(c(sd_dS_lmer, sd_dS_inla)) %>%
  ggplot() +
  geom_point(aes(x = nrec, y = value, color = name, size = vs30, shape = name)) +
  scale_x_log10(breaks = breaks, minor_breaks = minor_breaks) +
  ylim(c(0,0.18)) +
  labs(x = 'number of records per station', y = TeX("$\\psi(\\widehat{\\delta S})$")) +
  guides(color = guide_legend(title=NULL), size = guide_legend(title = 'VS30')) +
  guides(color = guide_legend(title=NULL, override.aes = list(size = sp)),
         size = guide_legend(title = 'M'),
         shape = guide_legend(title=NULL)) +
  scale_color_manual(values=c("blue", 'red'),
                     labels=c("inla", "lmer")) +
  scale_shape_manual(values=c(16,17),
                     labels=c("inla", "lmer")) +
  theme(legend.position = c(0.85,0.75)) +
  labs(tag = '(b)')

plot_name <- 'figure_1'
ggsave(file.path(path_plot, sprintf('%s.pdf', plot_name)), patchwork::wrap_plots(p1,p2),
       width = 2 * wid, height = asp * wid)
ggsave(file.path(path_plot, sprintf('%s_a.pdf', plot_name)), p1,
       width = wid, height = asp * wid)
ggsave(file.path(path_plot, sprintf('%s_b.pdf', plot_name)), p2,
       width = wid, height = asp * wid)


### figure 2
plot_name <- 'figure_2'
pl <- data.frame(inla = sd_deltaWS_inla, lmer = sd_deltaWS, M = data_reg$M,
                 nrec = as.numeric(table(stat))[stat]) %>%
  ggplot() +
  geom_point(aes(x = inla, y = lmer), size = sp) +
  geom_abline(color = 'red', linewidth = lw) +
  lims(x = c(0.03,0.21), y = c(0.03, 0.21)) +
  labs(x = TeX("$\\psi(\\widehat{\\delta WS}_{inla})$"),
       y = TeX("$\\psi(\\widehat{\\delta WS}_{lmer})$"))
ggsave(file.path(path_plot, sprintf('%s.pdf', plot_name)), pl,
       width = wid, height = asp * wid)



################################################################################
######### Figure 3
plot_name <- 'figure_3'
set.seed(8472)
dB1 <- 0.01603136
dB2 <- 0.1904922
dB3 <- -0.3002733

sd_dB1 <- 0.2422961
sd_dB2 <- 0.2700048
sd_dB3 <- 0.1084342

sam <- c(rnorm(1, mean = dB1, sd = sd_dB1),
         rnorm(1, mean = dB2, sd = sd_dB2),
         rnorm(1, mean = dB3, sd = sd_dB3))

xv <- seq(-0.8,0.8, by = 0.01)

pl <- rbind(data.frame(x = xv, y = dnorm(xv, mean = dB1, sd = sd_dB1), eq = '1'),
            data.frame(x = xv, y = dnorm(xv, mean = dB2, sd = sd_dB2), eq = '2'),
            data.frame(x = xv, y = dnorm(xv, mean = dB3, sd = sd_dB3), eq = '3')) |>
  ggplot() +
  geom_line(aes(x = x, y = y, color = eq), linewidth = lw) +
  geom_point(data.frame(x = c(dB1, dB2, dB3),
                        y = c(dnorm(dB1, mean = dB1, sd = sd_dB1),
                              dnorm(dB2, mean = dB2, sd = sd_dB2),
                              dnorm(dB3, mean = dB3, sd = sd_dB3))),
             mapping = aes(x = x, y = y), size = sp) +
  geom_point(data.frame(x = sam,
                        y = c(dnorm(sam[1], mean = dB1, sd = sd_dB1),
                              dnorm(sam[2], mean = dB2, sd = sd_dB2),
                              dnorm(sam[3], mean = dB3, sd = sd_dB3))),
             mapping = aes(x = x, y = y), size = sp, color = 'red') +
  labs(x = expression(paste(delta,'B')), y = 'density') +
  scale_color_manual(values = c('gray','gray','gray')) +
  theme(legend.position = 'none')
ggsave(file.path(path_plot, sprintf('%s.pdf', plot_name)), pl,
       width = wid, height = asp * wid)



################################################################################
######### Figure 4
plot_name <- 'figure_4'

tau_sim <- 0.4
phi_s2s_sim <- 0.43
phi_ss_sim <- 0.5

model_name <- 'CB'
load(file.path(path_res, sprintf('results_sim1_%s.Rdata', model_name)))



df <- data.frame(res_val[,c(1,4,7)], res_sd[,c(1,4)]) %>%
  set_names(c('lmer_max', 'stan_mean','stan_median','lmer_sd(dS)','stan_sd(dS)')) %>%
  pivot_longer(everything(), names_sep = '_', names_to = c('model','type'))

df_density <- df %>%
  group_by(model, type) %>%
  do({
    dens <- density(.$value)
    data.frame(value = dens$x, density = dens$y)
  })

p1 <- ggplot(df, aes(x = value, color = model, linetype = type)) +
  geom_density(linewidth = 1.5, key_glyph = draw_key_path) + 
  geom_point(data = df_density %>% filter(row_number() %% 20 == 0),
             aes(x = value, y = density, shape = model), size = sp) +
  geom_vline(xintercept = phi_s2s_sim, linewidth = lw) +
  labs(x = TeX('$\\hat{\\phi}_{S2S}$')) +
  scale_linetype_manual(values = c(1,2,3,4),
                        labels = c('max','mean','median',TeX("sd($\\widehat{\\delta S}$)"))) +
  scale_color_manual(values = c('red','blue'), name = NULL) +
  scale_shape_manual(values = c(16,17), name = NULL) +
  guides(color = guide_legend(title=NULL), linetype = guide_legend(title=NULL)) +
  theme(legend.position = "inside",
        legend.position.inside = c(0.5,0.77),
        legend.key.width = unit(2,'cm')) +
  labs(tag = '(a)')


df <- data.frame(res_val[,c(2,5,8)], res_sd[,c(2,5)]) %>%
  set_names(c('lmer_max', 'stan_mean','stan_median','lmer_sd(dB)','stan_sd(dB)')) %>%
  pivot_longer(everything(), names_sep = '_', names_to = c('model','type'))

df_density <- df %>%
  group_by(model, type) %>%
  do({
    dens <- density(.$value)
    data.frame(value = dens$x, density = dens$y)
  })

p2 <- ggplot(df, aes(x = value, color = model, linetype = type)) +
  geom_density(linewidth = lw, key_glyph = draw_key_path) + 
  geom_point(data = df_density %>% filter(row_number() %% 20 == 0),
             aes(x = value, y = density, shape = model), size = sp) +
  geom_vline(xintercept = tau_sim, linewidth = lw) +
  labs(x = TeX('$\\hat{\\tau}$')) +
  scale_linetype_manual(values = c(1,2,3,4),
                        labels = c('max','mean','median',TeX("sd($\\widehat{\\delta S}$)"))) +
  scale_color_manual(values = c('red','blue'), name = NULL) +
  scale_shape_manual(values = c(16,17), name = NULL) +
  guides(color = guide_legend(title=NULL), linetype = guide_legend(title=NULL)) +
  theme(legend.position = "none") +
  labs(tag = '(b)')


df <- data.frame(res_val[,c(3,6,9)], res_sd[,c(3,6)]) %>%
  set_names(c('lmer_max', 'stan_mean','stan_median','lmer_sd(dWS)','stan_sd(dWS)')) %>%
  pivot_longer(everything(), names_sep = '_', names_to = c('model','type'))

df_density <- df %>%
  group_by(model, type) %>%
  do({
    dens <- density(.$value)
    data.frame(value = dens$x, density = dens$y)
  })

p3 <- ggplot(df, aes(x = value, color = model, linetype = type)) +
  geom_density(linewidth = lw, key_glyph = draw_key_path) + 
  geom_point(data = df_density %>% filter(row_number() %% 20 == 0),
             aes(x = value, y = density, shape = model), size = sp) +
  geom_vline(xintercept = phi_ss_sim, linewidth = lw) +
  labs(x = TeX('$\\hat{\\phi}_{SS}$')) +
  scale_linetype_manual(values = c(1,2,3,4),
                        labels = c('max','mean','median',TeX("sd($\\widehat{\\delta S}$)"))) +
  scale_color_manual(values = c('red','blue'), name = NULL) +
  scale_shape_manual(values = c(16,17), name = NULL) +
  guides(color = guide_legend(title=NULL), linetype = guide_legend(title=NULL)) +
  theme(legend.position = "none") +
  labs(tag = '(c)')

ggsave(file.path(path_plot, sprintf('%s.pdf', plot_name)), wrap_plots(p1,p2,p3),
       width = 3 * wid, height = asp * wid)
ggsave(file.path(path_plot, sprintf('%s_a.pdf', plot_name)), p1,
       width = wid, height = asp * wid)
ggsave(file.path(path_plot, sprintf('%s_b.pdf', plot_name)), p2,
       width = wid, height = asp * wid)
ggsave(file.path(path_plot, sprintf('%s_c.pdf', plot_name)), p3,
       width = wid, height = asp * wid)

################################################################################
######### Figure 5
plot_name <- 'figure_5'

load(file = file.path(path_res, 'results_sim2_heteroscedastic_coeff_CB.Rdata'))
load(file = file.path(path_res, 'results_sim2_heteroscedastic_coeff_stan2_CB.Rdata'))

coeffs <- c(3.421046409, 0.193954090, -0.021982777, 0.287149291, -1.405635476, -0.002911264, -0.394575970)
names_coeffs <- c("intercept", "M1", "M2", "MlogR", "logR", "R", "logVS")

phi_s2s_sim <- 0.43
tau_sim_val <- c(0.4,0.25)
phi_sim_val <- c(0.55,0.4)
mb_tau <- c(5,6)
mb_phi <- c(4.5,5.5)

df <- data.frame(res_phi, res_phi_stan[,c(1,2)], res_phi_stan2[,c(1,2)]) %>%
  set_names(c('sd(dWS)_lowm','sd(dWS)+unc_lowm','sd(dWS)_largem','sd(dWS)+unc_largem',
              'stan_lowm','stan_largem','stanf_lowm','stanf_largem')) %>%
  pivot_longer(everything(), names_to = c('model','mag'),names_sep = '_')

df_density <- df %>%
  group_by(model, mag) %>%
  do({
    dens <- density(.$value)
    data.frame(value = dens$x, density = dens$y)
  })

sm <- c(5,4,17,16)

p1 <- df %>%
  ggplot() +
  geom_density(aes(x = value, color = model, linetype = mag), linewidth = lw, key_glyph = draw_key_path) +
  geom_point(data = df_density %>% filter(row_number() %% 20 == 0),
             aes(x = value, y = density, shape = model, color = model), size = sp) +
  geom_vline(xintercept = phi_sim_val[2], linewidth = lw) +
  geom_vline(xintercept = phi_sim_val[1], linetype = 'dashed', linewidth = lw) +
  scale_color_manual(values = c('orange','red','blue','cyan'),
                     labels = c(TeX("sd(\\widehat{\\delta WS})"),
                                TeX("sd(\\widehat{\\delta WS} + unc)"),
                                TeX('stan ($\\delta R$)'), 'stan (full)'),
                     name = NULL) +
  scale_shape_manual(values = sm, 
                     labels = c(TeX("sd(\\widehat{\\delta WS})"),
                                TeX("sd(\\widehat{\\delta WS} + unc)"),
                                TeX('stan ($\\delta R$)'), 'stan (full)'),
                     name = NULL) +
  scale_linetype_manual(values = c(1,2),
                        labels = c(TeX(sprintf("$M \\geq %.1f$",mb_phi[2])), 
                                   TeX(sprintf("$M \\leq %.1f$",mb_phi[1])))) +
  guides(linetype = guide_legend(title = NULL), 
         color = guide_legend(title = NULL),
         shape = guide_legend(title = NULL)) +
  theme(legend.position = c(0.5,0.8),
        legend.key.width = unit(2,'cm')) +
  labs(x = expression(paste(widehat(phi)[SS]))) +
  lims(y = c(0,100)) +
  labs(tag = '(a)')



df <- data.frame(res_tau, res_tau_stan[,c(1,2)], res_tau_stan2[,c(1,2)]) %>%
  set_names(c('sd(dB)_lowm','sd(dB)+unc_lowm','sd(dB)_largem','sd(dB)+unc_largem',
              'stan_lowm','stan_largem','stanf_lowm','stanf_largem')) %>%
  pivot_longer(everything(), names_to = c('model','mag'),names_sep = '_')

df_density <- df %>%
  group_by(model, mag) %>%
  do({
    dens <- density(.$value)
    data.frame(value = dens$x, density = dens$y)
  })

p2 <- df %>%
  ggplot() +
  geom_density(aes(x = value, color = model, linetype = mag), linewidth = lw, key_glyph = draw_key_path) +
  geom_point(data = df_density %>% filter(row_number() %% 20 == 0),
             aes(x = value, y = density, shape = model, color = model), size = sp) +
  geom_vline(xintercept = tau_sim_val[2], linewidth = lw) +
  geom_vline(xintercept = tau_sim_val[1], linetype = 'dashed', linewidth = lw) +
  scale_color_manual(values = c('orange','red','blue','cyan'),
                     labels = c(TeX("sd(\\widehat{\\delta B})"),
                                TeX("sd(\\widehat{\\delta B} + unc)"),
                                TeX('stan ($\\delta R$)'), 'stan (full)'),
                     name = NULL) +
  scale_shape_manual(values = sm, 
                     labels = c(TeX("sd(\\widehat{\\delta B})"),
                                TeX("sd(\\widehat{\\delta B} + unc)"),
                                TeX('stan ($\\delta R$)'), 'stan (full)'),
                     name = NULL) +
  scale_linetype_manual(values = c(1,2),
                        labels = c(TeX(sprintf("$M \\geq %.1f$",mb_tau[2])), 
                                   TeX(sprintf("$M \\leq %.1f$",mb_tau[1])))) +
  guides(color = guide_legend(title = NULL), linetype = guide_legend(title = NULL)) +
  theme(legend.position = c(0.2,0.8),
        legend.key.width = unit(2,'cm')) +
  labs(x = expression(paste(widehat(tau)))) +
  labs(tag = '(b)')




df <- data.frame(res_phis2s[,c(1,3)], res_phis2s_stan2[,1]) %>%
  set_names(c('lmer','stan','stanf')) %>%
  pivot_longer(everything())

df_density <- df %>%
  group_by(name) %>%
  do(data.frame(density = density(.$value)[c("x", "y")]))


p3 <- df %>%
  ggplot() +
  geom_density(aes(x = value, color = name), linewidth = lw, key_glyph = draw_key_path) +
  geom_point(data = df_density %>% filter(row_number() %% 30 == 0), 
             aes(x = density.x, y = density.y, shape = name, color = name), size = sp) +
  geom_vline(xintercept = phi_s2s_sim, linewidth = lw) +
  scale_color_manual(values = c('red','blue','cyan'),
                     labels = c('lmer', TeX('stan ($\\delta R$)'), 'stan (full'),
                     name = NULL) +
  scale_shape_manual(values = c(15,16,17),
                     labels = c('lmer', TeX('stan ($\\delta R$)'), 'stan (full'),
                     name = NULL) +
  guides(color = guide_legend(title = NULL)) +
  theme(legend.position = c(0.15,0.9),
        legend.key.width = unit(2,'cm')) +
  labs(x = expression(paste(widehat(phi)[S2S])), y = 'density') +
  labs(tag = '(c)')

ggsave(file.path(path_plot, sprintf('%s.pdf', plot_name)), wrap_plots(p1,p2,p3),
       width = 3 * wid, height = asp * wid)
ggsave(file.path(path_plot, sprintf('%s_a.pdf', plot_name)), p1,
       width = wid, height = asp * wid)
ggsave(file.path(path_plot, sprintf('%s_b.pdf', plot_name)), p2,
       width = wid, height = asp * wid)
ggsave(file.path(path_plot, sprintf('%s_c.pdf', plot_name)), p3,
       width = wid, height = asp * wid)

################################################################################
######### Figure 6
plot_name <- 'figure_6'

load(file = file.path(path_res, 'results_sim2_heteroscedastic_coeff_CB.Rdata'))
load(file = file.path(path_res, 'results_sim2_heteroscedastic_coeff_Italy.Rdata'))

phi_s2s_sim <- 0.43
tau_sim_val <- c(0.4,0.25)
phi_sim_val <- c(0.55,0.4)
mb_tau <- c(5,6)
mb_phi <- c(4.5,5.5)

col_cb14 <- 'orange'
col_it <- 'darkgreen'

df <- data.frame(res_phi[,c(1,3)], res_phi_it[,c(1,3)]) %>%
  set_names(c('CB14_lowm','CB14_largem','Italy_lowm','Italy_largem')) %>%
  pivot_longer(everything(), names_to = c('model','mag'),names_sep = '_')

df_density <- df %>%
  group_by(model, mag) %>%
  do({
    dens <- density(.$value)
    data.frame(value = dens$x, density = dens$y)
  })

p1 <- df %>%
  ggplot() +
  geom_density(aes(x = value, color = model, linetype = mag), linewidth = lw, key_glyph = draw_key_path) +
  geom_point(data = df_density %>% filter(row_number() %% 30 == 0),
             aes(x = value, y = density, shape = model, color = model), size = sp) +
  geom_vline(xintercept = phi_sim_val[2], linewidth = lw) +
  geom_vline(xintercept = phi_sim_val[1], linetype = 'dashed', linewidth = lw) +
  scale_color_manual(values = c(col_cb14,col_it),
                     name = NULL) +
  scale_shape_manual(values = c(16,17), name = NULL) +
  scale_linetype_manual(values = c(1,2),
                        labels = c(TeX(sprintf("$M \\geq %.1f$",mb_phi[2])), 
                                   TeX(sprintf("$M \\leq %.1f$",mb_phi[1])))) +
  guides(color = guide_legend(title = NULL), linetype = guide_legend(title = NULL)) +
  theme(legend.position = c(0.5,0.8),
        legend.key.width = unit(2,'cm')) +
  labs(x = expression(paste(widehat(phi)[SS]))) +
  lims(y = c(0,100)) +
  labs(tag = '(a)')



df <- data.frame(res_tau[,c(1,3)], res_tau_it[,c(1,3)]) %>%
  set_names(c('CB14_lowm','CB14_largem','Italy_lowm','Italy_largem')) %>%
  pivot_longer(everything(), names_to = c('model','mag'),names_sep = '_')

df_density <- df %>%
  group_by(model, mag) %>%
  do({
    dens <- density(.$value)
    data.frame(value = dens$x, density = dens$y)
  })

p2 <- df %>%
  ggplot() +
  geom_density(aes(x = value, color = model, linetype = mag), linewidth = lw, key_glyph = draw_key_path) +
  geom_point(data = df_density %>% filter(row_number() %% 30 == 0),
             aes(x = value, y = density, shape = model, color = model), size = sp) +
  geom_vline(xintercept = tau_sim_val[2], linewidth = lw) +
  geom_vline(xintercept = tau_sim_val[1], linetype = 'dashed', linewidth = lw) +
  scale_color_manual(values = c(col_cb14,col_it),
                     name = NULL) +
  scale_shape_manual(values = c(16,17), name = NULL) +
  scale_linetype_manual(values = c(1,2),
                        labels = c(TeX(sprintf("$M \\geq %.1f$",mb_tau[2])), 
                                   TeX(sprintf("$M \\leq %.1f$",mb_tau[1])))) +
  guides(color = guide_legend(title = NULL), linetype = guide_legend(title = NULL)) +
  theme(legend.position = c(0.2,0.8),
        legend.key.width = unit(2,'cm')) +
  labs(x = expression(paste(widehat(tau)))) +
  labs(tag = '(b)')


df <- data.frame(CB14 = res_phis2s[,1], Italy = res_sd_lmer_it[,1]) %>%
  pivot_longer(everything())

df_density <- df %>%
  group_by(name) %>%
  do(data.frame(density = density(.$value)[c("x", "y")]))

p3 <- df %>%
  ggplot() +
  geom_density(aes(x = value, color = name), linewidth = lw, key_glyph = draw_key_path) +
  geom_point(data = df_density %>% filter(row_number() %% 30 == 0), 
             aes(x = density.x, y = density.y, shape = name, color = name), size = sp) +
  geom_vline(xintercept = phi_s2s_sim, linewidth = lw) +
  scale_color_manual(values = c(col_cb14,col_it),
                     name = NULL) +
  scale_shape_manual(values = c(16,17), name = NULL) +
  guides(color = guide_legend(title = NULL)) +
  theme(legend.position = c(0.85,0.9),
        legend.key.width = unit(2,'cm')) +
  labs(x = expression(paste(widehat(phi)[S2S]))) +
  labs(tag = '(c)', y = 'density')

ggsave(file.path(path_plot, sprintf('%s.pdf', plot_name)), 
       patchwork::wrap_plots(p1,p2,p3),
       width = 3 * wid, height = asp * wid)
ggsave(file.path(path_plot, sprintf('%s_a.pdf', plot_name)), p1,
       width = wid, height = asp * wid)
ggsave(file.path(path_plot, sprintf('%s_b.pdf', plot_name)), p2,
       width = wid, height = asp * wid)
ggsave(file.path(path_plot, sprintf('%s_c.pdf', plot_name)), p3,
       width = wid, height = asp * wid)

################################################################################
######### Figure 7
plot_name <- 'figure_7'


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

tau_sim <- 0.17
phi_s2s_sim <- 0.23
phi_sim <- 0.2

set.seed(1701)
eqt <- rnorm(n_eq, mean =0, sd = tau_sim)
statt <- rnorm(n_stat, mean =0, sd =phi_s2s_sim)
rect <- rnorm(n_rec, mean = 0, sd = phi_sim)

data_reg$y_sim <- as.numeric(rowSums(t(t(data_reg[,names_coeffs]) * coeffs)) + eqt[eq] + statt[stat] + rect)

fit_sim <- lmer(y_sim ~ M1 + M2 + MlogR + logR + R + logVS + (1|eq) + (1|stat), data_reg)
fit_sim2 <- lmer(y_sim ~ M1 + M2 + MlogR + logR + R + (1|eq) + (1|stat), data_reg)
deltaS_sim <- ranef(fit_sim2)$stat$`(Intercept)`
deltaB_sim <- ranef(fit_sim2)$eq$`(Intercept)`
fit_sim2a <- lm(deltaS_sim ~ logVS, data.frame(unique(data_reg[,c('stat','logVS')]),
                                              deltaS_sim = deltaS_sim))

vsplot <- exp(seq(log(100),log(2000), by=(log(2000) - log(100)) / 20))
vsplot2 <- log10(vsplot/800)*(vsplot<=1500)+log10(1500/800)*(vsplot>1500)

df2 <- data.frame(unique(data_reg[,c('stat','logVS','VS_gmean')]),
                  deltaS_sim = deltaS_sim)

df2a <- data.frame(vs =vsplot, 
                   p1 = vsplot2 * coeffs[7] - mean(df2$logVS) * coeffs[7],
                   p2 = predict(fit_sim2a, data.frame(logVS=vsplot2))) %>%
  #p3 = vsplot2 * fixef(fit_sim)[7] - mean(df2$lnVS) * fixef(fit_sim)[7]) %>%
  pivot_longer(!vs)


pl <- ggplot(df2) +
  geom_point(aes(x = VS_gmean, y = deltaS_sim), size = sp) +
  geom_line(df2a, mapping = aes(x = vs, y = value, color = name), linewidth = lw) +
  geom_point(df2a, mapping = aes(x = vs, y = value, color = name, shape = name), size = sp) +
  scale_color_manual(values = c('red','cyan'),
                     labels = c('true',
                                TeX('fit from $\\widehat{\\delta S}$'))) +
  scale_x_log10(breaks = breaks, minor_breaks = minor_breaks) +
  guides(color = guide_legend(title = NULL), shape = 'none') +
  theme(legend.position = c(0.2,0.15)) +
  labs(x = TeX('$V_{S30}$ (m/s)'), y = TeX('$\\widehat{\\delta S}$'))
ggsave(file.path(path_plot, sprintf('%s.pdf', plot_name)), pl,
       width = wid, height = asp * wid)



################################################################################
######### Figure 8
plot_name <- 'figure_8'

names <- c('full','dS','dS(N>10)','dR')
set1 <- RColorBrewer::brewer.pal(7, "Set1")
coeff_vs <- -0.394575970

load(file = file.path(path_res, sprintf('res_vs_ita18_%s.Rdata', 'CB')))
res_val_cb <- res_val
res_ci_diff_cb <- res_ci_diff

load(file = file.path(path_res, sprintf('res_vs_ita18_%s.Rdata', 'italy')))
res_val_it <- res_val
res_ci_diff_it <- res_ci_diff

df <- rbind(data.frame(res_val_cb[,c(1,5,10,15)], data = 'CB14'),
            data.frame(res_val_it[,c(1,5,10,15)], data = 'Italy')) %>%
  set_names(c(names, 'data')) %>%
  pivot_longer(!data)
df$name <- factor(df$name, names)

df_density <- df %>%
  group_by(name, data) %>%
  do({
    dens <- density(.$value)
    data.frame(value = dens$x, density = dens$y)
  })

p1 <- ggplot(df, aes(x = value, color = name, linetype = data, shape = name)) +
  geom_density(linewidth = lw, key_glyph = draw_key_path) +
  geom_vline(xintercept = coeff_vs, linewidth = lw) +
  geom_point(data = df_density %>% filter(row_number() %% 20 == 0),
             aes(x = value, y = density), size = sp) +
  #scale_color_brewer(palette = 'Set1') +
  scale_color_manual(values=set1,
                     labels = c('full',TeX('$\\widehat{\\delta S}$'),
                                TeX('$\\widehat{\\delta S}$ ($N \\geq 10$)'),
                                TeX('$\\widehat{\\delta R}$'))) +
  scale_shape_manual(values = c(16,17,15,11), 
                     labels = c('full',TeX('$\\widehat{\\delta S}$'),
                                TeX('$\\widehat{\\delta S}$ ($N \\geq 10$)'),
                                TeX('$\\widehat{\\delta R}$')),
                     name = NULL) +
  guides(color = guide_legend(title=NULL), linetype = guide_legend(title=NULL)) +
  labs(x = TeX('$c_7$')) +
  theme(legend.position = c(0.2,0.8),
        legend.box = "horizontal") +
  labs(tag = '(a)')


df <- rbind(data.frame(res_ci_diff_cb, data = 'CB14'),
            data.frame(res_ci_diff_it, data = 'Italy')) %>%
  set_names(c(names, 'data')) %>%
  pivot_longer(!data)
df$name <- factor(df$name, names)

df_density <- df %>%
  group_by(name, data) %>%
  do({
    dens <- density(.$value)
    data.frame(value = dens$x, density = dens$y)
  })

xlab <- expression(paste(CI[0.9],'(',c[7],')'))
p2 <- ggplot(df, aes(x = value, color = name, linetype = data, shape = name)) +
  geom_density(linewidth = lw, key_glyph = draw_key_path) +
  geom_point(data = df_density %>% filter(row_number() %% 20 == 0),
             aes(x = value, y = density), size = sp) +
  #scale_color_brewer(palette = 'Set1') +
  scale_color_manual(values=set1,
                     labels = c('full',TeX('$\\widehat{\\delta S}$'),
                                TeX('$\\widehat{\\delta S}$ ($N \\geq 10$)'),
                                TeX('$\\widehat{\\delta R}$'))) +
  scale_shape_manual(values = c(16,17,15,11), 
                     labels = c('full',TeX('$\\widehat{\\delta S}$'),
                                TeX('$\\widehat{\\delta S}$ ($N \\geq 10$)'),
                                TeX('$\\widehat{\\delta R}$')),
                     name = NULL) +
  guides(color = guide_legend(title=NULL), linetype = guide_legend(title=NULL)) +
  labs(x = xlab) +
  theme(legend.position = c(0.7,0.8), legend.box = "horizontal") +
  labs(x = xlab) +
  labs(tag = '(b)')

ggsave(file.path(path_plot, sprintf('%s.pdf', plot_name)),
       wrap_plots(p1 + theme(legend.position = 'none'), p2),
       width = 2 * wid, height = asp * wid)
ggsave(file.path(path_plot, sprintf('%s_a.pdf', plot_name)),
       p1 + theme(legend.position = 'none'),
       width = wid, height = asp * wid)
ggsave(file.path(path_plot, sprintf('%s_b.pdf', plot_name)),p2,
       width = wid, height = asp * wid)

################################################################################
######### Figure 9
plot_name <- 'figure_9'

df_res_cor <- read.csv(file.path(path_res, 'res_sim_cor_CB_N50.csv'))

tau_sim <- 0.4
phi_s2s_sim <- 0.43
phi_ss_sim <- 0.5

rho <- 0.7
tau2 <- 1

pl <- df_res_cor %>% pivot_longer(c(cor_sim, cor_lme, cor_mean)) %>%
  ggplot() +
  geom_density(aes(x = value, color = name, linetype = name), linewidth = lw, key_glyph = draw_key_path) +
  geom_vline(aes(xintercept = rho), linewidth = lw) +
  scale_color_manual(values=c("blue", 'red', 'gray'),
                     labels=c("2-step lmer", "1-step stan", "sim"),
                     name = NULL
  ) +
  scale_linetype_manual(values=c(2,4,1),
                     labels=c("2-step lmer", "1-step stan", "sim"),
                     name = NULL
  ) +
  theme(legend.position = c(0.22,0.89),
        legend.key.width = unit(3, "cm")) +
  labs(x = expression(widehat(rho)))
ggsave(file.path(path_plot, sprintf('%s.pdf', plot_name)), pl,
       width = wid, height = asp * wid)



################################################################################
######### Figure 10
plot_name <- 'figure_10'

range <- 30
wvar <- 0.65
tau_sim <- 0.17
phi_s2s_sim <- 0.23
phi_s2s_0 <- sqrt((1 - wvar) * phi_s2s_sim^2)
phi_s2s_c <- sqrt(wvar * phi_s2s_sim^2)
phi_ss_sim <- 0.2

seed <- 1701
load(file = file.path(path_res, sprintf('res_spatial_ita18_italy_seed%d.Rdata', seed)))
load(file = file.path(path_res, sprintf('res_spatial_ita18b_italy_seed%d.Rdata', seed)))

p1 <- data.frame(res_spatial[,c(4,7)],res_spatial_tot[,4]) %>% set_names('m1','m3','m2') %>%
  pivot_longer(everything()) %>%
  ggplot() +
  geom_density(aes(x = value, color = name, linetype = name), linewidth = lw, key_glyph = draw_key_path) +
  geom_vline(aes(xintercept = range), linewidth = lw) +
  labs(x = 'spatial range (km)') +
  theme(legend.position = c(0.85,0.85),
        legend.key.width = unit(3, "cm")) +
  guides(color = guide_legend(title=NULL)) +
  scale_color_manual(values = c('blue','red','orange'),
                     labels = c('full',
                                TeX("$\\widehat{\\delta R}$"),
                                TeX("$\\widehat{\\delta S}$")),
                     name = NULL) +
  scale_linetype_manual(values = c(1,2,4),
                     labels = c('full',
                                TeX("$\\widehat{\\delta R}$"),
                                TeX("$\\widehat{\\delta S}$")),
                     name = NULL) +
  labs(tag = '(a)')

p2 <- data.frame(m1 = res_spatial[,5]^2 / (1/res_spatial[,3] + res_spatial[,5]^2),
                 m1 = res_spatial_tot[,5]^2 / (1/res_spatial_tot[,3] + res_spatial_tot[,5]^2),
                 m3 = res_spatial[,8]^2 / (1/res_spatial[,6] + res_spatial[,8]^2)) %>%
  pivot_longer(everything()) %>%
  ggplot() +
  geom_density(aes(x = value, color = name, linetype = name), linewidth = 1.5, key_glyph = draw_key_path) +
  geom_vline(aes(xintercept = wvar), linewidth = 1.5) +
  labs(x = TeX("$\\hat{\\phi}_{S2S,c}^2 / \\hat{phi}_{S2S}^2$")) +
  theme(legend.position = 'none') +
  guides(color = guide_legend(title=NULL)) +
  scale_color_manual(values = c('blue','red','orange'),
                     labels = c('full',
                                TeX("$\\widehat{\\delta R}$"),
                                TeX("$\\widehat{\\delta S}$")),
                     name = NULL) +
  scale_linetype_manual(values = c(1,2,4),
                        labels = c('full',
                                   TeX("$\\widehat{\\delta R}$"),
                                   TeX("$\\widehat{\\delta S}$")),
                        name = NULL) +
  labs(tag = '(b)')

ggsave(file.path(path_plot, sprintf('%s.pdf', plot_name)), 
       patchwork::wrap_plots(p1,p2),
       width = 2 * wid, height = asp * wid)
ggsave(file.path(path_plot, sprintf('%s_a.pdf', plot_name)), p1,
       width = wid, height = asp * wid)
ggsave(file.path(path_plot, sprintf('%s_b.pdf', plot_name)), p2,
       width = wid, height = asp * wid)




################################################################################
######### Figure 11
plot_name <- 'figure_11'

tau <- 0.17
phi_s2s <- 0.2
phi_0 <- 0.18
sigma_cell <- 0.35
coeffs <- c(3.421046409, 0.193954090, -0.021982777, 0.287149291, -1.405635476,
            -0.002911264, 0.085983743, 0.010500239, -0.394575970)

seed <- 5618
load(file = file.path(path_res, sprintf('res_cell_italy_ita18_seed%d.Rdata', seed)))

p1 <- data.frame(res_cell[,c(4,10,6)]) %>% set_names(c('full','dR','dWS')) %>%
  pivot_longer(everything()) %>%
  ggplot() +
  geom_density(aes(x = value, color = name, linetype = name), linewidth = 1.5, key_glyph = draw_key_path) +
  geom_vline(aes(xintercept = sigma_cell), linewidth = 1.5) +
  labs(x = TeX('$\\hat{\\sigma}_{cell}$')) +
  theme(legend.position = c(0.5,0.85),
        legend.key.width = unit(3, "cm")) +
  guides(color = guide_legend(title=NULL)) +
  scale_color_manual(values = c('red','orange','blue'),
                     labels = c(TeX("$\\delta R$"), TeX("$\\widehat{\\delta WS}$"), "full"),
                     name = NULL) +
  scale_linetype_manual(values = c(2,4,1),
                        labels = c(TeX("$\\delta R$"), TeX("$\\widehat{\\delta WS}$"), "full"),
                        name = NULL) +
  labs(tag = '(a)')

p2 <- data.frame(res_cell_fix[,c(1,7,4)]) %>% set_names(c('full','dR','dWS')) %>%
  pivot_longer(everything()) %>%
  ggplot() +
  geom_density(aes(x = value, color = name, linetype = name), linewidth = lw, key_glyph = draw_key_path) +
  geom_vline(aes(xintercept = coeffs[6]), linewidth = lw) +
  labs(x = TeX('$\\hat{c}_{attn}$')) +
  theme(legend.position = 'none') +
  guides(color = guide_legend(title=NULL)) +
  scale_color_manual(values = c('red','orange','blue'),
                     labels = c(TeX("$\\delta R$"), TeX("$\\widehat{\\delta WS}$"), "full"),
                     name = NULL) +
  scale_linetype_manual(values = c(2,4,1),
                        labels = c(TeX("$\\delta R$"), TeX("$\\widehat{\\delta WS}$"), "full"),
                        name = NULL) +
  labs(tag = '(b)')

p3 <- data.frame(full = res_cell[,1]^2/res_cell_sdlme[,3]^2,
                 dR = res_cell[,7]^2/res_cell_sdlme[,3]^2,
                 dS = res_cell[,5]^2/res_cell_sdlme[,3]^2) %>%
  pivot_longer(everything()) %>%
  ggplot() +
  geom_density(aes(x = value, color = name, linetype = name), linewidth = 1.5) +
  labs(x = TeX("$\\hat{\\phi}_{SS,0}^2 / \\hat{\\phi}_{SS}^2$")) +
  theme(legend.position = 'none') +
  guides(color = guide_legend(title=NULL)) +
  scale_color_manual(values = c('red','orange','blue'),
                     labels = c(TeX("$\\delta R$"), TeX("$\\widehat{\\delta WS}$"), "full"),
                     name = NULL) +
  scale_linetype_manual(values = c(2,4,1),
                        labels = c(TeX("$\\delta R$"), TeX("$\\widehat{\\delta WS}$"), "full"),
                        name = NULL) +
  labs(tag = '(c)')

p4 <- data.frame(res_cell_cor[,c(4,6,5)]) %>%
  set_names(c('full','dR','dWS')) %>%
  pivot_longer(everything()) %>%
  ggplot() +
  geom_density(aes(x = value, color = name, linetype = name), linewidth = 1.5) +
  labs(x = expression(paste(rho,'(c'[true],',c'[est],')'))) +
  theme(legend.position = 'none') +
  guides(color = guide_legend(title=NULL)) +
  scale_color_manual(values = c('red','orange','blue'),
                     labels = c(TeX("$\\delta R$"), TeX("$\\widehat{\\delta WS}$"), "full"),
                     name = NULL) +
  scale_linetype_manual(values = c(2,4,1),
                        labels = c(TeX("$\\delta R$"), TeX("$\\widehat{\\delta WS}$"), "full"),
                        name = NULL) +
  labs(tag = '(d)')

ggsave(file.path(path_plot, sprintf('%s.pdf', plot_name)), 
       wrap_plots(p1,p2,p3,p4),
       width = 2 * wid, height = 2 * asp * wid)
ggsave(file.path(path_plot, sprintf('%s_a.pdf', plot_name)), p1,
       width = wid, height = asp * wid)
ggsave(file.path(path_plot, sprintf('%s_b.pdf', plot_name)), p2,
       width = wid, height = asp * wid)
ggsave(file.path(path_plot, sprintf('%s_c.pdf', plot_name)), p3,
       width = wid, height = asp * wid)
ggsave(file.path(path_plot, sprintf('%s_d.pdf', plot_name)), p4,
       width = wid, height = asp * wid)



################################################################################
######### Figure 12 and 13

make_plot_cor <- function(plot_name, cor_name) {
  tau_sim1 <- 0.4
  phi_s2s_sim1 <- 0.43
  phi_ss_sim1 <- 0.5
  
  tau_sim2 <- 0.45
  phi_s2s_sim2 <- 0.4
  phi_ss_sim2 <- 0.55
  
  sigma_tot1 <- sqrt(tau_sim1^2 + phi_s2s_sim1^2 + phi_ss_sim1^2)
  sigma_tot2 <- sqrt(tau_sim2^2 + phi_s2s_sim2^2 + phi_ss_sim2^2)
  
  model_name <- 'CB14'
  if(cor_name == 'high') {
    rho_tau <- 0.95
    rho_ss <- 0.9
    rho_s2s <- 0.85
    
    load(file.path(file.path(path_res, sprintf('res_corrre_%s_%s.Rdata', model_name, cor_name))))
    load(file.path(path_res, sprintf('res_corrre_stan_%s_%s.Rdata', model_name, cor_name)))
    mat_cor <- mat_cor[1:nrow(mat_cor_stan),]
    mat_cor_sample <- mat_cor_sample[1:nrow(mat_cor_stan),]
    
  } else if(cor_name == 'low') {
    rho_tau <- 0.45
    rho_ss <- 0.5
    rho_s2s <- 0.55
    
    load(file.path(file.path(path_res, sprintf('res_corrre_%s_%s.Rdata', model_name, cor_name))))
    load(file.path(path_res, sprintf('res_corrre_stan_%s_%s.Rdata', model_name, cor_name)))
    mat_cor <- mat_cor[1:nrow(mat_cor_stan),]
    mat_cor_sample <- mat_cor_sample[1:nrow(mat_cor_stan),]
  } else if(cor_name == 'eas') {
    rho_tau <- 0.95
    rho_ss <- 0.54
    rho_s2s <- 0.77
    
    load(file.path(file.path(path_res, sprintf('res_corrre_%s_%s.Rdata', model_name, cor_name))))
  }
  
  
  rho_total <- (rho_tau * tau_sim1 * tau_sim2 +
                  rho_s2s * phi_s2s_sim1 * phi_s2s_sim2 +
                  rho_ss * phi_ss_sim1 * phi_ss_sim2) /
    (sigma_tot1 * sigma_tot2)
  
  rho_total_sample <- (mat_cor_sample[,2] * tau_sim1 * tau_sim2 + 
                         mat_cor_sample[,1] * phi_s2s_sim1 * phi_s2s_sim2 + 
                         mat_cor_sample[,3] * phi_ss_sim1 * phi_ss_sim2) /
    (sigma_tot1 * sigma_tot2)
  
  ltm <- c(2,4,1)

  p1 <- data.frame(a = mat_cor[,1],b = mat_cor_stan[,1], z = mat_cor_sample[,1]) %>%
    pivot_longer(everything()) %>%
    ggplot() +
    geom_density(aes(x = value, color = name, linetype = name), linewidth = lw, key_glyph = draw_key_path) +
    geom_vline(xintercept = rho_s2s, linewidth = lw) +
    labs(x = TeX('$\\hat{\\rho}(\\delta S_1, \\delta S_2)$')) +
    scale_color_manual(values = c('blue','red','gray'),
                       labels = c('2-step lmer', '1-step stan', 'sim'),
                       name = NULL) +
    scale_linetype_manual(values = ltm,
                          labels = c('2-step lmer', '1-step stan', 'sim'),
                          name = NULL) +
    guides(color = guide_legend(title = NULL)) +
    theme(legend.key.width = unit(3,'cm')) +
    legend_move(c(0.2,0.9)) +
    labs(tag = '(a)')
  
  
  p2 <- data.frame(a = mat_cor[,2],b = mat_cor_stan[,2], z = mat_cor_sample[,2]) %>%
    pivot_longer(everything()) %>%
    ggplot() +
    geom_density(aes(x = value, color = name, linetype = name), linewidth = lw, key_glyph = draw_key_path) +
    geom_vline(xintercept = rho_tau, linewidth = lw) +
    labs(x = TeX('$\\hat{\\rho}(\\delta B_1, \\delta B_2)$')) +
    scale_color_manual(values = c('blue','red','gray'),
                       labels = c('2-step lmer', '1-step stan', 'sim'),
                       name = NULL) +
    scale_linetype_manual(values = ltm,
                          labels = c('2-step lmer', '1-step stan', 'sim'),
                          name = NULL) +
    guides(color = guide_legend(title = NULL)) +
    theme(legend.position = 'none',
          legend.key.width = unit(3,'cm')) +
    labs(tag = '(b)')
  
  
  p3 <- data.frame(a = mat_cor[,3],b = mat_cor_stan[,3], z = mat_cor_sample[,3]) %>%
    pivot_longer(everything()) %>%
    ggplot() +
    geom_density(data.frame(x = mat_cor_sample[,3]),
                 mapping = aes(x = x), color = 'gray', linewidth = lw) +
    geom_density(aes(x = value, color = name, linetype = name), linewidth = lw, key_glyph = draw_key_path) +
    geom_vline(xintercept = rho_ss, linewidth = lw) +
    labs(x = TeX('$\\hat{\\rho}(\\delta WS_1, \\delta WS_2)$')) +
    scale_color_manual(values = c('blue','red','gray'),
                       labels = c('2-step lmer', '1-step stan', 'sim'),
                       name = NULL) +
    scale_linetype_manual(values = ltm,
                          labels = c('2-step lmer', '1-step stan', 'sim'),
                          name = NULL) +
    guides(color = guide_legend(title = NULL)) +
    theme(legend.position = 'none',
          legend.key.width = unit(3,'cm')) +
    labs(tag = '(c)')
  
  p4 <- data.frame(a = mat_cor[,5],b = mat_cor_stan[,4],c = mat_cor[,4], z = rho_total_sample) %>%
    pivot_longer(everything()) %>%
    ggplot() +
    geom_density(aes(x = value, color = name, linetype = name), linewidth = lw, key_glyph = draw_key_path) +
    geom_vline(xintercept = rho_total, linewidth = lw) +
    labs(x = TeX('$\\hat{\\rho}_{total}$')) +
    scale_color_manual(values = c('blue','red','cyan','gray'),
                       labels = c('2-step lmer', '1-step stan', TeX('$\\delta R$'), 'sim'),
                       name = NULL) +
    scale_linetype_manual(values = c(4,2,6,1),
                          labels = c('2-step lmer', '1-step stan', TeX('$\\delta R$'), 'sim'),
                          name = NULL) +
    guides(color = guide_legend(title = NULL)) +
    theme(legend.position = c(0.2,0.8),
          legend.key.width = unit(3,'cm')) +
    legend_move(c(0.2,0.8)) +
    labs(tag = '(d)')
  
  pl <- patchwork::wrap_plots(p1,p2,p3,p4)
  ggsave(file.path(path_plot, sprintf('%s.pdf', plot_name)), pl,
         width = 2 * wid, height = 2 * asp * wid)
  ggsave(file.path(path_plot, sprintf('%s_a.pdf', plot_name)), p1,
         width = wid, height = asp * wid)
  ggsave(file.path(path_plot, sprintf('%s_b.pdf', plot_name)), p2,
         width = wid, height = asp * wid)
  ggsave(file.path(path_plot, sprintf('%s_c.pdf', plot_name)), p3,
         width = wid, height = asp * wid)
  ggsave(file.path(path_plot, sprintf('%s_d.pdf', plot_name)), p4,
         width = wid, height = asp * wid)
}
make_plot_cor(plot_name = 'figure_12', cor_name = 'high')
make_plot_cor(plot_name = 'figure_13', cor_name = 'eas')
make_plot_cor(plot_name = 'figure_cor_low', cor_name = 'low')



################################################################################
######### Figure 14 and 15

coeffs <- c(3.421046409, 0.193954090, -0.021982777, 0.287149291, -1.405635476, -0.002911264, -0.394575970)
names_coeffs <- c("intercept", "M1", "M2", "MlogR", "logR", "R", "logVS")
tau_sim <- 0.17
phi_s2s_sim <- 0.23
phi_ss_sim <- 0.2
sds_sim <- c(phi_s2s_sim, tau_sim, phi_ss_sim)

load(file = file.path(path_res,
                      sprintf('res_twostep_ita18_%s.Rdata', 'CB')))

df1 <- data.frame(mat_fix) %>% set_names(names_coeffs)
df1$model <- '1-step'

df2 <- data.frame(mat_fix2) %>% set_names(names_coeffs)
df2$model <- '2-step'

df <- data.frame(name = names_coeffs,
                 true = coeffs)

names_tex <- c(TeX("$\\hat{c}_{1}$"),TeX("$\\hat{c}_{2}$"),TeX("$\\hat{c}_{3}$"),
               TeX("$\\hat{c}_{4}$"), TeX("$\\hat{c}_{5}$"),TeX("$\\hat{c}_{6}$"),
               TeX("$\\hat{c}_{7}$"))

names_tags <- c('a','b','c','d','e','f','g')

plot_name <- 'figure_14'

pl_list <- list()
for(i in 1:length(names_coeffs)) {
  pl_list[[i]] <- local({
    i <- i
    if(i == 1) {
      pos <- c(0.2,0.9)
    } else {pos <- 'none'}
    
    name <- names_coeffs[i]
    rbind(df1[,c(name,'model')],df2[,c(name,'model')]) %>%
      set_names(c('value','model')) %>%
      ggplot() +
      geom_density(aes(x = value, color = model, linetype = model), linewidth = 1.5, key_glyph = draw_key_path) +
      geom_vline(aes(xintercept = df[df$name == name,'true']), linewidth = 1.5) +
      guides(color = guide_legend(title = NULL)) +
      scale_color_manual(values = c('blue','red'), name = NULL) +
      scale_linetype_manual(values = c(1,2), name = NULL) +
      labs(x = names_tex[i]) +
      theme(legend.position = pos,
            legend.key.width = unit(3,'cm')) +
      labs(tags = paste0('(',names_tags[i],')'))
  })
  ggsave(file.path(path_plot, sprintf('%s_%s.pdf', plot_name, names_tags[i])), pl_list[[i]],
         width = wid, height = asp * wid)   
}
pl <- wrap_plots(pl_list)

ggsave(file.path(path_plot, sprintf('%s.pdf', plot_name)), pl,
       width = 3 * wid, height = 3* asp * wid)       


names_sds <- c('phi_s2s','tau','phi_ss')
df1 <- data.frame(mat_sd) %>% set_names(names_sds)
df1$model <- '1-step'

df2 <- data.frame(mat_sd2) %>% set_names(names_sds)
df2$model <- '2-step'

df <- data.frame(name = names_sds,
                 true = sds_sim)

names_tex <- c(TeX("$\\hat{\\phi}_{S2S}$"),TeX("$\\hat{\\tau}$"),TeX("$\\hat{\\phi}_{SS}$"))
plot_name <- 'figure_15'
pl_list <- list()
for(i in 1:length(names_sds)) {
  pl_list[[i]] <- local({
    i <- i
    if(i == 1) {
      pos <- c(0.8,0.9)
    } else {pos <- 'none'}
    
    name <- names_sds[i]
    rbind(df1[,c(name,'model')],df2[,c(name,'model')]) %>%
      set_names(c('value','model')) %>%
      ggplot() +
      geom_density(aes(x = value, color = model, linetype = model), linewidth = 1.5, key_glyph = draw_key_path) +
      geom_vline(aes(xintercept = df[df$name == name,'true']), linewidth = 1.5) +
      guides(color = guide_legend(title = NULL)) +
      scale_color_manual(values = c('blue','red'), name = NULL) +
      scale_linetype_manual(values = c(1,2), name = NULL) +
      labs(x = names_tex[i]) +
      theme(legend.position = pos,
            legend.key.width = unit(3,'cm')) +
      labs(tags = paste0('(',names_tags[i],')'))
  })
  ggsave(file.path(path_plot, sprintf('%s_%s.pdf', plot_name, names_tags[i])), pl_list[[i]],
         width = wid, height = asp * wid)   
}
pl <- wrap_plots(pl_list)

ggsave(file.path(path_plot, sprintf('%s.pdf', plot_name)), pl,
       width = 3 * wid, height = asp * wid)
