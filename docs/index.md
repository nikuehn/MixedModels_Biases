---
title: "Biases in Mixed-Effects Model GMMs"
author: "Nicolas Kuehn, Ken Campbell, Yousef Bozorgnia"
date: "22 March, 2024, first published 14 September 2023."
output:
  html_document:
    keep_md: true
    toc: true
    toc_depth: 3
    number_sections: true
    highlight: tango
link-citations: yes
linkcolor: blue
citecolor: blue
urlcolor: blue
bibliography: /Users/nico/BIBLIOGRAPHY/BIBTEX/references.bib
---




# Introduction

This page provides code for the simulations shown in ``Potential Biases in Empirical Ground-Motion Models by focusing on Point Estimates of Random Effects'', which highlights some biases that can occur when using point estimates of random effects/residuals in mixed effects ground-motion models.
For details, see the paper.

We use simulations from different models and/or using different data sets to illustrate potential biases.
In particular, standard deviations are underestimated when they are calculated from point estimates of random effects/residuals.
For the simulations, we randomly sample event terms, site terms, and within-event/within-site residuals from their respective distributions, and then perform regessions on the sampled data to see whether we recover get the parameters used in the simulations.
In this document, we generally do a single simulation for different cases, which typically highlight the points we want to made.
For the paper, we repeat these simulations multiple times, since due to the relative small sample size in ground-motion data sets there ca be variability from one sample to the next.

Simulation can be a pwoeful tool to gain understanding of different models [@DeBruine2021].
There exist several `R`-packages for simulation (e.g. [faux](https://debruine.github.io/faux/index.html), [simDesign](https://cran.r-project.org/web/packages/SimDesign/vignettes/SimDesign-intro.html), [simulator](https://github.com/jacobbien/simulator)), but since our models are simple, we code them up directly.

In general, a ground-motion model (GMM) can be written as
$$
Y_{es} = f(\vec{c}; \vec{x}) + \delta B_e + \delta S_s + \delta WS_{es}
$$
or in matrix form as
$$
\vec{Y} = f(\vec{c}; \mathbf{x}) + \mathbf{Z} \vec{u} + \vec{\delta WS} 
$$
where $\mathbf{Z}$ is the design matrix of the random effects.
It is importat to remember that in general the outcome of a mixed-effects regression will give point estimates of the random effects (the conditional modes),and that there is uncetainty around them.
The conditional variances of the random effects are the diagonal entries of the following matrix
$$
\begin{aligned}
V &= \phi_{SS}^2 \mathbf{\Lambda} \left(\mathbf{\Lambda}^T \mathbf{Z}^T \mathbf{Z} \mathbf{\Lambda} + \mathbf{I} \right)^{-1} \mathbf{\Lambda} \\
\psi(\hat{\vec{u}})^2 &= \mbox{diag}(V)
\end{aligned}
$$
where $\mathbf{\Lambda}$ is the relative covariance factor [@Bates2015].
If this uncertainty is ignored, biases can occur, as we deomstrate throughut this page.
In particular, the variances of the random effects are calculated as (example for $\tau$)
$$
\hat{\tau}^2 = \frac{1}{N_E}\sum_{i = 1}^{N_E} \widehat{\delta B}_i^2 + \frac{1}{N_E}\sum_{i = 1}^{N_E} \psi(\widehat{\delta B}_i)^2 
$$
which is the sum of the variance of the point estimates plus the average conditional variance.
Hence, just esimating the variance (or standard deviation) of the point estimates will lead to an underestmation.

## Set up

Load required libraries, and define some plot options for `ggplot2`.


```r
library(ggplot2)
library(lme4)
library(cmdstanr)
library(posterior)
library(bayesplot)
library(tidyverse)
library(INLA)
library(matrixStats)
library(latex2exp)
```


```
## CmdStan path set to: /Users/nico/GROUNDMOTION/SOFTWARE/cmdstan-2.33.1
```


```r
theme_set(theme_bw() + theme(
  axis.title = element_text(size = 30),
  axis.text = element_text(size = 20),
  plot.title = element_text(size = 30),
  legend.text = element_text(size = 20),
  legend.title = element_text(size = 20),
  legend.key.width = unit(1, "cm"),
  legend.box.background = element_rect(colour = "black"),
  panel.grid = element_line(color = "gray",linewidth = 0.75)
))

breaks <- 10^(-10:10)
minor_breaks <- rep(1:9, 21)*(10^rep(-10:10, each=9))

lw <- 1.5 # linewidth
sp <- 4 # pointsize
```

## Example

We start with an example using real data, just to get familiar with the concepts and code.
In later sections, we use simulations, which make it easy to compare the results from a regression to the true values of the parameters.
We use the Ialian data from the ITA18 GMM (@Lanzano2018, see also @Caramenti2022), and perform a regression on peak ground acceleration (PGA), using the functional form of ITA18.

First, we read in the data and prepare a data frame for the regression.
In total, there are 4784 records from 137 events and 923 stations.


```r
data_it <- read.csv(file.path('./Git/MixedModels_Biases/','/data','italian_data_pga_id_utm_stat.csv'))

# Set linear predictors
mh = 5.5
mref = 5.324
h = 6.924
attach(data_it)
b1 = (mag-mh)*(mag<=mh)
b2 = (mag-mh)*(mag>mh)
c1 = (mag-mref)*log10(sqrt(JB_complete^2+h^2))
c2 = log10(sqrt(JB_complete^2+h^2))
c3 = sqrt(JB_complete^2+h^2)
f1 = as.numeric(fm_type_code == "SS")
f2 = as.numeric(fm_type_code == "TF")
k = log10(vs30/800)*(vs30<=1500)+log10(1500/800)*(vs30>1500)
y = log10(rotD50_pga)
detach(data_it)

n_rec <- length(b1)
eq <- data_it$EQID
stat <- data_it$STATID
n_eq <- max(eq)
n_stat <- max(stat)
n_rec <- nrow(data_it)

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
print(paste0('Number of records: ',n_rec,'; number of events: ',n_eq,'; number of stations: ',n_stat))
```

```
## [1] "Number of records: 4784; number of events: 137; number of stations: 923"
```

Now we fit the model.
We fit the model using `lmer`, `inla`, and using Stan via `cmdstanr`.
The Stan code can be found at <https://github.com/nikuehn/MixedModels_Biases/tree/main/stan>.



```r
##################
# fit using lmer

fit_lmer <- lmer(Y ~ M1 + M2 + MlogR + logR + R + Fss + Frv + logVS + (1|eq) + (1|stat), data_reg)

tmp <- as.data.frame(VarCorr(fit_lmer))$sdcor
phi_s2s_lmer <- tmp[1]
tau_lmer <- tmp[2]
phi_ss_lmer <- tmp[3]

deltaB <- ranef(fit_lmer)$eq$`(Intercept)`
deltaS <- ranef(fit_lmer)$stat$`(Intercept)`
sd_deltaB <- as.numeric(arm::se.ranef(fit_lmer)$eq)
sd_deltaS <- as.numeric(arm::se.ranef(fit_lmer)$stat)
deltaWS <- data_reg$Y - predict(fit_lmer)
sd_deltaWS <- sqrt(sd_deltaB[eq]^2 + sd_deltaS[stat]^2) # approximately

##################
# fit using Inla
# priors for standard deviation paramters
prior_prec_tau    <- list(prec = list(prior = 'pc.prec', param = c(0.3, 0.01)))
prior_prec_phiS2S    <- list(prec = list(prior = 'pc.prec', param = c(0.3, 0.01))) 
prior_prec_phiSS    <- list(prec = list(prior = 'pc.prec', param = c(0.3, 0.01))) 

form <- Y ~ M1 + M2 + MlogR + logR + R + Fss + Frv + logVS +
  f(eq, model = "iid", hyper = prior_prec_tau) + 
  f(stat, model = "iid",hyper = prior_prec_phiS2S)

fit_inla <- inla(form, 
                 data = data_reg,
                 family="gaussian",
                 control.family = list(hyper = prior_prec_phiSS),
                 quantiles = c(0.05,0.5,0.95)
)

sd_deltaS_inla <-fit_inla$summary.random$stat$sd
sd_deltaB_inla <-fit_inla$summary.random$eq$sd
sd_deltaWS_inla <- fit_inla$summary.fitted.values$sd

##################
# fit using Stan
mod <- cmdstan_model(file.path('./Git/MixedModels_Biases/', 'stan', 'gmm_full_qr.stan'))
data_list <- list(
  N = n_rec,
  NEQ = n_eq,
  NSTAT = n_stat,
  K = 9,
  Y = as.numeric(data_reg$Y),
  X = data_reg[,c("M1", "M2", "MlogR", "logR", "R", "Fss", "Frv", "logVS")], # design matrix
  eq = eq,
  stat = stat,
  alpha = c(1,1,1)
)

fit_stan <- mod$sample(
  data = data_list,
  seed = 8472,
  chains = 4,
  iter_sampling = 200,
  iter_warmup = 200,
  refresh = 100,
  max_treedepth = 10,
  adapt_delta = 0.8,
  parallel_chains = 2,
  show_exceptions = FALSE
)
```

```
## Running MCMC with 4 chains, at most 2 in parallel...
## 
## Chain 1 Iteration:   1 / 400 [  0%]  (Warmup) 
## Chain 2 Iteration:   1 / 400 [  0%]  (Warmup) 
## Chain 1 Iteration: 100 / 400 [ 25%]  (Warmup) 
## Chain 2 Iteration: 100 / 400 [ 25%]  (Warmup) 
## Chain 2 Iteration: 200 / 400 [ 50%]  (Warmup) 
## Chain 2 Iteration: 201 / 400 [ 50%]  (Sampling) 
## Chain 1 Iteration: 200 / 400 [ 50%]  (Warmup) 
## Chain 1 Iteration: 201 / 400 [ 50%]  (Sampling) 
## Chain 2 Iteration: 300 / 400 [ 75%]  (Sampling) 
## Chain 1 Iteration: 300 / 400 [ 75%]  (Sampling) 
## Chain 2 Iteration: 400 / 400 [100%]  (Sampling) 
## Chain 2 finished in 26.5 seconds.
## Chain 3 Iteration:   1 / 400 [  0%]  (Warmup) 
## Chain 1 Iteration: 400 / 400 [100%]  (Sampling) 
## Chain 1 finished in 27.0 seconds.
## Chain 4 Iteration:   1 / 400 [  0%]  (Warmup) 
## Chain 3 Iteration: 100 / 400 [ 25%]  (Warmup) 
## Chain 4 Iteration: 100 / 400 [ 25%]  (Warmup) 
## Chain 3 Iteration: 200 / 400 [ 50%]  (Warmup) 
## Chain 3 Iteration: 201 / 400 [ 50%]  (Sampling) 
## Chain 4 Iteration: 200 / 400 [ 50%]  (Warmup) 
## Chain 4 Iteration: 201 / 400 [ 50%]  (Sampling) 
## Chain 3 Iteration: 300 / 400 [ 75%]  (Sampling) 
## Chain 3 Iteration: 400 / 400 [100%]  (Sampling) 
## Chain 3 finished in 29.2 seconds.
## Chain 4 Iteration: 300 / 400 [ 75%]  (Sampling) 
## Chain 4 Iteration: 400 / 400 [100%]  (Sampling) 
## Chain 4 finished in 34.4 seconds.
## 
## All 4 chains finished successfully.
## Mean chain execution time: 29.3 seconds.
## Total execution time: 61.7 seconds.
```

```r
draws <- fit_stan$draws()

sd_deltaB_stan <- colSds(subset(as_draws_matrix(draws), variable = 'eqterm'))
sd_deltaS_stan <- colSds(subset(as_draws_matrix(draws), variable = 'statterm'))
sd_deltaWS_stan <- colSds(subset(as_draws_matrix(draws), variable = 'resid'))
```

The estimated standard deviations are very similar between the three methods (we have used relatively weak priors, so their influence is not very strong).
For Inla, we transform the mean estimate of the precisions into an estimate of the standard deviations, which is technicaly not correct, but for the sake of smplicity we keep it.
The difference is small.
Note that for the Bayesian models (Inla and Stan), the full output is not just a point estimate of $\phi_{SS}$, $\tau$, and $\phi_{S2S}$, but the full posterior distribution.


```r
df <- data.frame(inla = 1/sqrt(fit_inla$summary.hyperpar$mean),
                 lmer = c(phi_ss_lmer, tau_lmer, phi_s2s_lmer),
                 stan = colMeans(subset(as_draws_matrix(draws), variable = c('phi_ss','tau','phi_s2s'))),
                 row.names = c('phi_ss','tau','phi_s2s'))
knitr::kable(df, digits = 5, row.names = TRUE,
             caption = "Comparison of standard deviation estimates.")
```



Table: Comparison of standard deviation estimates.

|        |    inla|    lmer|    stan|
|:-------|-------:|-------:|-------:|
|phi_ss  | 0.20407| 0.20413| 0.20415|
|tau     | 0.14141| 0.14327| 0.14436|
|phi_s2s | 0.23341| 0.23365| 0.23314|
Now we compare credible intervals (for the Bayesian models) and confidence intervals (for `lmer`).
For `lmer`, we calculate the profile confidence intervals, using function `confint`, while for the Bayesian credible intervals we take the 5% and 95% quantile of the posterior distribution.
Since `inla` internally uses precisions for the scale parameter, we convert the quantiles (precision to standard devation is a monotonic transformation, so the quantiles do not change).
We note that the interpretation of confidence intervals can be tricky [@Morey2016].
The confidence and credible intervals agree quite well.


```r
ci_lmer <- confint(fit_lmer, level = c(0.9))
```

```
## Computing profile confidence intervals ...
```

```r
df <- data.frame(inla_q05 = 1/sqrt(fit_inla$summary.hyperpar[,'0.95quant']),
                 inla_q95 = 1/sqrt(fit_inla$summary.hyperpar[,'0.05quant']),
                 lmer_c05 = ci_lmer[c(3,2,1),1],
                 lmer_c95 = ci_lmer[c(3,2,1),2],
                 stan_q05 = colQuantiles(subset(as_draws_matrix(draws), variable = c('phi_ss','tau','phi_s2s')), 
                                         probs = 0.05),
                 stan_q95 = colQuantiles(subset(as_draws_matrix(draws), variable = c('phi_ss','tau','phi_s2s')), 
                                         probs = 0.95),
                 row.names = c('phi_ss','tau','phi_s2s'))
knitr::kable(df, digits = 5, row.names = TRUE,
             caption = "Comparison of standard deviation credible/confidence intervals.")
```



Table: Comparison of standard deviation credible/confidence intervals.

|        | inla_q05| inla_q95| lmer_c05| lmer_c95| stan_q05| stan_q95|
|:-------|--------:|--------:|--------:|--------:|--------:|--------:|
|phi_ss  |  0.20023|  0.20800|  0.20025|  0.20798|  0.20041|  0.20800|
|tau     |  0.12580|  0.16383|  0.12404|  0.15880|  0.12719|  0.16362|
|phi_s2s |  0.22163|  0.24542|  0.22216|  0.24544|  0.22238|  0.24503|

The coefficient estimates are also very close.


```r
df <- data.frame(inla = fit_inla$summary.fixed$mean,
                 lmer = fixef(fit_lmer),
                 stan = colMeans(subset(as_draws_matrix(draws), variable = c('^c\\['), regex = TRUE)))
knitr::kable(df, digits = 5, row.names = TRUE,
             caption = "Comparison of coefficient estimates.")
```



Table: Comparison of coefficient estimates.

|            |     inla|     lmer|     stan|
|:-----------|--------:|--------:|--------:|
|(Intercept) |  3.40873|  3.40922|  3.40668|
|M1          |  0.20329|  0.20343|  0.20201|
|M2          |  0.00275|  0.00256|  0.01465|
|MlogR       |  0.28754|  0.28764|  0.28749|
|logR        | -1.39877| -1.39899| -1.39601|
|R           | -0.00309| -0.00309| -0.00311|
|Fss         |  0.11563|  0.11583|  0.11048|
|Frv         | -0.00142| -0.00107| -0.00363|
|logVS       | -0.42206| -0.42193| -0.42216|

As are the confidence and credible intervals of the fixed effects.


```r
df <- data.frame(inla_q05 = fit_inla$summary.fixed[,'0.05quant'],
                 inla_q95 = fit_inla$summary.fixed[,'0.95quant'],
                 lmer_c05 = ci_lmer[4:12,1],
                 lmer_c95 = ci_lmer[4:12,2],
                 stan_q05 = colQuantiles(subset(as_draws_matrix(draws), variable = c('^c\\['), regex = TRUE),
                                         probs = 0.05),
                 stan_q95 = colQuantiles(subset(as_draws_matrix(draws), variable = c('^c\\['), regex = TRUE),
                                         probs = 0.95))
knitr::kable(df, digits = 5, row.names = TRUE,
             caption = "Comparison of coefficient credible/confidence intervals.")
```



Table: Comparison of coefficient credible/confidence intervals.

|            | inla_q05| inla_q95| lmer_c05| lmer_c95| stan_q05| stan_q95|
|:-----------|--------:|--------:|--------:|--------:|--------:|--------:|
|(Intercept) |  3.32703|  3.49060|  3.32740|  3.49023|  3.33255|  3.48558|
|M1          |  0.13983|  0.26688|  0.14014|  0.26647|  0.14216|  0.26619|
|M2          | -0.11604|  0.12140| -0.11532|  0.12070| -0.09652|  0.12896|
|MlogR       |  0.26484|  0.31024|  0.26484|  0.31019|  0.26520|  0.31024|
|logR        | -1.44813| -1.34942| -1.44810| -1.34952| -1.44679| -1.34740|
|R           | -0.00342| -0.00275| -0.00342| -0.00275| -0.00343| -0.00278|
|Fss         |  0.05622|  0.17525|  0.05652|  0.17486|  0.04949|  0.17084|
|Frv         | -0.05706|  0.05456| -0.05676|  0.05421| -0.05467|  0.04708|
|logVS       | -0.49610| -0.34807| -0.49600| -0.34820| -0.49422| -0.34680|


Below we plot the standard deviations of the random effects (conditional standard deviations for `lmer`, standard deviations of the posterior distribution for the Bayesian models) against number of records per event/station.
In general, they are similar between all three fits.
We see larger standard deviations of the event terms for the Bayesian models for large magnitudes, which is due to the fact that uncertainty due to coefficients is included, which is larger for large magnitudes.


```r
p1 <- data.frame(unique(data_it[,c('EQID','mag')]),
                 lmer = sd_deltaB,
                 inla = sd_deltaB_inla,
                 stan = sd_deltaB_stan,
                 nrec = as.numeric(table(eq))) |>
  pivot_longer(c(lmer, inla, stan)) %>%
  ggplot() +
  geom_point(aes(x = nrec, y = value, color = name, size = mag)) +
  scale_x_log10(breaks = breaks, minor_breaks = minor_breaks) +
  ylim(c(0,0.125)) +
  labs(x = 'number of records per event', y = TeX("$\\psi(\\widehat{\\delta B})$")) +
  guides(color = guide_legend(title=NULL), size = guide_legend(title = 'M')) +
  theme(legend.position = c(0.85,0.75))

p2 <- data.frame(unique(data_it[,c('STATID','vs30')]),
           logvs = unique(data_reg[,c('stat','logVS')])[,2],
           lmer = sd_deltaS,
           inla = sd_deltaS_inla,
           stan = sd_deltaS_stan,
           nrec = as.numeric(table(stat))) |>
  pivot_longer(c(lmer, inla, stan)) %>%
  ggplot() +
  geom_point(aes(x = nrec, y = value, color = name, size = vs30)) +
  scale_x_log10(breaks = breaks, minor_breaks = minor_breaks) +
  ylim(c(0,0.18)) +
  labs(x = 'number of records per station', y = TeX("$\\psi(\\widehat{\\delta S})$")) +
  guides(color = guide_legend(title=NULL), size = guide_legend(title = 'VS30')) +
  theme(legend.position = c(0.85,0.75))

patchwork::wrap_plots(p1,p2)
```

```
## Warning: Removed 1 rows containing missing values (`geom_point()`).
```

<img src="pictures/example-plot-se-1.png" width="100%" />
Below we compare the standard deviations for $\delta WS$.
For `lmer`, this is an approximation, calculated as $\psi(\widehat{\delta WS})^2 = \psi(\widehat{\delta B})^2 + \psi(\widehat{\delta S})^2$.
At larger values of $\psi(\widehat{\delta WS})$, the values from `lmer` are larger compared to the ones from Inla or Stan, which is probably due to some correlations between (estimated) event terms and site terms.
These corelations are implicitly taken into account in the Bayesian models.


```r
data.frame(inla = sd_deltaWS_inla, 
           lmer = sd_deltaWS,
           stan = sd_deltaWS_stan) %>%
  pivot_longer(!inla) %>%
  ggplot() +
  geom_point(aes(x = inla, y = value, color = name), size = 4) +
  geom_abline(color = 'black', linewidth = 1.5) +
  lims(x = c(0.03,0.21), y = c(0.03, 0.21)) +
  labs(x = TeX("$\\psi(\\widehat{\\delta WS}_{inla})$"),
       y = TeX("$\\psi(\\widehat{\\delta WS})$")) +
  guides(color = guide_legend(title=NULL))
```

<img src="pictures/example-plot-se2-1.png" width="50%" />

We now calculate the conditional standard deviations of the random effects from the `lmer` fit according to Equation (3) of the paper, which makes use of the design matrix of the random effects $\mathbf{Z}$, and the relative covariance factor $\mathbf{\Lambda}$.
We compare the results against the values calculated by package `arm`, and find that the differences are negligible.


```r
Z <- getME(fit_lmer, 'Z') #sparse Z design matrix
lambda <- getME(fit_lmer, 'Lambda')

V <- sigma(fit_lmer)^2 * lambda %*% solve((t(lambda) %*% t(Z) %*% Z %*% lambda + diag(n_eq + n_stat))) %*% lambda

# station entries are first
print(c(sum((arm::se.ranef(fit_lmer)$stat)^2 - diag(V)[1:n_stat]),
        sum((arm::se.ranef(fit_lmer)$eq)^2 - diag(V)[(n_stat + 1):(n_eq + n_stat)])))
```

```
## [1] 7.350891e-16 2.257309e-16
```

# Simulations using CB14 Data

We now turn to simulations to show how biases can occur when the uncertainty of random effects (event and site terms) as well as residuals is neglected.
We first focus on standard deviations, which are underestimated when estimated from point estimates.
This becomes a problem when standard deviations are modeled as heteroscedastic, e.e. dependent on predictor variables such as magnitude and/or distance.

We illustrate the underestimation of standard deviations on the WUS data from the GMM of @Campbell2014.
In total, there are 12482 records from 274 events and 1519 stations.


```r
data_reg <- read.csv(file.path('./Git/MixedModels_Biases/','/data','data_cb.csv'))
print(dim(data_reg))
```

```
## [1] 12482    11
```

```r
print(head(data_reg))
```

```
##   intercept eqid statid    M     VS   Rrup    Rjb eq stat M_stat VS_gmean
## 1         1    7    133 6.60 219.31  91.22  91.15  1    1  6.600   219.31
## 2         1   12    326 7.36 316.46 117.75 114.62  2    2  6.656   316.46
## 3         1   25    131 6.19 408.93  17.64  17.64  3    3  6.190   408.93
## 4         1   25    127 6.19 289.56   9.58   9.58  3    4  6.190   289.56
## 5         1   25    129 6.19 256.82  12.90  12.90  3    5  6.190   256.82
## 6         1   25    162 6.19 527.92  15.96  15.96  3    6  6.190   527.92
```

```r
p1 <- ggplot(data_reg) +
  geom_point(aes(x = Rrup, y = M)) +
  scale_x_log10(breaks = breaks, minor_breaks = minor_breaks)
p2 <- ggplot(unique(data_reg[,c('eqid','M')])) +
  geom_histogram(aes(x = M))
patchwork::wrap_plots(p1, p2)
```

```
## `stat_bin()` using `bins = 30`. Pick better value with `binwidth`.
```

<img src="pictures/read-data-cb-1.png" width="100%" />


```r
n_rec <- nrow(data_reg)
n_eq <- max(data_reg$eq)
n_stat <- max(data_reg$stat)

eq <- data_reg$eq
stat <- data_reg$stat

mageq <- unique(data_reg[,c('eq','M')])$M # event-specific magnitude
magstat <- unique(data_reg[,c('stat','M_stat')])$M_stat # station-specific magnitude

print(paste0('Number of records: ',n_rec,'; number of events: ',n_eq,'; number of stations: ',n_stat))
```

```
## [1] "Number of records: 12482; number of events: 274; number of stations: 1519"
```

## Homoscedastic Standard Deviations

First, we simulate data using standard deviations that do not depend on any predictor variables, i.e. are homoscedastic.
We do not simulate any fixed effects structure, in this example we focus on biases in the estimation of standard deviations.

First, we fix the standard deviations:

```r
tau_sim <- 0.4
phi_s2s_sim <- 0.43
phi_ss_sim <- 0.5
```

Next, we randomly sample event terms, site terms, and withn-event/within-site residuals, and combine them into total residuals (our target variable for this example).

```r
set.seed(5618)
# randomly sample residals, event and site terms
dWS_sim <- rnorm(n_rec, sd = phi_ss_sim)
dS_sim <- rnorm(n_stat, sd = phi_s2s_sim)
dB_sim <- rnorm(n_eq, sd = tau_sim)

# combine into total residual/target variable
data_reg$y_sim <- dB_sim[eq] + dS_sim[stat] + dWS_sim
```

Now we perform the linear mixed effects regression using `lmer`.
We use maximum likelihood instead of restricted maximum likelihood in this case to show the equivalence of the calculations of standard deviations.
In the paper, we also use Stan to estimate the random effects and standard deviations, but we omit this here to save time and space.
As we have seen for the Italian data, we get very similar results using `lmer` and Stan, and this is also reflected by the results shown in the paper.
Frequentist methods are still overwhelmingly used in GMM development, so it makes sense to focus on them here.


```r
fit_sim <- lmer(y_sim ~ (1 | eq) + (1 | stat), data_reg, REML = FALSE)
summary(fit_sim)
```

```
## Linear mixed model fit by maximum likelihood  ['lmerMod']
## Formula: y_sim ~ (1 | eq) + (1 | stat)
##    Data: data_reg
## 
##      AIC      BIC   logLik deviance df.resid 
##  20896.2  20925.9 -10444.1  20888.2    12478 
## 
## Scaled residuals: 
##     Min      1Q  Median      3Q     Max 
## -3.6189 -0.6421  0.0026  0.6345  3.7999 
## 
## Random effects:
##  Groups   Name        Variance Std.Dev.
##  stat     (Intercept) 0.1940   0.4404  
##  eq       (Intercept) 0.1685   0.4105  
##  Residual             0.2473   0.4973  
## Number of obs: 12482, groups:  stat, 1519; eq, 274
## 
## Fixed effects:
##             Estimate Std. Error t value
## (Intercept) -0.03459    0.03011  -1.149
```

As we can see, the model parameters are quite well estimated.
The intercept is close to zero, and the standard deviations are close to the values used in the simulation.

Below, we extract the conditional modes and standard deviations of the random effects.
We also calculate the within-event/within-site residuals, and approximate their standard deviation.


```r
tmp <- as.data.frame(ranef(fit_sim))
dS_lmer <- tmp[tmp$grpvar == 'stat','condval']
dB_lmer <- tmp[tmp$grpvar == 'eq','condval']
dWS_lmer <- data_reg$y_sim - predict(fit_sim)

sd_dS_lmer <- tmp[tmp$grpvar == 'stat','condsd']
sd_dB_lmer <- tmp[tmp$grpvar == 'eq','condsd']
sd_dWS_lmer <- sqrt(sd_dB_lmer[eq]^2 + sd_dS_lmer[stat]^2) # approximately

# alternative way to extract the random effects and conditional standard deviations
# dS_lmer <- ranef(fit_sim)$stat$`(Intercept)`
# dB_lmer <- ranef(fit_sim)$eq$`(Intercept)`
# 
# sd_dB_lmer <- as.numeric(arm::se.ranef(fit_sim)$eq)
# sd_dS_lmer <- as.numeric(arm::se.ranef(fit_sim)$stat)
```

Next, we compare different standard deviations.
For all terms, we show the true value used in the simulations, the standard devation of the sampled terms, and then the value from the fit using `lmer`.
Then, we calculate the standard deviations according to Equation (4) in the paper (including uncertainty).
We also show the standard deviations calculated based on the conditional modes of the random effects/residuals, as well as calculated using a sample from the conditional distribution.


```r
# compare estiamtes of standard deviations
df <- data.frame(phi_s2s = c(phi_s2s_sim,
                       sd(dS_sim),
                       as.data.frame(VarCorr(fit_sim))$sdcor[1], 
                       sqrt(sum(dS_lmer^2)/n_stat + sum(sd_dS_lmer^2)/n_stat),
                       sd(dS_lmer),
                       sd(rnorm(n_stat, mean = dS_lmer, sd = sd_dS_lmer))),
           tau = c(tau_sim, 
                   sd(dB_sim),
                   as.data.frame(VarCorr(fit_sim))$sdcor[2], 
                   sqrt(sum(dB_lmer^2)/n_eq + sum(sd_dB_lmer^2)/n_eq),
                   sd(dB_lmer),
                   sd(rnorm(n_eq, mean = dB_lmer, sd = sd_dB_lmer))),
           phi_ss = c(phi_ss_sim,
                      sd(dWS_sim),
                      as.data.frame(VarCorr(fit_sim))$sdcor[3], 
                      sqrt(sum(dWS_lmer^2)/n_rec + sum(sd_dWS_lmer^2)/n_rec), 
                      sd(dWS_lmer),
                      sd(rnorm(n_rec, mean = dWS_lmer, sd = sd_dWS_lmer))),
           row.names = c('sim','sd(true)', 'lmer', 'with unc','sd(point estimate)','sd(sample)')
)
knitr::kable(df, digits = 5, row.names = TRUE,
             caption = "Comparison of standard deviation estimates.")
```



Table: Comparison of standard deviation estimates.

|                   | phi_s2s|     tau|  phi_ss|
|:------------------|-------:|-------:|-------:|
|sim                | 0.43000| 0.40000| 0.50000|
|sd(true)           | 0.43363| 0.40595| 0.49964|
|lmer               | 0.44042| 0.41048| 0.49731|
|with unc           | 0.44042| 0.41048| 0.49894|
|sd(point estimate) | 0.35544| 0.38813| 0.47214|
|sd(sample)         | 0.44312| 0.41292| 0.50019|

As we can see, the values from `lmer` and the ones calculated according to Equation (4) agree for $\tau$ and $\phi_{S2S}$.
For $\phi_{SS}$, there is a small discrepancy, since the conditional standard deviations are just an approximation.
These values are also close to the true ones, while the standard deviations calculated from the point estimates are underestimating the true values.
The differences is largest for $\phi_{S2S}$, since there are several stations with only few recordings and thus large conditional standard deviations.
Sampling from the conditional distrbution of the random effects/standard deviations leads to values that are closer to the true ones.

Since there are many stations with very few recordings, the value of $\phi_{S2S}$ is severely underestimated when calculated from the point estimates of the site terms.
Thus, we now test whether what happens if we only use stations with at least 5 or 10 recordings.
As we can see from the histogram (which shows 200 repeated simulations), on average the values are closer to the true value, but some bias remains.
If one chooses to go this route, one also has to account for the fact that the estimates are based on fewer stations.


```r
n_sam <- 200
res_s2s <- matrix(nrow = n_sam, ncol = 4)
set.seed(5618)
for(i in 1:n_sam) {
  rect <- rnorm(n_rec, sd = phi_ss_sim)
  statt <- rnorm(n_stat, sd = phi_s2s_sim)
  eqtt <- rnorm(n_eq, sd = tau_sim)
  
  data_reg$y_sim <- eqtt[eq] + statt[stat] + rect
  
  fit_sim <- lmer(y_sim ~ (1 | eq) + (1 | stat), data_reg)
  tmp <- ranef(fit_sim)$stat$`(Intercept)`
  res_s2s[i,] <- c(as.data.frame(VarCorr(fit_sim))$sdcor[1],
                   sd(tmp), sd(tmp[table(stat) >= 5]), sd(tmp[table(stat) >= 10]))
}

data.frame(res_s2s) |> set_names(c('lmer', 'all','N_rec >= 5','N_rec >= 10')) |>
  pivot_longer(everything()) |>
  ggplot() +
  geom_density(aes(x = value, color = name),linewidth = 1.5, key_glyph = draw_key_path) +
  geom_vline(xintercept = phi_s2s_sim, linewidth = 1.5) +
  guides(color = guide_legend(title = NULL)) +
  xlab('phi_S2S') +
  theme(legend.position = c(0.4,0.8))
```

<img src="pictures/sim1-phis2s-1.png" width="50%" />


In GMM development, the standard deviations are often modeled as dependent on some predictor variables such as magnitude.
@Bayless2018 contains a magnitude-dependent $\phi_{S2S}$, which is modeled using the mean magnitude of all records by station.
@Kotha2022 performed a Breusch-Pagan test [@Breusch1979] for heteroscedasticity to test for magnitude dependence of $\tau$ and $\phi_{SS}$.
Below, we calcuale the p-values for the simulated data (which we know is not heteroscedastic).
The null hypothesis is that the data is homoscedastc, and a low p-value is the probability of observing data if the null hypothesis were true.
Based on point estimates, one would conclude that site terms and within-event/within-site residuals are heteroscedastic.
In this context, be aware of hypothesis tests [@Wasserstein2019],[@Amrhein2019].


```r
# calculate p-value of Breusch-Pagan test, testing for dependence on magnitude
df <- data.frame(
  dS = c(lmtest::bptest(dS ~ M, data = data.frame(M = magstat, dS = dS_lmer))$p.value,
         lmtest::bptest(dS ~ M, data = data.frame(M = magstat, dS = rnorm(n_stat, mean = dS_lmer, sd = sd_dS_lmer)))$p.value,
         lmtest::bptest(dS ~ M, data = data.frame(M = magstat, dS = dS_sim))$p.value),
  
  dB = c(lmtest::bptest(dB ~ M, data = data.frame(M = mageq, dB = dB_lmer))$p.value,
         lmtest::bptest(dB ~ M, data = data.frame(M = mageq, dB = rnorm(n_eq, mean = dB_lmer, sd = sd_dB_lmer)))$p.value,
         lmtest::bptest(dB ~ M, data = data.frame(M = mageq, dB = dB_sim))$p.value),
  
  dWS = c(lmtest::bptest(dWS ~ M, data = data.frame(M = data_reg$M, dWS = dWS_lmer))$p.value,
          lmtest::bptest(dWS ~ M, data = data.frame(M = data_reg$M, dWS = rnorm(n_rec, mean = dWS_lmer, sd = sd_dWS_lmer)))$p.value,
          lmtest::bptest(dWS ~ M, data = data.frame(M = data_reg$M, dWS = dWS_sim))$p.value),
  row.names = c('point estimate','sample','true'))

knitr::kable(df, digits = 5, row.names = TRUE,
             caption = "P-values from Breusch-Pagan test.")
```



Table: P-values from Breusch-Pagan test.

|               |      dS|      dB|    dWS|
|:--------------|-------:|-------:|------:|
|point estimate | 0.00000| 0.99921| 0.0000|
|sample         | 0.80356| 0.53500| 0.6652|
|true           | 0.10186| 0.98637| 0.8881|

Below, we calculate the standard deviations of the site terms $\delta S$, the event terms $\delta B$, and the residuals $\delta WS$, for different magnitude bins.
We calculate the standard deviations for the simulated (true) values, the point estimates from the `lmer` fit, and for the estmated random effects/residuals including their uncertainty.
The true value of the standard deviation is shown as a horizontal black line.

We can decreasing values wth magnitude in the standard deviations estimated from the point estimates for $\phi_{S2S}$ and $\phi_{SS}$, while the values calculated from the true samples are more constant (as they should be).
While one always needs to be careful with different numbers of events/records/stations within each bin, plots like these (wth patterns as in these plots) are often used to conclude that standard deviations ($\phi_{S2S}$ and $\phi_{SS}$ in this case) should be modeled as magnitude dependent, which in this case is not true.
The standard deviations that include uncertainty of the random effects/residuals show a pattern that is closr to constant.


```r
# mamgnitude break points for bins
magbins <- c(3,4,5,6,7,8)

# site terms
df_stat <- data.frame(M = magstat, dS_sim, dS_lmer, sd_dS_lmer) %>% 
  mutate(bin = cut(M, breaks = magbins, labels = FALSE)) %>%
  group_by(bin) %>%
  mutate(sd_sim = sd(dS_sim),
         sd_lmer = sd(dS_lmer),
         sd_lmer_unc = sqrt((sum(dS_lmer^2) + sum(sd_dS_lmer^2))/length(dS_lmer^2)),
         meanM = mean(M)) %>%
  arrange(M)

p1 <- cbind(unique(df_stat[,c('sd_sim','sd_lmer','sd_lmer_unc','meanM')]), 
            m1 = magbins[1:(length(magbins)-1)],
            m2 = magbins[2:length(magbins)]) |>
  pivot_longer(!c(m1,m2,meanM)) |>
  ggplot() +
  geom_segment(aes(x = m1, xend = m2, y=value, yend = value, color = name), linewidth = 1.5) +
  scale_color_manual(values = c('red','orange','blue'),
                     labels = c('lmer','lmer with unc','sim')) +
  geom_hline(yintercept = phi_s2s_sim, linewidth = 1.5) +
  guides(color = guide_legend(title=NULL)) +
  labs(x = 'M', y = 'phi_S2S') +
  theme(legend.position = c(0.2,0.2))

# event terms
df_eq <- data.frame(M = mageq, dB_sim, dB_lmer, sd_dB_lmer) %>% 
  mutate(bin = cut(M, breaks = magbins, labels = FALSE)) %>%
  group_by(bin) %>%
  mutate(sd_sim = sd(dB_sim),
         sd_lmer = sd(dB_lmer),
         sd_lmer_unc = sqrt((sum(dB_lmer^2) + sum(sd_dB_lmer^2))/length(dB_lmer^2)),
         meanM = mean(M)) %>%
  arrange(M)

p2 <- cbind(unique(df_eq[,c('sd_sim','sd_lmer','sd_lmer_unc','meanM')]), 
      m1 = magbins[1:(length(magbins)-1)],
      m2 = magbins[2:length(magbins)]) |>
  pivot_longer(!c(m1,m2,meanM)) |>
  ggplot() +
  geom_segment(aes(x = m1, xend = m2, y=value, yend = value, color = name), linewidth = 1.5) +
  scale_color_manual(values = c('red','orange','blue'),
                     labels = c('lmer','lmer with unc','sim')) +
  geom_hline(yintercept = tau_sim, linewidth = 1.5) +
  guides(color = guide_legend(title=NULL)) +
  labs(x = 'M', y = 'tau') +
  theme(legend.position = 'none')

# residuals
df_rec <- data.frame(M = data_reg$M, dWS_sim, dWS_lmer, sd_dWS_lmer) %>% 
  mutate(bin = cut(M, breaks = magbins, labels = FALSE)) %>%
  group_by(bin) %>%
  mutate(sd_sim = sd(dWS_sim),
         sd_lmer = sd(dWS_lmer),
         sd_lmer_unc = sqrt((sum(dWS_lmer^2) + sum(sd_dWS_lmer^2))/length(dWS_lmer^2)),
         meanM = mean(M)) %>%
  arrange(M)

p3 <- cbind(unique(df_rec[,c('sd_sim','sd_lmer','sd_lmer_unc','meanM')]), 
      m1 = magbins[1:(length(magbins)-1)],
      m2 = magbins[2:length(magbins)]) |>
  pivot_longer(!c(m1,m2,meanM)) |>
  ggplot() +
  geom_segment(aes(x = m1, xend = m2, y=value, yend = value, color = name), linewidth = 1.5) +
  scale_color_manual(values = c('red','orange','blue'),
                     labels = c('lmer','lmer with unc','sim')) +
  geom_hline(yintercept = phi_ss_sim, linewidth = 1.5) +
  guides(color = guide_legend(title=NULL)) +
  labs(x = 'M', y = 'phi_SS') +
  theme(legend.position = 'none')

patchwork::wrap_plots(p1,p2,p3, ncol = 2)
```

<img src="pictures/sim1-plot-sd-bin-1.png" width="100%" />

## Magnitude-Dependent Tau and Phi_SS

In this section, we estimate magnitude dependent standard deviations.
We simulate data with magnitude dependent $\tau$ and $\phi_{SS}$.
The dependence has the form
$$
\tau(M) = \left\{
  \begin{array}{ll}
    {\tau}_1 & M \leq M_1 \\
    {\tau}_1 + ({\tau}_2 - {\tau}_1) \frac{M - M_1}{M_2 - M_1} & M_1 < M < M_2 \\
    {\tau}_2 & M \geq M_2
  \end{array}
  \right.
$$
with a form similar for $\phi_{SS}$.

For this simulation, we also generate median predictions from fixed effects, in order to check how well the coefficients are estimated.

First, we declare the values of the standard deviations for the simulations.

```r
# fix standard deviations and magnitude break points
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
```

Now, we declare the coefficients, which are taken from the ITA18 model of @Lanzano2019.
We also compute the linear predictors for the model.


```r
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
```

Now, we randomly sample event terms, site terms, and residuals, and combine with median predictions.


```r
set.seed(1701)
dB_sim <- rnorm(n_eq, sd = tau_sim)
dWS_sim <- rnorm(n_rec, sd = phi_ss_sim)
dS_sim <- rnorm(n_stat, sd = phi_s2s_sim)

data_reg$y_sim <- as.matrix(data_reg[,names_coeffs]) %*% coeffs + dB_sim[eq] + dS_sim[stat] + dWS_sim
```

Firs, we perform a linear mixed effects regression (which assumes homoscedastic standard deviations).
In general, the coefficients are estimated well, but the standard deviations are off.


```r
# linear mixed effects regression
fit_sim <- lmer(y_sim ~ M1 + M2 + MlogR + logR + R + logVS + (1 | eq) + (1 | stat), data_reg)
summary(fit_sim)
```

```
## Linear mixed model fit by REML ['lmerMod']
## Formula: y_sim ~ M1 + M2 + MlogR + logR + R + logVS + (1 | eq) + (1 |  
##     stat)
##    Data: data_reg
## 
## REML criterion at convergence: 22112.8
## 
## Scaled residuals: 
##     Min      1Q  Median      3Q     Max 
## -4.1164 -0.6395  0.0005  0.6258  3.8848 
## 
## Random effects:
##  Groups   Name        Variance Std.Dev.
##  stat     (Intercept) 0.1706   0.4130  
##  eq       (Intercept) 0.1458   0.3819  
##  Residual             0.2786   0.5278  
## Number of obs: 12482, groups:  stat, 1519; eq, 274
## 
## Fixed effects:
##               Estimate Std. Error t value
## (Intercept)  3.6438563  0.0981049  37.142
## M1           0.2919727  0.0548058   5.327
## M2          -0.0849910  0.0971780  -0.875
## MlogR        0.2676451  0.0178760  14.972
## logR        -1.4212048  0.0389614 -36.477
## R           -0.0029312  0.0001654 -17.723
## logVS       -0.3182135  0.0902805  -3.525
## 
## Correlation of Fixed Effects:
##       (Intr) M1     M2     MlogR  logR   R     
## M1     0.755                                   
## M2    -0.311 -0.248                            
## MlogR -0.328 -0.542 -0.385                     
## logR  -0.584 -0.241 -0.178  0.470              
## R      0.389  0.029  0.038 -0.119 -0.798       
## logVS  0.303  0.029  0.029 -0.011 -0.021 -0.035
```

```r
# extract conditional modes and standard deviations of residuals/random effects
tmp <- as.data.frame(ranef(fit_sim))
dS_lmer <- tmp[tmp$grpvar == 'stat','condval']
dB_lmer <- tmp[tmp$grpvar == 'eq','condval']
dWS_lmer <- data_reg$y_sim - predict(fit_sim)

sd_dS_lmer <- tmp[tmp$grpvar == 'stat','condsd']
sd_dB_lmer <- tmp[tmp$grpvar == 'eq','condsd']
sd_dWS_lmer <- sqrt(sd_dB_lmer[eq]^2 + sd_dS_lmer[stat]^2) # approximately

# calculate total resduals
dR_lmer <- data_reg$y_sim - predict(fit_sim, re.form=NA)
```

Next, we run a Stan model [@Carpenter2016], <https://mc-stan.org/>, on the total residuals.
In the Stan model, we can mdel the standard deviations to be magnitude dependent.
Below, we compile the Stan model, and print out its code.


```r
mod <- cmdstan_model(file.path('./Git/MixedModels_Biases/', 'stan', 'gmm_partition_tauM_phiM.stan'))
mod
```

```
## /*********************************************
##  ********************************************/
## 
## data {
##   int N;  // number of records
##   int NEQ;  // number of earthquakes
##   int NSTAT;  // number of stations
## 
##   vector[NEQ] MEQ;
##   vector[N] Y; // ln ground-motion value
## 
##   array[N] int<lower=1,upper=NEQ> eq; // event id (in numerical order from 1 to last)
##   array[N] int<lower=1,upper=NSTAT> stat; // station id (in numerical order from 1 to last)
## 
##   vector[NEQ] M1_eq;
##   vector[NEQ] M2_eq;
##   vector[N] M1_rec;
##   vector[N] M2_rec;
## }
## 
## parameters {
##   real ic;
##   
##   real<lower=0> tau_1;
##   real<lower=0> tau_2;
## 
##   real<lower=0> phi_ss_1;
##   real<lower=0> phi_ss_2;
## 
##   real<lower=0> phi_s2s;
## 
##   vector[NEQ] eqterm;
##   vector[NSTAT] statterm;
## }
## 
## 
## model {
##   ic ~ normal(0,0.1);
## 
##   phi_s2s ~ normal(0,0.5); 
## 
##   tau_1 ~ normal(0,0.5); 
##   tau_2 ~ normal(0,0.5);
## 
##   phi_ss_1 ~ normal(0,0.5); 
##   phi_ss_2 ~ normal(0,0.5);
## 
##   vector[N] phi_ss = M1_rec * phi_ss_1 + M2_rec * phi_ss_2;
##   vector[NEQ] tau = M1_eq * tau_1 + M2_eq * tau_2;
## 
##   eqterm ~ normal(0, tau);
##   statterm ~ normal(0, phi_s2s);
## 
##   Y ~ normal(ic + eqterm[eq] + statterm[stat], phi_ss);
## }
```

Now, we declare the data for Stan, and run the model.
To keep running time low, we only run 200 warm-up and 200 sampling iterations.


```r
data_list <- list(
  N = n_rec,
  NEQ = n_eq,
  NSTAT = n_stat,
  Y = as.numeric(dR_lmer),
  eq = eq,
  stat = stat,
  MEQ = mageq,
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
  refresh = 100,
  max_treedepth = 10,
  adapt_delta = 0.8,
  parallel_chains = 2,
  show_exceptions = FALSE
)
```

```
## Running MCMC with 4 chains, at most 2 in parallel...
## 
## Chain 1 Iteration:   1 / 400 [  0%]  (Warmup) 
## Chain 2 Iteration:   1 / 400 [  0%]  (Warmup) 
## Chain 1 Iteration: 100 / 400 [ 25%]  (Warmup) 
## Chain 2 Iteration: 100 / 400 [ 25%]  (Warmup) 
## Chain 1 Iteration: 200 / 400 [ 50%]  (Warmup) 
## Chain 1 Iteration: 201 / 400 [ 50%]  (Sampling) 
## Chain 2 Iteration: 200 / 400 [ 50%]  (Warmup) 
## Chain 2 Iteration: 201 / 400 [ 50%]  (Sampling) 
## Chain 1 Iteration: 300 / 400 [ 75%]  (Sampling) 
## Chain 2 Iteration: 300 / 400 [ 75%]  (Sampling) 
## Chain 1 Iteration: 400 / 400 [100%]  (Sampling) 
## Chain 1 finished in 132.0 seconds.
## Chain 3 Iteration:   1 / 400 [  0%]  (Warmup) 
## Chain 2 Iteration: 400 / 400 [100%]  (Sampling) 
## Chain 2 finished in 139.2 seconds.
## Chain 4 Iteration:   1 / 400 [  0%]  (Warmup) 
## Chain 3 Iteration: 100 / 400 [ 25%]  (Warmup) 
## Chain 4 Iteration: 100 / 400 [ 25%]  (Warmup) 
## Chain 3 Iteration: 200 / 400 [ 50%]  (Warmup) 
## Chain 3 Iteration: 201 / 400 [ 50%]  (Sampling) 
## Chain 4 Iteration: 200 / 400 [ 50%]  (Warmup) 
## Chain 4 Iteration: 201 / 400 [ 50%]  (Sampling) 
## Chain 3 Iteration: 300 / 400 [ 75%]  (Sampling) 
## Chain 4 Iteration: 300 / 400 [ 75%]  (Sampling) 
## Chain 3 Iteration: 400 / 400 [100%]  (Sampling) 
## Chain 3 finished in 203.4 seconds.
## Chain 4 Iteration: 400 / 400 [100%]  (Sampling) 
## Chain 4 finished in 201.9 seconds.
## 
## All 4 chains finished successfully.
## Mean chain execution time: 169.1 seconds.
## Total execution time: 341.5 seconds.
```

```r
print(fit$cmdstan_diagnose())
```

```
## Processing csv files: /var/folders/p3/r7vrsk6n2d15709vgcky_y880000gn/T/RtmpmoLBg2/gmm_partition_tauM_phiM-202310021107-1-0da5d7.csv, /var/folders/p3/r7vrsk6n2d15709vgcky_y880000gn/T/RtmpmoLBg2/gmm_partition_tauM_phiM-202310021107-2-0da5d7.csv, /var/folders/p3/r7vrsk6n2d15709vgcky_y880000gn/T/RtmpmoLBg2/gmm_partition_tauM_phiM-202310021107-3-0da5d7.csv, /var/folders/p3/r7vrsk6n2d15709vgcky_y880000gn/T/RtmpmoLBg2/gmm_partition_tauM_phiM-202310021107-4-0da5d7.csv
## 
## Checking sampler transitions treedepth.
## Treedepth satisfactory for all transitions.
## 
## Checking sampler transitions for divergences.
## No divergent transitions found.
## 
## Checking E-BFMI - sampler transitions HMC potential energy.
## E-BFMI satisfactory.
## 
## Effective sample size satisfactory.
## 
## Split R-hat values satisfactory all parameters.
## 
## Processing complete, no problems detected.
## $status
## [1] 0
## 
## $stdout
## [1] "Processing csv files: /var/folders/p3/r7vrsk6n2d15709vgcky_y880000gn/T/RtmpmoLBg2/gmm_partition_tauM_phiM-202310021107-1-0da5d7.csv, /var/folders/p3/r7vrsk6n2d15709vgcky_y880000gn/T/RtmpmoLBg2/gmm_partition_tauM_phiM-202310021107-2-0da5d7.csv, /var/folders/p3/r7vrsk6n2d15709vgcky_y880000gn/T/RtmpmoLBg2/gmm_partition_tauM_phiM-202310021107-3-0da5d7.csv, /var/folders/p3/r7vrsk6n2d15709vgcky_y880000gn/T/RtmpmoLBg2/gmm_partition_tauM_phiM-202310021107-4-0da5d7.csv\n\nChecking sampler transitions treedepth.\nTreedepth satisfactory for all transitions.\n\nChecking sampler transitions for divergences.\nNo divergent transitions found.\n\nChecking E-BFMI - sampler transitions HMC potential energy.\nE-BFMI satisfactory.\n\nEffective sample size satisfactory.\n\nSplit R-hat values satisfactory all parameters.\n\nProcessing complete, no problems detected.\n"
## 
## $stderr
## [1] ""
## 
## $timeout
## [1] FALSE
```

```r
print(fit$diagnostic_summary())
```

```
## $num_divergent
## [1] 0 0 0 0
## 
## $num_max_treedepth
## [1] 0 0 0 0
## 
## $ebfmi
## [1] 0.7268607 0.9408649 0.7289387 1.1460655
```

```r
draws_part <- fit$draws()

summarise_draws(subset(draws_part, variable = c('ic','phi','tau'), regex = TRUE))
```

```
## # A tibble: 6 × 10
##   variable     mean   median      sd     mad      q5    q95  rhat ess_bulk
##   <chr>       <num>    <num>   <num>   <num>   <num>  <num> <num>    <num>
## 1 ic       -0.00596 -0.00479 0.0260  0.0266  -0.0484 0.0370 1.02      91.2
## 2 phi_ss_1  0.550    0.550   0.00418 0.00399  0.543  0.557  1.00     812. 
## 3 phi_ss_2  0.395    0.395   0.00831 0.00839  0.381  0.409  1.00     893. 
## 4 phi_s2s   0.431    0.431   0.0109  0.0115   0.414  0.450  1.00     594. 
## 5 tau_1     0.391    0.390   0.0202  0.0187   0.357  0.426  0.999   1775. 
## 6 tau_2     0.319    0.314   0.0520  0.0507   0.240  0.412  1.00     912. 
## # ℹ 1 more variable: ess_tail <num>
```
In general, the parameters are well estimated.
There are not that many events for $M \geq 6$, so the value of $\tau_2$ is quite uncertain.

In the following, we run a Stan model which estimates coefficients and magitude-dependent standard deviations at the same time.
To improve sampling, we use the QR-decomposition of the desgn matrix.


```r
mod <- cmdstan_model(file.path('./Git/MixedModels_Biases/', 'stan', 'gmm_full_qr_tauM_phiM.stan'))
mod
```

```
## /*********************************************
##  ********************************************/
## 
## data {
##   int N;  // number of records
##   int NEQ;  // number of earthquakes
##   int NSTAT;  // number of stations
##   int K;
## 
##   vector[NEQ] MEQ;
##   matrix[N, K-1] X;
##   vector[N] Y; // ln ground-motion value
## 
##   array[N] int<lower=1,upper=NEQ> eq; // event id (in numerical order from 1 to last)
##   array[N] int<lower=1,upper=NSTAT> stat; // station id (in numerical order from 1 to last)
## 
##   vector[NEQ] M1_eq;
##   vector[NEQ] M2_eq;
##   vector[N] M1_rec;
##   vector[N] M2_rec;
## }
## 
## transformed data {
##   matrix[N, K-1] Q_ast = qr_thin_Q(X) * sqrt(N - 1);
##   matrix[K-1, K-1] R_ast = qr_thin_R(X) / sqrt(N - 1);
##   matrix[K-1, K-1] R_ast_inverse = inverse(R_ast);
## }
## 
## parameters {
##   vector[K-1] c_qr;
##   real ic;
## 
##   real<lower=0> tau_1;
##   real<lower=0> tau_2;
## 
##   real<lower=0> phi_ss_1;
##   real<lower=0> phi_ss_2;
## 
##   real<lower=0> phi_s2s;
## 
##   vector[NEQ] eqterm;
##   vector[NSTAT] statterm;
## }
## 
## model {
##   ic ~ normal(0,5);
##   c_qr ~ std_normal();
## 
##   phi_s2s ~ normal(0,0.5); 
## 
##   tau_1 ~ normal(0,0.5); 
##   tau_2 ~ normal(0,0.5);
## 
##   phi_ss_1 ~ normal(0,0.5); 
##   phi_ss_2 ~ normal(0,0.5);
## 
##   vector[N] phi_ss = M1_rec * phi_ss_1 + M2_rec * phi_ss_2;
##   vector[NEQ] tau = M1_eq * tau_1 + M2_eq * tau_2;
## 
##   eqterm ~ normal(0, tau);
##   statterm ~ normal(0, phi_s2s);
## 
##   Y ~ normal(ic + Q_ast * c_qr + eqterm[eq] + statterm[stat], phi_ss);
## 
## }
## 
## generated quantities {
##   vector[K] c;
##   c[1] = ic;
##   c[2:K] =  R_ast_inverse * c_qr;
## }
```

```r
data_list <- list(
  N = n_rec,
  NEQ = n_eq,
  NSTAT = n_stat,
  K = length(coeffs),
  Y = as.numeric(data_reg$y_sim),
  X = data_reg[,c("M1", "M2", "MlogR", "logR", "R", "logVS")], # design matrix
  eq = eq,
  stat = stat,
  MEQ = mageq,
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
  refresh = 100,
  max_treedepth = 10,
  adapt_delta = 0.8,
  parallel_chains = 2,
  show_exceptions = FALSE
)
```

```
## Running MCMC with 4 chains, at most 2 in parallel...
## 
## Chain 1 Iteration:   1 / 400 [  0%]  (Warmup) 
## Chain 2 Iteration:   1 / 400 [  0%]  (Warmup) 
## Chain 2 Iteration: 100 / 400 [ 25%]  (Warmup) 
## Chain 1 Iteration: 100 / 400 [ 25%]  (Warmup) 
## Chain 2 Iteration: 200 / 400 [ 50%]  (Warmup) 
## Chain 2 Iteration: 201 / 400 [ 50%]  (Sampling) 
## Chain 1 Iteration: 200 / 400 [ 50%]  (Warmup) 
## Chain 1 Iteration: 201 / 400 [ 50%]  (Sampling) 
## Chain 2 Iteration: 300 / 400 [ 75%]  (Sampling) 
## Chain 1 Iteration: 300 / 400 [ 75%]  (Sampling) 
## Chain 2 Iteration: 400 / 400 [100%]  (Sampling) 
## Chain 2 finished in 316.5 seconds.
## Chain 3 Iteration:   1 / 400 [  0%]  (Warmup) 
## Chain 1 Iteration: 400 / 400 [100%]  (Sampling) 
## Chain 1 finished in 327.2 seconds.
## Chain 4 Iteration:   1 / 400 [  0%]  (Warmup) 
## Chain 3 Iteration: 100 / 400 [ 25%]  (Warmup) 
## Chain 4 Iteration: 100 / 400 [ 25%]  (Warmup) 
## Chain 3 Iteration: 200 / 400 [ 50%]  (Warmup) 
## Chain 3 Iteration: 201 / 400 [ 50%]  (Sampling) 
## Chain 4 Iteration: 200 / 400 [ 50%]  (Warmup) 
## Chain 4 Iteration: 201 / 400 [ 50%]  (Sampling) 
## Chain 3 Iteration: 300 / 400 [ 75%]  (Sampling) 
## Chain 4 Iteration: 300 / 400 [ 75%]  (Sampling) 
## Chain 3 Iteration: 400 / 400 [100%]  (Sampling) 
## Chain 3 finished in 329.1 seconds.
## Chain 4 Iteration: 400 / 400 [100%]  (Sampling) 
## Chain 4 finished in 327.4 seconds.
## 
## All 4 chains finished successfully.
## Mean chain execution time: 325.0 seconds.
## Total execution time: 655.1 seconds.
```

```r
print(fit$cmdstan_diagnose())
```

```
## Processing csv files: /var/folders/p3/r7vrsk6n2d15709vgcky_y880000gn/T/RtmpmoLBg2/gmm_full_qr_tauM_phiM-202310021113-1-03f6bb.csv, /var/folders/p3/r7vrsk6n2d15709vgcky_y880000gn/T/RtmpmoLBg2/gmm_full_qr_tauM_phiM-202310021113-2-03f6bb.csv, /var/folders/p3/r7vrsk6n2d15709vgcky_y880000gn/T/RtmpmoLBg2/gmm_full_qr_tauM_phiM-202310021113-3-03f6bb.csv, /var/folders/p3/r7vrsk6n2d15709vgcky_y880000gn/T/RtmpmoLBg2/gmm_full_qr_tauM_phiM-202310021113-4-03f6bb.csv
## 
## Checking sampler transitions treedepth.
## Treedepth satisfactory for all transitions.
## 
## Checking sampler transitions for divergences.
## No divergent transitions found.
## 
## Checking E-BFMI - sampler transitions HMC potential energy.
## E-BFMI satisfactory.
## 
## Effective sample size satisfactory.
## 
## Split R-hat values satisfactory all parameters.
## 
## Processing complete, no problems detected.
## $status
## [1] 0
## 
## $stdout
## [1] "Processing csv files: /var/folders/p3/r7vrsk6n2d15709vgcky_y880000gn/T/RtmpmoLBg2/gmm_full_qr_tauM_phiM-202310021113-1-03f6bb.csv, /var/folders/p3/r7vrsk6n2d15709vgcky_y880000gn/T/RtmpmoLBg2/gmm_full_qr_tauM_phiM-202310021113-2-03f6bb.csv, /var/folders/p3/r7vrsk6n2d15709vgcky_y880000gn/T/RtmpmoLBg2/gmm_full_qr_tauM_phiM-202310021113-3-03f6bb.csv, /var/folders/p3/r7vrsk6n2d15709vgcky_y880000gn/T/RtmpmoLBg2/gmm_full_qr_tauM_phiM-202310021113-4-03f6bb.csv\n\nChecking sampler transitions treedepth.\nTreedepth satisfactory for all transitions.\n\nChecking sampler transitions for divergences.\nNo divergent transitions found.\n\nChecking E-BFMI - sampler transitions HMC potential energy.\nE-BFMI satisfactory.\n\nEffective sample size satisfactory.\n\nSplit R-hat values satisfactory all parameters.\n\nProcessing complete, no problems detected.\n"
## 
## $stderr
## [1] ""
## 
## $timeout
## [1] FALSE
```

```r
print(fit$diagnostic_summary())
```

```
## $num_divergent
## [1] 0 0 0 0
## 
## $num_max_treedepth
## [1] 0 0 0 0
## 
## $ebfmi
## [1] 0.8592778 1.0110550 0.7789643 0.8587937
```

```r
draws_full <- fit$draws()

summarise_draws(subset(draws_full, variable = c('phi','tau'), regex = TRUE))
```

```
## # A tibble: 5 × 10
##   variable  mean median      sd     mad    q5   q95  rhat ess_bulk ess_tail
##   <chr>    <num>  <num>   <num>   <num> <num> <num> <num>    <num>    <num>
## 1 phi_ss_1 0.550  0.550 0.00393 0.00397 0.544 0.557 1.01      957.     607.
## 2 phi_ss_2 0.395  0.395 0.00860 0.00822 0.380 0.409 1.00      661.     800.
## 3 phi_s2s  0.433  0.432 0.0110  0.0114  0.415 0.451 1.01      609.     674.
## 4 tau_1    0.392  0.391 0.0206  0.0192  0.358 0.430 0.999    1622.     600.
## 5 tau_2    0.322  0.314 0.0628  0.0582  0.235 0.436 1.01      601.     803.
```

```r
summarise_draws(subset(draws_full, variable = c('^c\\['), regex = TRUE))
```

```
## # A tibble: 7 × 10
##   variable     mean   median       sd      mad       q5      q95  rhat ess_bulk
##   <chr>       <num>    <num>    <num>    <num>    <num>    <num> <num>    <num>
## 1 c[1]      3.60     3.60    0.0986   0.0947    3.44     3.77     1.04     112.
## 2 c[2]      0.274    0.274   0.0521   0.0511    0.188    0.357    1.04     119.
## 3 c[3]     -0.0621  -0.0610  0.0885   0.0820   -0.206    0.0789   1.03     179.
## 4 c[4]      0.268    0.268   0.0156   0.0157    0.241    0.292    1.00    1239.
## 5 c[5]     -1.42    -1.42    0.0377   0.0399   -1.47    -1.36     1.01     734.
## 6 c[6]     -0.00295 -0.00295 0.000166 0.000171 -0.00323 -0.00268  1.01     998.
## 7 c[7]     -0.336   -0.333   0.0886   0.0880   -0.489   -0.192    1.01     209.
## # ℹ 1 more variable: ess_tail <num>
```

The standard deviations are well estimated (very similar to the values based on partitioning the total residuals from the `lmer` fit), and the coefficients are also well estimated.


Below, we plot the posterior distribution of $\tau_1$ and $\tau_2$, together with the true value (black) and the value estimated from point estimates of the event terms in the respective magnitude bins (red), with (solid) and without (dashed) uncertainty.


```r
tmp <- mageq <= mb_tau[1]
tmp <- sqrt((sum(dB_lmer[tmp]^2) + sum(sd_dB_lmer[tmp]^2)) / sum(tmp))

p1 <- data.frame(dR = subset(as_draws_matrix(draws_part), variable = 'tau_1', regex = FALSE),
           full = subset(as_draws_matrix(draws_full), variable = 'tau_1', regex = FALSE)) |>
  set_names(c('dR','full')) |>
  pivot_longer(everything()) |>
  ggplot() +
  geom_density(aes(x = value, color = name), linewidth = 1.5, key_glyph = draw_key_path) +
  geom_vline(xintercept = tau_sim_val[1], linewidth = 1.5) +
  geom_vline(xintercept = sd(dB_lmer[mageq <= mb_tau[1]]), linewidth = 1.5, color = 'red') +
  geom_vline(xintercept = tmp, , linewidth = 1.5, color = 'red', linetype = 'dashed') +
  guides(color = guide_legend(title = NULL)) +
  theme(legend.position = c(0.8,0.8)) +
  xlab('tau_1')


tmp <- mageq >= mb_tau[2]
tmp <- sqrt((sum(dB_lmer[tmp]^2) + sum(sd_dB_lmer[tmp]^2)) / sum(tmp))

p2 <- data.frame(dR = subset(as_draws_matrix(draws_part), variable = 'tau_2', regex = FALSE),
           full = subset(as_draws_matrix(draws_full), variable = 'tau_2', regex = FALSE)) |>
  set_names(c('dR','full')) |>
  pivot_longer(everything()) |>
  ggplot() +
  geom_density(aes(x = value, color = name), linewidth = 1.5, key_glyph = draw_key_path) +
  geom_vline(xintercept = tau_sim_val[2], linewidth = 1.5) +
  geom_vline(xintercept = sd(dB_lmer[mageq >= mb_tau[2]]), linewidth = 1.5, color = 'red') +
  geom_vline(xintercept = tmp, , linewidth = 1.5, color = 'red', linetype = 'dashed') +
  guides(color = guide_legend(title = NULL)) +
  theme(legend.position = c(0.8,0.8)) +
  xlab('tau_2')

patchwork::wrap_plots(p1,p2)
```

<img src="pictures/sim2-hs-plot-tau-1.png" width="100%" />

While we see here that the values of $\tau_1$ and $\tau_2$ are estimated ok from `lmer`, in the paper we show results from 100 simulations which reveal on average a strong bias, whereas estimates from the Stan models are on average better.

Below, we show posterior distributions of $\phi_{SS,1}$ and $\phi_{SS,2}$, similar to the plots for $\tau_1$ and $\tau_2$.
In this case, we see strong biases for the estimates from `lmer`.


```r
tmp <- data_reg$M <= mb_phi[1]
tmp <- sqrt((sum(dWS_lmer[tmp]^2) + sum(sd_dWS_lmer[tmp]^2)) / sum(tmp))

p1 <- data.frame(dR = subset(as_draws_matrix(draws_part), variable = 'phi_ss_1', regex = FALSE),
           full = subset(as_draws_matrix(draws_full), variable = 'phi_ss_1', regex = FALSE)) |>
  set_names(c('dR','full')) |>
  pivot_longer(everything()) |>
  ggplot() +
  geom_density(aes(x = value, color = name), linewidth = 1.5, key_glyph = draw_key_path) +
  geom_vline(xintercept = phi_ss_sim_val[1], linewidth = 1.5) +
  geom_vline(xintercept = sd(dWS_lmer[data_reg$M <= mb_phi[1]]), linewidth = 1.5, color = 'red') +
  geom_vline(xintercept = tmp, linewidth = 1.5, color = 'red', linetype = 'dashed') +
  guides(color = guide_legend(title = NULL)) +
  theme(legend.position = c(0.8,0.8)) +
  xlab('phi_ss_1')


tmp <- data_reg$M >= mb_phi[2]
tmp <- sqrt((sum(dWS_lmer[tmp]^2) + sum(sd_dWS_lmer[tmp]^2)) / sum(tmp))

p2 <- data.frame(dR = subset(as_draws_matrix(draws_part), variable = 'phi_ss_2', regex = FALSE),
           full = subset(as_draws_matrix(draws_full), variable = 'phi_ss_2', regex = FALSE)) |>
  set_names(c('dR','full')) |>
  pivot_longer(everything()) |>
  ggplot() +
  geom_density(aes(x = value, color = name), linewidth = 1.5, key_glyph = draw_key_path) +
  geom_vline(xintercept = phi_ss_sim_val[2], linewidth = 1.5) +
  geom_vline(xintercept = sd(dWS_lmer[data_reg$M >= mb_phi[2]]), linewidth = 1.5, color = 'red') +
  geom_vline(xintercept = tmp, linewidth = 1.5, color = 'red', linetype = 'dashed') +
  guides(color = guide_legend(title = NULL)) +
  theme(legend.position = c(0.8,0.8)) +
  xlab('phi_ss_2')

patchwork::wrap_plots(p1,p2)
```

<img src="pictures/sim2-hs-plot-phiss-1.png" width="100%" />

And finally, the posterior distribution of $\phi_{S2S}$.


```r
data.frame(dR = subset(as_draws_matrix(draws_part), variable = 'phi_s2s', regex = FALSE),
                 full = subset(as_draws_matrix(draws_full), variable = 'phi_s2s', regex = FALSE)) |>
  set_names(c('dR','full')) |>
  pivot_longer(everything()) |>
  ggplot() +
  geom_density(aes(x = value, color = name), linewidth = 1.5, key_glyph = draw_key_path) +
  geom_vline(xintercept = phi_s2s_sim, linewidth = 1.5) +
  geom_vline(xintercept = sd(dS_lmer), linewidth = 1.5, color = 'red') +
  geom_vline(xintercept = sqrt((sum(dS_lmer^2) + sum(sd_dS_lmer^2)) / n_stat),
             linewidth = 1.5, color = 'red', linetype = 'dashed') +
  guides(color = guide_legend(title = NULL)) +
  theme(legend.position = c(0.3,0.8)) +
  xlab('phi_S2S')
```

<img src="pictures/sim2-hs-plot-phis2s-1.png" width="50%" />

We can conclude from this smulation (and the repeated ones in the paper) that the magnitude-dependent standard deviations can be estimated using Stan from total residuals, but one should also account for uncertainty.
Estimating the values from binned random effects/residuals can work but leads to a larger bias.

Our focus is on estimating the magnitude-dependent standard deviations, but as a check we also plot the posterior distrbutions of the coefficients for the full stan model, together with the true values (black) and `lmer` estimates (red).


```r
df <- data.frame(Parameter = c('c[1]','c[2]','c[3]','c[4]','c[5]','c[6]','c[7]'),
           true = coeffs, lmer = fixef(fit_sim))

mcmc_hist(draws_full,regex_pars = 'c\\[') +
  geom_vline(aes(xintercept = true), data = df, linewidth = 1.5) +
  geom_vline(aes(xintercept = lmer), data = df, linewidth = 1.5, color = 'red')
```

```
## `stat_bin()` using `bins = 30`. Pick better value with `binwidth`.
```

<img src="pictures/sim2-hs-coeffs-1.png" width="100%" />

### Estimating Break Points

In the previous section, we have simulated data with magnitude-dependent $\tau$ and $\phi$, using a trilnear function for the magnitude dependence.
In the Stan code, we have used the same fuctional form, and assumed that the magnitude break points are known.
Here, we relax this assumption.
We cannot directly estimate the break points since the Hamiltonian Monte Carlo algorithm of Stan relies on differentiation the model with respect to the parameters (loosely speaking), which leads to problems for functions with sharp beaks.
Instead, we model the dependence using the ``logistic sigmoid'' function (called `inv_logit` in Stan), which is defined as
$$
\sigma(x) = \frac{1}{1 + \exp(-x)}
$$
Below, we run a Stan model where both $\phi_{SS}$ and $\tau$ are modeled using this function.

```r
mod <- cmdstan_model(file.path('./Git/MixedModels_Biases/', 'stan', 'gmm_partition_tauM_phiM_invlogit.stan'))
mod
```

```
## /*********************************************
##  ********************************************/
## 
## data {
##   int N;  // number of records
##   int NEQ;  // number of earthquakes
##   int NSTAT;  // number of stations
## 
##   vector[NEQ] MEQ;
##   vector[N] Y; // ln ground-motion value
## 
##   array[N] int<lower=1,upper=NEQ> eq; // event id (in numerical order from 1 to last)
##   array[N] int<lower=1,upper=NSTAT> stat; // station id (in numerical order from 1 to last)
## 
## }
## 
## parameters {
##   real ic;
##   
##   real<lower=0> tau_1;
##   real<upper=tau_1> tau_2;
##   real<lower=0> tau_scale;
##   real<lower=4> mb_tau;
## 
##   real<lower=0> phi_ss_1;
##   real<upper=phi_ss_1> phi_ss_2;
##   real<lower=0> phi_ss_scale;
##   real<lower=4> mb_phi_ss;
## 
##   real<lower=0> phi_s2s;
## 
##   vector[NEQ] eqterm;
##   vector[NSTAT] statterm;
## }
## 
## transformed parameters {
## }
## 
## 
## model {
##   ic ~ normal(0,0.1);
## 
##   phi_s2s ~ normal(0,0.5); 
## 
##   tau_1 ~ normal(0,0.5); 
##   tau_2 ~ normal(0,0.5);
##   mb_tau ~ normal(5.,1);
##   tau_scale ~ normal(6,1);
## 
##   phi_ss_1 ~ normal(0,0.5); 
##   phi_ss_2 ~ normal(0,0.5);
##   mb_phi_ss ~ normal(5.,1);
##   phi_ss_scale ~ normal(6,1);
## 
##   vector[NEQ] tau = tau_1 - tau_2 * inv_logit(tau_scale * (MEQ - mb_tau));
##   vector[N] phi_ss = phi_ss_1 - phi_ss_2 * inv_logit(phi_ss_scale * (MEQ[eq] - mb_phi_ss));
## 
##   eqterm ~ normal(0, tau);
##   statterm ~ normal(0, phi_s2s);
## 
##   Y ~ normal(ic + eqterm[eq] + statterm[stat], phi_ss);
## 
## }
```

```r
data_list <- list(
  N = n_rec,
  NEQ = n_eq,
  NSTAT = n_stat,
  Y = as.numeric(dR_lmer),
  eq = eq,
  stat = stat,
  MEQ = mageq
)

fit <- mod$sample(
  data = data_list,
  seed = 8472,
  chains = 4,
  iter_sampling = 200,
  iter_warmup = 200,
  refresh = 100,
  max_treedepth = 10,
  adapt_delta = 0.8, # increase to avoid divergences
  parallel_chains = 2,
  show_exceptions = FALSE
)
```

```
## Running MCMC with 4 chains, at most 2 in parallel...
## 
## Chain 1 Iteration:   1 / 400 [  0%]  (Warmup) 
## Chain 2 Iteration:   1 / 400 [  0%]  (Warmup) 
## Chain 1 Iteration: 100 / 400 [ 25%]  (Warmup) 
## Chain 2 Iteration: 100 / 400 [ 25%]  (Warmup) 
## Chain 1 Iteration: 200 / 400 [ 50%]  (Warmup) 
## Chain 1 Iteration: 201 / 400 [ 50%]  (Sampling) 
## Chain 1 Iteration: 300 / 400 [ 75%]  (Sampling) 
## Chain 2 Iteration: 200 / 400 [ 50%]  (Warmup) 
## Chain 2 Iteration: 201 / 400 [ 50%]  (Sampling) 
## Chain 1 Iteration: 400 / 400 [100%]  (Sampling) 
## Chain 1 finished in 209.0 seconds.
## Chain 3 Iteration:   1 / 400 [  0%]  (Warmup) 
## Chain 2 Iteration: 300 / 400 [ 75%]  (Sampling) 
## Chain 2 Iteration: 400 / 400 [100%]  (Sampling) 
## Chain 2 finished in 225.2 seconds.
## Chain 4 Iteration:   1 / 400 [  0%]  (Warmup) 
## Chain 3 Iteration: 100 / 400 [ 25%]  (Warmup) 
## Chain 3 Iteration: 200 / 400 [ 50%]  (Warmup) 
## Chain 3 Iteration: 201 / 400 [ 50%]  (Sampling) 
## Chain 4 Iteration: 100 / 400 [ 25%]  (Warmup) 
## Chain 3 Iteration: 300 / 400 [ 75%]  (Sampling) 
## Chain 4 Iteration: 200 / 400 [ 50%]  (Warmup) 
## Chain 4 Iteration: 201 / 400 [ 50%]  (Sampling) 
## Chain 3 Iteration: 400 / 400 [100%]  (Sampling) 
## Chain 3 finished in 135.4 seconds.
## Chain 4 Iteration: 300 / 400 [ 75%]  (Sampling) 
## Chain 4 Iteration: 400 / 400 [100%]  (Sampling) 
## Chain 4 finished in 133.0 seconds.
## 
## All 4 chains finished successfully.
## Mean chain execution time: 175.7 seconds.
## Total execution time: 359.3 seconds.
```

```r
print(fit$cmdstan_diagnose())
```

```
## Processing csv files: /var/folders/p3/r7vrsk6n2d15709vgcky_y880000gn/T/RtmpY2WJGx/gmm_partition_tauM_phiM_invlogit-202310021433-1-967859.csv, /var/folders/p3/r7vrsk6n2d15709vgcky_y880000gn/T/RtmpY2WJGx/gmm_partition_tauM_phiM_invlogit-202310021433-2-967859.csv, /var/folders/p3/r7vrsk6n2d15709vgcky_y880000gn/T/RtmpY2WJGx/gmm_partition_tauM_phiM_invlogit-202310021433-3-967859.csv, /var/folders/p3/r7vrsk6n2d15709vgcky_y880000gn/T/RtmpY2WJGx/gmm_partition_tauM_phiM_invlogit-202310021433-4-967859.csv
## 
## Checking sampler transitions treedepth.
## Treedepth satisfactory for all transitions.
## 
## Checking sampler transitions for divergences.
## No divergent transitions found.
## 
## Checking E-BFMI - sampler transitions HMC potential energy.
## E-BFMI satisfactory.
## 
## Effective sample size satisfactory.
## 
## The following parameters had split R-hat greater than 1.05:
##   ic, eqterm[47], eqterm[54], eqterm[90], eqterm[104], eqterm[143], eqterm[165]
## Such high values indicate incomplete mixing and biased estimation.
## You should consider regularizating your model with additional prior information or a more effective parameterization.
## 
## Processing complete.
## $status
## [1] 0
## 
## $stdout
## [1] "Processing csv files: /var/folders/p3/r7vrsk6n2d15709vgcky_y880000gn/T/RtmpY2WJGx/gmm_partition_tauM_phiM_invlogit-202310021433-1-967859.csv, /var/folders/p3/r7vrsk6n2d15709vgcky_y880000gn/T/RtmpY2WJGx/gmm_partition_tauM_phiM_invlogit-202310021433-2-967859.csv, /var/folders/p3/r7vrsk6n2d15709vgcky_y880000gn/T/RtmpY2WJGx/gmm_partition_tauM_phiM_invlogit-202310021433-3-967859.csv, /var/folders/p3/r7vrsk6n2d15709vgcky_y880000gn/T/RtmpY2WJGx/gmm_partition_tauM_phiM_invlogit-202310021433-4-967859.csv\n\nChecking sampler transitions treedepth.\nTreedepth satisfactory for all transitions.\n\nChecking sampler transitions for divergences.\nNo divergent transitions found.\n\nChecking E-BFMI - sampler transitions HMC potential energy.\nE-BFMI satisfactory.\n\nEffective sample size satisfactory.\n\nThe following parameters had split R-hat greater than 1.05:\n  ic, eqterm[47], eqterm[54], eqterm[90], eqterm[104], eqterm[143], eqterm[165]\nSuch high values indicate incomplete mixing and biased estimation.\nYou should consider regularizating your model with additional prior information or a more effective parameterization.\n\nProcessing complete.\n"
## 
## $stderr
## [1] ""
## 
## $timeout
## [1] FALSE
```

```r
print(fit$diagnostic_summary())
```

```
## $num_divergent
## [1] 0 0 0 0
## 
## $num_max_treedepth
## [1] 0 0 0 0
## 
## $ebfmi
## [1] 0.8298975 0.9716006 0.8808843 1.0236464
```

```r
draws_part2 <- fit$draws()

summarise_draws(subset(draws_part2, variable = c('ic','phi','tau'), regex = TRUE))
```

```
## # A tibble: 10 × 10
##    variable         mean   median      sd     mad      q5    q95  rhat ess_bulk
##    <chr>           <num>    <num>   <num>   <num>   <num>  <num> <num>    <num>
##  1 ic           -0.00338 -0.00238 0.0298  0.0306  -0.0542 0.0409 1.11      30.9
##  2 phi_ss_1      0.551    0.551   0.00473 0.00443  0.543  0.559  1.00    1191. 
##  3 phi_ss_2      0.152    0.152   0.00982 0.00968  0.136  0.168  1.01     692. 
##  4 phi_ss_scale  6.40     6.39    0.873   0.850    5.01   7.88   1.01    1034. 
##  5 mb_phi_ss     4.96     4.96    0.0595  0.0605   4.87   5.06   1.00     807. 
##  6 phi_s2s       0.431    0.431   0.0117  0.0118   0.413  0.451  0.997    500. 
##  7 tau_1         0.391    0.390   0.0210  0.0217   0.360  0.426  1.00    1481. 
##  8 tau_2         0.0679   0.0630  0.0791  0.0692  -0.0578 0.212  1.01     407. 
##  9 tau_scale     5.94     5.95    0.988   0.963    4.35   7.57   1.00     925. 
## 10 mb_tau        5.66     5.57    0.810   0.838    4.42   7.03   1.00     449. 
## # ℹ 1 more variable: ess_tail <num>
```
We get some warnings about incomplete mixing, which we will ignore here.
We ranonly 400 iterations in total, and some longer chains probably lead to R-hat values that are closer to 1.


We now plot the magnitude dependence of $\tau$ and $\phi_{SS}$, estimated using the correct functional form and the logistic sigmoid function.
For $\tau$, the end values are similiar, but the transition is not well estimated with the logistic sigmoid function.
Ths s due to the fact that there are not many events in total (and not many at large magnitudes), which makes the estimation difficult.
For $\phi_{SS}$, on the other hand, the two estimated functions agree quite well.


```r
# function to calculate lnear predictos for magnitude scaling with break points
func_sd_mag <- function(mag, mb) {
  m1 <- 1 * (mag < mb[2]) - (mag - mb[1]) / (mb[2] - mb[1]) * (mag > mb[1] & mag < mb[2])
  m2 <- 1 * (mag >= mb[2]) + (mag - mb[1]) / (mb[2] - mb[1]) * (mag > mb[1] & mag < mb[2])
  
  return(list(m1 = m1, m2 = m2))
  
}

# logistic sigmoid function
logsig <- function(x) {1/(1 + exp(-x))}

# magnitudes for plotting
mags_f <- seq(3,8,by=0.1)
mags <- func_sd_mag(mags_f, mb_tau)
tau_m <- as_draws_matrix(subset(draws_part, variable = 'tau_1')) %*% matrix(mags$m1, nrow = 1) +
  as_draws_matrix(subset(draws_part, variable = 'tau_2')) %*% matrix(mags$m2, nrow = 1)

# posterior distribution of tau values for different magnitudes
tau_m_sig <- sapply(mags_f,
                    function(m) {as_draws_matrix(subset(draws_part2, variable = 'tau_1')) - 
                        as_draws_matrix(subset(draws_part2, variable = 'tau_2')) *
                        logsig((m - as_draws_matrix(subset(draws_part2, variable = 'mb_tau'))) *
                                 as_draws_matrix(subset(draws_part2, variable = 'tau_scale')))})


p1 <- data.frame(M = mags_f, 
           mean_true = tau_sim_val[1] * mags$m1 + tau_sim_val[2] * mags$m2,
           mean_mod1 = colMeans(tau_m),
           q05_mod1 = colQuantiles(tau_m, probs = 0.05),
           q95_mod1 = colQuantiles(tau_m, probs = 0.95),
           mean_mod2 = colMeans(tau_m_sig),
           q05_mod2 = colQuantiles(tau_m_sig, probs = 0.05),
           q95_mod2 = colQuantiles(tau_m_sig, probs = 0.95)) |>
  pivot_longer(!M, names_sep = '_', names_to = c('par','mod')) |>
  ggplot() +
  geom_line(aes(x = M, y=value, color = mod, linetype = par), linewidth = 1.5) +
  guides(color = guide_legend(title = NULL), linetype = guide_legend(title = NULL)) +
  theme(legend.position = c(0.2,0.3)) +
  scale_color_manual(values = c('red','blue','black'),
                     labels = c('tri-linear','logistic sigmoid','true')) +
  labs(x = 'M', y = 'tau')

############ phi
mags <- func_sd_mag(mags_f, mb_phi)
phi_m <- as_draws_matrix(subset(draws_part, variable = 'phi_ss_1')) %*% matrix(mags$m1, nrow = 1) +
  as_draws_matrix(subset(draws_part, variable = 'phi_ss_2')) %*% matrix(mags$m2, nrow = 1)

# posterior distribution of tau values for different magnitudes
phi_m_sig <- sapply(mags_f,
                    function(m) {as_draws_matrix(subset(draws_part2, variable = 'phi_ss_1')) - 
                        as_draws_matrix(subset(draws_part2, variable = 'phi_ss_2')) *
                        logsig((m - as_draws_matrix(subset(draws_part2, variable = 'mb_phi_ss'))) *
                                 as_draws_matrix(subset(draws_part2, variable = 'phi_ss_scale')))})


p2 <- data.frame(M = mags_f, 
           mean_true = phi_ss_sim_val[1] * mags$m1 + phi_ss_sim_val[2] * mags$m2,
           mean_mod1 = colMeans(phi_m),
           q05_mod1 = colQuantiles(phi_m, probs = 0.05),
           q95_mod1 = colQuantiles(phi_m, probs = 0.95),
           mean_mod2 = colMeans(phi_m_sig),
           q05_mod2 = colQuantiles(phi_m_sig, probs = 0.05),
           q95_mod2 = colQuantiles(phi_m_sig, probs = 0.95)) |>
  pivot_longer(!M, names_sep = '_', names_to = c('par','mod')) |>
  ggplot() +
  geom_line(aes(x = M, y=value, color = mod, linetype = par), linewidth = 1.5) +
  guides(color = guide_legend(title = NULL), linetype = guide_legend(title = NULL)) +
  theme(legend.position = c(0.2,0.3)) +
  scale_color_manual(values = c('red','blue','black'),
                     labels = c('tri-linear','logistic sigmoid','true')) +
  labs(x = 'M', y = 'phi_SS')

patchwork::wrap_plots(p1, p2)
```

<img src="pictures/sim2-hs-logsig-results-1.png" width="100%" />

## Estimating Scaling from Point Estimates of Random Effects

Random effects are sometimes used to estimate scaling of ground motions wth respect to new parameters, such as parameters associated with horizontal-to-vertical ratios.
To assess potential biases, we simulate synthetic data using the ITA18 functional form, and then estimate a model without $V_{S30}$-scaling.
We then estimate the $V_{S30}$-scaling coefficient from site terms.

The coefficients and linear predictors are already set.
Here, we use standard deviations in $log_{10}$-units, which is what was used in @Lanzano2019.
We generate some data, and fit a full model (including $V_{S30}$-scaling).


```r
tau_sim <- 0.17
phi_s2s_sim <- 0.23
phi_sim <- 0.2

set.seed(1701)
dB_sim <- rnorm(n_eq, mean =0, sd = tau_sim)
dS_sim <- rnorm(n_stat, mean =0, sd = phi_s2s_sim)
dWS_sim <- rnorm(n_rec, mean = 0, sd = phi_sim)

data_reg$y_sim <- as.matrix(data_reg[,names_coeffs]) %*% coeffs + dB_sim[eq] + dS_sim[stat] + dWS_sim
fit_sim <- lmer(y_sim ~ M1 + M2 + MlogR + logR + R + logVS + (1|eq) + (1|stat), data_reg)
summary(fit_sim)
```

```
## Linear mixed model fit by REML ['lmerMod']
## Formula: y_sim ~ M1 + M2 + MlogR + logR + R + logVS + (1 | eq) + (1 |  
##     stat)
##    Data: data_reg
## 
## REML criterion at convergence: -1280.9
## 
## Scaled residuals: 
##     Min      1Q  Median      3Q     Max 
## -4.0218 -0.6377  0.0026  0.6275  3.9526 
## 
## Random effects:
##  Groups   Name        Variance Std.Dev.
##  stat     (Intercept) 0.05519  0.2349  
##  eq       (Intercept) 0.02733  0.1653  
##  Residual             0.03971  0.1993  
## Number of obs: 12482, groups:  stat, 1519; eq, 274
## 
## Fixed effects:
##               Estimate Std. Error t value
## (Intercept)  3.478e+00  4.206e-02  82.685
## M1           2.049e-01  2.313e-02   8.862
## M2          -1.067e-01  4.156e-02  -2.567
## MlogR        2.945e-01  7.160e-03  41.131
## logR        -1.399e+00  1.545e-02 -90.557
## R           -2.948e-03  6.396e-05 -46.099
## logVS       -4.171e-01  4.608e-02  -9.051
## 
## Correlation of Fixed Effects:
##       (Intr) M1     M2     MlogR  logR   R     
## M1     0.760                                   
## M2    -0.320 -0.275                            
## MlogR -0.330 -0.521 -0.363                     
## logR  -0.547 -0.253 -0.182  0.506              
## R      0.359  0.036  0.040 -0.134 -0.789       
## logVS  0.357  0.028  0.031 -0.020 -0.022 -0.025
```

Now, we fit the model without the `logVS` term, and then use linear regression on the estimated station term to estiamte the coefficient.
To account for stations with few recordings, we also use only estmated site terms from staions with at least 10 records.
We also fit linear mixed effects model on the total residuals.



```r
fit_sim2 <- lmer(y_sim ~ M1 + M2 + MlogR + logR + R + (1|eq) +(1|stat), data_reg)

deltaS_lmer2 <- ranef(fit_sim2)$stat$`(Intercept)`
deltaB_lmer2 <- ranef(fit_sim2)$eq$`(Intercept)`
data_reg$deltaR_lmer2 <- data_reg$y_sim - predict(fit_sim2, re.form=NA)
tmp <- data.frame(unique(data_reg[,c('stat','logVS')]), deltaS_lmer2 = deltaS_lmer2)

### all station terms
fit_sim2a <- lm(deltaS_lmer2 ~ logVS, tmp)
# summary(fit_sim2a)

### subset station ters
idx <- as.numeric(which(table(stat) >= 10))
fit_sim3a <- lm(deltaS_lmer2 ~ logVS, tmp[idx,])
# summary(fit_sim3a)

### total residuals
fit_sim4 <- lmer(deltaR_lmer2 ~ logVS + (1|eq) + (1|stat), data_reg)

df <- data.frame(name = c('true','full','dS(all)','dS(N>=10)','dR'),
           value = c(coeffs[7], fixef(fit_sim)[7], fit_sim2a$coefficients[2],
                     fit_sim3a$coefficients[2], fixef(fit_sim4)[2]))

knitr::kable(df, digits = 5, row.names = TRUE,
             caption = "Estimated VS30-scaling coefficient")
```



Table: Estimated VS30-scaling coefficient

|   |name      |    value|
|:--|:---------|--------:|
|1  |true      | -0.39458|
|2  |full      | -0.41709|
|3  |dS(all)   | -0.30915|
|4  |dS(N>=10) | -0.34638|
|5  |dR        | -0.41379|

As we can see, the $V_{S30}$-scaling coefficient estimated from site terms $\delta S$ is strongly biased.
The bias persists even if only stations with 10 or more recordings are used.
A mixed-effects model on the total residuals works well to estimate the coefficient.

Below we look at the estimated standard deviations.
The full fit and fit from total residuals estimate $\phi_{S2S}$ well, while the values estimated using linear regression are biased low (using well recorded stations reduces the bias).


```r
df <- data.frame(model = c('true', 'sample','full', 'dS','dS(N>=10)','dR'),
  phi_s2s = c(phi_s2s_sim, sd(dS_sim),as.data.frame(VarCorr(fit_sim))$sdcor[1],
  sigma(fit_sim2a), sigma(fit_sim3a), as.data.frame(VarCorr(fit_sim4))$sdcor[1]))

knitr::kable(df, digits = 5, row.names = TRUE,
             caption = "Phi_S2S")
```



Table: Phi_S2S

|   |model     | phi_s2s|
|:--|:---------|-------:|
|1  |true      | 0.23000|
|2  |sample    | 0.23696|
|3  |full      | 0.23493|
|4  |dS        | 0.20739|
|5  |dS(N>=10) | 0.22793|
|6  |dR        | 0.23493|

## Two-Step Regression

In the simulations, we consder two random effects: event terms $\delta B$ and site terms $\delta S$.
So far, we have estimated the random effects and their associated standard deviations in one single step (`lmer` allows to fit crossed random effects, the Bayesian methods as well).
Traditionally, in GMM development often the algorithm of @Abrahamson1992 is used, which does not work for crossed random effects (Similarly, package `nlme` does not work for crossed random effects).
In this case, one ofen first runs a egresson wh only event terms, and then a second regression which partitions the between-event residuals from the first regression into $\delta S$ and $\delta WS$.
In the following, we investigate the differences.
Again, we randomly sample even terms, site terms, and within-event/within-site residuals, and then fit regression models using `lmer`.
We repeat this process several times, and record the estimated values of coefficients and standard deviations, as well as the confidence intervals.


```r
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
```

Below we plot the desities of the estimated coefficients for the repeated simulations, for both the one-sep and two-step runs.
Overall, on average we can recover the coefficients using both methods reasonably well, but the two-step procedure leads to somewhat wider ranges of the estimated coefficients.


```r
df1 <- data.frame(mat_fix) %>% set_names(names_coeffs)
df1$model <- '1-step'

df2 <- data.frame(mat_fix2) %>% set_names(names_coeffs)
df2$model <- '2-step'

df <- data.frame(name = names_coeffs,
                 true = coeffs)

rbind(df1 %>% pivot_longer(!model),
      df2 %>% pivot_longer(!model)) %>%
  ggplot() +
  geom_density(aes(x = value, color = model), linewidth = 1.5, key_glyph = draw_key_path) +
  facet_wrap(vars(name), scales = "free") +
  geom_vline(aes(xintercept = true), data = df, linewidth = 1.5) +
  guides(color = guide_legend(title = NULL)) +
  labs(x = '') +
  theme(legend.position = c(0.8,0.2),
        strip.text = element_text(size = 20))
```

<img src="pictures/sim-two-step-results-fixef-1.png" width="100%" />

Generally, it is important to not only o have a good estimate of how well the coefficients are estimated, but also how well we ca quantify uncertainty.
Epistemc uncertainty associated with predctions is very important in PSHA.
We now look how often the true coefficient lies inside the 90% confidence interval.
For the one-step regression, the estimates are well calibrated (true coefficient values are inside the 90% confidence interval roughly 90% of the time), while they are not for the two-step procedure.


```r
data.frame(rbind(colSums(mat_ci)/n_sam,
                 colSums(mat_ci2)/n_sam),
           row.names = c('one-step','two-step')) %>%
  set_names(names_coeffs) %>%
  knitr::kable(digits = 5, row.names = TRUE,
               caption = "Fraction how many times the estimated coefficient is inside the 90% confidence interval.")
```



Table: Fraction how many times the estimated coefficient is inside the 90% confidence interval.

|         | intercept|   M1|   M2| MlogR| logR|    R| logVS|
|:--------|---------:|----:|----:|-----:|----:|----:|-----:|
|one-step |      0.91| 0.92| 0.88|  0.93| 0.91| 0.89|  0.86|
|two-step |      0.71| 0.87| 0.90|  0.79| 0.66| 0.57|  0.28|

Below, we plot the densities of the estimated standard deviations, with similar results and conclusions as for the coefficients.


```r
names_sds <- c('phi_s2s','tau','phi_ss')
df1 <- data.frame(mat_sd) %>% set_names(names_sds)
df1$model <- '1-step'

df2 <- data.frame(mat_sd2) %>% set_names(names_sds)
df2$model <- '2-step'

df <- data.frame(name = names_sds,
                 true = sds_sim)

rbind(df1 %>% pivot_longer(!model),
      df2 %>% pivot_longer(!model)) %>%
  ggplot() +
  geom_density(aes(x = value, color = model), linewidth = 1.5, key_glyph = draw_key_path) +
  facet_wrap(vars(name), scales = "free") +
  geom_vline(aes(xintercept = true), data = df, linewidth = 1.5) +
  guides(color = guide_legend(title = NULL)) +
  labs(x = '') +
  theme(legend.position = c(0.8,0.2),
        strip.text = element_text(size = 20))
```

<img src="pictures/sim-two-step-results-sd-1.png" width="100%" />

For the confidence intervals of the standard deviations, again the two-step procedure s not well calibrated.


```r
data.frame(rbind(colSums(mat_ci_sd)/n_sam,
                 colSums(mat_ci_sd2)/n_sam),
           row.names = c('one-step','two-step')) %>%
  set_names(names_sds) %>%
  knitr::kable(digits = 5, row.names = TRUE,
               caption = "Fraction how many times the estimated standard deviation is inside the 90% confidence interval.")
```



Table: Fraction how many times the estimated standard deviation is inside the 90% confidence interval.

|         | phi_s2s|  tau| phi_ss|
|:--------|-------:|----:|------:|
|one-step |    0.94| 0.83|   0.86|
|two-step |    0.80| 0.82|   0.88|

## Correlations between Random Effects

Here, we briefly show that estimating correlations between random effects/residuals can be well estimated from point estimates.
We simulate correlated terms from a bivariate normal distribution, perform a linear mixed-effects regression on each target variable separately, and then calculate the correlation.

The correlation coefficient is calculated as
$$
\rho(X,Y) = \frac{cov(X,Y)}{\sigma_X \sigma_Y}
$$
In general, in GMM modeling we are interested in the correlations of the different terms in the model (event terms, site terms, wthin-event/within-site residuals), which means we can only calculate the sample covariance of the estimates of these terms.
For the standard deviations, however, we can either use the sample standard deviations of the point estimates (which neglects uncertainty), or use the (RE)ML estimate (which takes uncertainty in the random effects/residuals into account).
We compare calculating $\rho$ using the standard deviations of the point estimates in the denominator (whch iswhatis done by usig function `cor`), as well as the ML estimate from `lmer`.
The correlations are well estimated by using point estimates, but are underestimated when using the (RE)ML value in the denominator.
The reason is that the sample covariance of the point estimates underestimates the true covariance.

Next to the individual correlations of the random effects and residual, we are also interested in the total correlations, which are calculated as
\begin{equation}
  \rho_{IM1,IM2} = \frac{\rho_{\delta B_1, \delta B_2} \tau_1 \tau_2 + \rho_{\delta S_1, \delta S_2} \phi_{S2S,1} \phi_{S2S,2} + \rho_{\delta W_1, \delta W_2} \phi_{SS,1} \phi_{SS,2}}{\sigma_{1}\sigma_2} \label{eq: corr total}
\end{equation}
where $IM1$ and $IM2$ are different ground-motion intensity measures.
There are again different possible choices for the use of the standard deviations.
We calculate the total correlation usng the best estimate (REML), the sample standard deviation, and also as the correlation of total residuals.



```r
tau_sim1 <- 0.4
phi_s2s_sim1 <- 0.43
phi_ss_sim1 <- 0.5

tau_sim2 <- 0.45
phi_s2s_sim2 <- 0.4
phi_ss_sim2 <- 0.55

rho_tau <- 0.95
rho_ss <- 0.9
rho_s2s <- 0.85

sigma_tot1 <- sqrt(tau_sim1^2 + phi_s2s_sim1^2 + phi_ss_sim1^2)
sigma_tot2 <- sqrt(tau_sim2^2 + phi_s2s_sim2^2 + phi_ss_sim2^2)

rho_total <- (rho_tau * tau_sim1 * tau_sim2 + 
                rho_s2s * phi_s2s_sim1 * phi_s2s_sim2 + 
                rho_ss * phi_ss_sim1 * phi_ss_sim2) / 
  (sigma_tot1 * sigma_tot2)

cov_tau <- matrix(c(tau_sim1^2, rho_tau * tau_sim1 * tau_sim2,
                    rho_tau * tau_sim1 * tau_sim2, tau_sim2^2), ncol = 2)
cov_s2s <- matrix(c(phi_s2s_sim1^2, rho_s2s * phi_s2s_sim1 * phi_s2s_sim2,
                    rho_s2s * phi_s2s_sim1 * phi_s2s_sim2, phi_s2s_sim2^2), ncol = 2)
cov_ss <- matrix(c(phi_ss_sim1^2, rho_ss * phi_ss_sim1 * phi_ss_sim2,
                   rho_ss * phi_ss_sim1 * phi_ss_sim2, phi_ss_sim2^2), ncol = 2)

eqt2 <- mvtnorm::rmvnorm(n_eq, sigma = cov_tau)
statt2 <- mvtnorm::rmvnorm(n_stat, sigma = cov_s2s)
rect2 <- mvtnorm::rmvnorm(n_rec, sigma = cov_ss)

data_reg$y_sim1 <- eqt2[eq,1] + statt2[stat,1] + rect2[,1]
data_reg$y_sim2 <- eqt2[eq,2] + statt2[stat,2] + rect2[,2]

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

rho_t <- (sds1[1] * sds2[1] * cor(dS1, dS2) +
            sds1[2] * sds2[2] * cor(dB1, dB2) +
            sds1[3] * sds2[3] * cor(dWS1, dWS2)) /
  (sqrt(sum(sds1^2)) * sqrt(sum(sds2^2)))

rho_ta <- (sds1a[1] * sds2a[1] * cor(dS1, dS2) +
             sds1a[2] * sds2a[2] * cor(dB1, dB2) +
             sds1a[3] * sds2a[3] * cor(dWS1, dWS2)) /
  (sqrt(sum(sds1a^2)) * sqrt(sum(sds2a^2)))

df <- data.frame(dS = c(rho_s2s, cor(dS1,dS2), cov(dS1,dS2)/(sd(dS1) * sd(dS2)), cov(dS1,dS2)/(sds1[1] * sds2[1])),
           dB = c(rho_tau, cor(dB1,dB2), cov(dB1,dB2)/(sd(dB1) * sd(dB2)), cov(dB1,dB2)/(sds1[2] * sds2[2])),
           dWS = c(rho_ss, cor(dWS1,dWS2), cov(dWS1,dWS2)/(sd(dWS1) * sd(dWS2)), cov(dWS1,dWS2)/(sds1[3] * sds2[3])),
           dR = c(rho_total, cor(dR1, dR2), rho_t, rho_ta),
           row.names = c('true','cor','cov/sd(point estimate)','cov()/hat()'))
knitr::kable(df, digits = 3, row.names = TRUE,
             caption = "Estimated correlation coefficients.")
```



Table: Estimated correlation coefficients.

|                       |    dS|    dB|   dWS|    dR|
|:----------------------|-----:|-----:|-----:|-----:|
|true                   | 0.850| 0.950| 0.900| 0.898|
|cor                    | 0.863| 0.952| 0.898| 0.905|
|cov/sd(point estimate) | 0.863| 0.952| 0.898| 0.904|
|cov()/hat()            | 0.518| 0.853| 0.815| 0.907|

## Correlation with e.g. Stress Drop


```r
tau_sim <- 0.4
phi_s2s_sim <- 0.43
phi_ss_sim <- 0.5

rho <- 0.7
tau2 <- 1

cov_tau <- matrix(c(tau_sim^2, rho * tau_sim * tau2, 
                    rho * tau_sim * tau2, tau2^2), ncol = 2)

set.seed(5618)
deltaWS_sim <- rnorm(n_rec, sd = phi_ss_sim)
deltaS_sim <- rnorm(n_stat, sd = phi_s2s_sim)
eqt2 <- mvtnorm::rmvnorm(n_eq, sigma = cov_tau)

y_sim <- eqt2[eq,1] + deltaS_sim[stat] + deltaWS_sim
data_reg$y_sim <- y_sim

fit_lmer_sim <- lmer(y_sim ~ (1 | eq) + (1 | stat), data_reg)
deltaB_lmer <- ranef(fit_lmer_sim)$eq$`(Intercept)`

cor(deltaB_lmer, eqt2[,2])
```

```
## [1] 0.6591753
```

The fit by itself is ok, but in this case the correlation is underestimated.
For the paper, we have repeated this simulation 100 times, and on average we observe a bias.

In the following, we fit a Stan model to the data in which we directly estimate the correlation between the event terns and the other event related variable $E$during te model fitting phase, i.e. during partitioning.
This implicitly takes the uncertainty in the estimated event terms into account.
We model the two variables as a bivariate normal distribution, and use the conditional distribution of $\delta B$ given $E$ as the prior distribution for $\delta B$
$$
\begin{aligned}
  \delta B &\sim N(\mu_s, \tau_s) \\
  \mu_s &= \frac{\tau}{\sigma_{E}} \; \rho \;E \\
  \tau_s &= \sqrt{(1 - \rho^2) \tau^2}
\end{aligned}
$$


```r
mod <- cmdstan_model(file.path('./Git/MixedModels_Biases/', 'stan', 'gmm_partition_wvar_corr.stan'))

data_list_cor <- list(
  N = n_rec,
  NEQ = n_eq,
  NSTAT = n_stat,
  Y = data_reg$y_sim,
  E = eqt2[,2],
  eq = eq,
  stat = stat,
  alpha = c(1,1,1)
)

fit <- mod$sample(
  data = data_list_cor,
  seed = 8472,
  chains = 4,
  iter_sampling = 200,
  iter_warmup = 200,
  refresh = 100,
  max_treedepth = 10,
  adapt_delta = 0.8,
  parallel_chains = 2,
  show_exceptions = FALSE
)
```

```
## Running MCMC with 4 chains, at most 2 in parallel...
## 
## Chain 1 Iteration:   1 / 400 [  0%]  (Warmup) 
## Chain 2 Iteration:   1 / 400 [  0%]  (Warmup) 
## Chain 1 Iteration: 100 / 400 [ 25%]  (Warmup) 
## Chain 2 Iteration: 100 / 400 [ 25%]  (Warmup) 
## Chain 1 Iteration: 200 / 400 [ 50%]  (Warmup) 
## Chain 1 Iteration: 201 / 400 [ 50%]  (Sampling) 
## Chain 2 Iteration: 200 / 400 [ 50%]  (Warmup) 
## Chain 2 Iteration: 201 / 400 [ 50%]  (Sampling) 
## Chain 1 Iteration: 300 / 400 [ 75%]  (Sampling) 
## Chain 2 Iteration: 300 / 400 [ 75%]  (Sampling) 
## Chain 1 Iteration: 400 / 400 [100%]  (Sampling) 
## Chain 1 finished in 35.6 seconds.
## Chain 3 Iteration:   1 / 400 [  0%]  (Warmup) 
## Chain 2 Iteration: 400 / 400 [100%]  (Sampling) 
## Chain 2 finished in 36.0 seconds.
## Chain 4 Iteration:   1 / 400 [  0%]  (Warmup) 
## Chain 4 Iteration: 100 / 400 [ 25%]  (Warmup) 
## Chain 3 Iteration: 100 / 400 [ 25%]  (Warmup) 
## Chain 4 Iteration: 200 / 400 [ 50%]  (Warmup) 
## Chain 4 Iteration: 201 / 400 [ 50%]  (Sampling) 
## Chain 3 Iteration: 200 / 400 [ 50%]  (Warmup) 
## Chain 3 Iteration: 201 / 400 [ 50%]  (Sampling) 
## Chain 4 Iteration: 300 / 400 [ 75%]  (Sampling) 
## Chain 3 Iteration: 300 / 400 [ 75%]  (Sampling) 
## Chain 4 Iteration: 400 / 400 [100%]  (Sampling) 
## Chain 4 finished in 39.4 seconds.
## Chain 3 Iteration: 400 / 400 [100%]  (Sampling) 
## Chain 3 finished in 41.7 seconds.
## 
## All 4 chains finished successfully.
## Mean chain execution time: 38.2 seconds.
## Total execution time: 77.5 seconds.
```

```r
print(fit$cmdstan_diagnose())
```

```
## Processing csv files: /var/folders/p3/r7vrsk6n2d15709vgcky_y880000gn/T/RtmpAkuaKi/gmm_partition_wvar_corr-202403221351-1-8208b1.csv, /var/folders/p3/r7vrsk6n2d15709vgcky_y880000gn/T/RtmpAkuaKi/gmm_partition_wvar_corr-202403221351-2-8208b1.csv, /var/folders/p3/r7vrsk6n2d15709vgcky_y880000gn/T/RtmpAkuaKi/gmm_partition_wvar_corr-202403221351-3-8208b1.csv, /var/folders/p3/r7vrsk6n2d15709vgcky_y880000gn/T/RtmpAkuaKi/gmm_partition_wvar_corr-202403221351-4-8208b1.csv
## 
## Checking sampler transitions treedepth.
## Treedepth satisfactory for all transitions.
## 
## Checking sampler transitions for divergences.
## No divergent transitions found.
## 
## Checking E-BFMI - sampler transitions HMC potential energy.
## E-BFMI satisfactory.
## 
## Effective sample size satisfactory.
## 
## Split R-hat values satisfactory all parameters.
## 
## Processing complete, no problems detected.
## $status
## [1] 0
## 
## $stdout
## [1] "Processing csv files: /var/folders/p3/r7vrsk6n2d15709vgcky_y880000gn/T/RtmpAkuaKi/gmm_partition_wvar_corr-202403221351-1-8208b1.csv, /var/folders/p3/r7vrsk6n2d15709vgcky_y880000gn/T/RtmpAkuaKi/gmm_partition_wvar_corr-202403221351-2-8208b1.csv, /var/folders/p3/r7vrsk6n2d15709vgcky_y880000gn/T/RtmpAkuaKi/gmm_partition_wvar_corr-202403221351-3-8208b1.csv, /var/folders/p3/r7vrsk6n2d15709vgcky_y880000gn/T/RtmpAkuaKi/gmm_partition_wvar_corr-202403221351-4-8208b1.csv\n\nChecking sampler transitions treedepth.\nTreedepth satisfactory for all transitions.\n\nChecking sampler transitions for divergences.\nNo divergent transitions found.\n\nChecking E-BFMI - sampler transitions HMC potential energy.\nE-BFMI satisfactory.\n\nEffective sample size satisfactory.\n\nSplit R-hat values satisfactory all parameters.\n\nProcessing complete, no problems detected.\n"
## 
## $stderr
## [1] ""
## 
## $timeout
## [1] FALSE
```

```r
print(fit$diagnostic_summary())
```

```
## $num_divergent
## [1] 0 0 0 0
## 
## $num_max_treedepth
## [1] 0 0 0 0
## 
## $ebfmi
## [1] 0.8160571 0.8943540 0.7220071 0.8905378
```

```r
draws_corr <- fit$draws()


summarise_draws(subset(draws_corr, variable = c('rho', 'phi', 'tau'), regex = TRUE))
```

```
## # A tibble: 5 × 10
##   variable  mean median      sd     mad    q5   q95  rhat ess_bulk ess_tail
##   <chr>    <dbl>  <dbl>   <dbl>   <dbl> <dbl> <dbl> <dbl>    <dbl>    <dbl>
## 1 rho      0.686  0.687 0.0325  0.0323  0.629 0.736  1.00     872.     717.
## 2 phi_ss   0.497  0.497 0.00335 0.00331 0.492 0.503  1.01     649.     528.
## 3 phi_s2s  0.441  0.441 0.0115  0.0119  0.422 0.459  1.00     568.     684.
## 4 tau2     0.992  0.991 0.0434  0.0453  0.926 1.07   1.00     867.     657.
## 5 tau      0.422  0.420 0.0205  0.0207  0.389 0.456  1.00     590.     466.
```

The fit looks good, and we also get a good estimate of $\rho$.

Below, we show the posterior distribution of the correlation coefficient $\rho$, together with the true value (black), the correlation between point estimates $\widehat{\delta B}$ and $E$ from `lmer` (red), and the correlation between the mean event terms (point estimates) from the Stan fit (blue).
The black dashed line is the mean of he posterior distribution of `rho`.


```r
mcmc_hist(draws_corr, pars = 'rho') +
  vline_at(rho, linewidth = 1.5) +
  vline_at(colMeans(subset(as_draws_matrix(draws_corr), variable = 'rho', regex = FALSE)),
                    linewidth = 1.5, linetype = 'dashed') +
  vline_at(cor(deltaB_lmer, eqt2[,2]), linewidth = 1.5, color = 'red') +
  vline_at(cor(eqt2[,2], colMeans(subset(as_draws_matrix(draws_corr), variable = 'eqterm', regex = TRUE))),
           linewidth = 1.5, color = 'blue')
```

```
## `stat_bin()` using `bins = 30`. Pick better value with `binwidth`.
```

<img src="pictures/sm-corr-stan-1.png" width="50%" />

# Simulations using Italian Data

Now, we use the Italian data of @Lanzano2019 (see also @Caramenti2022) for simulations with spatial/nonergodic models.
This data was used in @Kuehn2022b for nonergodic model comparison, which made it easy to set up.

First, we read and prepare the data.


```r
data_it <- read.csv(file.path('./Git/MixedModels_Biases/','/data','italian_data_pga_id_utm_stat.csv'))

# Set linear predictors
mh = 5.5
mref = 5.324
h = 6.924
attach(data_it)
b1 = (mag-mh)*(mag<=mh)
b2 = (mag-mh)*(mag>mh)
c1 = (mag-mref)*log10(sqrt(JB_complete^2+h^2))
c2 = log10(sqrt(JB_complete^2+h^2))
c3 = sqrt(JB_complete^2+h^2)
f1 = as.numeric(fm_type_code == "SS")
f2 = as.numeric(fm_type_code == "TF")
k = log10(vs30/800)*(vs30<=1500)+log10(1500/800)*(vs30>1500)
y = log10(rotD50_pga)
detach(data_it)

n_rec <- length(b1)
eq <- data_it$EQID
stat <- data_it$STATID
n_eq <- max(eq)
n_stat <- max(stat)
n_rec <- nrow(data_it)

mageq <- unique(data_it[,c('EQID','mag')])[,2]

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
                       intercept = 1,
                       M = data_it$mag
)

# coefficients from ITA18
coeffs <- c(3.421046409, 0.193954090, -0.021982777, 0.287149291,
            -1.405635476, -0.002911264, 0.085983743, 0.010500239,
            -0.394575970)
names_coeffs <- c("intercept", "M1", "M2", "MlogR", "logR", "R", "Fss", "Frv", "logVS")
```



```r
p1 <- ggplot(data_it) +
  geom_point(aes(x = JB_complete, y = mag)) +
  scale_x_log10(breaks = breaks, minor_breaks = minor_breaks)
p2 <- ggplot(unique(data_it[,c('EQID','mag')])) +
  geom_histogram(aes(x = mag))
patchwork::wrap_plots(p1, p2)
```

```
## Warning: Transformation introduced infinite values in continuous x-axis
```

```
## `stat_bin()` using `bins = 30`. Pick better value with `binwidth`.
```

<img src="pictures/italy-data-plots-1.png" width="100%" />
We simulate data for some spatial models on the Italian data, and we use INLA (<https://www.r-inla.org/>) to estimate the models.
Below, we set the penalized complexity prior [@Simpson2017] for the standard deviations, used throughout.


```r
prior_prec_tau <- list(prec = list(prior = 'pc.prec', param = c(0.3, 0.01)))
prior_prec_phiS2S    <- list(prec = list(prior = 'pc.prec', param = c(0.3, 0.01))) 
prior_prec_phiSS    <- list(prec = list(prior = 'pc.prec', param = c(0.3, 0.01)))
```

## Spatial correlations of Site Terms

In nonergodic models, site terms are often modeled as spatially correlated.
The spatial correlation structure can be assessed from point estimates of the site terms.
Here, we simulate some data with spatially correlated site terms, to check whether we can get the model parameters back.

We use the Mat\'ern covariance function for the spatial correlation of the site terms, which is defined below.
In general, we follow @Krainski2019 when seting up the spatial models.


```r
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
```

Next we define the coefficients, spatial range, and standard deviations for the simulation.


```r
# unique station coordinates
co_stat_utm <- unique(data_it[,c("STATID", "X_stat","Y_stat")])[,c(2,3)]

range <- 30 # spatial range
nu <- 1
kappa <- sqrt(8*nu)/range

# standard deviations
wvar <- 0.65 # variance ratio for phi_s2s
tau_sim <- 0.17
phi_s2s_sim <- 0.23
phi_s2s_0 <- sqrt((1 - wvar) * phi_s2s_sim^2)
phi_s2s_c <- sqrt(wvar * phi_s2s_sim^2)
phi_ss_sim <- 0.2

cov <- phi_s2s_c^2 * cMatern(as.matrix(dist(co_stat_utm)), nu, kappa) + diag(10^-9, n_stat)
```

To use R-INLA with the ``stochastic partial differential equation'' (SPDE) approach, we need to define a mesh.


```r
max.edge2    <- 5 
bound.outer2 <- 40 
mesh = inla.mesh.2d(loc=co_stat_utm,
                    max.edge = c(1,5)*max.edge2,
                    cutoff = max.edge2,
                    offset = c(5 * max.edge2, bound.outer2))
print(mesh$n) # print number of mesh nodes
```

```
## [1] 26702
```

Now we define priors for the standard deviations (based on @Simpson2017), the SPDE prior, the projecion matrix `A`, and the formula.


```r
spde_stat <- inla.spde2.pcmatern(
  # Mesh and smoothness parameter
  mesh = mesh, alpha = 2,
  # P(practic.range < 0.3) = 0.5
  prior.range = c(100, 0.9),
  # P(sigma > 1) = 0.01
  prior.sigma = c(.3, 0.01))

A_stat <- inla.spde.make.A(mesh, loc = as.matrix(co_stat_utm[stat,]))
A_stat_unique   <- inla.spde.make.A(mesh, loc = as.matrix(co_stat_utm))
idx_stat   <- inla.spde.make.index("idx_stat",spde_stat$n.spde)

# formula to be used on the total residuals
form_spatial_total <- y ~ 0 + intercept + 
  f(eq, model = "iid", hyper = prior_prec_tau) + 
  f(stat, model = "iid",hyper = prior_prec_phiS2S) +
  f(idx_stat, model = spde_stat)

# formula for the full fit
form_spatial_stat <- y ~ 0 + intercept + 
  M1 + M2 + MlogR + logR + R + Fss + Frv + logVS +
  f(eq, model = "iid", hyper = prior_prec_tau) + 
  f(stat, model = "iid",hyper = prior_prec_phiS2S) +
  f(idx_stat, model = spde_stat)

# formula for fit from site terms
form_spatial_stat_u <- y ~ 0 + intercept + f(idx_stat, model = spde_stat)
```


Now, we sample the event terms, site terms, spatially correlated site terms, and within-event/within-site residuals, and combine them with the median predictions from ITA18.
We then fit a `lmer` model to the data.


```r
set.seed(8472)
dB_sim <- rnorm(n_eq, mean =0, sd = tau_sim)
dS_sim <- rnorm(n_stat, mean =0, sd = phi_s2s_0)
dWS_sim <- rnorm(n_rec, mean = 0, sd = phi_ss_sim)

data_reg$y_sim <- as.matrix(data_reg[,names_coeffs]) %*% coeffs +
  dB_sim[eq] + dS_sim[stat] + dWS_sim + rmvnorm0(1, cov)[stat]

fit_sim <- lmer(y_sim ~ M1 + M2 + MlogR + logR + R + Fss + Frv + logVS + (1|eq) + (1|stat), data_reg)
dR_lmer <- data_reg$y_sim - predict(fit_sim, re.form=NA)
dS_lmer <- ranef(fit_sim)$stat$`(Intercept)`
```

Now, we fit the Inla models.
We fit the full model (fixed and random effects), a model on the total residuals, and a model on the site terms from the `lmer` fit.


```r
#### full model
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

### ft from site terms
data_reg_stat <- data.frame(
  dS = dS_lmer,
  intercept = 1
)

# create the stack
stk_spatial_stat_u <- inla.stack(
  data = list(y = data_reg_stat$dS),
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

#### fit from total residuals
data_reg$deltaR <- dR_lmer
# create the stack
stk_spatial_total <- inla.stack(
  data = list(y = data_reg$deltaR),
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
```

Below, we plot the posterior distributions of the spatial range as well as the associated standard deviation.


```r
p1 <- rbind(
  data.frame(inla.tmarginal(function(x) exp(x), 
                                fit_inla_spatial_stat$internal.marginals.hyperpar$`log(Range) for idx_stat`),
             mod = "full"),
  data.frame(inla.tmarginal(function(x) exp(x), 
                                fit_inla_spatial_stat_u$internal.marginals.hyperpar$`log(Range) for idx_stat`),
             mod = "dS"),
  data.frame(inla.tmarginal(function(x) exp(x), 
                                fit_inla_spatial_total$internal.marginals.hyperpar$`log(Range) for idx_stat`),
             mod = "dR")
) %>%
  ggplot() +
  geom_line(aes(x = x, y = y, color = mod), linewidth = 1.5) +
  geom_vline(xintercept = range, linewidth = 1.5) +
  labs(x = 'spatial range (km)', 'density') +
  theme(legend.position = c(0.8,0.8)) +
  guides(color = guide_legend(title = NULL))

p2 <- rbind(
  data.frame(inla.tmarginal(function(x) exp(x), 
                            fit_inla_spatial_stat$internal.marginals.hyperpar$`log(Stdev) for idx_stat`),
             mod = "full", type = 'phi_s2s_c'),
  data.frame(inla.tmarginal(function(x) exp(x), 
                            fit_inla_spatial_stat_u$internal.marginals.hyperpar$`log(Stdev) for idx_stat`),
             mod = "dS", type = 'phi_s2s_c'),
  data.frame(inla.tmarginal(function(x) exp(x), 
                            fit_inla_spatial_total$internal.marginals.hyperpar$`log(Stdev) for idx_stat`),
             mod = "dR", type = 'phi_s2s_c'),
  data.frame(inla.tmarginal(function(x) sqrt(exp(-x)), 
                            fit_inla_spatial_stat$internal.marginals.hyperpar$`Log precision for stat`),
             mod = "full", type = 'phi_s2s_0'),
  data.frame(inla.tmarginal(function(x) sqrt(exp(-x)), 
                            fit_inla_spatial_stat_u$internal.marginals.hyperpar$`Log precision for the Gaussian observations`),
             mod = "dS", type = 'phi_s2s_0'),
  data.frame(inla.tmarginal(function(x) sqrt(exp(-x)), 
                            fit_inla_spatial_total$internal.marginals.hyperpar$`Log precision for stat`),
             mod = "dR", type = 'phi_s2s_0')
) %>%
  ggplot() +
  geom_line(aes(x = x, y = y, color = mod, linetype = type), linewidth = 1.5) +
  geom_vline(xintercept = phi_s2s_c, linewidth = 1.5, linetype = 'dashed') +
  geom_vline(xintercept = phi_s2s_0, linewidth = 1.5) +
  labs(x = 'phi_S2S_c', 'density') +
  theme(legend.position = c(0.8,0.8)) +
  guides(color = guide_legend(title = NULL), linetype = guide_legend(title = NULL))

patchwork::wrap_plots(p1, p2)
```

<img src="pictures/sim-it-spatial-results-1.png" width="100%" />

We can see that the spatial range is quite well estimated for all approaches, and that the full model and the model based on total residuals give almost the same results.
The model based on site terms does not lead to good results for the standard deviations, in particular for $\phi_{S2S,c}$, which is severely underestimated.
The relative sizes of standard devations based on the fit from $\delta S$ are wrongly estimated.


Below, we plot differences predictions of the spatially correlated site terms between the different models.
We show differences with respect to the full model.
As we can see, there are strong differences in the predicted means of the spatially correlated site terms of the full model and the model estmated from site terms.


```r
diff <- fit_inla_spatial_stat_u$summary.random$idx_stat$mean - fit_inla_spatial_stat$summary.random$idx_stat$mean
diff2 <- fit_inla_spatial_total$summary.random$idx_stat$mean - fit_inla_spatial_stat$summary.random$idx_stat$mean

p1 <- ggplot() + inlabru::gg(mesh, color = diff, nx = 500, ny = 500) +
  labs(x="X (km)", y="Y (km)", title = "Difference in Mean Predictions",
       subtitle = "dS - full") +
  scale_fill_gradient2(name = "", limits = c(-0.15,0.15))

p2 <- ggplot() + inlabru::gg(mesh, color = diff2, nx = 500, ny = 500) +
  labs(x="X (km)", y="Y (km)", title = "Difference in Mean Predictions",
       subtitle = "dR - full") +
  scale_fill_gradient2(name = "", limits = c(-0.15,0.15))
patchwork::wrap_plots(p1, p2)
```

<img src="pictures/sim-it-spatial-plot-diff-1.png" width="100%" />

## Cell-specific attenuation

In this section, we simulate data based on the cell-specific attenuation model [@Kuehn2019,@Dawood2013], and estimate the model parameters using the total model, as well as from within-event/within-site residuals.

First, we read in the cell-specific distances, and some definitions.


```r
# read in cell-specific attenuation
data_dm <- rstan::read_rdump(file.path('./Git/MixedModels_Biases/','/data','dm_25x25.Rdata'))
dm_sparse <- as(data_dm$RC,"dgCMatrix") / 100 # divide by 100 to avoid small values
n_cell <- data_dm$NCELL

prior_prec_cell    <- list(prec = list(prior = 'pc.prec', param = c(0.5, 0.01))) 
data_reg$idx_cell <- 1:n_rec
```

Now we define the parameters for the simulation, and sample.


```r
tau_sim <- 0.17
phi_s2s_sim <- 0.2
phi_ss_sim <- 0.18
sigma_cell_sim <- 0.35

set.seed(5618)
dB_sim <- rnorm(n_eq, mean =0, sd = tau_sim)
dS_sim <- rnorm(n_stat, mean =0, sd =phi_s2s_sim)
dWS_sim <- rnorm(n_rec, mean = 0, sd = phi_ss_sim)
dC_sim <- rnorm(n_cell, mean = 0, sd = sigma_cell_sim)

data_reg$y_sim <- as.numeric(as.matrix(data_reg[,names_coeffs]) %*% coeffs + dm_sparse %*% dC_sim +
  dB_sim[eq] + dS_sim[stat] + dWS_sim)
```

Now we fit the different models using Inla.


```r
# full fit
fit_sim_cell <- inla(y_sim ~ 0 + intercept + M1 + M2 + logR + MlogR + R + Fss + Frv + logVS +
                       f(eq, model = "iid", hyper = prior_prec_tau) + 
                       f(stat, model = "iid",hyper = prior_prec_phiS2S) +
                       f(idx_cell, model = "z", Z = dm_sparse, hyper = prior_prec_cell),
                     data = data_reg,
                     family="gaussian",
                     control.family = list(hyper = prior_prec_phiSS),
                     quantiles = c(0.05,0.5,0.95)
)
```

```
## Warning in inla.model.properties.generic(inla.trim.family(model), mm[names(mm) == : Model 'z' in section 'latent' is marked as 'experimental'; changes may appear at any time.
##   Use this model with extra care!!! Further warnings are disabled.
```

```r
fit_sim <- lmer(y_sim ~ M1 + M2 + logR + MlogR + R + Fss + Frv + logVS + (1|eq) + (1|stat), data_reg)
data_reg$dWS_lmer <- data_reg$y_sim - predict(fit_sim)
fit_sim_cell_dws <- inla(dWS_lmer ~ 0 + intercept +
                            f(idx_cell, model = "z", Z = dm_sparse, hyper = prior_prec_cell),
                          data = data_reg,
                          family="gaussian",
                          control.family = list(hyper = prior_prec_phiSS),
                          quantiles = c(0.05,0.5,0.95)
)

# total residuals, but also take out linear R term
data_reg$dR_lmer <- data_reg$y_sim - (predict(fit_sim,re.form=NA) -
                           fixef(fit_sim)[6] * data_reg$R)

fit_sim_cell_dR <- inla(dR_lmer ~ 0 + intercept + R +
                          f(eq, model = "iid", hyper = prior_prec_tau) + 
                          f(stat, model = "iid",hyper = prior_prec_phiS2S) +
                          f(idx_cell, model = "z", Z = dm_sparse, hyper = prior_prec_cell),
                        data = data_reg,
                        family="gaussian",
                        control.family = list(hyper = prior_prec_phiSS),
                        quantiles = c(0.05,0.5,0.95)
)
```

Ths shows the posterior distribution of the standard deviation of the cell-specific attenuation coefficients, which is understimated from $\delta WS$.


```r
rbind(data.frame(inla.tmarginal(function(x) sqrt(exp(-x)), 
                                fit_sim_cell$internal.marginals.hyperpar$`Log precision for idx_cell`),
                 mod = "full"),
      data.frame(inla.tmarginal(function(x) sqrt(exp(-x)), 
                                fit_sim_cell_dws$internal.marginals.hyperpar$`Log precision for idx_cell`),
                 mod = "dWS"),
      data.frame(inla.tmarginal(function(x) sqrt(exp(-x)), 
                                fit_sim_cell_dR$internal.marginals.hyperpar$`Log precision for idx_cell`),
                 mod = "dR")
) %>%
  ggplot() +
  geom_line(aes(x = x, y = y, color = mod), linewidth = 1.5) +
  geom_vline(xintercept = sigma_cell_sim, linewidth = 1.5) +
  labs(x = 'sigma_cell', 'density') +
  theme(legend.position = c(0.8,0.8)) +
  guides(color = guide_legend(title = NULL))
```

<img src="pictures/sim-it-inla-results-1.png" width="50%" />


# Plots of Repeated Simulations

In the paper, we show results from many repeated simulations, typically as density plots of estimated parameters.
The simulations are carried out using code as in this document, looping over the simulations and keeping the estimated parameters.
Here, we generate some of the plots of the paper.
Code to run the simulations can be found at <https://github.com/nikuehn/MixedModels_Biases/tree/main/r>.

## Simulations with Homoscedastic Standard Deviations


```r
# Results for simulations based on CB14 data
load(file = file.path('./Git/MixedModels_Biases/', 'results', 'results_sim1_CB.Rdata'))

# simulation
tau_sim <- 0.4
phi_s2s_sim <- 0.43
phi_ss_sim <- 0.5

# look at values of standard deviations estimated from lmer and Stan (using mean and median of posterior)
df <- data.frame(res_val) |>
  set_names(c('phis2s_lmer_mode','tau_lmer_mode','phiss_lmer_mode',
              'phis2s_stan_mean','tau_stan_mean','phiss_stan_mean',
              'phis2s_stan_median','tau_stan_median','phiss_stan_median'))
knitr::kable(head(df), digits = 5, row.names = TRUE,
             caption = "Comparison of standard deviation estimates.")
```



Table: Comparison of standard deviation estimates.

|   | phis2s_lmer_mode| tau_lmer_mode| phiss_lmer_mode| phis2s_stan_mean| tau_stan_mean| phiss_stan_mean| phis2s_stan_median| tau_stan_median| phiss_stan_median|
|:--|----------------:|-------------:|---------------:|----------------:|-------------:|---------------:|------------------:|---------------:|-----------------:|
|1  |          0.44047|       0.41113|         0.49731|          0.44057|       0.41290|         0.49736|            0.44043|         0.41206|           0.49739|
|2  |          0.43806|       0.38546|         0.50110|          0.43856|       0.38625|         0.50106|            0.43880|         0.38556|           0.50103|
|3  |          0.41809|       0.39821|         0.49917|          0.41878|       0.40063|         0.49908|            0.41830|         0.40080|           0.49911|
|4  |          0.44819|       0.39679|         0.49623|          0.44831|       0.39811|         0.49633|            0.44800|         0.39716|           0.49623|
|5  |          0.42661|       0.37745|         0.49996|          0.42682|       0.37868|         0.49990|            0.42693|         0.37818|           0.49985|
|6  |          0.43857|       0.42567|         0.50033|          0.43869|       0.42727|         0.50032|            0.43828|         0.42638|           0.50028|



```r
p1 <- data.frame(res_val[,c(1,4,7)], res_sd[,c(1,4)]) %>%
  set_names(c('lmer_max', 'stan_mean','stan_median','lmer_sd(dS)','stan_sd(dS)')) %>%
  pivot_longer(everything(), names_sep = '_', names_to = c('model','type')) %>%
  ggplot() +
  geom_density(aes(x = value, color = model, linetype = type), linewidth = 1.5, key_glyph = draw_key_path) +
  labs(x = expression(paste(widehat(phi)[S2S]))) +
  guides(color = guide_legend(title = NULL), linetype = guide_legend(title = NULL)) +
  theme(legend.position = 'none',
        legend.key.width = unit(2,'cm')) +
  geom_vline(xintercept = phi_s2s_sim, linewidth = 1.5) +
  scale_linetype_manual(values = c(1,2,3,4),
                        labels = c('max','mean','median',TeX("sd($\\delta S$)"))) +
  scale_color_manual(values = c('red','blue'))

p2 <- data.frame(res_val[,c(2,5,8)], res_sd[,c(2,5)]) %>%
  set_names(c('lmer_max', 'stan_mean','stan_median','lmer_sd(dB)','stan_sd(dB)')) %>%
  pivot_longer(everything(), names_sep = '_', names_to = c('model','type')) %>%
  ggplot() +
  geom_density(aes(x = value, color = model, linetype = type), linewidth = 1.5, key_glyph = draw_key_path) +
  labs(x = expression(widehat(tau))) +
  guides(color = guide_legend(title = NULL), linetype = guide_legend(title = NULL)) +
  theme(legend.position = c(0.3,0.65),
        legend.key.width = unit(1.5,'cm')) +
  geom_vline(xintercept = tau_sim, linewidth = 1.5) +
  scale_linetype_manual(values = c(1,2,3,4),
                        labels = c('max','mean','median',TeX("sd($\\delta B$)"))) +
  scale_color_manual(values = c('red','blue'))

p3 <- data.frame(res_val[,c(3,6,9)], res_sd[,c(3,6)]) %>%
  set_names(c('lmer_max', 'stan_mean','stan_median','lmer_sd(dWS)','stan_sd(dWS)')) %>%
  pivot_longer(everything(), names_sep = '_', names_to = c('model','type')) %>%
  ggplot() +
  geom_density(aes(x = value, color = model, linetype = type), linewidth = 1.5, key_glyph = draw_key_path) +
  labs(x = expression(paste(widehat(phi)[SS]))) +
  guides(color = guide_legend(title = NULL), linetype = guide_legend(title = NULL)) +
  theme(legend.position = 'none',
        legend.key.width = unit(2,'cm')) +
  geom_vline(xintercept = phi_ss_sim, linewidth = 1.5) +
  scale_linetype_manual(values = c(1,2,3,4),
                        labels = c('max','mean','median',TeX("sd($\\delta WS$)"))) +
  scale_color_manual(values = c('red','blue'))

leg <- ggpubr::get_legend(p2)
patchwork::wrap_plots(p1,p2 + theme(legend.position = 'none'),p3,ggpubr::as_ggplot(leg), ncol = 2)
```

<img src="pictures/res-sim1-all-plots-1.png" width="100%" />


## Heteroscedastic Standard Deviations


```r
# Results for simulations based on CB14 data
load(file = file.path('./Git/MixedModels_Biases/', 'results', 'results_sim2_heteroscedastic_coeff_CB.Rdata'))
load(file = file.path('./Git/MixedModels_Biases/', 'results', 'results_sim2_heteroscedastic_coeff_stan2_CB.Rdata'))

coeffs <- c(3.421046409, 0.193954090, -0.021982777, 0.287149291, -1.405635476, -0.002911264, -0.394575970)
names_coeffs <- c("intercept", "M1", "M2", "MlogR", "logR", "R", "logVS")

phi_s2s_sim <- 0.43
tau_sim_val <- c(0.4,0.25)
phi_sim_val <- c(0.55,0.4)
mb_tau <- c(5,6)
mb_phi <- c(4.5,5.5)
```


```r
p1 <- data.frame(res_phi, res_phi_stan[,c(1,2)], res_phi_stan2[,c(1,2)]) %>%
  set_names(c('sd(dWS)_lowm','sd(dWS)+unc_lowm','sd(dWS)_largem','sd(dWS)+unc_largem',
              'stan_lowm','stan_largem','stanf_lowm','stanf_largem')) %>%
  pivot_longer(everything(), names_to = c('model','mag'),names_sep = '_') %>%
  ggplot() +
  geom_density(aes(x = value, color = model, linetype = mag), linewidth = lw, key_glyph = draw_key_path) +
  geom_vline(xintercept = phi_sim_val[2], linewidth = lw) +
  geom_vline(xintercept = phi_sim_val[1], linetype = 'dashed', linewidth = lw) +
  scale_color_manual(values = c('orange','red','blue','cyan'),
                     labels = c(TeX("sd(\\widehat{\\delta WS})"),
                                TeX("sd(\\widehat{\\delta WS} + unc)"),
                                TeX('stan ($\\delta R$)'), 'stan (full)')) +
  scale_linetype_manual(values = c(1,2),
                        labels = c(TeX(sprintf("$M \\geq %.1f$",mb_phi[2])), 
                                   TeX(sprintf("$M \\leq %.1f$",mb_phi[1])))) +
  guides(color = guide_legend(title = NULL), linetype = guide_legend(title = NULL)) +
  theme(legend.position = c(0.5,0.8),
        legend.key.width = unit(2,'cm')) +
  labs(x = expression(paste(widehat(phi)[SS]))) +
  lims(y = c(0,100))

p2 <- data.frame(res_tau, res_tau_stan[,c(1,2)], res_tau_stan2[,c(1,2)]) %>%
  set_names(c('sd(dB)_lowm','sd(dB)+unc_lowm','sd(dB)_largem','sd(dB)+unc_largem',
              'stan_lowm','stan_largem','stanf_lowm','stanf_largem')) %>%
  pivot_longer(everything(), names_to = c('model','mag'),names_sep = '_') %>%
  ggplot() +
  geom_density(aes(x = value, color = model, linetype = mag), linewidth = lw, key_glyph = draw_key_path) +
  geom_vline(xintercept = tau_sim_val[2], linewidth = lw) +
  geom_vline(xintercept = tau_sim_val[1], linetype = 'dashed', linewidth = lw) +
  scale_color_manual(values = c('orange','red','blue','cyan'),
                     labels = c(TeX("sd(\\widehat{\\delta B})"),
                                TeX("sd(\\widehat{\\delta B} + unc)"),
                                TeX('stan ($\\delta R$)'), 'stan (full)')) +
  scale_linetype_manual(values = c(1,2),
                        labels = c(TeX(sprintf("$M \\geq %.1f$",mb_tau[2])), 
                                   TeX(sprintf("$M \\leq %.1f$",mb_tau[1])))) +
  guides(color = guide_legend(title = NULL), linetype = guide_legend(title = NULL)) +
  theme(legend.position = 'none') +
  labs(x = expression(paste(widehat(tau))))

p3 <- data.frame(res_phis2s[,c(1,3)], res_phis2s_stan2[,1]) %>%
  set_names(c('lmer','stan','stanf')) %>%
  pivot_longer(everything()) %>%
  ggplot() +
  geom_density(aes(x = value, color = name), linewidth = lw, key_glyph = draw_key_path) +
  geom_vline(xintercept = phi_s2s_sim, linewidth = lw) +
  scale_color_manual(values = c('red','blue','cyan'),
                     labels = c('lmer', TeX('stan ($\\delta R$)'), 'stan (full')) +
  guides(color = guide_legend(title = NULL)) +
  theme(legend.position = 'none') +
  labs(x = expression(paste(widehat(phi)[S2S])))

leg <- ggpubr::get_legend(p1)
patchwork::wrap_plots(p1 + theme(legend.position = 'none'),p2,p3,ggpubr::as_ggplot(leg), ncol = 2)
```

<img src="pictures/res-sim2-hs-all-plots-1.png" width="100%" />


```r
df1 <- data.frame(res_coeffs) %>% set_names(names_coeffs)
df1$model <- 'lmer'

df2 <- data.frame(res_coeffs_stan2) %>% set_names(names_coeffs)
df2$model <- 'stan'

df <- data.frame(name = names_coeffs,
                true = coeffs)

rbind(df1 %>% pivot_longer(!model),
      df2 %>% pivot_longer(!model)) %>%
  ggplot() +
  geom_density(aes(x = value, color = model), linewidth = 1.5, key_glyph = draw_key_path) +
  facet_wrap(vars(name), scales = "free") +
  geom_vline(aes(xintercept = true), data = df, linewidth = 1.5) +
  guides(color = guide_legend(title = NULL)) +
  labs(x = '') +
  theme(legend.position = c(0.8,0.2),
        strip.text = element_text(size = 20))
```

<img src="pictures/res-sim2-hs-all-plots-coeffs-1.png" width="100%" />

## $V_{S30}$-Scaling from Site Terms

Here, we show results of estimating the coefficient for the $V_{S30}$-scaling from the full model, site terms, and a mixed-effects regression of total residuals.
We performed the repeated simulations for both the CB141 data and the Ialian data, and below we show results for both.
We see very similar results, with a larger bias for the estimaton from well-recorded stations (more than 9 records) for the Italian data.
There are less such stations in the Italian data set, which results in this larger bias.
This is a reminder that the size of the biases that can occur depends on the data set.


```r
set1 <- RColorBrewer::brewer.pal(7, "Set1")
coeff_vs <- -0.394575970

# Results for simulations based on CB14 data
load(file = file.path('./Git/MixedModels_Biases/', 'results', 'res_vs_ita18_CB.Rdata'))
xlab <- expression(paste(c[vs]))
names <- c('full','dS','dS(N>10)','dR')
df <- data.frame(res_val[,c(1,5,10,15)]) %>% set_names(names) %>%
  pivot_longer(everything())
df$name <- factor(df$name, names)

p1 <- ggplot(df) +
  geom_density(aes(x = value, color = name), linewidth = 1.5, key_glyph = draw_key_path) +
  geom_vline(xintercept = coeff_vs, linewidth = 1.5) +
  scale_color_manual(values=set1,
                     labels = c('full',TeX('$\\widehat{\\delta S}$'),
                                TeX('$\\widehat{\\delta S}$ ($N \\geq 10$)'),
                                TeX('$\\widehat{\\delta R}$'))) +
  guides(color = guide_legend(title=NULL)) +
  labs(x = xlab, title = 'CB14 Data') +
  theme(legend.position = c(0.2,0.8))


# Results for simulations based on Italian
load(file = file.path('./Git/MixedModels_Biases/', 'results', 'res_vs_ita18_italy.Rdata'))
xlab <- expression(paste(c[vs]))
names <- c('full','dS','dS(N>10)','dR')
df <- data.frame(res_val[,c(1,5,10,15)]) %>% set_names(names) %>%
  pivot_longer(everything())
df$name <- factor(df$name, names)

p2 <- ggplot(df) +
  geom_density(aes(x = value, color = name), linewidth = 1.5, key_glyph = draw_key_path) +
  geom_vline(xintercept = coeff_vs, linewidth = 1.5) +
  scale_color_manual(values=set1,
                     labels = c('full',TeX('$\\widehat{\\delta S}$'),
                                TeX('$\\widehat{\\delta S}$ ($N \\geq 10$)'),
                                TeX('$\\widehat{\\delta R}$'))) +
  guides(color = guide_legend(title=NULL)) +
  labs(x = xlab, title = 'Italian Data') +
  theme(legend.position = c(0.2,0.8))

patchwork::wrap_plots(p1, p2)
```

<img src="pictures/plot-vs-1.png" width="100%" />

## Correlations of Random Effects

```r
tau_sim1 <- 0.4
phi_s2s_sim1 <- 0.43
phi_ss_sim1 <- 0.5

tau_sim2 <- 0.45
phi_s2s_sim2 <- 0.4
phi_ss_sim2 <- 0.55

sigma_tot1 <- sqrt(tau_sim1^2 + phi_s2s_sim1^2 + phi_ss_sim1^2)
sigma_tot2 <- sqrt(tau_sim2^2 + phi_s2s_sim2^2 + phi_ss_sim2^2)
```


```r
load(file.path('./Git/MixedModels_Biases/', 'results',
                                        'res_corrre_CB14_high.Rdata'))

rho_tau <- 0.95
rho_ss <- 0.9
rho_s2s <- 0.85
rho_total <- (rho_tau * tau_sim1 * tau_sim2 +
                rho_s2s * phi_s2s_sim1 * phi_s2s_sim2 +
                rho_ss * phi_ss_sim1 * phi_ss_sim2) /
  (sigma_tot1 * sigma_tot2)

patchwork::wrap_plots(
  data.frame(mat_cor[,c(1,6)]) %>%
    pivot_longer(everything()) %>%
    ggplot() +
    geom_density(data.frame(x = mat_cor_sample[,1]),
                 mapping = aes(x = x), color = 'gray', linewidth = lw) +
    geom_density(aes(x = value, color = name), linewidth = lw, key_glyph = draw_key_path) +
    geom_vline(xintercept = rho_s2s, linewidth = lw) +
    labs(x = TeX('$\\rho(\\delta S_1, \\delta S_2)$')) +
    scale_color_manual(values = c('red','blue'),
                       labels = c(TeX('$\\rho_{sample}$'),
                                  TeX('$cov_{sample}/(\\hat{\\sigma}_{1} \\; \\hat{\\sigma}_{2})$'))) +
    guides(color = guide_legend(title = NULL)) +
    theme(legend.position = c(0.4,0.8),
          legend.key.width = unit(2,'cm')) +
    #lims(x = c(0.25,0.6)),
    lims(x = c(0.45,0.95)),
  
  data.frame(mat_cor[,c(2,7)]) %>%
    pivot_longer(everything()) %>%
    ggplot() +
    geom_density(data.frame(x = mat_cor_sample[,2]),
                 mapping = aes(x = x), color = 'gray', linewidth = lw) +
    geom_density(aes(x = value, color = name), linewidth = lw, key_glyph = draw_key_path) +
    geom_vline(xintercept = rho_tau, linewidth = lw) +
    labs(x = TeX('$\\rho(\\delta B_1, \\delta B_2)$')) +
    scale_color_manual(values = c('red','blue'),
                       labels = c(TeX('$\\rho_{sample}$'),'re2')) +
    guides(color = guide_legend(title = NULL)) +
    theme(legend.position = 'none',
          legend.key.width = unit(2,'cm')) +
    #lims(x = c(0.3,0.6)),
    lims(x = c(0.8,1.0)),
  
  data.frame(mat_cor[,c(3,8)]) %>%
    pivot_longer(everything()) %>%
    ggplot() +
    geom_density(data.frame(x = mat_cor_sample[,3]),
                 mapping = aes(x = x), color = 'gray', linewidth = lw) +
    geom_density(aes(x = value, color = name), linewidth = lw, key_glyph = draw_key_path) +
    geom_vline(xintercept = rho_ss, linewidth = lw) +
    labs(x = TeX('$\\rho(\\delta WS_1, \\delta WS_2)$')) +
    scale_color_manual(values = c('red','blue'),
                       labels = c('re','re2')) +
    guides(color = guide_legend(title = NULL)) +
    theme(legend.position = 'none',
          legend.key.width = unit(2,'cm')) +
    #lims(x = c(0.4,0.55)),
    lims(x = c(0.8,0.95)),
  
  data.frame(mat_cor[,c(4,5,9)]) %>%
    pivot_longer(everything()) %>%
    ggplot() +
    geom_density(aes(x = value, color = name), linewidth = lw, key_glyph = draw_key_path) +
    geom_vline(xintercept = rho_total, linewidth = lw) +
    labs(x = TeX('$\\rho_{total}$')) +
    scale_color_manual(values = c('red','blue','cyan'),
                       labels = c("(1)","(2)",'(3)')) +
    guides(color = guide_legend(title = NULL)) +
    theme(legend.position = c(0.2,0.8),
          legend.key.width = unit(2,'cm')) +
    #lims(x = c(0.45,0.55))
    lims(x = c(0.875,0.925))
)
```

<img src="pictures/res-sim6-corrre-high-1.png" width="100%" />



```r
load(file.path('./Git/MixedModels_Biases/', 'results',
                                        'res_corrre_CB14_low.Rdata'))

rho_tau <- 0.45
rho_ss <- 0.5
rho_s2s <- 0.55
rho_total <- (rho_tau * tau_sim1 * tau_sim2 +
                rho_s2s * phi_s2s_sim1 * phi_s2s_sim2 +
                rho_ss * phi_ss_sim1 * phi_ss_sim2) /
  (sigma_tot1 * sigma_tot2)

patchwork::wrap_plots(
  data.frame(mat_cor[,c(1,6)]) %>%
    pivot_longer(everything()) %>%
    ggplot() +
    geom_density(data.frame(x = mat_cor_sample[,1]),
                 mapping = aes(x = x), color = 'gray', linewidth = lw) +
    geom_density(aes(x = value, color = name), linewidth = lw, key_glyph = draw_key_path) +
    geom_vline(xintercept = rho_s2s, linewidth = lw) +
    labs(x = TeX('$\\rho(\\delta S_1, \\delta S_2)$')) +
    scale_color_manual(values = c('red','blue'),
                       labels = c(TeX('$\\rho_{sample}$'),
                                  TeX('$cov_{sample}/(\\hat{\\sigma}_{1} \\; \\hat{\\sigma}_{2})$'))) +
    guides(color = guide_legend(title = NULL)) +
    theme(legend.position = c(0.4,0.8),
          legend.key.width = unit(2,'cm')) +
    lims(x = c(0.25,0.6)),
  
  data.frame(mat_cor[,c(2,7)]) %>%
    pivot_longer(everything()) %>%
    ggplot() +
    geom_density(data.frame(x = mat_cor_sample[,2]),
                 mapping = aes(x = x), color = 'gray', linewidth = lw) +
    geom_density(aes(x = value, color = name), linewidth = lw, key_glyph = draw_key_path) +
    geom_vline(xintercept = rho_tau, linewidth = lw) +
    labs(x = TeX('$\\rho(\\delta B_1, \\delta B_2)$')) +
    scale_color_manual(values = c('red','blue'),
                       labels = c(TeX('$\\rho_{sample}$'),'re2')) +
    guides(color = guide_legend(title = NULL)) +
    theme(legend.position = 'none',
          legend.key.width = unit(2,'cm')) +
    lims(x = c(0.3,0.6)),
  
  data.frame(mat_cor[,c(3,8)]) %>%
    pivot_longer(everything()) %>%
    ggplot() +
    geom_density(data.frame(x = mat_cor_sample[,3]),
                 mapping = aes(x = x), color = 'gray', linewidth = lw) +
    geom_density(aes(x = value, color = name), linewidth = lw, key_glyph = draw_key_path) +
    geom_vline(xintercept = rho_ss, linewidth = lw) +
    labs(x = TeX('$\\rho(\\delta WS_1, \\delta WS_2)$')) +
    scale_color_manual(values = c('red','blue'),
                       labels = c('re','re2')) +
    guides(color = guide_legend(title = NULL)) +
    theme(legend.position = 'none',
          legend.key.width = unit(2,'cm')) +
    lims(x = c(0.4,0.55)),
  
  data.frame(mat_cor[,c(4,5,9)]) %>%
    pivot_longer(everything()) %>%
    ggplot() +
    geom_density(aes(x = value, color = name), linewidth = lw, key_glyph = draw_key_path) +
    geom_vline(xintercept = rho_total, linewidth = lw) +
    labs(x = TeX('$\\rho_{total}$')) +
    scale_color_manual(values = c('red','blue','cyan'),
                       labels = c("(1)","(2)",'(3)')) +
    guides(color = guide_legend(title = NULL)) +
    theme(legend.position = c(0.2,0.8),
          legend.key.width = unit(2,'cm')) +
    lims(x = c(0.45,0.55))
)
```

```
## Warning: Removed 2 rows containing non-finite values (`stat_density()`).
```

```
## Warning: Removed 3 rows containing non-finite values (`stat_density()`).
```

<img src="pictures/res-sim6-corrre-low-1.png" width="100%" />

## Correlations with Stress Drop


```r
# Results for simulations based on CB14 data
df_res_cor <- read.csv(file.path('./Git/MixedModels_Biases/', 'results',
                                        'res_sim_cor_CB_N50.csv'))

tau_sim <- 0.4
phi_s2s_sim <- 0.43
phi_ss_sim <- 0.5

rho <- 0.7
tau2 <- 1

df_res_cor %>% pivot_longer(c(cor_sim, cor_lme, cor_mean)) %>%
  ggplot() +
  geom_density(aes(x = value, color = name), linewidth = lw, key_glyph = draw_key_path) +
  geom_vline(aes(xintercept = rho), linewidth = lw) +
  scale_color_manual(values=c("blue", 'red', 'gray'),
                     labels=c("2-step lmer", "1-step stan", "sim")
  ) +
  guides(color = guide_legend(title=NULL)) +
  theme(legend.position = c(0.18,0.89)) +
  labs(x = expression(widehat(rho)))
```

<img src="pictures/res-sim4-corr-all-1.png" width="50%" />


## Spatial Correlations of Site Terms


```r
seed <- 1701
load(file = file.path('./Git/MixedModels_Biases/', 'results', sprintf('res_spatial_ita18_italy_seed%d.Rdata', seed)))
load(file = file.path('./Git/MixedModels_Biases/', 'results', sprintf('res_spatial_ita18b_italy_seed%d.Rdata', seed)))


range <- 30
wvar <- 0.65
tau_sim <- 0.17
phi_s2s_sim <- 0.23
phi_s2s_0 <- sqrt((1 - wvar) * phi_s2s_sim^2)
phi_s2s_c <- sqrt(wvar * phi_s2s_sim^2)
phi_ss_sim <- 0.2

p1 <- data.frame(res_spatial[,c(4,7)],res_spatial_tot[,4]) %>% set_names('m1','m3','m2') %>%
  pivot_longer(everything()) %>%
  ggplot() +
  geom_density(aes(x = value, color = name), linewidth = lw, key_glyph = draw_key_path) +
  geom_vline(aes(xintercept = range), linewidth = lw) +
  labs(x = 'spatial range (km)') +
  theme(legend.position = c(0.85,0.85)) +
  guides(color = guide_legend(title=NULL)) +
  scale_color_manual(values = c('blue','red','orange'),
                     labels = c('full',
                                TeX("$\\widehat{\\delta R}$"),
                                TeX("$\\widehat{\\delta S}$")))

p2 <- data.frame(m1 = res_spatial[,5]^2 / (1/res_spatial[,3] + res_spatial[,5]^2),
                 m1 = res_spatial_tot[,5]^2 / (1/res_spatial_tot[,3] + res_spatial_tot[,5]^2),
                 m3 = res_spatial[,8]^2 / (1/res_spatial[,6] + res_spatial[,8]^2)) %>%
  pivot_longer(everything()) %>%
  ggplot() +
  geom_density(aes(x = value, color = name), linewidth = 1.5, key_glyph = draw_key_path) +
  geom_vline(aes(xintercept = wvar), linewidth = 1.5) +
  labs(x = TeX("$\\hat{\\phi}_{S2S,c}^2 / \\hat{phi}_{S2S}^2$")) +
  theme(legend.position = 'none') +
  guides(color = guide_legend(title=NULL)) +
  scale_color_manual(values = c('blue','red','orange'),
                     labels = c('full',
                                TeX("$\\widehat{\\delta R}$"),
                                TeX("$\\widehat{\\delta S}$")))

p3 <- data.frame(res_spatial[,c(5,8)],res_spatial_tot[,5]) %>% set_names('m1','m3','m2') %>%
  pivot_longer(everything()) %>%
  ggplot() +
  geom_density(aes(x = value, color = name), linewidth = lw, key_glyph = draw_key_path) +
  geom_vline(aes(xintercept = phi_s2s_c), linewidth = lw) +
  labs(x = TeX("$\\hat{\\phi}_{S2S,c}$")) +
  theme(legend.position = 'none') +
  guides(color = guide_legend(title=NULL)) +
  scale_color_manual(values = c('blue','red','orange'),
                     labels = c('full',
                                TeX("$\\widehat{\\delta R}$"),
                                TeX("$\\widehat{\\delta S}$")))

p4 <- data.frame(1/sqrt(res_spatial[,c(3,6)]),1/sqrt(res_spatial_tot[,3])) %>% set_names('m1','m3','m2') %>%
  pivot_longer(everything()) %>%
  ggplot() +
  geom_density(aes(x = value, color = name), linewidth = lw, key_glyph = draw_key_path) +
  geom_vline(aes(xintercept = phi_s2s_0), linewidth = lw) +
  labs(x = TeX("$\\hat{\\phi}_{S2S,0}$")) +
  theme(legend.position = 'none') +
  guides(color = guide_legend(title=NULL)) +
  scale_color_manual(values = c('blue','red','orange'),
                     labels = c('full',
                                TeX("$\\widehat{\\delta R}$"),
                                TeX("$\\widehat{\\delta S}$")))

patchwork::wrap_plots(p1,p2,p3,p4, ncol = 2)
```

<img src="pictures/res-sim5-spatial-all-1.png" width="100%" />

## Cell-Specific Attenuation


```r
tau <- 0.17
phi_s2s <- 0.2
phi_0 <- 0.18
sigma_cell <- 0.35
coeffs <- c(3.421046409, 0.193954090, -0.021982777, 0.287149291, -1.405635476,
            -0.002911264, 0.085983743, 0.010500239, -0.394575970)

seed <- 5618
load(file = file.path('./Git/MixedModels_Biases/', 'results',
                      sprintf('res_cell_italy_ita18_seed%d.Rdata', seed)))

p1 <- data.frame(res_cell[,c(4,10,6)]) %>% set_names(c('full','dR','dWS')) %>%
  pivot_longer(everything()) %>%
  ggplot() +
  geom_density(aes(x = value, color = name), linewidth = 1.5, key_glyph = draw_key_path) +
  geom_vline(aes(xintercept = sigma_cell), linewidth = 1.5) +
  labs(x = expression(paste(sigma[cell]))) +
  theme(legend.position = c(0.5,0.85)) +
  guides(color = guide_legend(title=NULL)) +
  scale_color_manual(values = c('red','orange','blue'),
                     labels = c(TeX("$\\delta R$"), TeX("$\\widehat{\\delta WS}$"), "full"))

p2 <- data.frame(res_cell_fix[,c(1,7,4)]) %>% set_names(c('full','dR','dWS')) %>%
  pivot_longer(everything()) %>%
  ggplot() +
  geom_density(aes(x = value, color = name), linewidth = lw, key_glyph = draw_key_path) +
  geom_vline(aes(xintercept = coeffs[6]), linewidth = lw) +
  labs(x = expression(paste(c[attn]))) +
  theme(legend.position = 'none') +
  guides(color = guide_legend(title=NULL)) +
  scale_color_manual(values = c('red','orange','blue'),
                     labels = c(TeX("$\\delta R$"), TeX("$\\widehat{\\delta WS}$"), "full"))

p3 <- data.frame(full = res_cell[,1]^2/res_cell_sdlme[,3]^2,
                 dR = res_cell[,7]^2/res_cell_sdlme[,3]^2,
                 dS = res_cell[,5]^2/res_cell_sdlme[,3]^2) %>%
  pivot_longer(everything()) %>%
  ggplot() +
  geom_density(aes(x = value, color = name), linewidth = 1.5) +
  labs(x = TeX("$\\hat{\\phi}_{SS,0}^2 / \\hat{\\phi}_{SS}^2$")) +
  theme(legend.position = 'none') +
  guides(color = guide_legend(title=NULL)) +
  scale_color_manual(values = c('red','orange','blue'),
                     labels = c(TeX("$\\delta R$"), TeX("$\\widehat{\\delta WS}$"), "full"))

p4 <- data.frame(res_cell_cor[,c(4,6,5)]) %>%
  set_names(c('full','dR','dWS')) %>%
  pivot_longer(everything()) %>%
  ggplot() +
  geom_density(aes(x = value, color = name), linewidth = 1.5) +
  labs(x = expression(paste(rho,'(c'[true],',c'[est],')'))) +
  theme(legend.position = 'none') +
  guides(color = guide_legend(title=NULL)) +
  scale_color_manual(values = c('red','orange','blue'),
                     labels = c(TeX("$\\delta R$"), TeX("$\\widehat{\\delta WS}$"), "full"))

patchwork::wrap_plots(p1,p2,p3,p4, ncol = 2)
```

<img src="pictures/res-sim6-cell-all-1.png" width="100%" />

# References
