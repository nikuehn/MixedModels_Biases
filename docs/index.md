---
title: "Biases in Mixed-Effects Model GMMs"
author: "Nicolas Kuehn, Ken Campbell, Yousef Bozorgnia"
date: "2023-09-14"
output:
  html_document:
    keep_md: true
    toc: true
    toc_depth: 2
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
In particular, standard devations are underestimated when they are calculated from point estimates of random effects/residuals.
For the simulations, we randomly sample event terms, site terms, and within-event/within-site residuals from their respective distributions, and then perform regessions on the sampled data to see whether we recover get the parameters used in the simulations.
In this document, we generally do a single simuation for different cases, which typically highlight the points we want to made.
For the paper, we repeat these simulations multiple times, since due to the relative small sample size in ground-motion data sets there ca be variability from one sample to the next.

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
\psi(\hat{\vec{u}})^2 = \phi_{SS}^2 \mathbf{\Lambda} \left(\mathbf{\Lambda}^T \mathbf{Z}^T \mathbf{Z} \mathbf{\Lambda} + \mathbf{I} \right)^{-1} \mathbf{\Lambda}
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
## CmdStan path set to: /Users/nico/GROUNDMOTION/SOFTWARE/cmdstan-2.32.2
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
```

# Simlations using CB14 Data

Use Californian data from @Campbell2014.


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
n_rec <- nrow(data_reg)
n_eq <- max(data_reg$eq)
n_stat <- max(data_reg$stat)

eq <- data_reg$eq
stat <- data_reg$stat

mageq <- unique(data_reg[,c('eq','M')])$M # event-specific magnitude
magstat <- unique(data_reg[,c('stat','M_stat')])$M_stat # station-specific magnitude
```

## Homoscedastic Standard Deviations

First, we simulate data using standard deviations that do not depend on any predictor variables, i.e. are homoscedastic.
We do not simulate any fixed effects structure, in this example we focus biases in the estimation of standard deviations.

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
We use maximum likelihood instead of restricted maximum likelihood in this case to show the equivalnce of the calculations of standard deviations.


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

As we can see, the values from `lmer` and calculated according to Equation (4) agree for $\tau$ and $\phi_{S2S}$.
For $\phi_{SS}$, there is a small discrepancy, since the conditional standard deviations are just an approximation.
These values are also close to the true ones, while the standard deviations calculated from the point estimates are underestimating the true values.
The differences is largest for $\phi_{S2S}$, since there are several stations with only few recordings and thus large conditional standard deviations.
Sampling from the conditional distrbution of the random effects/standard deviations leads to values that are close to the true ones.

Since there are many stations with very few recordings, the value of $\phi_{S2S}$ is severel underestimated when calculated from the point estimates of the site terms.
Thus, we now test whether what happesf we only use stations with at least 5 or 10 recordings.
As we canseefrom the hstogram (which shows 200 repeated simulations), on average the values are closer to the true value, but some bias remains.
If one chooses to go thsroute, one also has to account for the fact that the estimates are based on fewer stations.


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
  xlab('phi_S2S')
```

<img src="pictures/sim1-phis2s-1.png" width="50%" />


In GMM development, the standard deviations are often modeled as dependent on some predictor variables such as magnitude.
@Bayless2018 contains a magnitude-dependent $\phi_{S2S}$, which is modeled using the mean magnitude of all records by station.
@Kotha2022 performed a Breusch-Pagan test [@Breusch1979] for heteroscedasticity to test for magnitude dependence of $\tau$ and $\phi_{SS}$.
Below, we calcuale the p-values for the simulated data (which we know is not heterosceastic).
The nul hypothesis is that the data is homscedastc, and a low p-value is the probability of observingdata if the null hypothesis were true.
Based on point estimates, one would conclude that site terms and within-event/within-site residuals are heteroscedastic.
In this context, be aware of hypothesis tests [@Wasserstein2019,@Amrhein2019].


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

For this simulation, we also generate median predictions from fixed effects, in order to checkhow well the coefficients are estimated.

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
We alsocompute the linear predictors for the model.


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
In general, thecoefficients are estimated well, but the standard deviations are off.


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
mod <- cmdstan_model(file.path('./Git/MixedModels_Biases/', 'stan', 'gmm_partition_tauM2_phiM2.stan'))
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
##   vector[2] tau_mb;
##   vector[2] phi_mb;
## 
## }
## 
## transformed data {
##   vector[NEQ] M_tau = (MEQ - tau_mb[1]) / (tau_mb[2] - tau_mb[1]);
##   vector[N] M_phi = (MEQ[eq] - phi_mb[1]) / (phi_mb[2] - phi_mb[1]);
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
##   vector[N] phi_ss;
##   for(i in 1:N) {
##     if(MEQ[eq[i]] <= phi_mb[1]) phi_ss[i] = phi_ss_1;
##     else if(MEQ[eq[i]] >= phi_mb[2]) phi_ss[i] = phi_ss_2;
##     else
##       phi_ss[i] = phi_ss_1 + (phi_ss_2 - phi_ss_1) * M_phi[i];
##   }
## 
##   vector[NEQ] tau;
##   for(i in 1:NEQ) {
##     if(MEQ[i] <= tau_mb[1]) tau[i] = tau_1;
##     else if(MEQ[i] >= tau_mb[2]) tau[i] = tau_2;
##     else
##       tau[i] = tau_1 + (tau_2 - tau_1) * M_tau[i];
##   }
## 
##   eqterm ~ normal(0, tau);
##   statterm ~ normal(0, phi_s2s);
## 
##   Y ~ normal(ic + eqterm[eq] + statterm[stat], phi_ss);
## }
```

Now, we declare the data for Stan, and run the model.
To keep running time low,we only run 200 warm-up and 200sampling iterations.


```r
data_list <- list(
  N = n_rec,
  NEQ = n_eq,
  NSTAT = n_stat,
  Y = as.numeric(dR_lmer),
  eq = eq,
  stat = stat,
  MEQ = mageq,
  tau_mb = mb_tau,
  phi_mb = mb_phi
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
  show_messages = FALSE
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
## Chain 1 finished in 111.1 seconds.
## Chain 3 Iteration:   1 / 400 [  0%]  (Warmup) 
## Chain 2 Iteration: 400 / 400 [100%]  (Sampling) 
## Chain 2 finished in 114.3 seconds.
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
## Chain 4 finished in 126.3 seconds.
## Chain 3 Iteration: 400 / 400 [100%]  (Sampling) 
## Chain 3 finished in 130.5 seconds.
## 
## All 4 chains finished successfully.
## Mean chain execution time: 120.6 seconds.
## Total execution time: 242.0 seconds.
```

```r
print(fit$cmdstan_diagnose())
```

```
## Processing csv files: /var/folders/p3/r7vrsk6n2d15709vgcky_y880000gn/T/RtmpfkuOtN/gmm_partition_tauM2_phiM2-202309261529-1-0e553f.csv, /var/folders/p3/r7vrsk6n2d15709vgcky_y880000gn/T/RtmpfkuOtN/gmm_partition_tauM2_phiM2-202309261529-2-0e553f.csv, /var/folders/p3/r7vrsk6n2d15709vgcky_y880000gn/T/RtmpfkuOtN/gmm_partition_tauM2_phiM2-202309261529-3-0e553f.csv, /var/folders/p3/r7vrsk6n2d15709vgcky_y880000gn/T/RtmpfkuOtN/gmm_partition_tauM2_phiM2-202309261529-4-0e553f.csv
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
## [1] "Processing csv files: /var/folders/p3/r7vrsk6n2d15709vgcky_y880000gn/T/RtmpfkuOtN/gmm_partition_tauM2_phiM2-202309261529-1-0e553f.csv, /var/folders/p3/r7vrsk6n2d15709vgcky_y880000gn/T/RtmpfkuOtN/gmm_partition_tauM2_phiM2-202309261529-2-0e553f.csv, /var/folders/p3/r7vrsk6n2d15709vgcky_y880000gn/T/RtmpfkuOtN/gmm_partition_tauM2_phiM2-202309261529-3-0e553f.csv, /var/folders/p3/r7vrsk6n2d15709vgcky_y880000gn/T/RtmpfkuOtN/gmm_partition_tauM2_phiM2-202309261529-4-0e553f.csv\n\nChecking sampler transitions treedepth.\nTreedepth satisfactory for all transitions.\n\nChecking sampler transitions for divergences.\nNo divergent transitions found.\n\nChecking E-BFMI - sampler transitions HMC potential energy.\nE-BFMI satisfactory.\n\nEffective sample size satisfactory.\n\nSplit R-hat values satisfactory all parameters.\n\nProcessing complete, no problems detected.\n"
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
## [1] 0.8081726 0.8814360 0.8563872 1.0001283
```

```r
draws_part <- fit$draws()

summarise_draws(subset(draws_part, variable = c('ic','phi','tau'), regex = TRUE))
```

```
## # A tibble: 6 × 10
##   variable      mean    median      sd     mad      q5    q95  rhat ess_bulk
##   <chr>        <num>     <num>   <num>   <num>   <num>  <num> <num>    <num>
## 1 ic       -0.000684 -0.000298 0.0266  0.0272  -0.0442 0.0426 1.02      86.7
## 2 phi_ss_1  0.550     0.550    0.00402 0.00408  0.544  0.557  1.01    1014. 
## 3 phi_ss_2  0.395     0.395    0.00837 0.00839  0.382  0.409  1.01     688. 
## 4 phi_s2s   0.433     0.433    0.0110  0.0117   0.416  0.451  1.00     494. 
## 5 tau_1     0.391     0.390    0.0191  0.0197   0.361  0.424  0.998   1397. 
## 6 tau_2     0.319     0.315    0.0563  0.0548   0.238  0.420  1.00     727. 
## # ℹ 1 more variable: ess_tail <num>
```
In general, the parameters are well estimated.
There are not that many events for $M \geq 6$, so the value of $\tau_2$ is quite uncertain.

In the following, we run a Stan model which estimates coefficients and magitude-dependent standard deviations at the same time.
To improve sampling, we use the QR-decomposition of the desgn matrix.


```r
mod <- cmdstan_model(file.path('./Git/MixedModels_Biases/', 'stan', 'gmm_full_qr_tauM2_phiM2.stan'))
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
##   vector[2] tau_mb;
##   vector[2] phi_mb;
## 
## }
## 
## transformed data {
## 
##   matrix[N, K-1] Q_ast = qr_thin_Q(X) * sqrt(N - 1);
##   matrix[K-1, K-1] R_ast = qr_thin_R(X) / sqrt(N - 1);
##   matrix[K-1, K-1] R_ast_inverse = inverse(R_ast);
## 
##   vector[NEQ] M_tau = (MEQ - tau_mb[1]) / (tau_mb[2] - tau_mb[1]);
##   vector[N] M_phi = (MEQ[eq] - phi_mb[1]) / (phi_mb[2] - phi_mb[1]);
## 
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
##   vector[N] phi_ss;
##   for(i in 1:N) {
##     if(MEQ[eq[i]] <= phi_mb[1]) phi_ss[i] = phi_ss_1;
##     else if(MEQ[eq[i]] >= phi_mb[2]) phi_ss[i] = phi_ss_2;
##     else
##       phi_ss[i] = phi_ss_1 + (phi_ss_2 - phi_ss_1) * M_phi[i];
##   }
## 
##   vector[NEQ] tau;
##   for(i in 1:NEQ) {
##     if(MEQ[i] <= tau_mb[1]) tau[i] = tau_1;
##     else if(MEQ[i] >= tau_mb[2]) tau[i] = tau_2;
##     else
##       tau[i] = tau_1 + (tau_2 - tau_1) * M_tau[i];
##   }
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
  tau_mb = mb_tau,
  phi_mb = mb_phi
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
  show_messages = FALSE
)
```

```
## Running MCMC with 4 chains, at most 2 in parallel...
## 
## Chain 2 Iteration:   1 / 400 [  0%]  (Warmup) 
## Chain 1 Iteration:   1 / 400 [  0%]  (Warmup) 
## Chain 2 Iteration: 100 / 400 [ 25%]  (Warmup) 
## Chain 1 Iteration: 100 / 400 [ 25%]  (Warmup) 
## Chain 1 Iteration: 200 / 400 [ 50%]  (Warmup) 
## Chain 1 Iteration: 201 / 400 [ 50%]  (Sampling) 
## Chain 2 Iteration: 200 / 400 [ 50%]  (Warmup) 
## Chain 2 Iteration: 201 / 400 [ 50%]  (Sampling) 
## Chain 1 Iteration: 300 / 400 [ 75%]  (Sampling) 
## Chain 2 Iteration: 300 / 400 [ 75%]  (Sampling) 
## Chain 1 Iteration: 400 / 400 [100%]  (Sampling) 
## Chain 1 finished in 320.4 seconds.
## Chain 3 Iteration:   1 / 400 [  0%]  (Warmup) 
## Chain 2 Iteration: 400 / 400 [100%]  (Sampling) 
## Chain 2 finished in 364.1 seconds.
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
## Chain 3 finished in 357.3 seconds.
## Chain 4 Iteration: 400 / 400 [100%]  (Sampling) 
## Chain 4 finished in 369.0 seconds.
## 
## All 4 chains finished successfully.
## Mean chain execution time: 352.7 seconds.
## Total execution time: 733.8 seconds.
```

```r
print(fit$cmdstan_diagnose())
```

```
## Processing csv files: /var/folders/p3/r7vrsk6n2d15709vgcky_y880000gn/T/RtmpfkuOtN/gmm_full_qr_tauM2_phiM2-202309261534-1-04a623.csv, /var/folders/p3/r7vrsk6n2d15709vgcky_y880000gn/T/RtmpfkuOtN/gmm_full_qr_tauM2_phiM2-202309261534-2-04a623.csv, /var/folders/p3/r7vrsk6n2d15709vgcky_y880000gn/T/RtmpfkuOtN/gmm_full_qr_tauM2_phiM2-202309261534-3-04a623.csv, /var/folders/p3/r7vrsk6n2d15709vgcky_y880000gn/T/RtmpfkuOtN/gmm_full_qr_tauM2_phiM2-202309261534-4-04a623.csv
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
## [1] "Processing csv files: /var/folders/p3/r7vrsk6n2d15709vgcky_y880000gn/T/RtmpfkuOtN/gmm_full_qr_tauM2_phiM2-202309261534-1-04a623.csv, /var/folders/p3/r7vrsk6n2d15709vgcky_y880000gn/T/RtmpfkuOtN/gmm_full_qr_tauM2_phiM2-202309261534-2-04a623.csv, /var/folders/p3/r7vrsk6n2d15709vgcky_y880000gn/T/RtmpfkuOtN/gmm_full_qr_tauM2_phiM2-202309261534-3-04a623.csv, /var/folders/p3/r7vrsk6n2d15709vgcky_y880000gn/T/RtmpfkuOtN/gmm_full_qr_tauM2_phiM2-202309261534-4-04a623.csv\n\nChecking sampler transitions treedepth.\nTreedepth satisfactory for all transitions.\n\nChecking sampler transitions for divergences.\nNo divergent transitions found.\n\nChecking E-BFMI - sampler transitions HMC potential energy.\nE-BFMI satisfactory.\n\nEffective sample size satisfactory.\n\nSplit R-hat values satisfactory all parameters.\n\nProcessing complete, no problems detected.\n"
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
## [1] 0.7742797 0.8354595 0.8414977 0.9012932
```

```r
draws_full <- fit$draws()

summarise_draws(subset(draws_full, variable = c('phi','tau'), regex = TRUE))
```

```
## # A tibble: 5 × 10
##   variable  mean median      sd     mad    q5   q95  rhat ess_bulk ess_tail
##   <chr>    <num>  <num>   <num>   <num> <num> <num> <num>    <num>    <num>
## 1 phi_ss_1 0.550  0.550 0.00411 0.00426 0.544 0.557  1.01     942.     466.
## 2 phi_ss_2 0.396  0.396 0.00805 0.00785 0.382 0.409  1.00     899.     731.
## 3 phi_s2s  0.433  0.433 0.0108  0.0110  0.414 0.449  1.00     633.     722.
## 4 tau_1    0.392  0.390 0.0205  0.0207  0.359 0.428  1.01    1156.     675.
## 5 tau_2    0.317  0.310 0.0563  0.0552  0.237 0.422  1.01     693.     621.
```

```r
summarise_draws(subset(draws_full, variable = c('^c\\['), regex = TRUE))
```

```
## # A tibble: 7 × 10
##   variable     mean   median       sd      mad       q5      q95  rhat ess_bulk
##   <chr>       <num>    <num>    <num>    <num>    <num>    <num> <num>    <num>
## 1 c[1]      3.59     3.59    0.0957   0.0896    3.43     3.74    1.02      65.5
## 2 c[2]      0.268    0.267   0.0537   0.0512    0.177    0.354   1.03      89.1
## 3 c[3]     -0.0581  -0.0588  0.0828   0.0845   -0.195    0.0752  1.03     115. 
## 4 c[4]      0.268    0.267   0.0157   0.0156    0.241    0.294   0.998    729. 
## 5 c[5]     -1.42    -1.42    0.0364   0.0376   -1.48    -1.36    0.998    496. 
## 6 c[6]     -0.00295 -0.00296 0.000158 0.000150 -0.00320 -0.00268 1.00     765. 
## 7 c[7]     -0.337   -0.338   0.0910   0.0914   -0.483   -0.181   1.02      97.4
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

And finally, the posterior dstribution of $\phi_{S2S}$.


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

We can conclude from this smulation (and the repeated ones in the paper) that to the magnitude-dependent standard deviations can be estimated using Stan from total residuals, but one should also account for uncertainty.
Estimating the values from binned random effects/resduals can work but leads to a larger bias.

Our focus is on estmating the magnitude-dependent standard deviations, but as a check we also plot the posterior distrbutions of the coefficients for the full stan model, together with the true values (black) and `lmer` estimates.


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

## Scaling from Point Estimates of Random Effects

Random effects are sometimes used to estimate scaling of ground motions wth respect to new parameters, such as parameters associated with horizontal-to-vertical ratios.
To assess poential biases, we simulate synthetic data using the ITA18 functional form, and then estimate a model without $V_{S30}$-scaling.
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

Now, we fit the model with the `logVS` term, and then use linear regression on the estimated station term to estiamte the coefficient.
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

## Correlations between Random Effects

Here, we briefly show that estimating correlations between random effects/residuals can be well estimated from point estimates.
We simulate correlated terms from a bivariate normal distribution, perform a linear mixed-effects regression on each target variable separael, and then calcualte the correlation.
The correlation coefficient is
$$
\rho(X,Y) = \frac{cov(X,Y)}{\sigma_X \sigma_Y}
$$
We compare calculating $\rho$ using the standard deviations of the point estimates in the denominator, as well as the ML estimate from `lmer`.
The correlations are well estimated by using point estimates, but are underestimated when using the (RE)ML value in the denominator.



```r
tau_sim <- 0.4
phi_s2s_sim <- 0.43
phi_ss_sim <- 0.5

rho <- 0.9
cov_tau <- matrix(c(tau_sim^2, rho * tau_sim * tau_sim,
                    rho * tau_sim * tau_sim, tau_sim^2), ncol = 2)
cov_s2s <- matrix(c(phi_s2s_sim^2, rho * phi_s2s_sim * phi_s2s_sim,
                    rho * phi_s2s_sim * phi_s2s_sim, phi_s2s_sim^2), ncol = 2)
cov_ss <- matrix(c(phi_ss_sim^2, rho * phi_ss_sim * phi_ss_sim,
                    rho * phi_ss_sim * phi_ss_sim, phi_ss_sim^2), ncol = 2)

eqt2 <- mvtnorm::rmvnorm(n_eq, sigma = cov_tau)
statt2 <- mvtnorm::rmvnorm(n_stat, sigma = cov_s2s)
rect2 <- mvtnorm::rmvnorm(n_rec, sigma = cov_ss)

data_reg$y_sim1 <- eqt2[eq,1] + statt2[stat,1] + rect2[,1]
data_reg$y_sim2 <- eqt2[eq,2] + statt2[stat,2] + rect2[,2]

fit_sim1 <- lmer(y_sim1 ~ (1 | eq) + (1 | stat), data_reg)
fit_sim2 <- lmer(y_sim2 ~ (1 | eq) + (1 | stat), data_reg)

dB1 <- ranef(fit_sim1)$eq$`(Intercept)`
dS1 <- ranef(fit_sim1)$stat$`(Intercept)`
dWS1 <-data_reg$y_sim1 - predict(fit_sim1)

dB2 <- ranef(fit_sim2)$eq$`(Intercept)`
dS2 <- ranef(fit_sim2)$stat$`(Intercept)`
dWS2 <-data_reg$y_sim2 - predict(fit_sim2)

sds1 <- as.data.frame(VarCorr(fit_sim1))$sdcor
sds2 <- as.data.frame(VarCorr(fit_sim2))$sdcor


df <- data.frame(dS = c(rho, cor(dS1,dS2), cov(dS1,dS2)/(sd(dS1) * sd(dS2)), cov(dS1,dS2)/(sds1[1] * sds2[1])),
           dB = c(rho, cor(dB1,dB2), cov(dB1,dB2)/(sd(dB1) * sd(dB2)), cov(dB1,dB2)/(sds1[2] * sds2[2])),
           dWS = c(rho, cor(dWS1,dWS2), cov(dWS1,dWS2)/(sd(dWS1) * sd(dWS2)), cov(dWS1,dWS2)/(sds1[3] * sds2[3])),
           row.names = c('true','cor','cov/sd(point estimate)','cov()/hat()'))
knitr::kable(df, digits = 3, row.names = TRUE,
             caption = "Estimated correlation coefficients.")
```



Table: Estimated correlation coefficients.

|                       |    dS|    dB|   dWS|
|:----------------------|-----:|-----:|-----:|
|true                   | 0.900| 0.900| 0.900|
|cor                    | 0.906| 0.921| 0.897|
|cov/sd(point estimate) | 0.906| 0.921| 0.897|
|cov()/hat()            | 0.600| 0.815| 0.807|

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
  show_messages = FALSE
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
## Chain 1 finished in 35.7 seconds.
## Chain 3 Iteration:   1 / 400 [  0%]  (Warmup) 
## Chain 2 Iteration: 400 / 400 [100%]  (Sampling) 
## Chain 2 finished in 36.1 seconds.
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
## Chain 4 finished in 49.1 seconds.
## Chain 3 Iteration: 400 / 400 [100%]  (Sampling) 
## Chain 3 finished in 54.0 seconds.
## 
## All 4 chains finished successfully.
## Mean chain execution time: 43.7 seconds.
## Total execution time: 90.0 seconds.
```

```r
print(fit$cmdstan_diagnose())
```

```
## Processing csv files: /var/folders/p3/r7vrsk6n2d15709vgcky_y880000gn/T/RtmpphFq1o/gmm_partition_wvar_corr-202309280946-1-810a48.csv, /var/folders/p3/r7vrsk6n2d15709vgcky_y880000gn/T/RtmpphFq1o/gmm_partition_wvar_corr-202309280946-2-810a48.csv, /var/folders/p3/r7vrsk6n2d15709vgcky_y880000gn/T/RtmpphFq1o/gmm_partition_wvar_corr-202309280946-3-810a48.csv, /var/folders/p3/r7vrsk6n2d15709vgcky_y880000gn/T/RtmpphFq1o/gmm_partition_wvar_corr-202309280946-4-810a48.csv
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
## [1] "Processing csv files: /var/folders/p3/r7vrsk6n2d15709vgcky_y880000gn/T/RtmpphFq1o/gmm_partition_wvar_corr-202309280946-1-810a48.csv, /var/folders/p3/r7vrsk6n2d15709vgcky_y880000gn/T/RtmpphFq1o/gmm_partition_wvar_corr-202309280946-2-810a48.csv, /var/folders/p3/r7vrsk6n2d15709vgcky_y880000gn/T/RtmpphFq1o/gmm_partition_wvar_corr-202309280946-3-810a48.csv, /var/folders/p3/r7vrsk6n2d15709vgcky_y880000gn/T/RtmpphFq1o/gmm_partition_wvar_corr-202309280946-4-810a48.csv\n\nChecking sampler transitions treedepth.\nTreedepth satisfactory for all transitions.\n\nChecking sampler transitions for divergences.\nNo divergent transitions found.\n\nChecking E-BFMI - sampler transitions HMC potential energy.\nE-BFMI satisfactory.\n\nEffective sample size satisfactory.\n\nSplit R-hat values satisfactory all parameters.\n\nProcessing complete, no problems detected.\n"
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
##   <chr>    <num>  <num>   <num>   <num> <num> <num> <num>    <num>    <num>
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

We simulate data for some spatial models on the Italian data, and we use INLA (<https://www.r-inla.org/>) to estimate the models.
Below, we set the penalize complexity prior [@Simpson2017] for the standard deviations, used throughout.


```r
prior_prec_tau <- list(prec = list(prior = 'pc.prec', param = c(0.3, 0.01)))
prior_prec_phiS2S    <- list(prec = list(prior = 'pc.prec', param = c(0.3, 0.01))) 
prior_prec_phiSS    <- list(prec = list(prior = 'pc.prec', param = c(0.3, 0.01)))
```

## Spatial correlations of Site Terms

In nonergodic models, site terms are often modeled as spatially correlated.
The spatial correlaion structure can be assess from point estimates of the site terms.
Here, we simulate some data with spatally correlated site terms, to check whether we can get the model parameters back.

We use the Mat\'ern covariance function for the spatial correlation of the site terms, which is defined below.


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

To se R-INLA with the ``stochastic partial differential equation'' (SPDE) approach, we need to define a mesh.

```r
max.edge2    <- 5 
bound.outer2 <- 40 
mesh = inla.mesh.2d(loc=co_stat_utm,
                    max.edge = c(1,5)*max.edge2,
                    cutoff = max.edge2,
                    offset = c(5 * max.edge2, bound.outer2))
```

```
## Please note that rgdal will be retired during October 2023,
## plan transition to sf/stars/terra functions using GDAL and PROJ
## at your earliest convenience.
## See https://r-spatial.org/r/2023/05/15/evolution4.html and https://github.com/r-spatial/evolution
## rgdal: version: 1.6-7, (SVN revision 1203)
## Geospatial Data Abstraction Library extensions to R successfully loaded
## Loaded GDAL runtime: GDAL 3.4.2, released 2022/03/08
## Path to GDAL shared files: /Users/nico/Library/R/x86_64/4.2/library/rgdal/gdal
## GDAL binary built with GEOS: FALSE 
## Loaded PROJ runtime: Rel. 8.2.1, January 1st, 2022, [PJ_VERSION: 821]
## Path to PROJ shared files: /Users/nico/Library/R/x86_64/4.2/library/rgdal/proj
## PROJ CDN enabled: FALSE
## Linking to sp version:1.6-1
## To mute warnings of possible GDAL/OSR exportToProj4() degradation,
## use options("rgdal_show_exportToProj4_warnings"="none") before loading sp or rgdal.
```

```r
print(mesh$n)
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

# formua for fit from site terms
form_spatial_stat_u <- y ~ 0 + intercept + f(idx_stat, model = spde_stat)
```


Now, we sample the event terms, ste terms, spatially corrlated site terms, and within-event/within-site residuals, and combine them with the median predictions from ITA18.
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
We fit the full model (fixed and random effects), a model onthe total residuals, and a model on the site terms from the `lmer` fit.


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
The model based on site terms does not lead to good results for the standard deviations, n particular for $\phi_{S2S,c}$, which is severely underestimated.
The relative sizes ofstandard devations based on the fit from $\delta S$ are wrongly estimated.

## Cell-specific atteuation

In this section, we simulate data based on the cell-specific attenuation model [@Kuehn2019,@Dawood2013], and estima the model parameters using the total model, aswell as from within-event/within-site residuals.

Read in the cell-specific distances, and some definitions.


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


# References