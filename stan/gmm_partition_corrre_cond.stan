/* ***********************************************
 Model that partitions total resduals of multiple target variables
 into even terms, site terms, and singe-site-residuals, 
 and estmates correlations between them.
 
 This model is for two target variables, and the second is
 modeled contional on the first.
 *********************************************** */

data {
  int<lower=1> N;
  int<lower=1> NEQ;
  int<lower=1> NSTAT;

  matrix[N, 2] Y;    // log psa values - 1 is PGA

  array[N] int<lower=1,upper=NEQ> eq;
  array[N] int<lower=1,upper=NSTAT> stat;

}

transformed data {
  int N_target = 2;
  vector[N_target] zerovec = rep_vector(0,N_target);
}

parameters {
  vector[N_target] c0;

  vector<lower=0>[N_target] phi_ss;
  vector<lower=0>[N_target] phi_s2s;
  vector<lower=0>[N_target] tau;

  real<lower=-1,upper=1> rho_rec;
  real<lower=-1,upper=1> rho_eq;
  real<lower=-1,upper=1> rho_stat;

  matrix[NEQ, N_target] deltaB;
  matrix[NSTAT, N_target] deltaS;
}

model {
  // // prior distrributions for standard deviations
  phi_ss ~ normal(0, 0.5);
  phi_s2s ~ normal(0, 0.5);
  tau ~ normal(0, 0.5);

  c0 ~ normal(0,0.5);

  deltaB[:,1] ~ normal(0, tau[1]);
  deltaS[:,1] ~ normal(0, phi_s2s[1]);

  // frst target
  vector[N] mu_1 = c0[1] + deltaB[eq, 1] + deltaS[stat,1];
  Y[:,1]  ~ normal(mu_1, phi_ss[1]);
  vector[N] deltaWS_1 = Y[:,1] - mu_1;

  // second target
  vector[NEQ] mu_deltaB = tau[2] / tau[1] * rho_eq * deltaB[:,1];
  real tau_cond = sqrt((1 - square(rho_eq)) * square(tau[2]));
  deltaB[:,2] ~ normal(mu_deltaB, tau_cond);


  vector[NSTAT] mu_deltaS = phi_s2s[2] / phi_s2s[1] * rho_stat * deltaS[:,1];
  real phi_s2s_cond = sqrt((1 - square(rho_stat)) * square(phi_s2s[2]));
  deltaS[:,2] ~ normal(mu_deltaS, phi_s2s_cond);

  vector[N] mu_2 = c0[2] + deltaB[eq, 2] + deltaS[stat,2] +
                 phi_ss[2] / phi_ss[1] * rho_rec * deltaWS_1;
  real phi_ss_cond = sqrt((1 - square(rho_rec)) * square(phi_ss[2]));

  Y[:,2]  ~ normal(mu_2, phi_ss_cond);
}

