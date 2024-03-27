/*********************************************
 ********************************************/

data {
  int N;  // number of records
  int NEQ;  // number of earthquakes
  int NSTAT;  // number of stations
  int NP; // number of periods

  array[N] vector[NP] Y;

  array[N] int<lower=1,upper=NEQ> eq; // event id (in numerical order from 1 to last)
  array[N] int<lower=1,upper=NSTAT> stat; // station id (in numerical order from 1 to last)


}

transformed data {
  vector[NP] zerovec = rep_vector(0.0, NP);
}

parameters {
  vector[NP] ic;

  vector<lower=0>[NP] phi_ss;
  vector<lower=0>[NP] tau;
  vector<lower=0>[NP] phi_s2s;

  array[NEQ] vector[NP] eqterm;
  array[NSTAT] vector[NP] statterm;

  cholesky_factor_corr[NP] L_eq;
  cholesky_factor_corr[NP] L_stat;
  cholesky_factor_corr[NP] L_rec;
}

model {
  ic ~ normal(0, 0.1);

  phi_ss ~ normal(0,0.5);
  tau ~ normal(0,0.5);
  phi_s2s ~ normal(0,0.5);

  eqterm ~ multi_normal_cholesky(zerovec, diag_pre_multiply(tau, L_eq));
  statterm ~ multi_normal_cholesky(zerovec, diag_pre_multiply(phi_s2s, L_stat));

  array[N] vector[NP] mu;
  for(i in 1:N)
    mu[i] = ic + eqterm[eq[i]] + statterm[stat[i]];

  Y ~ multi_normal_cholesky(mu, diag_pre_multiply(phi_ss, L_rec));
}

generated quantities {
  matrix[NP,NP] C_rec = multiply_lower_tri_self_transpose(L_rec);
  matrix[NP,NP] C_eq = multiply_lower_tri_self_transpose(L_eq);
  matrix[NP,NP] C_stat = multiply_lower_tri_self_transpose(L_stat);
}
