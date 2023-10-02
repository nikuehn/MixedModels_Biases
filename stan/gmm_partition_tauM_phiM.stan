/*********************************************
 ********************************************/

data {
  int N;  // number of records
  int NEQ;  // number of earthquakes
  int NSTAT;  // number of stations

  vector[NEQ] MEQ;
  vector[N] Y; // ln ground-motion value

  array[N] int<lower=1,upper=NEQ> eq; // event id (in numerical order from 1 to last)
  array[N] int<lower=1,upper=NSTAT> stat; // station id (in numerical order from 1 to last)

  vector[NEQ] M1_eq;
  vector[NEQ] M2_eq;
  vector[N] M1_rec;
  vector[N] M2_rec;
}

parameters {
  real ic;
  
  real<lower=0> tau_1;
  real<lower=0> tau_2;

  real<lower=0> phi_ss_1;
  real<lower=0> phi_ss_2;

  real<lower=0> phi_s2s;

  vector[NEQ] eqterm;
  vector[NSTAT] statterm;
}


model {
  ic ~ normal(0,0.1);

  phi_s2s ~ normal(0,0.5); 

  tau_1 ~ normal(0,0.5); 
  tau_2 ~ normal(0,0.5);

  phi_ss_1 ~ normal(0,0.5); 
  phi_ss_2 ~ normal(0,0.5);

  vector[N] phi_ss = M1_rec * phi_ss_1 + M2_rec * phi_ss_2;
  vector[NEQ] tau = M1_eq * tau_1 + M2_eq * tau_2;

  eqterm ~ normal(0, tau);
  statterm ~ normal(0, phi_s2s);

  Y ~ normal(ic + eqterm[eq] + statterm[stat], phi_ss);
}

