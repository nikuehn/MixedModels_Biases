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

}

parameters {
  real ic;
  
  real<lower=0> tau_1;
  real<upper=tau_1> tau_2;
  real<lower=0> tau_scale;
  real<lower=4> mb_tau;

  real<lower=0> phi_ss_1;
  real<upper=phi_ss_1> phi_ss_2;
  real<lower=0> phi_ss_scale;
  real<lower=4> mb_phi_ss;

  real<lower=0> phi_s2s;

  vector[NEQ] eqterm;
  vector[NSTAT] statterm;
}

transformed parameters {
}


model {
  ic ~ normal(0,0.1);

  phi_s2s ~ normal(0,0.5); 

  tau_1 ~ normal(0,0.5); 
  tau_2 ~ normal(0,0.5);
  mb_tau ~ normal(5.,1);
  tau_scale ~ normal(6,1);

  phi_ss_1 ~ normal(0,0.5); 
  phi_ss_2 ~ normal(0,0.5);
  mb_phi_ss ~ normal(5.,1);
  phi_ss_scale ~ normal(6,1);

  vector[NEQ] tau = tau_1 - tau_2 * inv_logit(tau_scale * (MEQ - mb_tau));
  vector[N] phi_ss = phi_ss_1 - phi_ss_2 * inv_logit(phi_ss_scale * (MEQ[eq] - mb_phi_ss));

  eqterm ~ normal(0, tau);
  statterm ~ normal(0, phi_s2s);

  Y ~ normal(ic + eqterm[eq] + statterm[stat], phi_ss);

}
