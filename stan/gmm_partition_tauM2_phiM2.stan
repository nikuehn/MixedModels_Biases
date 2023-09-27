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

  vector[2] tau_mb;
  vector[2] phi_mb;

}

transformed data {
  vector[NEQ] M_tau = (MEQ - tau_mb[1]) / (tau_mb[2] - tau_mb[1]);
  vector[N] M_phi = (MEQ[eq] - phi_mb[1]) / (phi_mb[2] - phi_mb[1]);
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

  vector[N] phi_ss;
  for(i in 1:N) {
    if(MEQ[eq[i]] <= phi_mb[1]) phi_ss[i] = phi_ss_1;
    else if(MEQ[eq[i]] >= phi_mb[2]) phi_ss[i] = phi_ss_2;
    else
      phi_ss[i] = phi_ss_1 + (phi_ss_2 - phi_ss_1) * M_phi[i];
  }

  vector[NEQ] tau;
  for(i in 1:NEQ) {
    if(MEQ[i] <= tau_mb[1]) tau[i] = tau_1;
    else if(MEQ[i] >= tau_mb[2]) tau[i] = tau_2;
    else
      tau[i] = tau_1 + (tau_2 - tau_1) * M_tau[i];
  }

  eqterm ~ normal(0, tau);
  statterm ~ normal(0, phi_s2s);

  Y ~ normal(ic + eqterm[eq] + statterm[stat], phi_ss);
}

