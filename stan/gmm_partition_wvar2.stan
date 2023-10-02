/*********************************************
 ********************************************/

data {
  int N;  // number of records
  int NEQ;  // number of earthquakes
  int NSTAT;  // number of stations

  vector[N] Y; // ln ground-motion value

  array[N] int<lower=1,upper=NEQ> eq; // event id (in numerical order from 1 to last)
  array[N] int<lower=1,upper=NSTAT> stat; // station id (in numerical order from 1 to last)

  vector<lower=0>[3] alpha; // dirichlet hyperparameter

}

parameters {
  real ic;

  real<lower=0> sigma_total;
  simplex[3] omega;

  vector[NEQ] eqterm;
  vector[NSTAT] statterm;
}

transformed parameters {
  real<lower=0> phi_0 = sqrt(omega[1] * square(sigma_total));
  real<lower=0> tau_0 = sqrt(omega[2] * square(sigma_total));
  real<lower=0> phi_S2S = sqrt(omega[3] * square(sigma_total));
}

model {
  ic ~ normal(0, 0.1);

  sigma_total ~ std_normal();
  omega ~ dirichlet(alpha);

  statterm ~ normal(0, phi_S2S);
  eqterm ~ normal(0, tau_0);

  Y ~ normal(ic + eqterm[eq] + statterm[stat], phi_0);
}

generated quantities {
  vector[N] resid = Y - (ic + eqterm[eq] + statterm[stat]);
}
