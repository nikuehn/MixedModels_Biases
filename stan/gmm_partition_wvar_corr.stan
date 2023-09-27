/*********************************************
 ********************************************/

data {
  int N;  // number of records
  int NEQ;  // number of earthquakes
  int NSTAT;  // number of stations

  vector[N] Y; // ln ground-motion value

  vector[NEQ] E;

  array[N] int<lower=1,upper=NEQ> eq; // event id (in numerical order from 1 to last)
  array[N] int<lower=1,upper=NSTAT> stat; // station id (in numerical order from 1 to last)

  vector<lower=0>[3] alpha; // dirichlet hyperparameter

}

parameters {
  real ic;
  real ic2;

  real<lower=0> sigma_total;
  simplex[3] omega;
  real<lower=-1,upper=1> rho;
  real<lower=0> tau2;

  vector[NEQ] eqterm;
  vector[NSTAT] statterm;
}

transformed parameters {
  real<lower=0> phi_ss = sqrt(omega[1] * square(sigma_total));
  real<lower=0> tau = sqrt(omega[2] * square(sigma_total));
  real<lower=0> phi_s2s = sqrt(omega[3] * square(sigma_total));
}

model {
  ic ~ normal(0, 0.1);
  ic2 ~ normal(0, 0.1);

  sigma_total ~ std_normal();
  omega ~ dirichlet(alpha);

  statterm ~ normal(0, phi_s2s);

  E ~ normal(ic2, tau2);
  vector[NEQ] mu_cond = tau / tau2 * rho * E;
  real tau_cond= sqrt((1 - square(rho)) * square(tau));
  eqterm ~ normal(mu_cond, tau_cond);

  Y ~ normal(ic + eqterm[eq] + statterm[stat], phi_ss);

}
