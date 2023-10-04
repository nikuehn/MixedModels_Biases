/*********************************************
 ********************************************/

data {
  int N;  // number of records
  int NEQ;  // number of earthquakes
  int NSTAT;  // number of stations
  int K;

  matrix[N, K-1] X;
  vector[N] Y; // ln ground-motion value

  array[N] int<lower=1,upper=NEQ> eq; // event id (in numerical order from 1 to last)
  array[N] int<lower=1,upper=NSTAT> stat; // station id (in numerical order from 1 to last)

  vector<lower=0>[3] alpha; // dirichlet hyperparameter
}

transformed data {
  matrix[N, K-1] Q_ast = qr_thin_Q(X) * sqrt(N - 1);
  matrix[K-1, K-1] R_ast = qr_thin_R(X) / sqrt(N - 1);
  matrix[K-1, K-1] R_ast_inverse = inverse(R_ast);
}

parameters {
  vector[K-1] c_qr;
  real ic;

  real<lower=0> sigma_total;
  simplex[3] omega;

  vector[NEQ] eqterm;
  vector[NSTAT] statterm;
}

transformed parameters {
  real<lower=0> phi_ss = sqrt(omega[1] * square(sigma_total));
  real<lower=0> tau = sqrt(omega[2] * square(sigma_total));
  real<lower=0> phi_s2s = sqrt(omega[3] * square(sigma_total));
}

model {
  ic ~ normal(0,5);
  c_qr ~ std_normal();

  sigma_total ~ std_normal();
  omega ~ dirichlet(alpha);

  eqterm ~ normal(0, tau);
  statterm ~ normal(0, phi_s2s);

  Y ~ normal(ic + Q_ast * c_qr + eqterm[eq] + statterm[stat], phi_ss);

}

generated quantities {
  vector[K] c;
  c[1] = ic;
  c[2:K] =  R_ast_inverse * c_qr;
  vector[N] resid = Y - (ic + Q_ast * c_qr + eqterm[eq] + statterm[stat]);
}
