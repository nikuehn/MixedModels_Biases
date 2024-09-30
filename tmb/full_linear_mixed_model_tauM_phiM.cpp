#include <TMB.hpp>

template<class Type>
Type objective_function<Type>::operator() ()
{
    // Data inputs
    DATA_VECTOR(Y);        // Response variable
    DATA_FACTOR(eq);       // Factor for eq random effect
    DATA_FACTOR(stat);     // Factor for stat random effect
    DATA_VECTOR(M1_eq);
    DATA_VECTOR(M2_eq);
    DATA_VECTOR(M1_rec);
    DATA_VECTOR(M2_rec);
    DATA_MATRIX(X);
    
    // Parameters to estimate
    PARAMETER_VECTOR(u_eq);      // Random effects for eq
    PARAMETER_VECTOR(u_stat);    // Random effects for stat
    PARAMETER_VECTOR(beta);             // Fixed intercept
    PARAMETER(log_phi_ss_sm);
    PARAMETER(log_phi_ss_lm);
    PARAMETER(log_tau_sm);
    PARAMETER(log_tau_lm);
    PARAMETER(log_phi_s2s);

    int n = Y.size();
    //int n_eq = eq.maxCoeff() + 1;   // Number of levels in eq
    //int n_stat = stat.maxCoeff() + 1; // Number of levels in stat

    // Derived variables
    vector<Type> phi_ss = exp(log_phi_ss_sm * M1_rec + log_phi_ss_lm * M2_rec);
    vector<Type> tau = exp(log_tau_sm * M1_eq + log_tau_lm * M2_eq);
    Type phi_s2s = exp(log_phi_s2s);   // Standard deviation for stat random effects
    Type phi_ss_sm = exp(log_phi_ss_sm);
    Type phi_ss_lm = exp(log_phi_ss_lm);
    Type tau_sm = exp(log_tau_sm);
    Type tau_lm = exp(log_tau_lm);

    vector<Type> mean_pred = X * beta;

    // Report the values of sigma, sigma_eq, and sigma_stat
    REPORT(phi_s2s);
    REPORT(phi_ss_sm);
    REPORT(phi_ss_lm);
    REPORT(tau_sm);
    REPORT(tau_lm);
    
    // Random effects contributions
    Type nll = 0.0; // Initialize negative log likelihood
    nll -= dnorm(u_eq, Type(0.0), tau, true).sum();     // eq random effects
    nll -= dnorm(u_stat, Type(0.0), phi_s2s, true).sum(); // stat random effects

    // Likelihood for the data
    for (int i = 0; i < n; i++) {
        Type mu = mean_pred(i) + u_eq(eq(i)) + u_stat(stat(i)); // Linear predictor
        nll -= dnorm(Y(i), mu, phi_ss(i), true);           // Add data likelihood contribution
    }

    return nll;
}

