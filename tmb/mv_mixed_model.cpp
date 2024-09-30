#include <TMB.hpp>

template<class Type>
Type objective_function<Type>::operator() ()
{
    using namespace density;
    // Data inputs
    DATA_MATRIX(Y);        // Response variable s
    DATA_FACTOR(eq);       // Factor for eq random effect
    DATA_FACTOR(stat);     // Factor for stat random effect
    
    // Parameters to estimate
    PARAMETER_MATRIX(u_eq);      // Random effects for eq
    PARAMETER_MATRIX(u_stat);    // Random effects for stat
    PARAMETER_VECTOR(beta);             // Fixed intercept
    PARAMETER_VECTOR(log_sigma_rec);        // Log of residual standard deviation
    PARAMETER_VECTOR(log_sigma_eq);     // Log of standard deviation for eq random effects
    PARAMETER_VECTOR(log_sigma_stat);   // Log of standard deviation for stat random effects
    PARAMETER_VECTOR(rho_eq);           // correlations of event terms
    PARAMETER_VECTOR(rho_stat);           // correlations of station terms
    PARAMETER_VECTOR(rho_rec);           // correlations of record terms

    int n = Y.rows();
    int p = Y.cols();

    // Get the number of levels for eq and stat
    int n_eq = eq.maxCoeff() + 1;   // Number of levels in eq
    int n_stat = stat.maxCoeff() + 1; // Number of levels in stat

    // Derived variables
    vector<Type> sigma_rec = exp(log_sigma_rec);             // Residual standard deviation
    vector<Type> sigma_eq = exp(log_sigma_eq);       // Standard deviation for eq random effects
    vector<Type> sigma_stat = exp(log_sigma_stat);   // Standard deviation for stat random effects

    Type nll = 0; // Initialize negative log likelihood
    vector<Type> res(p);
    for (int j = 0; j<n; j++){ //Density for response
      for(int i = 0; i <p; i++){
        Type mu = beta(i) + u_eq(eq(j), i) + u_stat(stat(j), i);
        res(i) = Y(j, i) - mu;
      }
      nll += VECSCALE(UNSTRUCTURED_CORR(rho_rec), sigma_rec)(res);
    }

    for(int i = 0; i < n_eq; i++) // Density for R.E.
      nll += VECSCALE(UNSTRUCTURED_CORR(rho_eq), sigma_eq)(u_eq.row(i));

    for(int i = 0; i < n_stat; i++) // Density for R.E.
      nll += VECSCALE(UNSTRUCTURED_CORR(rho_stat), sigma_stat)(u_stat.row(i));

    matrix<Type> Cor_eq(p,p);
    matrix<Type> Cor_stat(p,p);
    matrix<Type> Cor_rec(p,p);
    Cor_eq = UNSTRUCTURED_CORR(rho_eq).cov();
    Cor_stat = UNSTRUCTURED_CORR(rho_stat).cov();
    Cor_rec = UNSTRUCTURED_CORR(rho_rec).cov();
    REPORT(Cor_eq);
    REPORT(Cor_stat);
    REPORT(Cor_rec);

    return nll;
}

