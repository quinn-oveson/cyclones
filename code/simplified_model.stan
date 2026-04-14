data {
    // data
    int <lower=1> N;                    // number of observed cyclones
    int <lower=1> B;                    // number of ocean basins to model
    int <lower=1> D;                    // number of predictors to model
    array[N] real W;                    // logged Max Wind Speeds (future)
    matrix[N, D] X;                     // design matrix
    array[N] int <lower=0> basins;      // categorical vector, indicates which ocean basin each cyclone is in

    // prior parameters 
    real <lower=0> a_sig;               // shape parameter for inverse-gamma prior on sig2
    real <lower=0> b_sig;               // scale parameter for inverse-gamma prior on sig2
    real <lower=0> v2;                  // variance for normal prior on nu 
    real <lower=0> a_tau;               // shape parameter for inverse-gamma prior on tau2
    real <lower=0> b_tau;               // scale parameter for inverse-gamma prior on tau2
}

parameters {
    vector <lower=0> [B] sig2;          // wind speed variances for each basin
    matrix[B, D] Beta;                  // regression parameters
    vector <lower=0> [D] tau2;          // parameter variances for each predictor
    vector [D] nu;                      // parameter means for each predictor
}

transformed parameters {
   vector[N] all_sig2;
   vector[N] all_mu;

   for (i in 1:N) {
    all_sig2[i] = sig2[basins[i]];
    all_mu[i] = dot_product(X[i], Beta[basins[i]]);
   }  
}

model {
    // likelihood 
    W ~ normal(all_mu, sqrt(all_sig2));

    // prior on regression parameters 
    for (d in 1:D) {
        Beta[:, d] ~ multi_normal(rep_vector(nu[d], B), diag_matrix(rep_vector(tau2[d], B)));
    }

    // hyperpriors 
    sig2 ~ inv_gamma(a_sig, b_sig);
    nu ~ multi_normal(rep_vector(0.0, D), diag_matrix(rep_vector(v2, D)));
    tau2 ~ inv_gamma(a_tau, b_tau);
}

generated quantities {
   
}