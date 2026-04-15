data {
    // data
    int <lower=1> N;                    // number of observed cyclones
    int <lower=1> B;                    // number of ocean basins to model
    int <lower=1> D;                    // number of predictors to model
    array[N] real W;                    // logged Max Wind Speeds (future)
    matrix[N, D] X;                     // design matrix
    array[N] int <lower=0> basins;      // categorical vector, indicates which ocean basin each cyclone is in

    // prior parameters
    real <lower=0> a;                   // shape parameter for inverse-gamma prior on tau2
    real <lower=0> y_alpha;             // shape parameter for gamma prior on alpha
    real <lower=0> z_alpha;             // scale parameter for gamma prior on alpha
    real <lower=0> y_gamma;             // shape parameter for gamma prior on gamma
    real <lower=0> z_gamma;             // scale parameter for gamma prior on gamma
    real <lower=0> v2;                  // variance for normal prior on nu
    real <lower=0> y_lambda;            // shape parameter for gamma prior on lambda
    real <lower=0> z_lambda;            // scale parameter for gamma prior on lambda
}

parameters {
    // vector <lower=0> [B] sig2;          // wind speed variances for each basin
    matrix[B, D] Beta;                  // regression parameters
    vector <lower=0> [D] tau2;          // parameter variances for each predictor
    real <lower=0> alpha;               // shape parameter for inverse-gamma prior on sig2
    real <lower=0> gamma;               // scale parameter for inverse-gamma prior on sig2
    vector [D] nu;                      // parameter means for each predictor
    vector <lower=0> [D] lambda;        // scale parameter for inverse-gamma prior on tau2
    vector <lower = 0> [B] sig;
}

transformed parameters {
   vector[N] all_sig2;
   vector[N] all_mu;
    vector[B] sig2;
    for(i in 1:B) {
        sig2[i] = sig[i]^2;
}

   for (i in 1:N) {
    all_sig2[i] = sig2[basins[i]];
    all_mu[i] = dot_product(X[i], Beta[basins[i]]);
   }
}

model {
    // likelihood 
    W ~ normal(all_mu, sqrt(all_sig2));

    // priors
    sig ~ cauchy(alpha, gamma);
    for (d in 1:D) {
        Beta[:, d] ~ multi_normal(rep_vector(nu[d], B), diag_matrix(rep_vector(tau2[d], B)));
    }
    tau2 ~ inv_gamma(a, lambda);

    // hyperpriors
    alpha ~ gamma(y_alpha, 1.0/z_alpha);
    gamma ~ gamma(y_gamma, 1.0/z_gamma);
    nu ~ multi_normal(rep_vector(0.0, D), diag_matrix(rep_vector(v2, D)));
    lambda ~ gamma(y_lambda, 1.0/z_lambda);
}

generated quantities {
      vector[N] log_lik;
    for (i in 1:N){
        log_lik[i] = normal_lpdf(W[i] | all_mu, sqrt(all_sig2));
}
}