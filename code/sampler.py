import numpy as np 
from scipy import stats

class CycloneDataSimulator:
    ''' Generate simulated cyclone data according to the specified hierarichical model. 
    '''

    def __init__(self, B, a, y_alpha, z_alpha, y_gamma, z_gamma, v2, y_lam, z_lam, X=None, basins=None):
        ''' Constructor. Accepts necessary parameters to specify the model's hyperpriors. 

            Parameters:
                B (int) : number of ocean basins to be modeled
                y_alpha (float) : shape parameter for gamma hyperprior (must be positive)
                z_alpha (float) : scale parameter for gamma hyperprior (must be positive)
                y_gamma (float) : shape parameter for gamma hyperprior (must be positive)
                z_gamma (float) : scale parameter for gamma hyperprior (must be positive)
                kappa2 (float) : variance parameter for multivariate-normal hyperprior (must be positive)
                omega (float) : scale parameter for half-Cauchy hyperprior (must be positive)
                X (ndarray or None) : matrix of observed predictors 
                    **if no X is provided, generate_X() must be called to simulate predictors prior to sampling
                basins (ndarray) : array of ints/indices indicating the ocean basin for each observation
                    ** if not provided, basins will be specified during sampling and will assume 
                        an even number of samples from each basin 
        '''
        # save data inputs as class attributes 
        self.B = B 
        self.X = X 
        if X is not None: self.N_sim, self.D = X.shape
        else: self.N_sim, self.D = None, None
        self.basins = basins

        # save hyperparameters as class attributes 
        self.a = a
        self.y_alpha = y_alpha 
        self.z_alpha = z_alpha 
        self.y_gamma = y_gamma 
        self.z_gamma = z_gamma 
        self.v2 = v2 
        self.y_lam = y_lam 
        self.z_lam = z_lam


    def _assign_basins(self, N_sim):      
        ''' Create an array of integers/indices indicating the ocean basin for each associated observation. Assume roughly the same 
            number of cyclone observations from each ocean basin. Save this array as an attribute. 

            Parameters:
                N_sim (int) : desired number of simulated cyclone observations 
        '''
        # generate roughly evenly distributed basin assignments for observations
        # this is an efficient way to do this according to Gemini
        self.basins = np.arange(N_sim) * self.B // N_sim 


    def generate_X(self, N_sim):
        ''' Generate data for predictors. Does not return anything, but sets the class attribute X with a matrix simulated 
            data for the following features:
                - El Nino Index (between -2 to 2??)
                - Maximum Sustained Wind Speed (log-scale) over the first 12 hours of the cyclone
                - Minimum Sea Pressure over the first 12 hours of the cyclone
                - Wind Speed Gradient over the most recent 6 hours of the cyclone
                - Potential Intensity over the first 12 hours of the cyclone (same units as wind speed)

            Parameters: 
                N_sim (int) : desired number of simulated observations
        '''
        # we assume that if X isn't specified, the basins aren't either. Call the helper function to assign basins for each observation 
        self._assign_basins(N_sim)

        # generate El Nino index data 
        # normally between -2 and 2, with extreme values up to -3 or 3
        # use a normal distribution centered at zero with std=1
        el_nino_sims = stats.norm.rvs(loc=0., scale=1., size=N_sim)
        
        # generate max sustained wind speed 
        # normally between 25 and 65 knots 
        max_wind_sims = np.log(stats.norm.rvs(loc=40., scale=5., size=N_sim))       # log scale!!
        
        # generate minimum sea pressure
        # generally between 640 and 1000 hPa
        min_sea_press_sims = np.log(stats.norm.rvs(loc=990, scale=20., size=N_sim)) # log scale!!

        # generate wind speed gradient
        # subtract normals, one higher than the other
        wind_grad_sims = np.log(stats.norm.rvs(loc=40., scale=5., size=N_sim)) - np.log(stats.norm.rvs(loc=30, scale=5., size=N_sim))

        # Potential Intensity
        # same as max wind speed, just with a higher mean
        POT_sims = np.log(stats.norm.rvs(loc=100., scale=20., size=N_sim))       # log scale!!

        # stack data into a design matrix
        X = np.vstack([el_nino_sims, max_wind_sims, min_sea_press_sims, wind_grad_sims, POT_sims]).T

        self.X = X
        self.N_sim, self.D = X.shape

    
    def simulate(self):
        ''' Generate observations of Maximum Sustained Wind Speed over hours 12-60 of a cyclone, given predictors 
            and the described Bayesian hierarhical model.

            Returns: 
                W (ndarray) : array of length N_sim containing simulated observations of Maximum Sustained Wind Speed (log-scale)
        '''
        # check that all of the necessary model parameters are defined 
        if self.X is None or self.N_sim is None or self.basins is None or self.D is None: 
            raise ValueError("At least one of X, N_sim, and basins is not specified!")

        # draw from hyperpriors, save these as attributes
        self.alpha = stats.gamma.rvs(a=self.y_alpha, scale=self.z_alpha)
        self.gamma = stats.gamma.rvs(a=self.y_gamma, scale=self.z_gamma)
        self.nu = stats.multivariate_normal.rvs(mean=np.zeros(self.D), cov=(self.v2 * np.eye(self.D)))
        # self.lam = stats.halfcauchy.rvs(loc=0., scale=self.omega, size=self.D)
        self.lam = stats.gamma.rvs(a=self.y_lam, scale=self.z_lam, size=self.D)

        # draw from hierarchical priors, save these as attributes 
        self.sig2 = stats.invgamma.rvs(a=self.alpha, scale=self.gamma, size=self.B)
        # self.tau = stats.halfcauchy.rvs(loc=0., scale=self.lam)
        self.tau2 = stats.invgamma.rvs(a=self.a, scale=self.lam)
        self.Beta = np.vstack(
            [stats.multivariate_normal.rvs(
                mean = self.nu[d] * np.ones(self.B),
                cov = self.tau2[d] * np.eye(self.B),
            ) for d in range(self.D)]
        ).T

        # expand mu and sig2 by observation so that the process can be vectorized
        all_sig2 = np.array([self.sig2[self.basins[i]] for i in range(self.N_sim)])
        all_mu = np.array([self.X[i] @ self.Beta[self.basins[i]] for i in range(self.N_sim)])

        # draw observations from the likelihood 
        self.W = stats.norm.rvs(loc=all_mu, scale=np.sqrt(all_sig2))

        return self.W