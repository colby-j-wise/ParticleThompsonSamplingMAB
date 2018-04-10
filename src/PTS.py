# -*- coding: utf-8 -*-


from scipy.stats import invgamma


class PTS(object):
    """
    Does the heavy lifting lol... missing good description.
    Does most of Thompson Sampling and Prob. Matrix Factorization

    Parameters
    ----------
    alpha : float, default=2.
        shape hyper-parameter for Gamma dist.
    beta : float, default=0.5
        shape hyper-parameter for Gamma dist.
    var_star : float, default=0.5
        model variance hyper-parameter
    k : int, default=3 
        number of latent features
    """
    def __init__(self, alpha=2, beta=0.5, var_star=1.0, k=3):
        self.alpha = alpha
        self.beta = beta
        self.var_star = var_star
        self.K = k 
        self.history_log = {}
        
    def sample_var(self, matrix, N):
        """ 
        Calculates the posterior of precision which is closed
        form assuming Gamma prior => inverse gamma
        Pr(lamba_u | U, alpha, beta) = IG(*)

        Params
        ------
        matrix : type Numpy matrix
            U or V matrix depending on if var_u or var_v
        N : type int 
            ** N (M) = number of users (or M=items)

        Returns
        -------
        var : type float, 
            var_u or var_v
        """
        shape = ( (N * self.K) / 2.0 ) + self.alpha
        scale = 0.5 * np.linalg.norm(matrix, 'fro') + self.beta
        # Generate random value from inverse gamma(shape,scale)
        var = round( invgamma.rvs(shape, scale=scale), 3 )
        return var
    
    def sample_Ui(self, V_j, r_ij, var_u): # verify
        """ 
        Sample Ui from the posterior:
        Pr(Ui | V, R, var*, var_u) ~ N( Ui | mu_ui, (prec_ui)^-1 )

        Params
        ------
        V_j : type Numpy matrix, shape = O(M) x K 
            matrix of items rated by user from 1:t-1
        r_ij : type array, shape = 1 x O(M)
            array of observed ratings for user i, items j
            from 1:t-1
        var_u : type float,
            sampled var_u for time step t

        Returns
        -------
        U_i : type numpy vector, shape= 1 x K 
            sampled latent feature vector for user i
        """
        # Calculate eta_ui
        eta_ui = np.sum( np.dot(r_ij, V_j) )
        
        # Calculate precision_ui
        precision_ui = (1./self.var_star) * np.sum(np.dot(V_j, V_j.T)) + (var_u * np.eye(self.K))
        _invP = np.linalg.inv(precision_ui)
        #assert( _invP > 0).all(), "ERROR: precision can't be negative"
        inv_prec_ui = _invP if ( _invP > 0).all() else np.eye(self.K) # hack, keep getting neg's!!!?? ** 
        
        # calculate mu_ui 
        mu_ui = (1./self.var_star) * inv_prec_ui * eta_ui
        
        # sample U_i from N( Ui | mu_ui, (prec_ui)^-1 )
        U_i = np.random.normal( mu_ui, inv_prec_ui ) #should be 1 x k
        return U_i

    def sample_Vj(self, U_j, r_ji, var_v): # Not thinking about this right yet :(
        raise NotImplementedError
        # """ 
        # Sample Vj from the posterior:
        # Pr(Vj | U, R, var*, var_v) ~ N( Vj | mu_vj, (prec_vj)^-1 )

        # Params
        # ------
        # U_j : type Numpy matrix, shape = O(N) x K 
        #     N = # users | matrix of users who rated item 1:t-1
        # r_ji : type array, shape = 1 x O(N)
        #     array of observed ratings for item j
        # var_v : type float,
        #     sampled var_v for time step t

        # Returns
        # -------
        # V_j : type numpy matrix, shape= N x K ???  
        #     sampled latent feature vector for user i
        # """
        # # Calculate eta_vi
        # eta_vj = np.sum( np.dot(r_ji, U_j) )
        # # Calculate precision_ui
        # precision_vj = (1./self.var_star) * np.sum(np.dot(V_j, V_j.T)) + (var_v * np.eye(self.K))
        # # calculate U_ui 
        # mu_vj = (1./self.var_star) * np.linalg.inv(precision_vj) * eta_vj
        # # sample V_j from N( Vj | mu_vj, (prec_vj)^-1 )
        # V_j = np.random.normal( mu_vj, np.linalg.inv(precision_vj) ) 
        # return V_j
    
    def recommend_item(self, U_i, V):
        """ 
        Choose an item to recommend and 
        predicts rating

        Params
        ------
        U_i : type Numpy vector, shape = 1 x K 
            matrix of items rated by user from 1:t-1
        V : type Numpy matrix, shape = M x K
            inventory of items

        Returns
        -------
        rating
        j_hat : type int,
            index of item with highest predicted rating
        pred_rt : type float,
            predicted rating for item
        """
        ratings = np.dot(U_i, V.T)
        j_hat = np.argmax(ratings)
        pred_rt =  ratings[j_hat]
        return j_hat, pred_rt[0]
    
    def calculate_reward(self, r_hat, r_true):
        return 1 if r_hat == r_true else -1
    
    def update_posterior(self):
        raise NotImplementedError
    
