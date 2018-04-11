# -*- coding: utf-8 -*-

import numpy as np # move to notebook
import random 	   # move to notebook + set seed


class Environment(object):
    """
    Generates random users at each time step in the MAB process. 
    Logs user recommendations.

    Parameters
    ----------
    filePath : str, default=None
        Path to dataset
    k : int, default=3 
        number of latent features
    var_star : float, default=0.5
        model variance hyper-parameter
    genData : bool, default=False
        create synthetic dataset

    """
    def __init__(self, filePath=None, k=3, var_star=0.5, genData=False):
        self.K = k
        self.var_star = var_star
        # dictionary tracks which users/items have been seen
        self.history = {}

        # load and process data from file
        if genData:
            self.num_users = 20
            self.num_items = 50
            self.mu_u = 0 
            self.mu_v = 0
            self.var_u = 1
            self.var_v = 1
            self.mu_star = 0
            self.data = self.gen_syn_data()
        else: # generate synthetic dataset
            # 1. read in data
            # 2. preprocess data to get user/items/ratings dicts
            # 3. set num_users/num_movies
            # 4. set default var_u, var_v (to be used in bayesian PTS)
            pass

        self.userItemsCount = self.items_per_user(genData)
        self.userRewards = { k: [] for k,v in enumerate(range(self.num_users)) } 
        self.userRecItems = { k: [] for k,v in enumerate(range(self.num_users)) } 
        


    def gen_syn_data(self):
        """ 
        Generates synthetic dataset according to sec. 5.1, pg.6 [1]

        Returns
        -------
        data : dict, 
            dictionary of data containing U,V, R_true, R_obs (noisy R_true)
        """

        #U = np.random.normal(self.mu_u, self.var_u, (self.num_users, self.K)) 
        #V = np.random.normal(self.mu_v, self.var_v, (self.num_items, self.K))
        U = np.zeros((self.num_users, self.K))
        V = np.zeros((self.num_items, self.K))
        R_true = np.dot( U, V.T )
        R_obs = R_true + np.random.normal(self.mu_star, self.var_star) # add gaussian noise to R_true
        return { 'U': U, 'V': V, 'R_true': R_true, 'R_obs': R_obs }

    def get_new_user(self):
        """ 
        Randomly samples new user from user/item pairs not yet seen in data 

        Returns
        -------
        user_id : int, 
            new user_id generated at random
        """

        # List of user ids with rated items we haven't yet seen
        available_users = [key for key, v in self.userItemsCount.items() if v > 0]
        if len(available_users) > 0:
            user_id = random.choice(available_users)
            # decrease num of remaining movies
            self.userItemsCount[user_id] -= 1 
            #print("user_id: {} | unseen movie count: {}".format(user_id, self.userItemsCount[user_id]))
            return user_id
        else:
            print("ERROR: All user/item pairs have been seen!")
            print("Restarting user log...exit to cancel!")
            self.userItemsCount = self.items_per_user()
            user_id = self.get_new_user()
            return user_id


    def items_per_user(self, genData=False):
        """ 
        maps user ids to count of num of items they've rated in data 

        Returns
        -------
        userItemsCount : dict, userId : item count
            dictionary of userIds to count of rated items
        """
        if genData:
            userItemsCount = {}
            for user in range(len(self.data['R_true'])):
                movie_idxs = np.where(self.data['R_true'][user, :] > .0)
                userItemsCount[user] = len(movie_idxs)
            return userItemsCount
        else:
            raise NotImplementedError

    def itemsRatedByUser(self, user_id):
        """ 
        useful for synthetic data but probably need to reimplement 
        with real data 

        Returns
        -------
        V_j : Numpy matrix, shape ( O(M) x K )
            matrix of items rated by user. M upper bounded total items
        """
        item_ids = self.userRecItems[user_id]
        V_j = self.data['V'][item_ids]
        if len(V_j) == 0:
            return np.zeros((1,self.K))
        return V_j

    def usersWhoRatedItem(self, item_id):
        raise NotImplementedError
    
    def get_reward_vector(self, user_id):
        _r = self.userRewards[user_id]
        if len(_r) == 0:
            return 1.0
        # reward for user i from 1:t-1
        rPred_History = np.asarray(_r).reshape(1,-1) 
        return rPred_History




