# -*- coding: utf-8 -*-

import  numpy as np


class Particles(object):
    
    def __init__(self, num_particles, num_users, num_items, k=3):
        
        self.num_particles = num_particles
        self.particles = self.init_particles(num_users, num_items)
        self.weights = [1./self.num_particles] * self.num_particles
        
    def init_particles(self, num_users, 
                               num_items, 
                               mu_u=0, 
                               var_u=1, 
                               mu_v=0, 
                               var_v=1, 
                               k=3):
        
        particles = {}
        for i in range(self.num_particles):
            particles[i] = {
                        'U': np.random.normal(mu_u, var_u, (num_users, k)),
                        'V': np.random.normal(mu_v, var_v, (num_items, k)),
                        'var_u': var_u,
                        'var_v': var_v }
        return particles
    
    def sample(self, t=1):
        if t == 1:
            return random.choice( list(self.particles.keys()) )
        raise NotImplementedError
    
    def reweighting(self):
        raise NotImplementedError
    
    def update_weights(self):
        raise NotImplementedError

