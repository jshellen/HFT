# -*- coding: utf-8 -*-
import numpy as np
from enum import Enum
from copy import deepcopy

__author__ = "Juha-Samuli Hellen"

class MM_Model_Parameters:

    def __init__(self, lambda_m, lambda_p, kappa_m, kappa_p, delta, phi, alpha, q_min, q_max, T, cost, rebate):
        
        if(not isinstance(lambda_m,(float,int,np.int32,np.int64))):
            raise TypeError(f'lambda_m has to be type of <float> or <int>, not {type(lambda_m)}')

        if(not isinstance(lambda_p,(float,int,np.int32,np.int64))):
            raise TypeError(f'lambda_p has to be type of <float> or <int>, not {type(lambda_m)}')       

        if(not isinstance(delta,(float,int,np.int32,np.int64))):
            raise TypeError('delta has to be type of <float> or <int>')

        if(not isinstance(phi,(float,int,np.int32,np.int64))):
            raise TypeError('phi has to be type of <float> or <int>')

        if(not isinstance(alpha,(float,int,np.int32,np.int64))):
            raise TypeError('alpha has to be type of <float> or <int>')            

        if(not isinstance(q_min,(int,np.int32,np.int64))):
            raise TypeError('q_min has to be type of <int>')  
            
        if(not isinstance(q_max,(int,np.int32,np.int64))):
            raise TypeError('q_max has to be type of <int>')  
        
        if(q_max <= q_min):
            raise ValueError('q_max has to be larger than q_min!')

        if(not isinstance(cost,(float,int,np.int32,np.int64))):
            raise TypeError('cost has to be type of <int>')

        if(not isinstance(rebate,(float,int,np.int32,np.int64))):
            raise TypeError('rebate has to be type of <int>')

        self.m_lambda_m = lambda_m # Order-flow at be bid
        self.m_lambda_p = lambda_p # Order flow at the offer
        self.m_kappa_m = lambda_m # Order-flow decay at be bid
        self.m_kappa_p = lambda_p # Order flow decay at the offer
        self.m_T = T
        
        self.m_delta = delta # average "edge" with respect to mid-price
        self.m_phi = phi # running inventory penalty 
        self.m_alpha = alpha # terminal inventory penalty
        
        self.m_q_min = q_min 
        self.m_q_max = q_max
        self.m_cost = cost
        self.m_rebate = rebate
    
    @property
    def lambda_m(self):
        """
        Return copy of the order flow parameter at the bid
        """
        return deepcopy(self.m_lambda_m)
    
    @property
    def lambda_p(self):
        """
        Return copy of the order flow parameter at the ask
        """        
        return deepcopy(self.m_lambda_p)

    @property
    def kappa_m(self):
        """
        Return copy of the order flow parameter at the bid
        """
        return deepcopy(self.m_kappa_m)

    @property
    def kappa_p(self):
        """
        Return copy of the order flow parameter at the ask
        """        
        return deepcopy(self.m_kappa_p)
    
    @property
    def T(self):
        """
        Return copy of the order flow parameter at the ask
        """        
        return deepcopy(self.m_T)
    
    @property
    def delta(self):
        """
        Return copy of the average "edge".
        """        
        return deepcopy(self.m_delta)

    @property
    def phi(self):
        """
        Return copy of the running inventory penalty.
        """        
        return deepcopy(self.m_phi)

    @property
    def alpha(self):
        """
        Return copy of the terminal inventory penalty.
        """        
        return deepcopy(self.m_alpha)

    @property
    def q_min(self):
        """
        Return copy of the minimum inventory level.
        """        
        return deepcopy(self.m_q_min)    

    @property
    def q_max(self):
        """
        Return copy of the maximum inventory level.
        """        
        return deepcopy(self.m_q_max) 
    
    @property
    def cost(self):
        """
        Return copy of the trading cost.
        """        
        return deepcopy(self.m_cost) 

    @property
    def rebate(self):
        """
        Return copy of the trading cost.
        """
        return deepcopy(self.m_rebate)


class AS_Model_Output:
 
    def __init__(self, l_p,  l_m, h, q_lookup, q_grid, t_grid, N_steps, params):
        
        self.m_l_p = l_p
        self.m_l_m = l_m
        self.m_h = h
        
        self.m_q_lookup = q_lookup
        self.m_q_grid = q_grid
        self.m_t_grid = t_grid
        self.m_N_steps = N_steps
        self.m_params = params
    
    @property
    def params(self):
        """
        Returns a deepcopy of the model parameters
        """
        return deepcopy(self.m_params)
    
    @property
    def h(self):
        """
        Returns a copy of the value function h at each (q,t) node.
        """
        return deepcopy(self.m_h)
    
    @property
    def l_p(self):
        """
        Returns a copy of the decision variables l^{+} at each (q,t) node
        """
        return deepcopy(self.m_l_p)
 
    @property
    def l_m(self):
        """
        Returns a copy of the decision variables l^{-} at each (q,t) node
        """
        return deepcopy(self.m_l_m)
    
    @property
    def q_grid(self):
        """

        """
        return deepcopy(self.m_q_grid)
        
    @property
    def t_grid(self):
        """

        """
        return deepcopy(self.m_t_grid)

    @property
    def N_steps(self):
        
        return deepcopy(self.m_N_steps)

    def get_l_plus(self, q):

        return self.m_l_p[q]

    def get_l_minus(self, q):
        return self.m_l_m[q]

    def get_l_plus_q_t(self, q, t):
        """
        Returns SELL decision given inventory q and time remaining till end of
        trading day t.
        """
        t_idx = next(filter(lambda x: x[1] >= t, enumerate(self.m_t_grid)))[0]

        return self.m_l_p[q][t_idx]

    def get_l_minus_q_t(self, q, t):
        """
        Returns BUY decision given inventory q and time remaining till end of
        trading day t.
        """        
        t_idx = filter(lambda x: x >= t, self.m_t_grid)[0]

        return self.m_l_m[q][t_idx]


_EFF_ZERO = 1E-10


class AS2P_Finite_Difference_Solver:
    """
    Avellaneda Stoikov ++ model with terminal and running inventory penalties
    """
    @staticmethod
    def solve(params, N_steps=500):
        """
        Solves the optimal bid and ask spreads for a mm algorithm 
        using backward Euler finite difference scheme.
        """    
        n = params.q_max - params.q_min + 1 
        q_grid = [q for q in range(params.q_max, params.q_min-1, -1)]
        q_map = dict((q, i) for i, q in enumerate(q_grid))
        q_lookup = lambda q: q_map[q]

        C1 = (params.lambda_p / np.e) * (1.0/params.kappa_p - params.rebate)
        C2 = (params.lambda_m / np.e) * (1.0/params.kappa_m - params.rebate)
    
        # Terminal time
        T = params.T
        
        # Time step for finite difference
        dt = T/N_steps
        
        # Value function 
        h = np.zeros((n, N_steps))
        h[:, -1] = np.array([-params.alpha*q**2 for q in q_grid])
        
        # Time points
        t_grid = np.zeros(N_steps)
        t_grid[-1] = T
        
        # Compute 
        for idx in range(N_steps-1, 0, -1):
            
            # Update time 
            t_grid[idx-1] = t_grid[idx] - dt
            
            # Value function vector for current time step
            h_cur = np.zeros(n)
            
            # Value function vector for previous time step
            h_prev = h[:, idx]
            
            # Loop all inventory levels
            for q in range(params.q_max, params.q_min-1, -1):
                if q == params.q_max:
                    h_cur[q_lookup(q)] = h_prev[q_lookup(q)] + (-(params.phi*q**2)
                                            + C1 * np.exp(params.kappa_p*(params.rebate+h_prev[q_lookup(q-1)]-h_prev[q_lookup(q)])))*dt
                elif q == params.q_min:
                    h_cur[q_lookup(q)] = h_prev[q_lookup(q)] + (-(params.phi*q**2) 
                                         + C2 * np.exp(params.kappa_m*(params.rebate+h_prev[q_lookup(q+1)]-h_prev[q_lookup(q)])))*dt
                else:
                    h_cur[q_lookup(q)] = h_prev[q_lookup(q)] + (-(params.phi*q**2) 
                                         + C1 * np.exp(params.kappa_p*(params.rebate+h_prev[q_lookup(q-1)]-h_prev[q_lookup(q)]))
                                         + C2 * np.exp(params.kappa_m*(params.rebate+h_prev[q_lookup(q+1)]-h_prev[q_lookup(q)])))*dt
            
            # Set euler approximated value for h_cur
            h[:, idx-1] = h_cur

        # Solve optimal bid-ask spreads
        d_p = {}
        d_m = {}
        for q in range(params.q_max, params.q_min, -1):
            d_p[q] = (h[q_lookup(q)] - h[q_lookup(q-1)]) + (1. / params.kappa_p) - params.rebate

        for q in range(params.q_min, params.q_max):
            d_m[q] = (h[q_lookup(q)] - h[q_lookup(q+1)]) + (1. / params.kappa_m) - params.rebate
        
        return AS_Model_Output(d_p, d_m, h, q_lookup, q_grid, t_grid, N_steps, params)


class AS3P_Finite_Difference_Solver:
    """
    Avellaneda Stoikov +++ model with terminal and running inventory penalties
    and hedging
    
    @ Juha Hellén
    """
    @staticmethod
    def solve(params, N_steps=500):
        """
        Runs implicit backward Euler finite difference scheme for HJB part
        and then solves the double obstacle QVI problem given information
        of the value function h
        
        
        (1) Solve HJB assuming no impulses
        (2) Run again satisfying QVIs
        (3) For continuation region solve the bid ask spread
        """    
        
        n = params.q_max - params.q_min + 1 
        q_grid = [q for q in range(params.q_max, params.q_min-1, -1)]
        q_map = dict( (q, i) for i, q in enumerate(q_grid))
        q_lookup = lambda q : q_map[q]

        C1 = (params.lambda_p / np.e) * (1.0 / params.kappa_p - params.rebate)
        C2 = (params.lambda_m / np.e) * (1.0 / params.kappa_m - params.rebate)
    
        # Terminal time
        T = params.T
        
        # Time step for finite difference
        dt = T/N_steps
        
        # Function h in V(t, x) = x_t + s_t * q_t + h(t, q) 
        h_hjb = np.zeros((n, N_steps))
        
        h_hjb[:, -1] = np.array([-params.alpha*q**2 for q in q_grid])
        
        # Time points (it is useful to know the time points for which we have
        # approximated the h(t, x)
        t_grid = np.zeros(N_steps)
        t_grid[-1] = T
        
        # (1) Solve HJB by running the backward euler
        for idx in range(N_steps-1, 0, -1):
            
            # Update time 
            t_grid[idx-1] = t_grid[idx] - dt
            
            # Value function vector for current time step
            h_cur = np.zeros(n)
            
            # Value function vector for previous time step
            h_prev = h_hjb[:, idx]
            
            # Loop all inventory levels
            for q in range(params.q_max, params.q_min-1, -1):
                
                # At upper inventory limit post only sell LO
                if q == params.q_max:
                    h_cur[q_lookup(q)] = h_prev[q_lookup(q)] + (-(params.phi*q**2)
                                            + C1 * np.exp(params.kappa_p*(params.rebate+
                                                h_prev[q_lookup(q-1)]-h_prev[q_lookup(q)])))*dt
                    
                # At lower inventory limit post only buy LO
                elif q == params.q_min:
                    h_cur[q_lookup(q)] = h_prev[q_lookup(q)] + (-(params.phi*q**2) 
                                         + C2 * np.exp(params.kappa_m*(params.rebate+
                                             h_prev[q_lookup(q+1)]-h_prev[q_lookup(q)])))*dt
                # Post both sides
                else:
                    h_cur[q_lookup(q)] = h_prev[q_lookup(q)] + (-(params.phi*q**2) 
                                         + C1 * np.exp(params.kappa_p*(params.rebate+
                                             h_prev[q_lookup(q-1)]-h_prev[q_lookup(q)])) 
                                         + C2 * np.exp(params.kappa_m*(params.rebate+
                                             h_prev[q_lookup(q+1)]-h_prev[q_lookup(q)])))*dt
            
            # Set euler approximated value for h_cur
            h_hjb[:, idx-1] = h_cur
        
        
        
        # (2) Run QVI checking: max(HJB part, Send buy MO, Send sell MO)
        h = h_hjb.copy()
        impulses = np.zeros((n, N_steps))
        for idx in range(N_steps-1, 0, -1):
            
            # Vector for impulses
            i_ = np.zeros(n)
            
            # Vector for HJB-QVI consistent h
            h_qvi = np.zeros(n)
            
            # Loop all inventory levels
            for q in range(params.q_max, params.q_min-1, -1):
                
                qvi_values = -100000000000*np.ones(3)
                
                # Do nothing - let the system evolve according to HJB
                qvi_values[0] = h[q_lookup(q), idx]
                
                # Inventory at the upper boundary
                # Check if we send a sell MO
                if q == params.q_max:
                    
                    # Hedge by sending sell MO
                    qvi_values[2] = h[q_lookup(q-1), idx] - params.cost
                    
                # Inventory at the lower boundary
                # Check if we send a buy MO    
                elif q == params.q_min:
                    
                    # Hedge by sending buy MO
                    qvi_values[1] = h[q_lookup(q+1), idx] - params.cost
                
                # Inventory is between upper and lower boundaries
                # Just solve the HJB-QVI    
                else:
                    
                    # Hedge by sending buy MO
                    qvi_values[1] = h[q_lookup(q+1), idx] - params.cost
                    
                    # Hedge by sending sell MO
                    qvi_values[2] = h[q_lookup(q-1), idx] - params.cost
                    
                
                # Satisfy HJB-QVI conditions
                h_qvi[q_lookup(q)] = np.max(qvi_values)    
                i_[q_lookup(q)] = np.argmax(qvi_values)
                
            h_hjb[:, idx] = h_qvi
            impulses[:, idx] = i_
        
        
        # (3) Solve optimal bid-ask spreads for the continuation region
        d_p = {}
        d_m = {}
        for q in range(params.q_max, params.q_min, -1):
            d_p[q] = (h[q_lookup(q)] - h[q_lookup(q-1)]) + (1. / params.kappa_p) - params.rebate

        for q in range(params.q_min, params.q_max):
            d_m[q] = (h[q_lookup(q)] - h[q_lookup(q+1)]) + (1. / params.kappa_m) - params.rebate
        
        return impulses, AS_Model_Output(d_p, d_m, h, q_lookup, q_grid, t_grid, N_steps, params)    
    

