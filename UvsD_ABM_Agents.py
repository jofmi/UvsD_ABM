""" UvsD ABM - Agents Module """

import numpy as np
import copy

from operator import itemgetter
from UvsD_ABM_Parameters import *

#################      
### REGULATOR ###
#################

class c_regulator:  
    
    def __init__(self,p):
        
        self.qp, self.pe, self.R = np.zeros((3,p.T+1)) 
        self.quota = np.ones(p.T+1)
        
        self.permit_market = False
        self.emission_tax = False
        self.direct_regulation = False
        
        self.x = 0 # Step of gradual implementation
        self.E_max = 0 # Emission target
   
    def update_policy(self,economy,p,t):  
        
        u_sec, d_sec = economy
        
        if t == p.t_start: 
            
            self.E0 = d_sec.E_cov[t-1] 
            
            # Calculate E_max from existing covered emissions
            if p.regpoint == "upstream":   
                self.E_max = p.E_up_target 
            elif p.regpoint == "downstream": 
                #self.E0 = d_sec.E_cov[t-1] 
                self.E_max = p.E_down_target 
            
            # Activate policy
            if p.mode == "Tax": self.emission_tax = True
            elif p.mode == "Permits": self.permit_market = True
            elif p.mode == "Direct_Regulation": self.direct_regulation = True
                
        if t >= p.t_start: 
            
            # Calculate step of gradual implementation
            self.x = min( t - p.t_start + 1 , p.t_impl ) 
            
            # Update policy
            if p.mode == "Tax": self.set_tax(economy,p,t)     
            elif p.mode == "Permits": self.set_permits(economy,p,t)
            elif p.mode == "Direct_Regulation": self.set_direct_regulation(economy,p,t)
                
    def set_permits(self,economy,p,t): 
        
        u_sec, d_sec = economy
        if p.regpoint == "upstream": self.pe[t] = sum([j.pe[t-1] for j in u_sec if j.covered]) / len([j for j in u_sec if j.covered])
        else: self.pe[t] = sum([j.pe[t-1] for j in d_sec if j.covered]) / len([j for j in d_sec if j.covered])
    
        self.qp[t] += ( self.E0 - ( self.E0 - self.E_max ) * self.x / p.t_impl )   
        if t == p.t_start: self.qp[t] *= p.supply_forecast # Extra permits in first round 
        
    def set_tax(self,economy,p,t): 
        
        u_sec, d_sec = economy
        
        if p.regpoint == "upstream":
            
            self.pe[t] = p.tax * self.x / p.t_impl 
            for j in u_sec: 
                if j.covered == True:
                    j.pe[t] = self.pe[t]
                
        elif p.regpoint == "downstream":
            
            self.pe[t] = p.tax * self.x / p.t_impl 
            for j in d_sec: 
                if j.covered == True:
                    j.pe[t] = self.pe[t] 
                
    def set_direct_regulation(self,economy,p,t):
        
        u_sec, d_sec = economy
        if p.regpoint == "upstream": self.pe[t] = sum([j.pe[t-1] for j in u_sec if j.covered]) / len([j for j in u_sec if j.covered])
        else: self.pe[t] = sum([j.pe[t-1] for j in d_sec if j.covered]) / len([j for j in d_sec if j.covered])
        
        self.quota[t] = ( 1 - ( 1 - p.final_quota ) * self.x / p.t_impl )  
         
            
        
###############      
### SECTORS ###
###############

class c_economy(list):
    
    def apply(self,method,p,t):
        
        # Run method in each sector
        for obj in self: obj.apply(method,p,t)      

class c_sector(list):
     
    def __init__(self,p,pos,N,N_c):
        
        super().__init__()
        self.N = N # Number of firms
        self.pos = pos # Position (upstream/downstream)
        
        # Demand, Emissions, Covered emissions, Production level, Covered production level, Average price
        self.D, self.E, self.E_cov, self.Q, self.Q_cov, self.p = np.zeros((6,p.T+1)) 
        
        # Create agents/firms
        if pos == "upstream": firm = c_upfirm
        elif pos == "downstream": firm = c_downfirm
        for j in range(N): 
            if j <= N_c: self.append(firm(p,j,N,self)) # Covered
            else: self.append(firm(p,j,N,self,covered=False))
        
        # Sublist of covered firms
        self.covered = [j for j in self if j.covered] 
            
    def production(self,p,t):
        
        self.apply("production",p,t)
        
        # Total/covered emissions
        self.E[t] = sum([j.e[t] for j in self])
        self.E_cov[t] = sum([j.e[t] for j in self.covered])
        
        # Total/covered production
        self.Q[t] = sum([j.qg[t] for j in self])
        self.Q_cov[t] = sum([j.qg[t] for j in self.covered])
        
        # Production shares
        for j in self:
            if self.Q[t] > 0: j.sq[t] = j.qg[t] / self.Q[t]
            else: j.sq[t] = 1/self.N
        for j in self.covered:
            if self.Q_cov[t] > 0: j.sq_cov[t] = j.qg[t]/self.Q_cov[t]
            else: j.sq_cov[t] = 1/self.N
        
    def apply(self, method,p,t):
        
        # Run method in each list element
        for obj in self:
            getattr(obj, method)(p,t)         


            
#############      
### FIRMS ###
############# 

class c_firm:
    
    def __init__(self,p,j,N,sec,covered=True):
        
        # Parameters
        self.j = j # Firm index 
        self.sec = sec # Firm sector
        self.α = p.α[j] # Abatement cost factor
        self.δ = p.δ[j] # Emission price adaption rate
        self.η = p.eta[j] # Profitability target
        self.λ = copy.deepcopy(p.λ[j]) # List of abatement options [[a,b],[a,b],...]
        self.qI_d = p.qI_d # Desired inventory 
        self.covered = covered # Affected by policy (bool)
        
        # Dynamic variables
        self.s, self.sq, self.sq_cov, self.f, self.D, self.Dl = np.zeros((6,p.T+1)) # Market-Share, Fitness & Demand
        self.qg, self.qg_s, self.qg_d, self.qg_i, self.q_max = np.zeros((5,p.T+1)) # Production, Sales, Goal, Inventory
        self.pg, self.m, self.A, self.B, self.π = np.zeros((5,p.T+1)) # Prices & Costs 
        self.e, self.pe, self.qp_d, self.u_i, self.u_t, self.cu_t, self.MAC = np.zeros((7,p.T+1)) # Emission & Permits 
        self.qf_d,self.pf,self.f_i,self.f_q,self.f_pq = np.zeros((5,p.T+1)) # Fuels
    
        # Initial values
        self.s[0] = 1 / N # Market Share
        self.m[0] = p.m0 # Mark-up rate
    
    def update_vars(self,p,t):
        
        # Take values from last round
        self.pf[t] = self.pf[t-1]
        self.f_i[t] = self.f_i[t-1] 
        self.qg_i[t] = self.qg_i[t-1]
        self.u_i[t] = self.u_i[t-1]
        self.A[t] += self.A[t-1] 
        self.B[t] += self.B[t-1] 
        
        if (p.reg.permit_market == True or p.reg.direct_regulation == True) and self.covered == True and p.regpoint == self.pos:
            if t == p.t_start: self.pe[t] = p.pe0
            else: self.pe[t] = self.pe[t-1]
        
    def set_mark_up(self,p,t): 
        
        # Set dosi mark-up rate            
        if t != 1 and self.s[t-2] > 0.01: self.m[t] = self.m[t-1] * ( 1 +  p.ϑ * ( self.s[t-1] - self.s[t-2] ) / self.s[t-2] ) 
        else: self.m[t] = self.m[t-1] 
        
        # Set scarcity mark-up rate
        if p.reg.direct_regulation == True and p.regpoint == self.pos and self.covered == True: 
            if self.q_max[t] < self.D_exp: 
                if p.multipl_adapt: self.pe[t] *= (1+self.δ) 
                else: self.pe[t] += self.δ * (1+self.pe[t])
            else: 
                if p.multipl_adapt: self.pe[t] *= (1-self.δ)
                else: self.pe[t] = max(0, self.pe[t] - self.δ * (1+self.pe[t]) )

    def order_permits(self,p,t):
        
        if self.covered == True:
            
            # Desired permits
            if self.pos == "upstream": self.qp_d[t] = qp_d = ( self.D_exp * p.supply_forecast ) - self.u_i[t] 
            else: self.qp_d[t] = qp_d = ( self.D_exp * p.supply_forecast ) * self.A[t] - self.u_i[t]
                
            # Submitting orders
            if qp_d > 0:
                order=[self.pe[t], qp_d ,self]
                p.p_ex.orders.append(order) 
            else: 
                if p.multipl_adapt: self.pe[t] *= ( 1 - self.δ ) 
                else: self.pe[t] = max(0, self.pe[t] - self.δ * (1+self.pe[t]) )

         
        
            
########################        
### DOWNSTREAM FIRMS ###
########################            
            
class c_downfirm(c_firm):
    
    def __init__(self,p,j,N,sec,covered=True):

        super().__init__(p,j,N,sec,covered)
        self.pos = "downstream"
        self.A[0] =  p.A0[j] # Emission intensity 
        self.B[0] = p.B0[j] # Production costs    
        self.c_ab = 0 # Abatement costs
        self.suppliers = [ 1 / len(p.u_sec) ] * len(p.u_sec)
    
    def set_goal(self,p,t):
        
        # Expected Demand
        self.D_exp = self.D[t-1]
        
        # Set production goal
        self.qg_d[t] = max( 0 , self.D_exp * ( 1 + self.qI_d ) - self.qg_i[t] )  
        
    def order_fuel(self,p,t):
         
        # Calculate desired fuels
        self.qf_d[t] = qf_d = ( ( self.D_exp * p.supply_forecast ) * self.A[t] ) - self.f_i[t] 
        
        # Send orders to suppliers
        if qf_d > 0:
            for i,s in enumerate(self.suppliers):
                if s > 0: p.u_sec[i].orders.append( [qf_d*s,self] )
                elif s < 0: raise ValueError('Error in order_fuel(): Negative value in downfirm.suppliers')
        
    def production(self,p,t):
        
        # Production restricted by fuel stock 
        f_limitation = self.f_i[t] /  ( p.supply_forecast * self.A[t] ) 
        self.qg[t] = min( self.qg_d[t] , f_limitation )  
        
        # Production restricted by permit market
        if p.reg.permit_market == True and p.regpoint == self.pos and self.covered == True: 
            if self.A[t] > 0: 
                self.qg[t] = min( self.qg[t] , self.u_i[t] / ( p.supply_forecast * self.A[t] ) ) 
            else: raise ValueError('Error in production(): self.A[t] * tl(p,t) <= 0')
        
        # Restriction through direct regulation
        if p.mode == "Direct_Regulation" and p.regpoint == self.pos and self.covered == True and t >= p.t_start:
            self.q_max[t] = self.e[p.t_start-1] * p.reg.quota[t] / self.A[t] 
            self.qg[t] = min( self.qg[t] , self.q_max[t] )  
        
        self.π[t] -= self.B[t] * self.qg[t] # Remove production costs from profit
        self.qg_i[t] += self.qg[t] # Add production to goods inventory
            
        # Fuel use and Emissions  
        self.e[t] = self.qg[t] * self.A[t] # Internal Emissions
        self.f_i[t] -= self.qg[t] * self.A[t] # Fuel use
        
        # Tax collection and permit submission
        if p.regpoint == self.pos and self.covered == True: 
            if p.reg.emission_tax == True:
                p.reg.R[t] += self.e[t] * self.pe[t]
                self.π[t] -= self.e[t] * self.pe[t]
            if p.reg.permit_market == True: 
                self.u_i[t] = self.u_i[t+1] = self.u_i[t] - self.e[t]  
            
    def set_price(self,p,t):
        
        self.pg[t] = max( 0 , ( self.B[t] + self.A[t] * ( self.pe[t] + self.pf[t] ) ) * ( 1 + self.m[t] )  ) 
    
    def abatement(self,p,t):
        
        o,a,b = [0] * 3
        
        if len(self.λ) > 0: # Check if there are abatement options left
            a,b = self.λ[0] # Extract best abatement option
            self.MAC[t] = MAC = b / a # Marginal costs of abatement
            
            if MAC * ( 1 + self.η ) <= ( self.pe[t] + self.pf[t] ): 
                o = 1 # Activate abatement
                self.c_ab += b # Abatement costs
                self.λ.pop(0) # Remove used option from list
        
        # Update production factors
        self.A[t+1] -= o * a
        self.B[t+1] += o * b  

    def shift_suppliers(self,p,t):
     
        supplier_fitness = [ - p.u_sec[i].pg[t] - p.u_sec[i].Dl[t] for i in range(len(p.u_sec)) ]
        
        f_mean = sum([f * self.suppliers[i] for i,f in enumerate([j.f[t] for j in p.u_sec]) ]) # Average fitness
        
        if f_mean != 0:
            for i in range(len(self.suppliers)): 
                self.suppliers[i] = max( 0 , self.suppliers[i] * (1 - p.χup * (p.u_sec[i].f[t] - f_mean) / f_mean) )

        # Correct for numerical errors
        x = 1 - sum(self.suppliers)
        if x != 0:
            for i in range(len(self.suppliers)): 
                self.suppliers[i] *= 1 / (1-x)          

                    

######################        
### UPSTREAM FIRMS ###
######################
            
class c_upfirm(c_firm):

    def __init__(self,p,j,N,sec,covered=True):

        super().__init__(p,j,N,sec,covered)
        self.pos = "upstream"
        self.B[0] = p.B0_up[j] # Production costs
        self.orders = [] # Fuel orders 
        
    def set_goal(self,p,t):
        
        # Record demand from orders
        self.D[t] = sum([ order[0] for order in self.orders ])
        self.D_exp = self.D[t] 
        
        # Set production goal
        self.qg_d[t] = max( 0 , self.D_exp * ( 1 + self.qI_d ) - self.qg_i[t] ) 
            
    def production(self,p,t):           
        
        # Production
        self.qg[t] = self.qg_d[t]
        
        # Restriction through permits
        if p.reg.permit_market == True and p.regpoint == self.pos and self.covered == True: 
            self.qg[t] = min( self.qg[t] , self.u_i[t] / p.supply_forecast )  
        
        # Restriction through direct regulation
        if p.mode == "Direct_Regulation" and p.regpoint == self.pos and self.covered == True and t >= p.t_start:
            self.q_max[t] = self.qg[p.t_start-1] * p.reg.quota[t]
            self.qg[t] = min( self.qg[t] , self.q_max[t] )  
        
        self.π[t] -= self.B[t] * self.qg[t] # Remove production costs from profit
        self.B[t+1] += self.qg[t] * p.B_up_incr # Increase in production costs 
        self.qg_i[t] += self.qg[t] # Fuel inventory
        self.e[t] = self.qg[t] # Embedded emissions
        
        if p.regpoint == self.pos and self.covered == True: 
            if p.reg.emission_tax == True:
                p.reg.R[t] += self.qg[t] * self.pe[t]
                self.π[t] -= self.qg[t] * self.pe[t]
            elif p.reg.permit_market == True: 
                self.u_i[t] = self.u_i[t+1] = self.u_i[t] - self.qg[t] 
        
    def set_price(self,p,t):
        
        self.pg[t] = max( 0 , ( self.B[t] + self.pe[t] ) * ( 1 + self.m[t] )  )
            
    def sell_fuel(self,p,t):
        
        D = self.D_exp
        
        # Determine demand/supply match
        if D > self.qg_i[t] and D > 0:
            frac = self.qg_i[t] / D
            self.Dl[t] = D - self.qg_i[t]
            self.qg_s[t] = self.qg_i[t]
        else:
            frac = 1
            left = 0
            self.qg_s[t] = D
        
        # Document profits 
        self.π[t] += self.qg_s[t] * self.pg[t]
        
        # Deliver orders
        for order in self.orders:
            
            D, firm = order
            q = D * frac
            self.qg_i[t] -= q 
            firm.f_i[t] += q 
            firm.f_q[t] += q
            firm.f_pq[t] += ( q * self.pg[t] )
            firm.π[t] -= ( q * self.pg[t] )
            
        # Correct for numerical error 
        if frac != 1: self.qg_i[t] = 0 
        
        # Clear orders
        self.orders = [] 

        

###############      
### MARKETS ###
###############

# Permit Market
            
class c_exchange:
    
    def __init__(self,p):
        self.orders = [] # Market orders [Price, Quantity, Firm]
        self.pe, self.u_t, self.uA, self.uB = np.zeros((4,p.T+1)) # Permits: Market price, traded volume, supply, demand
        self.pf, self.f_t , self.t_r = np.zeros((3,p.T+1)) # Fuels: Market price, traded volume, number of trading rounds

    def auction_permits(self,sec,p,t):
        
        # Prepare bids
        np.random.shuffle(self.orders) # Shuffle list for same-price orders
        self.orders.sort(key=itemgetter(0)) # Sort list by price (lowest first) 
        bids = [x for x in self.orders if x[1] >= 0] 
        bids.reverse() # Highest first
        self.uA[t] = Q = p.reg.qp[t] 
        self.uB[t] = sum([b[1] for b in bids])
        sbids = []
        no_sbids = True
        
        # Select successful bids
        while len(bids) > 0 and Q > bids[0][1]:     
            Q -= bids[0][1]
            sbids.append(bids[0])
            self.pe[t] = bids[0][0]
            bids.pop(0)
            no_sbids = False
            
        # Last bid if it is the only succesful bid (does count as successfull for learning)
        if no_sbids == True and len(bids) > 0 and Q <= bids[0][1]:
            self.pe[t] = bids[0][0]
            sbids.append([bids[0][0],Q,bids[0][2]])
            bids.pop(0)
            Q = 0 
            
        # Learning 1: Successfull orders
        for b in sbids: 
            if p.multipl_adapt: b[2].pe[t]  *= ( 1 - b[2].δ ) 
            else: b[2].pe[t]  = max(0, b[2].pe[t] - b[2].δ * (1+b[2].pe[t]) )             
        
        # Last bid if there were other succesfull bids (does not count as successfull for learning)
        if no_sbids == False and len(bids) > 0 and Q <= bids[0][1]:
            self.pe[t] = bids[0][0]
            sbids.append([bids[0][0],Q,bids[0][2]])
            Q = 0 
        
        # Process successful bids
        for b in sbids: 
            pr = b[0] # Discriminatory pricing
            self.u_t[t] += b[1] # Trade log
            b[2].u_i[t] += b[1] # Inventory
            p.reg.R[t] += b[1] * pr # Revenue
            b[2].u_t[t] += b[1] # Trade log
            b[2].cu_t[t] += b[1] * pr # Trade log
            b[2].π[t] -= b[1] * pr # Document profit
        
        # Learning 2: Unsuccessfull orders
        for b in bids: 
            if p.multipl_adapt: b[2].pe[t] *= ( 1 + b[2].δ ) 
            else: b[2].pe[t] += b[2].δ * (1+b[2].pe[t]) 
        
        # Reset orders
        self.orders = [] 
        
        # Report market price
        if self.u_t[t] > 0: p.reg.pe[t] = sum([b[0]*b[1] for b in sbids])/self.u_t[t]
        else: p.reg.pe[t] = p.reg.pe[t-1]
        
        p.reg.qp[t+1] += Q # Keep unauctioned permits for next round 


# Fuel Market

def end_of_fuel_trade(dsec,usec,p,t):
    
    # Record average fuel prices for manufacturers
    for j in dsec:
        if j.f_q[t] > 0:
            j.pf[t] = j.f_pq[t] / j.f_q[t]
        else:
            j.pf[t] = j.pf[t-1]
            
    # Record fitness
    for j in usec:
        j.f[t] = - p.ω[0] * j.pg[t] - p.ω[1] * j.Dl[t]
        
    # Record average fuel price
    sum_qg_s = sum(j.qg_s[t] for j in usec)
    if sum_qg_s > 0: usec.p[t] = sum(j.qg_s[t]*j.pg[t] for j in usec) / sum_qg_s  
    else: usec.p[t] = usec.p[t-1]  
    
    # Record market shares
    sum_D = sum(j.D[t] for j in usec)    
    if sum_D > 0:
        for j in usec:
            j.s[t] = j.D[t] / sum_D
    else:
        for j in usec:
            j.s[t] = j.s[t-1]


# Commodity Market

def trade_commodities(sec,p,t):
    
    for j in sec: j.f[t] = - p.ω[0] * j.pg[t] - p.ω[1] * j.Dl[t-1] # Fitness
    f_mean = sum([j.f[t] * j.s[t-1] for j in sec]) # Average fitness
    for j in sec: j.s[t] = max( 0 , j.s[t-1] * (1 -  p.χ * (j.f[t] - f_mean) / f_mean) ) # Market-share evolution
    sec.p[t] = p_mean = sum([j.s[t] * j.pg[t] for j in sec]) # Average price
    sec.D[t] = D = p.D0 * np.exp(- p_mean * p.γ) # Total demand 
        
    for j in sec:
        j.D[t] = j.s[t] * D # Demand allocation    
        j.qg_s[t] = min(j.D[t],j.qg_i[t]) # Sold goods
        j.qg_i[t] -= j.qg_s[t] # Sold goods removed from inventory
        j.Dl[t] = j.D[t] - j.qg_s[t] # Unfilled demand
        j.π[t] += j.qg_s[t] * j.pg[t] # Document profit 
                
    # Correct for numerical errors
    x = 1 - sum([j.s[t] for j in sec])
    if x != 0:
        for i in sec: i.s[t] = i.s[t] * 1 / (1-x)                    
                    