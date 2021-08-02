""" UvsD ABM - Parameter Module """

import numpy as np

# Simulation Settings

scenarios = ["No_Policy","Downtax","Downmarket","DownDR","Uptax","Upmarket","UpDR"] # Which scenarios should be compared? 
fixed_seed = True # Use fixed seed for random values?
glob_seed = 13 # Seed for random parameters
print_errors = True # Display error messages?

## 000 Parameters

if fixed_seed == True: np.random.seed(glob_seed) 

param_range = { 
    'num_vars': 15,
    'names': ['$λ$', '$Δθ$','$ΔA_0$,$ΔB_0$', 'Δμ', 'Δη','θ','ϑ','$χ^M$','μ','η','γ','β','$t^*$','ϕ','$χ^S$'],
    'bounds': [[0.5  , 0.9 ], #0  - λ - Abatement potential 
               [0.1  , 0.5 ], #1  - Δθ - Abatement Cost Heterogeneity 
               [0.1  , 0.5 ], #2  - ΔA_0,ΔB_0 - Heterogeneity of production factors
               [0.1  , 0.5 ], #3  - Δμ - Heterogeneity of emission price adaption rate 
               [0.1  , 0.5 ], #4  - Δη - Heterogeneity of profitability target  
               [15   , 20  ], #5  - θ - Abatement cost factor 
               [0.1  , 0.5 ], #6 - ϑ (vartheta) - Dosi Mark-up adaptation rate
               [0.1  , 0.5 ], #7 - χ - Market share adaptation rate 
               [0.05 , 0.1 ], #8 - μ - Price adaption rate    
               [0.1  , 0.5 ], #9 - η - Profitability rate
               [0.4  , 0.5 ], #10 - γ - Demand sensitivity 
               [0.01 , 0.1 ], #11 - β - Production cost increase upstream 
               [3    , 10  ], #12 - t* - supply forecast
               [0.3  , 0.7 ], #13 - ϕ - coverage factor
               [0.1  , 0.5 ]] #7 - χup - Market share adaptation rate upstream
}    

class c_parameters:
    def __init__(self,variable_parameters,open_economy):    
        
        # Toggle model dynamics
        self.calibrate = True  
        self.multipl_adapt = False
        self.open_economy = open_economy  
        
        # Fixed parameters
        self.T = 200 # Number of rounds
        self.Np = self.Nf = 30 # Number of final goods producers
        self.t_start= 50 # Delay until policy starts
        self.t_impl = 100 # Number of implementation periods         
        self.D0 = 1 # Maximum demand
        self.A00 = 1 # Fuel and Emission intensity 
        self.B00 = 1 # Downstream Production costs
        self.B00_up = 1 # Upstream Production costs 
        self.qI_d = 0.1 # Desired inventory 
        self.ω = [1, 1] # Market-share evolution weights        
        self.λ_n = 20 # Number of technological steps    
        self.α0 = 0 # Abatement cost ground level 
        self.E_max = 0.1 # Emission Target 
        self.m0 = 0.1 # Dosi mark-up rate in the beginning          
        #self.eta = 0.1 # Profitability target   
        self.pe0 = 0.1 # Initial trading price for permits  
        
        # Calibration parameters
        self.calibration_treshold = 10**(-2) * 0.2 
        self.calibration_max_runs = 30
        self.tax = 100 # Upper bound for tax
        
        # Variable parameters
        self.λ_max, self.Δα, self.ΔAB, self.Δδ, self.Δeta, self.α, self.ϑ,self.χ,self.δ,self.eta, self.γ, self.B_up_incr, self.supply_forecast, self.η_c,self.χup = variable_parameters
        self.δ_up = self.δ 
        self.Nf = int(self.Nf) 
        
        # Implement limited coverage
        if self.open_economy == False: self.η_c = 0
        self.Nf_c = int(round(self.Nf * (1 - self.η_c)))
        self.Np_c = int(round(self.Np * (1 - self.η_c))) 
        
        # Error log
        self.error = False
        self.log = []

    def generate_random_par(self):
               
        a = [self.δ * ( 1 + self.Δδ * ( np.random.random() - 0.5 ) ) for i in range(self.Np)]
        b = [self.A00 * ( 1 + self.ΔAB * ( np.random.random() - 0.5 ) ) for i in range(self.Np)]
        c = [self.B00 * ( 1 + self.ΔAB * ( np.random.random() - 0.5 ) ) for i in range(self.Np)]
        d = [self.α * ( 1 + self.Δα * ( np.random.random() - 0.5 ) ) for i in range(self.Np)]
        e = [self.B00_up * ( 1 + self.ΔAB * ( np.random.random() - 0.5 ) ) for i in range(self.Nf)]
        f = [self.δ_up * ( 1 + self.Δδ * ( np.random.random() - 0.5 ) ) for i in range(self.Nf)]
        g = [self.eta * ( 1 + self.Δeta * ( np.random.random() - 0.5 ) ) for i in range(self.Np)] 
    
        return [a,b,c,d,e,f,g]
    
    def load_random_par(self,random_par):
        
        self.δ,self.A0,self.B0,self.α,self.B0_up,self.δ_up,self.eta = random_par  
        
        self.λ = [] # Abatement List
        for i in range(self.Np): self.λ.append( self.generate_λ(self.α[i],self.A0[i]) )
        
    # Abatement cost curve
    def generate_λ(self,α,A0):     
        λ =[] 
        for i in range(self.λ_n):
            a=(A0*self.λ_max)/self.λ_n     
            MAC=a*α*(i+1) + self.α0 
            b=a*MAC
            λ.append([a,b])
            
        return λ       
        
    # Manage errors
    def report_error(self,statement):
        self.error = True
        if statement not in self.log:
            self.log.append(statement) 
                 
    # Scenarios 
    def load_sc(self,scenario):
        # Load Scenario
        getattr(self,scenario)() 
        
    def No_Policy(self):
        self.mode = "No Policy"
        self.label = "No Policy"
        self.regpoint = "None"
    
    def UpDR(self):
        self.mode = "Direct_Regulation"
        self.label = "Upstream direct regulation" 
        self.regpoint = "upstream" 
        
    def DownDR(self):
        self.mode = "Direct_Regulation"
        self.label = "Downstream direct regulation" 
        self.regpoint = "downstream" 
        
    def Upmarket(self):
        self.mode = "Permits"
        self.label = "Upstream permit market"
        self.regpoint = "upstream"
        
    def Downmarket(self):
        self.mode = "Permits"
        self.label = "Downstream permit market"
        self.regpoint = "downstream"
        
    def Uptax(self):
        self.mode = "Tax"
        self.label = "Upstream tax"
        self.regpoint = "upstream"
        
    def Downtax(self):
        self.mode = "Tax"
        self.label = "Downstream tax"
        self.regpoint = "downstream"
          