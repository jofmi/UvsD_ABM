""" UvsD ABM - Model Module """

from UvsD_ABM_Parameters import *
from UvsD_ABM_Agents import *

## Main Dynamics ##

# Single Run

def run_model(scenarios,param_values=0,calibrating=False,p_cal=0,open_economy=False): 
    
    results = []
    E_up_target = None
    E_down_target = None
    
    # Set up random parameters
    if calibrating == False: 
        pr = c_parameters(param_values,open_economy)
        random_par = pr.generate_random_par()
    
    # Iterate through all scenarios
    for scenario in scenarios: 

        # Load parameters
        if calibrating == False: 
            p = c_parameters(param_values,open_economy)
            p.load_sc(scenario)
            p.load_random_par(random_par)
            p.E_up_target = E_up_target
            p.E_down_target = E_down_target
        else: p = p_cal 
            
        # Calibrate tax or direct regulation  
        if scenario == "Uptax" or scenario == "Downtax":
            if p.calibrate == True and calibrating == False: calibrate_tax(p)
        if scenario == "UpDR" or scenario == "DownDR":
            if p.calibrate == True and calibrating == False: calibrate_direct_regulation(p)
        
        # Initialize agents
        p.u_sec = u_sec = c_sector(p,"upstream",p.Nf,p.Nf_c) # Upstream Sector
        p.d_sec = d_sec = c_sector(p,"downstream",p.Np,p.Np_c) # Downstream Sector
        p.economy = economy = c_economy([u_sec,d_sec]) # Both sectors
        p.reg = reg = c_regulator(p) # Regulator
        p.f_ex = f_ex = c_exchange(p) # Fuel exchange
        p.p_ex = p_ex = c_exchange(p) # Permit exchange

        # Run simulation
        t = 1

        while t < p.T:
            
            # STEP 0 - Document emission level at policy start time for target calculation
            if scenario == 'No_Policy' and t == p.t_start:
                E_up_target = u_sec.E_cov[t-1] * p.E_max
                E_down_target = d_sec.E_cov[t-1] * p.E_max
            
            # STEP 1 - The regulator updates their climate policy.
            reg.update_policy(economy,p,t)
            
            # STEP 2 - Firms form their goals and order fuel.
            economy.apply("update_vars",p,t)
            d_sec.apply("set_goal",p,t)
            d_sec.apply("order_fuel",p,t)
            u_sec.apply("set_goal",p,t)
            
            # STEP 3 - Either suppliers or manufacturers trade at the permit market.
            if reg.permit_market == True: 
                if p.regpoint == "upstream":
                    u_sec.apply("order_permits",p,t) 
                    p_ex.auction_permits(u_sec,p,t)                
                else:
                    d_sec.apply("order_permits",p,t) 
                    p_ex.auction_permits(d_sec,p,t)   
            
            # STEP 4 - Suppliers acquire fossil fuels and try to sell them at the fuel market.
            u_sec.production(p,t)
            u_sec.apply("set_mark_up",p,t)
            u_sec.apply("set_price",p,t)
            u_sec.apply("sell_fuel",p,t)
            end_of_fuel_trade(d_sec,u_sec,p,t)
            
            # STEP 6 - Producers create consumption goods and try to sell them at the goods market.
            d_sec.production(p,t) # Downstream production   
            d_sec.apply("set_mark_up",p,t)
            d_sec.apply("set_price",p,t)
            trade_commodities(d_sec,p,t) 
            
            # STEP 7 & 8
            #economy.apply("expand",p,t) # Suppliers decide whether to expand production. 
            d_sec.apply("abatement",p,t) # Producers decide whether to adopt less emission-intensive technology.
            d_sec.apply("shift_suppliers",p,t) # Producers shift towards better fuel suppliers
            
            t += 1 # Move to next round
        
        results.append([d_sec,p,u_sec])
        
        # Check for Errors
        if print_errors == True and p.error == True and calibrating == False: 
            print("Errors found in Scenario ",scenario)
            print(p.log)
        
    return results


# Calibration for Tax and Direct Regulation

def calibrate_tax(p_cal):
    c=0
    mintax = 0
    maxtax = p_cal.tax
    
    p_cal.tax = (mintax + maxtax) / 2

    results = run_model(["X"],p_cal=p_cal,calibrating=True)
    sec,p,usec = results[0]
    
    if p_cal.regpoint == "upstream": sec = usec 
    
    while abs(sec.E_cov[p.T-1] - p.reg.E_max) > p.calibration_treshold: 
        
        if sec.E_cov[p.T-1] >= p.reg.E_max: mintax = p_cal.tax 
        else: maxtax = p_cal.tax
        p_cal.tax = (mintax + maxtax) / 2
        
        results = run_model(["X"],p_cal=p_cal,calibrating=True)
        sec,p,usec = results[0]
        if p_cal.regpoint == "upstream": sec = usec 
        c+=1
        
        if c>p.calibration_max_runs:
            p_cal.report_error("Error in calibrate_tax: c_max reached")
            break       
    
    return 

def calibrate_direct_regulation(p_cal):
    c=0
    tx=10
    t1=p_cal.T-1
    t0=t1-tx
    mintax = 0
    maxtax = 1
    
    p_cal.final_quota = (mintax + maxtax) / 2

    results = run_model(["X"],p_cal=p_cal,calibrating=True)
    sec,p,usec = results[0]
    
    if p_cal.regpoint == "upstream": sec = usec
    
    while abs(sec.E_cov[p.T-1] - p.reg.E_max)  > p.calibration_treshold: 
        
        if sec.E_cov[p.T-1] >= p.reg.E_max : maxtax = p_cal.final_quota
        else: mintax = p_cal.final_quota
        p_cal.final_quota = (mintax + maxtax) / 2
        
        results = run_model(["X"],p_cal=p_cal,calibrating=True)
        sec,p,usec = results[0]
        
        if p_cal.regpoint == "upstream": sec = usec 
        
        c+=1
        
        if c>p.calibration_max_runs:
            p_cal.report_error("Error in calibrate_direct_regulation: c_max reached")
            break       
    
    return 


# Prepare Evaluation Measures - Decomposition of abatement

def calc_abatement_analysis(sc,t0):

    sec,p,usec = sc
    
    T = p.T # End here

    ΔE = [sum([j.e[t]-j.e[t0] for j in sec]) for t in range(t0,T)]
    ab_tot = [-x for x in ΔE]

    ΔQ = [sum([j.qg[t]-j.qg[t0] for j in sec]) for t in range(t0,T)]

    # Decomposition into production and technology

    def x1(sec,j,t): # Technology change
        return (j.qg[t0] + j.qg[t]) / 2 * ( j.A[t] - j.A[t0] )

    def x2(sec,j,t): # Production change
        return (j.A[t0] + j.A[t]) / 2 * ( j.sq[t] * sec.Q[t] - j.sq[t0] * sec.Q[t0]) 

    ab_1 = [- sum([ x1(sec,j,t) for j in sec]) for t in range(t0,T)] 
    ab_2 = [- sum([ x2(sec,j,t) for j in sec]) for t in range(t0,T)] 

    # Further decomposition of production level

    def x21(sec,j,t): # Compositional change
        return (sec.Q[t0] + sec.Q[t]) / 2 * (j.sq[t] - j.sq[t0]) 

    def x22(sec,j,t): # Overall production level change
        return (j.sq[t0] + j.sq[t]) / 2 * (sec.Q[t] - sec.Q[t0]) 

    def A_mean(sec,j,t):
        return (j.A[t0] + j.A[t]) / 2    

    ab_21 = [ - sum([ x21(sec,j,t) * A_mean(sec,j,t) for j in sec]) for t in range(t0,T)] # Compositional change
    ab_22 = [ - sum([ x22(sec,j,t) * A_mean(sec,j,t) for j in sec]) for t in range(t0,T)] # Overall production level change
    
    return [ab_21, ab_1,ab_22,ab_tot]


def calc_local_abatement_analysis(sc,t0):

    sec,p,usec = sc
    
    T = p.T # End here

    ΔE = [sum([j.e[t]-j.e[t0] for j in sec.covered]) for t in range(t0,T)]
    ab_tot = [-x for x in ΔE]

    ΔQ = [sum([j.qg[t]-j.qg[t0] for j in sec.covered]) for t in range(t0,T)]

    # Decomposition into production and technology

    def x1(sec,j,t): # Technology change
        return (j.qg[t0] + j.qg[t]) / 2 * ( j.A[t] - j.A[t0] )

    def x2(sec,j,t): # Production change
        return (j.A[t0] + j.A[t]) / 2 * ( j.sq_cov[t] * sec.Q_cov[t] - j.sq_cov[t0] * sec.Q_cov[t0]) 

    ab_1 = [- sum([ x1(sec,j,t) for j in sec.covered]) for t in range(t0,T)] 
    ab_2 = [- sum([ x2(sec,j,t) for j in sec.covered]) for t in range(t0,T)] 

    # Further decomposition of production level

    def x21(sec,j,t): # Compositional change
        return (sec.Q_cov[t0] + sec.Q_cov[t]) / 2 * (j.sq_cov[t] - j.sq_cov[t0]) 

    def x22(sec,j,t): # Overall production level change
        return (j.sq_cov[t0] + j.sq_cov[t]) / 2 * (sec.Q_cov[t] - sec.Q_cov[t0]) 

    def A_mean(sec,j,t):
        return (j.A[t0] + j.A[t]) / 2    

    ab_21 = [ - sum([ x21(sec,j,t) * A_mean(sec,j,t) for j in sec.covered]) for t in range(t0,T)] # Compositional change
    ab_22 = [ - sum([ x22(sec,j,t) * A_mean(sec,j,t) for j in sec.covered]) for t in range(t0,T)] # Overall production level change
    
    return [ab_21, ab_1,ab_22,ab_tot]

# Prepare Evaluation Measures - Calculate Measures

measure_names = ['Run','Scenario',
                 'Emissions',
                 'Technology adoption','Compositional change','Production decline','Abatement, relative to target',
                 'Sales price',
                 'Upstream production costs','Downstream production costs','Upstream profits','Downstream profits','Policy revenue',
                 'Consumer Impact',
                 'Upstream Market Concentration','Downstream Market Concentration','Sales',
                 'Emission Price','Emission & Fuel Costs',
                 'Local Emissions',
                 'Local Technology adoption','Local Compositional change','Local Production decline','Local Abatement, relative to target',
                 'Local Sales price',
                 'Local Upstream production costs','Local Downstream production costs','Local Upstream profits','Local Downstream profits','Local Policy revenue',
                 'Local Consumer Impact',
                 'Local Upstream Market Concentration','Local Downstream Market Concentration','Local Sales'
                ]

measure_names2 = ['Run','Scenario',
                 'Emissions',
                 'Technology \n adoption','Compositional \n change','Production \n decline','Abatement, relative to target',
                 'Sales \n price',
                 'Upstream \n production \n costs','Downstream \n production \n costs','Upstream \n profits','Downstream \n profits','Policy \n revenue',
                'Consumer \n Impact',
                 'Upstream \n Market \n Concentration','Downstream \n Market \n Concentration','Sales',
                  'Emission \n Price','Emission & \n Fuel Costs',
                  'Local Emissions',
                 'Local Technology \n adoption','Local Compositional \n change','Local Production \n decline','Local Abatement, relative to target',
                 'Local Sales \n price',
                 'Local Upstream \n production \n costs','Local Downstream \n production \n costs','Local Upstream \n profits','Local Downstream \n profits','Local Policy \n revenue',
                 'Local \n Consumer \n Impact',
                 'Local Upstream \n Market \n Concentration','Local Downstream \n Market \n Concentration','Local Sales'
                 
                ]


def evaluation_measures(results,i):
    
    tt = 10
    measures = []
    
    for sc in results:
        
        sec,p,usec = sc
        t=p.T
        t0=p.t_start-1
        
        # Effectiveness & Economic Impact
        E =   sum([ sec.E[ti]                   for ti in range(t-tt,t)  ]) # Emissions
        PE =  sum([ p.reg.pe[ti]                for ti in range(t-tt,t)  ]) # Emission Price
        PEF = sum([ sum([j.pe[ti] + j.pf[ti] for j in sec]) for ti in range(t-tt,t)  ]) # Fuel and Emission Price 
        
        # Abatement Decomposition
        ac,at,ar,ab_tot = calc_abatement_analysis(sc,t0 )
        AT = at[t-t0-1] # Technology adoption
        AC = ac[t-t0-1] # Compositional Change
        AR = ar[t-t0-1] # Productin decline
        ABTOT = ab_tot[t-t0-1]
        
        # Profit distribution
        S = sum([ sum([j.qg_s[ti]            for j in sec ]) for ti in range(t-tt,t)  ]) # Sold Goods
        PG = sum([ sum([j.pg[ti] * j.qg_s[ti] for j in sec ]) for ti in range(t-tt,t)  ]) / S # Goods price per sold good
        PF = sum([ sum([j.pg[ti] * j.qg_s[ti] for j in usec]) for ti in range(t-tt,t)  ]) / S # Fuel costs per sold good
        PRd= sum([ sum([j.π[ti]               for j in sec ]) for ti in range(t-tt,t)  ]) / S # D Profit per sold good
        PRu= sum([ sum([j.π[ti]               for j in usec]) for ti in range(t-tt,t)  ]) / S # U Profit per sold good
        R = sum([ p.reg.R[ti] for ti in range(t-tt,t) ]) / S #(S*sec.p[t0]) # Revenue per sold good, relative to starting price
        Cd= PG - PF - PRd 
        Cu= PF - PRu 
        if p.regpoint == 'upstream': Cu -= R
        elif p.regpoint == 'downstream': Cd -= R
            
        # Consumer impact
        CC = ( sum([ sum([j.s[ti] * j.pg[ti]  for j in sec]) for ti in range(t-tt,t)  ]) / tt 
             - sum([ p.reg.R[ti] for ti in range(t-tt,t) ]) / S ) 

        # Market Concentration
        HHI_up = sum([ sum([j.s[ti]**2   for j in usec])       for ti in range(t-tt,t)  ])
        HHI_down = sum([ sum([j.s[ti]**2   for j in sec])       for ti in range(t-tt,t)  ])    
        
        ### REPEAT ABOVE FOR COVERED FIRMS
        # Effectiveness & Economic Impact
        oE =   sum([ sec.E_cov[ti]               for ti in range(t-tt,t)  ]) # Emissions
        
        # Abatement Decomposition
        ac,at,ar,ab_tot = calc_local_abatement_analysis(sc,t0) 
        oAT = at[t-t0-1] # Technology adoption
        oAC = ac[t-t0-1] # Compositional Change
        oAR = ar[t-t0-1] # Productin decline
        oABTOT = ab_tot[t-t0-1]
        
        # Profit distribution
        oS = sum([ sum([j.qg_s[ti]            for j in sec  if j.covered == True]) for ti in range(t-tt,t)  ]) # Sold Goods
        oPG = sum([ sum([j.pg[ti] * j.qg_s[ti] for j in sec if j.covered == True ]) for ti in range(t-tt,t)  ]) / oS # # Goods price per sold good
        oPF = sum([ sum([j.pg[ti] * j.qg_s[ti] for j in usec if j.covered == True]) for ti in range(t-tt,t)  ]) / oS # Fuel costs per sold good
        oPRd= sum([ sum([j.π[ti]               for j in sec  if j.covered == True]) for ti in range(t-tt,t)  ]) / oS # D Profit per sold good
        oPRu= sum([ sum([j.π[ti]               for j in usec if j.covered == True]) for ti in range(t-tt,t)  ]) / oS # U Profit per sold good
        oR = sum([ p.reg.R[ti] for ti in range(t-tt,t) ]) / oS # Revenue per sold good
        oCd= oPG - oPF - oPRd 
        oCu= oPF - oPRu 
        if p.regpoint == 'upstream': oCu -= oR
        elif p.regpoint == 'downstream': oCd -= oR

        # Consumer impact
        oCC = ( sum([ sum([j.s[ti] * j.pg[ti]  for j in sec  if j.covered == True]) for ti in range(t-tt,t)  ]) / tt 
             - sum([ p.reg.R[ti] for ti in range(t-tt,t) ]) / oS ) 
            
        # Market Concentration
        oHHI_up = sum([ sum([j.s[ti]**2   for j in usec if j.covered == True])       for ti in range(t-tt,t)  ])
        oHHI_down = sum([ sum([j.s[ti]**2   for j in sec if j.covered == True])       for ti in range(t-tt,t)  ])  
        
        # Corresponds with measure_names
        measures.append([i,p.label,
                         E,
                         AT,AC,AR,ABTOT,
                         PG,
                         Cu,Cd,PRu,PRd,R,CC,
                         HHI_up,HHI_down,S,
                         PE,PEF,
                         oE,
                         oAT,oAC,oAR,oABTOT,
                         oPG,
                         oCu,oCd,oPRu,oPRd,oR,oCC,
                         oHHI_up,oHHI_down,oS                      
                        ]) 
    
    return measures

