from IPython import get_ipython as gi
gi().run_line_magic('reset', '-sf')
from tqdm import trange # see looping progress bar %

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shaftstore_5a
DegC = u'\N{DEGREE SIGN}C'
    
#%% Read data
Dat = np.loadtxt('C:\\Users\cmb22235\OneDrive - University of Strathclyde\Desktop\STEaM WP4 team\MSTES-CHP\Win 1D Model\en_flows_Win_data.csv', delimiter=',') # Descriptions for Column A to D == Column [0] to [3] in en_flow_Win_data file is listed below
# A [0] Yearly Coylton GSP Electricity Demand (MWh/0.5h)
# B [1] Yearly Coylton 33kV Wind (MWh/0.5h)
# C [2] Yearly West Whitlawburn Housing Co-operative Heat Demand (kWh/0.5h)
# D [3] Yearly ambient air temp (DegC)

# XTC = np.loadtxt('C:\\Users\cmb22235\OneDrive - University of Strathclyde\Desktop\STEaM WP4 team\Energy Flow & MTES\Win Model\mth_geo_1.csv',delimiter=',')
# Geo = np.loadtxt('C:\\Users\cmb22235\OneDrive - University of Strathclyde\Desktop\STEaM WP4 team\Energy Flow & MTES\Win Model\mth_geo2_1.csv',delimiter=',')

#%%% Fixed variable
Mod_Year = 2 # Modeling years
lifetime = 20 # lifetime of system, years
ts = 1 # timesteps per half-hour of base data
MTS = int(1800/ts) # timestep in seconds per half-hour, 0.5hr*60min*60sec)
nt = int(17520 * ts * Mod_Year) # total number of timesteps (1yr*365days*24hr*2data/hr * model yr)

Tamb = Dat[:,3]
Dat = np.repeat(Dat,ts,0) / ts # Set demand and surplus data for sub-half-hour timesteps
Tamb = np.repeat(Tamb,ts,0) # Ambient temperature for CHP
PeakHD = 4300 # Peak Heat demand, kW from WWDH 4088 kWp
tariff_select = 2 # '1' = fixed tariff, '2' = wind-based tariff

# Store
RLA = 20 # Consolidated Air Layer, m
radius_TS = 3.65 # radius of the thermal store, m
mu = 50. # Buoyancy model factor (set lower (20-50) if 'overflow encountered in exp' warnings occur)
top_wat = 5 # top layer of heated water section
number_layers = 11 # includes one air layer above water
number_nodes  = 10 # from outer ground ring to water inside shaft
Rx = np.array([256.,128.,64.,32.,16.,8.,4.,2.,1.,0.,-3.65,-3.65,-3.65]) + 3.65 # Node outer radii (m)
XTC = np.array([[2.5,2.5,2.5,2.5,2.5,2.5,2.5,2.5,3.,0.026],[2.5,2.5,2.5,2.5,2.5,2.5,2.5,2.5,3.,0.6]]) # Node thermal conductivity (W/mK)

# CHP + Gas Boiler
eff_GB_thermal = 0.88 
deltaT = 20 # temperature difference between DH supply and return pipe
Size_CHP = PeakHD
Size_GB = PeakHD

# Outputs
plot = 1 # '1' generate all results; else only generate store temp and kpi_last 
combine_kpi_last = 0 # '1' combine kpi_last csv file in specific folder for combination of parameters

#%%% Sensitivity Inputs
heat_demand_temperatures = [50]  # DegC [50]
capex_factors = [1]  # CAPEX adjustment factor [0.5,1] b1
store_temperatures = [55]  # Heat store temp, DegC [30,55,70] b55
water_layer_h = [20]  # TES Water layer heights, m [0,20,40] b20
fixed_tariff = 0.5 # £/kWh
tariff_mapping = { # (Wind Surplus/Low export tariff, Wind Shortfall/High export tariff), £/kWh
1: (0.1, 0.1),  # Mapping for tariff_price = 1, 
2: (0.1, 0.3),  # Mapping for tariff_price = 2, 
3: (0.1, 0.6),  # Mapping for tariff_price = 3, 
4: (0.1, 1.2)   # Mapping for tariff_price = 4, 
}     
tariff_mapping_choice = [2] # select tariff_mapping [1,2,3,4] b2
eff_CHP_mapping = { # efficiency of CHP elec/heat (%)
1: (0.38, 0.45),  # gas turbine
2: (0.25, 0.58),  # steam / gas turbine
3: (0.13, 0.70),  # steam
4: (0.50, 0.33),  # fuel cell
}     
eff_CHP_choice = [1] # select eff_CHP [1,2,3,4] b1
chp_price = [0.075] # (£/kWh) for biomass [0.075,0.375] b0.075
# boil_cost = chp_price # (£/kWh)

CHPmode = 1 # 0 = power (max electrical output), 1 = heat (priority heat demand)
CHPcase = 1 # 0 = chp only; 1 = chp + gas boiler 

if CHPcase == 0:
    Size_GB = 0

#%%% Looping
for T_HD in heat_demand_temperatures:
    for CAfactor in capex_factors:
        for store_temp in store_temperatures:
            for RLW in water_layer_h:
                water_height_TS = RLW * (number_layers - 1)  # water height of TS (m)
                TS_volume = np.pi * radius_TS**2 * water_height_TS  # Volume of the thermal store (m3)
                for tariff_price in tariff_mapping_choice:  # To iterate over multiple scenarios, include them in the list
                    WS_tariff, NW_tariff = tariff_mapping.get(tariff_price, (None, None)) 
                    for eff_CHP in eff_CHP_choice:  # To iterate over multiple scenarios, include them in the list
                        eff_CHP_electrical, eff_CHP_thermal = eff_CHP_mapping.get(eff_CHP, (None, None)) 
                        for chp_cost in chp_price:
                            boil_cost = chp_cost
                            
                            #%%% Data Initialisation
                            Wind_Gen             = np.zeros((nt,1)) # Wind Gen (kWh e)
                            Elec_Dem             = np.zeros((nt,1)) # Electric Demand (kWh e)
                            WS                   = np.zeros((nt,1)) # Wind Surplus (kWh e)
                            DemH                 = np.zeros((nt,1)) # Heat Demand (kWh th) 
                            Tariff               = np.zeros((nt,1)) # tariff (£/kWh)
                            chp2HD               = np.zeros((nt,1)) # heat from CHP to HD (kWh th)
                            chp_used             = np.zeros((nt,1)) # chp fuel used (kWh)
                            boil_used            = np.zeros((nt,1)) # gas boiler fuel used (kWh th)
                            chp_elec_out         = np.zeros((nt,1)) # CHP electric output (kWh e)
                            chp_export_income    = np.zeros((nt,1)) # chp export income (£)
                            gb2HD                = np.zeros((nt,1)) # heat from GB to HD (kWh th)
                            chp2tes              = np.zeros((nt,1)) # heat from CHP to store (kWh th)
                            tesD2HD              = np.zeros((nt,1)) # heat from store direct to HD (kWh th)
                            tes2HD               = np.zeros((nt,1)) # heat from store to HD (kWh th)
                            chpBH2HD             = np.zeros((nt,1)) # CHP boost heat from store to HD (kWh th) 
                            gbBH2HD              = np.zeros((nt,1)) # GB boost heat from store to HD (kWh th) 
                            Un_H                 = np.zeros((nt,1)) # unmet heat demand (kWh th)
                            waste_heat           = np.zeros((nt,1)) # chp waste heat (kWh th)
                            Res_GB               = np.zeros((nt,1)) # GB residual heat (kWh th)
                            total_chp_heat_out   = np.zeros((nt,1)) # total heatout from chp (kWh th)   
                            total_gb_heat_out    = np.zeros((nt,1)) # total heatout from chp (kWh th)   
                            total_Qc             = np.zeros((nt,1)) # total heat charge to store (kWh th)
                            total_Qex            = np.zeros((nt,1)) # total heat extract from store (kWh th)
                            total_fuel_used      = np.zeros((nt,1)) # total fuel used (kWh)
                            total_fuel_cost      = np.zeros((nt,1)) # total fuel cost (£)
                            Qloss_S_g            = np.zeros((nt,1)) # heat loss from store (kWh th)
                            chp_elec_out_NW      = np.zeros((nt,1)) # chp electric output in Non-Wind Period (kWh e)
                                                        
                            #%%% Nodes Temp Initialisation
                            if RLW > 0:     
                                nodes_temp = np.ones(number_layers * number_nodes) * 10. # Initial node temperatures (DegC) 
                                # nodes_temp[0]   to nodes_temp[9]   = Nodes(from outer ground ring [0]   to top air nodes [9])        in Top Layer = 0  
                                # nodes_temp[10]  to nodes_temp[19]  = Nodes(from outer ground ring [10]  to top water nodes [19])     in Layer = 1  
                                # nodes_temp[100] to nodes_temp[109] = Nodes(from outer ground ring [100] to bottom water nodes [109]) in Bottom Layer = 10  
                                
                                # Below are water nodes only in mineshaft store from Layer 1 to 10
                                # nodes_temp[19]  = 69.9 # Top Water Node (Layer = 1)
                                # nodes_temp[29]  = 69.7
                                # nodes_temp[39]  = 69.5
                                # nodes_temp[49]  = 69.3
                                # nodes_temp[59]  = 69.1
                                # nodes_temp[69]  = 68.9
                                # nodes_temp[79]  = 68.7
                                # nodes_temp[89]  = 68.5
                                # nodes_temp[99]  = 68.3
                                # nodes_temp[109] = 68.1 # Bottom Water Node (Layer = 10)
                                nodes_temp[19:110:10] = 10 # change store water node temp (DegC)
                                
                                f_n_t = np.zeros((nt, number_layers * number_nodes)) # Results setup, array that store temp for each node at each timestep
                            
                            #%% Basic Store Analysis (all kWh per half-hour)
                            tr = 0 # counter variable for storing current timestep
                            for t in trange(nt,desc='Timestep'): # timestep
                                td = np.remainder(t,17520) # timestep remainder
                                    
                                #%%% Begin Analysis Loop
                                Wind_Gen[t,0] = Dat[td,1] * 1000 # Wind Gen from Coylton 33kV Wind (kWh e)
                                Elec_Dem[t,0] = Dat[td,0] * 1000 # Current Electric Demand from Coylton (kWh e)
                                Av_WS = max(0, Wind_Gen[t,0] - Elec_Dem[t,0]) # Available Wind Surplus (kWh e)
                                WS[t,0] = Av_WS # Wind Surplus (kWh e)
                                DemH[t,0] = Dat[td,2] * 3 # WWHD factor of 3, ~10GWh (kWh th) 
                                Res_HD = DemH[t,0] # Residual Heat Demand (kWh th)
                                R_CHP = Size_CHP / (3600 / MTS) # Residual CHP (kWh th)
                                R_GB = Size_GB / (3600 / MTS) # Residual GB (kWh th)
                                
                                #%%% Tariff
                                # Fixed Tariff
                                if tariff_select == 1:
                                    tariff = fixed_tariff # £/kWh
                                    Tariff[t,0] = tariff # £/kWh
                            
                                # Wind-based Tariff          
                                if tariff_select == 2:
                                    if WS[t,0]> 0: # WS period / Low Export Tariff
                                        tariff = WS_tariff # £/kWh
                                    else: # Non-Wind Period / High Export Tariff
                                        tariff = NW_tariff # £/kWh
                                    Tariff[t,0] = tariff # £/kWh
                                    
                                #%%% CHP only case
                                if CHPcase == 0:
                                    
                                    if Av_WS == 0: # if wind shortfall (CHP on)
                                    
                                        #%%%% Priority 1: CHP supply heat to HD
                                        if R_CHP * Res_HD > 0: # if Residual CHP and HD exists
                                            chp2HD[t,0] = min(R_CHP, Res_HD) # heat from CHP to HD (kWh th)
                                            Res_HD -= chp2HD[t,0] # Residual HD (kWh th)
                                            R_CHP -= chp2HD[t,0] # Remaining CHP heatout (kWh th)
                                            chp_used[t,0] += chp2HD[t,0] / eff_CHP_thermal # fuel used (kWh)
                                            
                                        #%%%% Priority 2: Store supply heat to HD
                                        if RLW * Res_HD > 0: # if store and residual HD exists
                                        
                                            if nodes_temp[19] >= T_HD: # if Residual HD exists and top store >= supply HD temp
                                                tes2HD[t,0] = Res_HD # heat from store to HD (kWh th)
                                                Res_HD -= tes2HD[t,0] # Residual HD (kWh th)
                                                tesD2HD[t,0] = tes2HD[t,0] # heat from store direct to HD (kWh th)

                                        #%%%% Priority 3: CHP supply heat to Store    
                                        if RLW * R_CHP > 0 and nodes_temp[109] < (store_temp-2): # if store and Residual CHP exists and btm store < heating store temp -2
                                            R_Ht = np.pi * radius_TS**2 * water_height_TS * 1000. * (4.181 / 3600.) * (store_temp - np.mean(nodes_temp[19:110:10]))     
                                            chp2tes[t,0] = max(0.,min(R_Ht, R_CHP)) # heat from CHP to store (kWh th)
                                            R_CHP -= chp2tes[t,0] # Remaining CHP heatout (kWh th)
                                            chp_used[t,0] += chp2tes[t,0] / eff_CHP_thermal # fuel used (kWh)
                                                                                         
                                        if R_CHP > 0: # if redisual CHP heat exists    
                                            waste_heat[t,0] = R_CHP # chp waste heat (kWh th) to optimize CHP size
                                            
                                        if CHPmode == 0: # Power    
                                            chp_used[t,0] += R_CHP / eff_CHP_thermal # Runs at 100% in power priority mode
                                        
                                        chp_elec_out_NW[t,0] = eff_CHP_electrical * chp_used[t,0] # CHP electric output (kWh e)
                                        
                                    if Av_WS > 0: # if wind surplus (CHP on)
                                    
                                        #%%%% Priority 4: Store supply heat to HD
                                        if RLW * Res_HD > 0: # if store and residual HD exists
                                        
                                            if nodes_temp[19] >= T_HD: # if Residual HD exists and top store >= supply HD temp
                                                tes2HD[t,0] = Res_HD # heat from store to HD (kWh th)
                                                Res_HD -= tes2HD[t,0] # Residual HD (kWh th)
                                                tesD2HD[t,0] = tes2HD[t,0] # heat from store direct to HD (kWh th)
                                            
                                            #%%%% Priority 5: Store + CHP supply heat to HD
                                            elif R_CHP * Res_HD > 0 and (T_HD - deltaT) < nodes_temp[19] < T_HD: # if Residual CHP and HD exists and return HD < top store < supply HD temp      
                                                tes2HD[t,0] = (nodes_temp[19] - (T_HD - deltaT)) / (T_HD - (T_HD - deltaT)) * Res_HD # heat from store to HD (kWh th)
                                                chpBH2HD[t,0] = min(R_CHP, Res_HD - tes2HD[t,0]) # CHP boost heat from store to HD (kWh th)                                      
                                                Res_HD -= (tes2HD[t,0] + chpBH2HD[t,0]) # Residual HD (kWh th)
                                                chp_used[t,0] += chpBH2HD[t,0] / eff_CHP_thermal # fuel used (kWh)
                                                R_CHP -= chpBH2HD[t,0]
                                        
                                        #%%%% Priority 6: CHP supply heat to HD
                                        if R_CHP * Res_HD > 0 : # if Residual CHP and HD exists
                                            chp2HD[t,0] = min(R_CHP, Res_HD) # heat from CHP to HD (kWh th)
                                            Res_HD -= chp2HD[t,0] # Residual HD (kWh th)
                                            R_CHP -= chp2HD[t,0] # Remaining CHP Qout (kWh th) 
                                            chp_used[t,0] += chp2HD[t,0] / eff_CHP_thermal # fuel used (kWh th)
                                            
                                        if R_CHP > 0: # if redisual CHP heat exists    
                                            waste_heat[t,0] = R_CHP # chp waste heat (kWh th) to optimize CHP size
                                            
                                        if CHPmode == 0: # Power    
                                            chp_used[t,0] += R_CHP / eff_CHP_thermal # Runs at 100% in power priority mode
                                                                                         
                                #%%% CHP + post GB case    
                                elif CHPcase == 1:
                                    
                                    if Av_WS == 0: # if wind shortfall (CHP on)
                                    
                                        #%%%% Priority 1: CHP supply heat to HD
                                        if R_CHP * Res_HD > 0: # if Residual CHP and HD exists
                                            chp2HD[t,0] = min(R_CHP, Res_HD) # heat from CHP to HD (kWh th)
                                            Res_HD -= chp2HD[t,0] # Residual HD (kWh th)
                                            R_CHP -= chp2HD[t,0] # Remaining CHP heatout (kWh th)
                                            chp_used[t,0] += chp2HD[t,0] / eff_CHP_thermal # fuel used (kWh)
                                            
                                        #%%%% Priority 2: Store supply heat to HD
                                        if RLW * Res_HD > 0: # if store and residual HD exists
                                        
                                            if nodes_temp[19] >= T_HD: # if Residual HD exists and top store >= supply HD temp
                                                tes2HD[t,0] = Res_HD # heat from store to HD (kWh th)
                                                Res_HD -= tes2HD[t,0] # Residual HD (kWh th)
                                                tesD2HD[t,0] = tes2HD[t,0] # heat from store direct to HD (kWh th)
                                            
                                            #%%%% Priority 3: Store+GB supply heat to HD
                                            elif R_GB * Res_HD > 0 and (T_HD - deltaT) < nodes_temp[19] < T_HD: # if Residual GB and HD exists and return HD < top store < supply HD temp      
                                                tes2HD[t,0] = (nodes_temp[19] - (T_HD - deltaT)) / (T_HD - (T_HD - deltaT)) * Res_HD # heat from store to HD (kWh th)
                                                gbBH2HD[t,0] = min(R_GB, Res_HD - tes2HD[t,0]) # GB boost heat from store to HD (kWh th)                                     
                                                Res_HD -= (tes2HD[t,0] + gbBH2HD[t,0]) # Residual HD (kWh th)
                                                boil_used[t,0] += gbBH2HD[t,0] / eff_GB_thermal # gas boiler fuel used (kWh th)
                                                R_GB -= gbBH2HD[t,0]
                                        
                                        # #%%%% Priority 3: GB supply heat to HD
                                        # if R_GB * Res_HD > 0: # if Residual GB and HD exists
                                        #     gb2HD[t,0] = min(R_GB, Res_HD) # heat from GB to HD (kWh th)
                                        #     Res_HD -= gb2HD[t,0] # Residual HD (kWh th) 
                                        #     R_GB -= gb2HD[t,0] # Remaining GB heatout (kWh th) 
                                        #     boil_used[t,0] += gb2HD[t,0] / eff_GB_thermal # gas boiler fuel used (kWh th)
                                            
                                        #%%%% Priority 4: CHP supply heat to Store 
                                        if RLW * R_CHP > 0 and nodes_temp[109] < (store_temp-2): # if store and Residual CHP exists and btm store < heating store temp -2
                                            R_Ht = np.pi * radius_TS**2 * water_height_TS * 1000. * (4.181 / 3600.) * (store_temp - np.mean(nodes_temp[19:110:10]))     
                                            chp2tes[t,0] = max(0.,min(R_Ht, R_CHP)) # heat from CHP to store (kWh th)
                                            R_CHP -= chp2tes[t,0] # Remaining CHP heatout (kWh th)
                                            chp_used[t,0] += chp2tes[t,0] / eff_CHP_thermal # fuel used (kWh)
                                                                                
                                        if R_CHP > 0: # if redisual CHP heat exists    
                                            waste_heat[t,0] = R_CHP # chp waste heat (kWh th) to optimize CHP size
                                            
                                        if CHPmode == 0: # Power    
                                            chp_used[t,0] += R_CHP / eff_CHP_thermal # Runs at 100% in power priority mode
                                        
                                        chp_elec_out_NW[t,0] = eff_CHP_electrical * chp_used[t,0] # CHP electric output (kWh e)
                                                
                                    if Av_WS > 0: # if wind surplus (CHP off)
                                    
                                        #%%%% Priority 5: Store supply heat to HD
                                        if RLW * Res_HD > 0: # if store and residual HD exists
                                        
                                            if nodes_temp[19] >= T_HD: # if Residual HD exists and top store >= supply HD temp
                                                tes2HD[t,0] = Res_HD # heat from store to HD (kWh th)
                                                Res_HD -= tes2HD[t,0] # Residual HD (kWh th)
                                                tesD2HD[t,0] = tes2HD[t,0] # heat from store direct to HD (kWh th)
                                            
                                            #%%%% Priority 6: Store + GB supply heat to HD
                                            elif R_GB * Res_HD > 0 and (T_HD - deltaT) < nodes_temp[19] < T_HD: # if Residual GB and HD exists and return HD < top store < supply HD temp      
                                                tes2HD[t,0] = (nodes_temp[19] - (T_HD - deltaT)) / (T_HD - (T_HD - deltaT)) * Res_HD # heat from store to HD (kWh th)
                                                gbBH2HD[t,0] = min(R_GB, Res_HD - tes2HD[t,0]) # GB boost heat from store to HD (kWh th)                                      
                                                Res_HD -= (tes2HD[t,0] + gbBH2HD[t,0]) # Residual HD (kWh th)
                                                boil_used[t,0] += gbBH2HD[t,0] / eff_GB_thermal # fuel used (kWh)
                                                R_GB -= gbBH2HD[t,0]
                                        
                                        #%%%% Priority 7: GB supply heat to HD
                                        if R_GB * Res_HD > 0 : # if Residual GB and HD exists
                                            gb2HD[t,0] = min(R_GB, Res_HD) # heat from GB to HD (kWh th)
                                            Res_HD -= gb2HD[t,0] # Residual HD (kWh th)
                                            R_GB -= gb2HD[t,0] # Remaining GB Qout (kWh th) 
                                            boil_used[t,0] += gb2HD[t,0] / eff_GB_thermal # gas boiler fuel used (kWh th)
                                                                                                                                     
                                Un_H[t,0] = Res_HD # Unmet heat demand (kWh th)  
                                if Un_H[t,0] > 1e-10:
                                    print(t, Un_H[t,0]) # check unmet heat demand                                
                                                               
                                if R_GB > 0: # if residual GB heat exists
                                    Res_GB[t,0] = R_GB # GB residual heat (kWh th) to optimize GB size
                                    
                                chp_elec_out[t,0] = eff_CHP_electrical * chp_used[t,0] # CHP electric output (kWh e)
                                chp_export_income[t,0] = Tariff[t,0] * chp_elec_out[t,0] # chp export income (£)
                                
                                #%%% Results                         
                                total_chp_heat_out[t,0] = chp2HD[t,0] + chp2tes[t,0] + chpBH2HD[t,0] # total heatout from chp (kWh th) 
                                total_gb_heat_out[t,0] = gb2HD[t,0] + gbBH2HD[t,0] # total heatout from gb (kWh th) 
                                total_Qc[t,0] = chp2tes[t,0] # total heat charge to store (kWh th)
                                total_Qex[t,0] = tes2HD[t,0] # total heat extract from store (kWh th)
                                total_fuel_used[t,0] = chp_used[t,0] + boil_used[t,0] # (kWh)
                                total_fuel_cost[t,0] = chp_cost * chp_used[t,0] + boil_cost * boil_used[t,0] # (£)
                                
                                #%%% Set configurations for shatstore model
                                # Extra Charge/Discharge Condition in Store
                                if RLW > 0: 
                                    if total_Qc[t,0] - total_Qex[t,0] > 0.: # Charge > Discharge
                                        charge = total_Qc[t,0] - total_Qex[t,0]
                                        discharge = 0.
                                    elif total_Qc[t,0] - total_Qex[t,0] < 0.: # Discharge > Charge
                                        charge = 0.
                                        discharge = -(total_Qc[t,0] - total_Qex[t,0])
                                    else:
                                        charge = 0.
                                        discharge = 0.
                                
                                # Inputs for shatstore model
                                    return_temp = max(T_HD - deltaT, nodes_temp[19] - deltaT) # Return temperature to store (DegC) 
                                    
                                    if charge == 0:                                 
                                        next_nodes_temp, Hloss = shaftstore_5a.ShaftStore(Rx, XTC, number_nodes, number_layers, RLA, RLW, deltaT).new_nodes_temp(nodes_temp, store_temp, return_temp, charge, discharge, MTS, deltaT) # Calculate new store temperatures at end of timestep
                                        nodes_temp = next_nodes_temp[1]
                                    else:
                                        R_Ch = charge
                                        Hloss = 0.
                                        while R_Ch > 0.:
                                            IntC = min(R_Ch, 0.5 * np.pi * radius_TS**2 * RLW * 1000. * (4.181 / 3600.) * (store_temp - nodes_temp[109]))
                                            xMTS = MTS * (IntC / charge)
                                            next_nodes_temp, IntHs = shaftstore_5a.ShaftStore(Rx, XTC, number_nodes, number_layers, RLA, RLW, deltaT).new_nodes_temp(nodes_temp, store_temp, return_temp, IntC, discharge, xMTS, deltaT) # Calculate new store temperatures at end of timestep
                                            nodes_temp = next_nodes_temp[1]
                                            R_Ch -= IntC
                                            Hloss += IntHs
                                    
                                    Qloss_S_g[t,0] = Hloss # heat loss from store (kWh th)
                                    
                                    f_n_t[tr,:] = nodes_temp # Full Temperature Results: 11 layers with 8 x earth nodes, 1 x concrete wall node, 1 air/stored water node
                                                        
                                #%%% End of the for loop
                                # print(tr) # print no. of simulation/ timestep counter
                                tr = tr + 1 # add no. of simulation
                                
                            #%%% Specific node temperature
                            if RLW > 0: 
                                # f_n_t = [0:109]
                                
                                # all exclude the top air layer nodes [0] to [9]
                                # f_gor_t = f_n_t[:,10:101:10]  # Store Ground Outer Ring (131.5-259.5m) Temperature Results only
                                # f_gr2_t = f_n_t[:,11:102:10]  # Store Ground Ring 2 (67.5-131.5m) Temperature Results only
                                # f_gr3_t = f_n_t[:,12:103:10]  # Store Ground Ring 3 (35.5-67.5m) Temperature Results only
                                # f_gr4_t = f_n_t[:,13:104:10]  # Store Ground Ring 4 (19.5-35.5m) Temperature Results only
                                # f_gr5_t = f_n_t[:,14:105:10]  # Store Ground Ring 5 (11.5-19.5m) Temperature Results only
                                # f_gr6_t = f_n_t[:,15:106:10]  # Store Ground Ring 6 (7.-11.5m) Temperature Results only
                                # f_gr7_t = f_n_t[:,16:107:10]  # Store Ground Ring 7 (5.5-7.5m) Temperature Results only
                                f_gr8_t = f_n_t[:,17:108:10]  # Store Ground Ring 8 (4.5-5.5m) Temperature Results only
                                f_csw_t = f_n_t[:,18:109:10]  # Store Concrete Shaft Wall (3.5-4.5m) Temperature Results only
                                f_s_t   = f_n_t[:,19:110:10]  # Store Fluid inside shaft (0-3.5m) Temperature Results only
                                
                                # if want to include top air layer nodes
                                # f_gor_t = f_n_t[:,0:108:10] # Store Ground Outer Ring (131.5-259.5m) Temperature Results only
                                # f_gr2_t = f_n_t[:,1:109:10] # Store Ground Ring 2 (67.5-131.5m) Temperature Results only
                                # f_s_t   = f_n_t[:,9:110:10] # Store Fluid inside shaft (0-3.5m) Temperature Results only
                        
                            #%%% Economic Analysis
                            HeatOpex = total_fuel_cost - chp_export_income
                            CHP_CAPEX = 4000 # CAPEX of CHP (£/kW) 3-5.5k
                            GB_CAPEX = 100 # CAPEX of Gas Boiler (£/kW) 1.5-3.4k
                            
                            if RLW == 0:
                                TS_CAPEX = 0
                            else:
                                TS_CAPEX = 7982*TS_volume**-0.483 # (£100/m3) fixed estimated price but should be exponential graph
                                
                            CAPEX = Size_CHP * CHP_CAPEX + Size_GB * GB_CAPEX + TS_volume * TS_CAPEX # Total Capital Expenses (£)
                            HPEX = Size_CHP * CHP_CAPEX + Size_GB * GB_CAPEX
                            
                            Maintenance = CAPEX * 0.02 # O&M costs usually 2-5% of CAPEX (£/yr.)
                        
                            CP = np.zeros(lifetime)
                            OP = np.zeros(lifetime)
                            OM = np.zeros(lifetime)
                            ET = np.zeros(lifetime)

                            OpH = np.zeros(lifetime) # Yearly operating costs (£)
                            for yr1 in range(Mod_Year):
                                OpH[yr1] = np.sum(HeatOpex[yr1*17520:(yr1+1)*17520,0]) # Op cost for modelled years
                              
                            for yr2 in range(Mod_Year,lifetime):
                                OpH[yr2] = OpH[Mod_Year-1] # Copy last op cost for unmodelled years      

                            CapI = CAPEX * CAfactor # Capital Costs
                            CP[0] = CapI
                            if lifetime > 20:
                                CP[20] = HPEX # replacement CHPs after year 20
                            if lifetime > 40:
                                CP[40] = HPEX # replacement CHPs after year 40
                            
                            # Calculation
                            # for yr in range(lifetime):
                            #     CP[yr] = CP[yr] * (((1+Inf)**(yr+0.5)) / ((1+DR)**(yr+0.5))) # Additional capital after operation begins
                            #     OP[yr] = OpH[yr] * (((1+Inf)**(yr+0.5)) / ((1+DR)**(yr+0.5))) # Operating (Power) Costs
                            #     OM[yr] = Maintenance * (((1+Inf)**(yr+0.5)) / ((1+DR)**(yr+0.5))) # Maintenance
                            # LCOH = ((CapI + np.sum(CP) + np.sum(OP) + np.sum(OM))) / (sum(DemH[17520*(Mod_Year-1):17520*Mod_Year]) * lifetime) # £/kWh
                        
                            DR = 0.05 # Discount Rate
                            for yr in range(lifetime):
                                CP[yr] = CP[yr] / ((1+DR)**(yr+1))
                                OP[yr] = OpH[yr] / ((1+DR)**(yr+1)) # Operating (Power) Costs
                                OM[yr] = Maintenance / ((1+DR)**(yr+1)) # Maintenance
                                ET[yr] = (sum(DemH[17520*(Mod_Year-1):17520*Mod_Year]))/((1+DR)**(yr+1)) # Total annual heat demand
                            LCOH = (np.sum(CP) + np.sum(OP) + np.sum(OM)) / np.sum(ET) # £/kWh
                            
                            DR = 0.05 # Discount Rate
                            Inf = 0.02 # Inflation  
                            DR = ((1+DR)*(1+Inf))-1 # nominal discounts rates with inflation
                            for yr in range(lifetime):
                                CP[yr] = CP[yr] / ((1+DR)**(yr+1))
                                OP[yr] = OpH[yr] / ((1+DR)**(yr+1)) # Operating (Power) Costs
                                OM[yr] = Maintenance / ((1+DR)**(yr+1)) # Maintenance
                                ET[yr] = (sum(DemH[17520*(Mod_Year-1):17520*Mod_Year]))/((1+DR)**(yr+1)) # Total annual heat demand
                            LCOH2 = (np.sum(CP) + np.sum(OP) + np.sum(OM)) / np.sum(ET) # £/kWh
                        
                            DR = 0.078 # Discount Rate
                            for yr in range(lifetime):
                                CP[yr] = CP[yr] / ((1+DR)**(yr+1))
                                OP[yr] = OpH[yr] / ((1+DR)**(yr+1)) # Operating (Power) Costs
                                OM[yr] = Maintenance / ((1+DR)**(yr+1)) # Maintenance
                                ET[yr] = (sum(DemH[17520*(Mod_Year-1):17520*Mod_Year]))/((1+DR)**(yr+1)) # Total annual heat demand
                            LCOH3 = (np.sum(CP) + np.sum(OP) + np.sum(OM)) / np.sum(ET) # £/kWh

                            DR = 0.078 # Discount Rate
                            Inf = 0.02 # Inflation  
                            DR = ((1+DR)*(1+Inf))-1 # nominal discounts rates with inflation
                            for yr in range(lifetime):
                                CP[yr] = CP[yr] / ((1+DR)**(yr+1))
                                OP[yr] = OpH[yr] / ((1+DR)**(yr+1)) # Operating (Power) Costs
                                OM[yr] = Maintenance / ((1+DR)**(yr+1)) # Maintenance
                                ET[yr] = (sum(DemH[17520*(Mod_Year-1):17520*Mod_Year]))/((1+DR)**(yr+1)) # Total annual heat demand
                            LCOH4 = (np.sum(CP) + np.sum(OP) + np.sum(OM)) / np.sum(ET) # £/kWh
                            
                            #%% Generate outputs
                            #%%% Create folder in specific location
                            def process_folder(location, folder_name):
                                folder_path = os.path.join(location, folder_name)
                                # Check if the folder exists
                                folder_exists = os.path.exists(folder_path)
                                if not folder_exists:
                                    os.makedirs(folder_path)
                            
                            # Set base location and define folder names
                            # base_location = "/Users/cmb22235/OneDrive - University of Strathclyde/Desktop/STEaM WP4 team/MSTES-CHP/Win 1D Model/results/" 
                            base_location = "/Users/cmb22235/OneDrive - University of Strathclyde/Desktop/" 
                            folder_name = f'MY{Mod_Year},CM{CHPmode},CC{CHPcase},CA{CAfactor},ST{store_temp},TES{water_height_TS},XTf{int(NW_tariff*100)},eff{int(eff_CHP_electrical*100)}&{int(eff_CHP_thermal*100)},FC{(chp_cost*100)}'
                            location_1 = os.path.join(base_location, folder_name)
                            process_folder(base_location, folder_name)
                            
                            folder_kpi = 'kpi'
                            kpi_location = os.path.join(location_1, folder_kpi)
                            process_folder(location_1, folder_kpi)
                            
                            folder_full = 'full'
                            full_location = os.path.join(location_1, folder_full)
                            process_folder(location_1, folder_full)
                            
                            folder_year = 'year'
                            year_location = os.path.join(location_1, folder_year)
                            process_folder(location_1, folder_year)
                            
                            folder_summer = 'summer'
                            summer_location = os.path.join(location_1, folder_summer)
                            process_folder(location_1, folder_summer)
                            
                            folder_winter = 'winter'
                            winter_location = os.path.join(location_1, folder_winter)
                            process_folder(location_1, folder_winter)
                            
                            folder_summer3d = 'summer3d'
                            summer3d_location = os.path.join(location_1, folder_summer3d)
                            process_folder(location_1, folder_summer3d)
                            
                            folder_winter3d = 'winter3d'
                            winter3d_location = os.path.join(location_1, folder_winter3d)
                            process_folder(location_1, folder_winter3d)
                            
                            results_location = base_location + folder_name
                            
                            #%%% Time For Each Season
                            # 1 day  = 24h * 2data/hr = 48  
                            # 3 day  = 48*3 = 144   
                            # 1 week = 48*7 = 336
                            # 1 year = 365 days * 24hr = 8760 hr * 2 data per hour -> 17520   
                            # data start from [0]
                            # spring = [ 2832 :  7247]
                            # summer = [ 7248 : 11663]
                            # fall   = [11664 : 16031] 
                            # winter = [16032:17519] & [0:2831]
                            
                            # def time(start_hour,end_hour):
                            #     time  = range(start_hour,end_hour)
                            #     return time                            
                            
                            def pick_time(start_hour,end_hour):
                                pick_time  = list(range(start_hour,end_hour))
                                return pick_time
                            
                            #%%% node temp csv header + data
                            if RLW > 0:    
                                header_node=[]
                                # # Option 1
                                # for layer in range(number_layers):
                                #     for node in range(number_nodes):
                                #         nm = "L" + str(layer) + "_N" + str(node)
                                #         header = np.append(header,nm)
                                # Option 2    
                                ndnm = ["Grd1", "Grd2", "Grd3", "Grd4", "Grd5", "Grd6", "Grd7", "Grd8", "Wall", "Store","Down","Up"]
                                for layer in range(number_layers):
                                    for node in range(number_nodes):
                                        nm = "L" + str(layer) + "_" + str(ndnm[node])
                                        header_node = np.append(header_node,nm)
                                filename = f'f_n_t{Mod_Year}yrs{store_temp}{DegC}.csv'        
                                np.savetxt(os.path.join(location_1, filename), 
                                            f_n_t, delimiter=',', header=', '.join(header_node), fmt='%f', comments='')

                            #%%% Extend data arrays for the entire lifetime
                            for exyr in range(lifetime - Mod_Year):
                                DemH = np.append(DemH, DemH[int(nt - (nt / Mod_Year)):nt])
                                WS = np.append(WS, WS[int(nt - (nt / Mod_Year)):nt])
                                chp2HD = np.append(chp2HD, chp2HD[int(nt - (nt / Mod_Year)):nt])
                                gb2HD = np.append(gb2HD, gb2HD[int(nt - (nt / Mod_Year)):nt])
                                tes2HD = np.append(tes2HD, tes2HD[int(nt - (nt / Mod_Year)):nt])
                                tesD2HD = np.append(tesD2HD, tesD2HD[int(nt - (nt / Mod_Year)):nt])
                                chpBH2HD = np.append(chpBH2HD, chpBH2HD[int(nt - (nt / Mod_Year)):nt])
                                gbBH2HD = np.append(gbBH2HD, gbBH2HD[int(nt - (nt / Mod_Year)):nt])
                                chp2tes = np.append(chp2tes, chp2tes[int(nt - (nt / Mod_Year)):nt])
                                total_chp_heat_out = np.append(total_chp_heat_out, total_chp_heat_out[int(nt - (nt / Mod_Year)):nt])
                                total_gb_heat_out = np.append(total_gb_heat_out, total_gb_heat_out[int(nt - (nt / Mod_Year)):nt])
                                chp_used = np.append(chp_used, chp_used[int(nt - (nt / Mod_Year)):nt])
                                boil_used = np.append(boil_used, boil_used[int(nt - (nt / Mod_Year)):nt])
                                total_fuel_used = np.append(total_fuel_used, total_fuel_used[int(nt - (nt / Mod_Year)):nt])
                                waste_heat = np.append(waste_heat, waste_heat[int(nt - (nt / Mod_Year)):nt])
                                Qloss_S_g = np.append(Qloss_S_g, Qloss_S_g[int(nt - (nt / Mod_Year)):nt])
                                total_Qc = np.append(total_Qc, total_Qc[int(nt - (nt / Mod_Year)):nt])
                                total_fuel_cost = np.append(total_fuel_cost, total_fuel_cost[int(nt - (nt / Mod_Year)):nt])
                                chp_elec_out = np.append(chp_elec_out, chp_elec_out[int(nt - (nt / Mod_Year)):nt])
                                chp_export_income = np.append(chp_export_income, chp_export_income[int(nt - (nt / Mod_Year)):nt])
                                chp_elec_out_NW = np.append(chp_elec_out_NW, chp_elec_out_NW[int(nt - (nt / Mod_Year)):nt])
    
                            #%%% Kpi list
                            kpi_header = [ 'Modelling year',
                                           'CHP Mode',
                                           'CHP case',
                                           'Store height (m)',
                                           'Size of CHP (MW)',
                                           'Size of GB (MW)',
                                           f'Heat Demand Temperature ({DegC})',
                                           f'Heating Store Temperature ({DegC})',
                                           'Wind Surplus/Low export tariff (£/kWh)',
                                           'Wind Shortfall/High export tariff (£/kWh)',
                                           'CHP electrical eff (%)',
                                           'CHP thermal eff (%)',
                                           'Fuel cost (£/kWh)',
                            
                                           'Wind Surplus (GWh e)',
                                           'Heat Demand (GWh th)',
                                           'Heatout from CHP to HD (GWh th)',
                                           'Heatout from gb to hd (GWh th)',
                                           'Heatout from store to hd (GWh th)',
                                           'Heatout from store direct to hd (GWh th)',
                                           'CHP boost to Heatout from store(GWh th)',
                                           'GB boost to Heatout from store(GWh th)',
                                           'Heatout from CHP to Store (GWh th)',
                                           'Total heatout from CHP (GWh th)',
                                           'Total heatout from gb (GWh th)',
                                           'CHP Fuel used (GWh)',
                                           'gas boiler fuel used (GWh)',
                                           'Total fuel used (GWh)',
                                           'CHP waste heat (GWh th)',
                                           'Store heat loss (GWh th)',
                                           'Total heat charge to store (GWh th)',
                                           'Total fuel cost (£ mil)',
                                           'Elec export from CHP (GWh e)',
                                           'CHP export income (£ mil)',
                                           'Elec export from CHP in Non-Wind Periods (GWh e)',
                                           'FlexPA(CHP)',
                                           
                                           'CAPEX (£ mil)',
                                           'heatout from CHP to hd / total heatout from chp (%)',
                                           'heatout from CHP to Store / total heatout from chp (%)',
                                           'heatout from CHP to hd / total heat demand (%)',
                                           'heatout from GB to hd / total heat demand (%)',
                                           'heatout from Store to hd / total heat demand (%)',
                                           'heatout from Store direct to hd / total heat demand (%)',
                                           'GB boost to heatout from Store / total heat demand (%)',
                                           'total heatout from CHP/ fuel used (%)',
                                           'total heatout from GB / fuel used (%)',
                                           'elec export from CHP / fuel used (%)',
                                           'Store heat loss / charge (%)',
                                           'heat supply eff = heat demand/fuel used (%)',
                                           'FLEX(CHP) (%)',
                                           'LCOHwd5 (£/kWh)', 
                                           'LCOHwd5Inf (£/kWh)',
                                           'LCOHwd8 (£/kWh)', 
                                           'LCOHwd8Inf (£/kWh)'
                                           ]
                            
                            def kpi_stack_last(pick_time):       
                                kpi_stack_last = np.column_stack((Mod_Year,
                                    CHPmode,   
                                    CHPcase,
                                    water_height_TS, # TES height (m)
                                    Size_CHP/1000, # Size of CHP (MW)
                                    Size_GB/1000,  # Size of GB (MW)
                                    T_HD,          # Heat Demand Temperature ({DegC})
                                    store_temp,    # Heating Store Temperature ({DegC})
                                    WS_tariff,     # Wind Surplus/Low export tariff (£/kWh)
                                    NW_tariff,     # Wind Shortfall/High export tariff (£/kWh)
                                    int(eff_CHP_electrical*100),
                                    int(eff_CHP_thermal*100),
                                    chp_cost,
                                     
                                    # np.sum(WS[pick_time])/(10**6)*lifetime,      # Wind Surplus (GWh e)              
                                    # np.sum(DemH[pick_time])/(10**6)*lifetime,    # Heat Demand (GWh th)
                                    # np.sum(chp2HD[pick_time])/(10**6)*lifetime,  # Heatout from CHP to hd (GWh th)
                                    # np.sum(gb2HD[pick_time])/(10**6)*lifetime,   # Heatout from gb to hd (GWh th)
                                    # np.sum(tes2HD[pick_time])/(10**6)*lifetime,  # Heatout from Store to hd (GWh th)
                                    # np.sum(tesD2HD[pick_time])/(10**6)*lifetime, # Heatout from Store direct to hd (GWh th)
                                    # np.sum(chpBH2HD[pick_time])/(10**6)*lifetime,# CHP boost heat from store to HD (kWh th) 
                                    # np.sum(gbBH2HD[pick_time])/(10**6)*lifetime, # GB boost heat from store to HD (kWh th) 
                                    # np.sum(chp2tes[pick_time])/(10**6)*lifetime, # Heatout from CHP to Store (GWh th)
                                    # np.sum(total_chp_heat_out[pick_time])/(10**6)*lifetime, # Total heatout from CHP (GWh th)
                                    # np.sum(total_gb_heat_out[pick_time])/(10**6)*lifetime,  # total heatout from gb (GWh th) 
                                    # np.sum(chp_used[pick_time])/(10**6)*lifetime,     # chp fuel used (GWh)
                                    # np.sum(boil_used[pick_time])/(10**6)*lifetime,    # gas boiler fuel used (GWh)
                                    # np.sum(total_fuel_used[pick_time])/(10**6)*lifetime, # total fuel used (GWh)
                                    # np.sum(waste_heat[pick_time])/(10**6)*lifetime,   # chp waste heat (GWh th)
                                    # np.sum(Qloss_S_g[pick_time])/(10**6)*lifetime,    # store heat loss (GWh th)
                                    # np.sum(total_Qc[pick_time])/(10**6)*lifetime,     # total heat charge to store (GWh th)
                                    # np.sum(total_fuel_cost[pick_time])/(10**6)*lifetime,   # total fuel cost (£) x (10**6)                                
                                    # np.sum(chp_elec_out[pick_time])/(10**6)*lifetime,      # elec export from CHP (GWh e)
                                    # np.sum(chp_export_income[pick_time])/(10**6)*lifetime, # chp export income (£)
                                    # np.sum(chp_elec_out_NW[pick_time])/(10**6)*lifetime,      # elec export from CHP in Non-Wind Periods (GWh e)
    
                                    np.sum(WS[pick_time])/(10**6),      # Wind Surplus (GWh e)              
                                    np.sum(DemH[pick_time])/(10**6),    # Heat Demand (GWh th)
                                    np.sum(chp2HD[pick_time])/(10**6),  # Heatout from CHP to hd (GWh th)
                                    np.sum(gb2HD[pick_time])/(10**6),   # Heatout from gb to hd (GWh th)
                                    np.sum(tes2HD[pick_time])/(10**6),  # Heatout from Store to hd (GWh th)
                                    np.sum(tesD2HD[pick_time])/(10**6), # Heatout from Store direct to hd (GWh th)
                                    np.sum(chpBH2HD[pick_time])/(10**6),# CHP boost heat from store to HD (kWh th) 
                                    np.sum(gbBH2HD[pick_time])/(10**6), # GB boost heat from store to HD (kWh th) 
                                    np.sum(chp2tes[pick_time])/(10**6), # Heatout from CHP to Store (GWh th)
                                    np.sum(total_chp_heat_out[pick_time])/(10**6), # Total heatout from CHP (GWh th)
                                    np.sum(total_gb_heat_out[pick_time])/(10**6),  # total heatout from gb (GWh th) 
                                    np.sum(chp_used[pick_time])/(10**6),     # chp fuel used (GWh)
                                    np.sum(boil_used[pick_time])/(10**6),    # gas boiler fuel used (GWh)
                                    np.sum(total_fuel_used[pick_time])/(10**6), # total fuel used (GWh)
                                    np.sum(waste_heat[pick_time])/(10**6),   # chp waste heat (GWh th)
                                    np.sum(Qloss_S_g[pick_time])/(10**6),    # store heat loss (GWh th)
                                    np.sum(total_Qc[pick_time])/(10**6),     # total heat charge to store (GWh th)
                                    np.sum(total_fuel_cost[pick_time])/(10**6),   # total fuel cost (£) x (10**6)                                
                                    np.sum(chp_elec_out[pick_time])/(10**6),      # elec export from CHP (GWh e)
                                    np.sum(chp_export_income[pick_time])/(10**6), # chp export income (£)
                                    np.sum(chp_elec_out_NW[pick_time])/(10**6),      # elec export from CHP in Non-Wind Periods (GWh e)   
                                    (np.sum(chp_elec_out_NW[pick_time])/(10**6))/lifetime,      # FlexPA(CHP) (GWh e)   
    
                                    CapI/(10**6),                                                             # CAPEX (£)
                                    (np.sum(chp2HD[pick_time])/np.sum(total_chp_heat_out[pick_time]))*100,    # heatout from CHP to hd / total heatout from chp (%)
                                    (np.sum(chp2tes[pick_time])/np.sum(total_chp_heat_out[pick_time]))*100,   # heatout from CHP to Store / total heatout from chp (%)
                                    (np.sum(chp2HD[pick_time])/np.sum(DemH[pick_time]))*100,                  # heatout from CHP to hd / total heat demand (%)
                                    (np.sum(gb2HD[pick_time])/np.sum(DemH[pick_time]))*100,                   # heatout from GB to hd / total heat demand (%)
                                    (np.sum(tes2HD[pick_time])/np.sum(DemH[pick_time]))*100,                  # heatout from Store to hd / total heat demand (%)
                                    (np.sum(tesD2HD[pick_time])/np.sum(DemH[pick_time]))*100,                 # heatout from Store direct to hd / total heat demand (%)
                                    (np.sum(gbBH2HD[pick_time])/np.sum(DemH[pick_time]))*100,                 # GB boost heat from store to HD/ total heat demand (%)
                                    (np.sum(total_chp_heat_out[pick_time])/np.sum(chp_used[pick_time]))*100,  # total heatout from CHP/ fuel used (%)
                                    (np.sum(total_gb_heat_out[pick_time])/np.sum(boil_used[pick_time]))*100,  # total heatout from GB / fuel used (%)
                                    (np.sum(chp_elec_out[pick_time])/np.sum(chp_used[pick_time]))*100,        # elec export from CHP / fuel used (%)
                                    (np.sum(Qloss_S_g[pick_time])/np.sum(total_Qc[pick_time]))*100,           # store heat loss / total heat charge to store (%)
                                    (np.sum(DemH[pick_time])/np.sum(total_fuel_used[pick_time]))*100,         # heat supply eff, heat demand/fuel used (%)
                                    (np.sum(chp_elec_out_NW[pick_time])/np.sum(chp_elec_out[pick_time]))*100, # CHP elec export in NW / total CHP elec export (%)
                                    LCOH,                                                                     # LCOHwd5 (£/kWh) 
                                    LCOH2,                                                                    # LCOHwd5Inf (£/kWh) 
                                    LCOH3,                                                                    # LCOHwd8 (£/kWh) 
                                    LCOH4                                                                     # LCOHwd8Inf (£/kWh) 
                                    ))
                                return kpi_stack_last
    
                            def kpi_run_last(start_hour, end_hour, duration):
                                header = kpi_header
                                filename = f'kpi_{duration}.csv'
                                np.savetxt(f"{kpi_location}/{filename}", 
                                           kpi_stack_last(pick_time(start_hour, end_hour)), 
                                           delimiter=',', 
                                           header=', '.join(header), 
                                           fmt='%f', 
                                           comments='')
                            
                            def kpi_last():
                                start_hour = 0
                                end_hour = 17520 * lifetime
                                duration = 'last'
                                kpi_run_last(start_hour, end_hour, duration)
                            kpi_last()
    
                            #%%% enflow csv header+stack
                            if plot == 1:
                                enflow_header = [ 'Wind Gen (kWh e)',
                                'Electric Demand (kWh e)',
                                'Wind Surplus (kWh e)',
                                'Heat Demand (kWh th) ',
                                'tariff (£/kWh)',
                                'heat from CHP to HD (kWh th)',
                                'chp fuel used (kWh)',
                                'gas boiler fuel used (kWh)',
                                'CHP electric output (kWh e)',
                                'chp export income (£)',
                                'heat from GB to HD (kWh th)',
                                'heat from CHP to store (kWh th)',
                                'heat from store direct to HD (kWh th)',
                                'heat from store to HD (kWh th)',
                                'GB boost heat from store to HD (kWh th)',
                                'unmet heat demand (kWh th)',
                                'chp waste heat (kWh th)',
                                'GB residual heat (kWh th)',
                                'total heatout from chp (kWh th)',
                                'total heatout from gb (kWh th)',
                                'total heat charge to store (kWh th)',
                                'total heat extract from store (kWh th)',
                                'total fuel cost (£)',
                                'heat loss from store (kWh th)'
                                                ]
                                
                                def enflow_stack(pick_time):
                                    enflow_stack = np.column_stack ((Wind_Gen[pick_time], # Wind Gen (kWh e)
                                    Elec_Dem[pick_time],          # Electric Demand (kWh e)
                                    WS[pick_time],                # Wind Surplus (kWh e)
                                    DemH[pick_time],              # Heat Demand (kWh th) 
                                    Tariff[pick_time],            # tariff (£/kWh)
                                    chp2HD[pick_time],            # heat from CHP to HD (kWh th)
                                    chp_used[pick_time],          # fuel used (kWh)
                                    boil_used[pick_time],         # gas boiler fuel used (kWh th)
                                    chp_elec_out[pick_time],      # CHP electric output (kWh e)
                                    chp_export_income[pick_time], # chp export income (£)
                                    gb2HD[pick_time],             # heat from GB to HD (kWh th)
                                    chp2tes[pick_time],           # heat from CHP to store (kWh th)
                                    tesD2HD[pick_time],           # heat from store direct to HD (kWh th)
                                    tes2HD[pick_time],            # heat from store to HD (kWh th)
                                    gbBH2HD[pick_time],           # GB boost heat from store to HD (kWh th) 
                                    Un_H[pick_time],              # unmet heat demand (kWh th)
                                    waste_heat[pick_time],        # chp waste heat (kWh th)
                                    Res_GB[pick_time],            # GB residual heat (kWh th)
                                    total_chp_heat_out[pick_time],# total heatout from chp (kWh th)   
                                    total_gb_heat_out[pick_time], # total heatout from chp (kWh th)   
                                    total_Qc[pick_time],          # total heat charge to store (kWh th)
                                    total_Qex [pick_time],        # total heat extract from store (kWh th)
                                    total_fuel_cost[pick_time],   # total fuel cost (£)
                                    Qloss_S_g[pick_time],         # heat loss from store (kWh th)
                                        ))
                                    return enflow_stack
                               
                                #%%% plot graphs   
                                plt.style.use('ggplot')
                                plt.rcParams.update({'font.size': 6})
                                 
                                def plot_f_s_t(pick_time,x_ticks,x_labels,duration,xaxis): 
                                    if RLW > 0: # if store exists
                                        
                                        fig = plt.figure(figsize=(5,3))
                                        plt.plot(f_s_t[pick_time], ls = '-', lw = '0.5')
                                        plt.title(f"Water Temp for {store_temp}{DegC} Store ({water_height_TS}m)")
                                        plt.xticks(ticks=x_ticks, labels=x_labels)
                                        plt.xlabel(f"Time {xaxis}")
                                        plt.ylabel(f"Node tempearture for each layer ({DegC})")
                                        # plt.legend(['L1','L2','L3','L4','L5','L6','L7','L8','L9','L10'],loc='center left', bbox_to_anchor=(1,0.5), fancybox = True, prop={'size': 8})
                                        plt.legend(['L1','L2','L3','L4','L5','L6','L7','L8','L9','L10'],loc = 'upper left', bbox_to_anchor=(0,-0.15), ncol = 5, fancybox = True, prop={'size': 8})   
                                        
                                        filename = f'{results_location}/{duration}/f_s_t {duration}.png'
                                        plt.savefig(filename, format = 'png',dpi=300, bbox_inches='tight')
                                        fig.clear()
                                        plt.close(fig) 
                                        
                                def plot_f_s_t_half(pick_time,x_ticks,x_labels,duration,xaxis): 
                                    if RLW > 0: # if store exists
                                        
                                        fig = plt.figure(figsize=(5,3))
                                        plt.plot(f_n_t[:,19][pick_time], ls = '-', lw = '0.5')
                                        plt.plot(f_n_t[:,59][pick_time], ls = '-', lw = '0.5')
                                        plt.plot(f_n_t[:,109][pick_time], ls = '-', lw = '0.5')
                                        plt.title(f"Water Temp for {store_temp}{DegC} Store ({water_height_TS}m)")
                                        plt.xticks(ticks=x_ticks, labels=x_labels)
                                        plt.xlabel(f"Time {xaxis}")
                                        plt.ylabel(f"Node tempearture for each layer ({DegC})")
                                        # plt.legend(['L1','L5','L10'],loc='center left', bbox_to_anchor=(1,0.5), fancybox = True, prop={'size': 8})
                                        plt.legend(['L1','L5','L10'],loc='upper left', ncols=3, bbox_to_anchor=(0,-0.15), fancybox = True, prop={'size': 8})
                                        
                                        filename = f'{results_location}/{duration}/f_s_t_half {duration}.png'
                                        plt.savefig(filename, format = 'png',dpi=300, bbox_inches='tight')
                                        fig.clear()
                                        plt.close(fig)         
                                      
                                
                                def plot_chp(pick_time,x_ticks,x_labels,duration,xaxis): 
                                    fig = plt.figure(figsize=(5,7))
                                    
                                    plt.subplot(3,1,1)
                                    plt.plot(Tariff[pick_time], label='Tariff', ls = '-', lw = '0.5', c = c[0])
                                    plt.title(f"Tariff {duration}", )
                                    plt.ylabel("Tariff (£/kWh e)", )
                                    plt.xticks(ticks=x_ticks, labels="", )
    
                                    plt.subplot(3,1,2)
                                    plt.plot(chp_used[pick_time]*2, label='Fuel', ls = '-', lw = '0.5', c = c[1])
                                    plt.plot(total_chp_heat_out[pick_time]*2, label='Heat Export', ls = '-', lw = '0.5', c = c[0])
                                    plt.plot(chp_elec_out[pick_time]*2, label='Elec Export', ls = '-', lw = '0.5', c = c[2])
                                    plt.title(f"CHP {duration}", )
                                    plt.ylabel("Power (kW)", )
                                    plt.xticks(ticks=x_ticks, labels="", )
                                    plt.legend(loc = 'center left', bbox_to_anchor=(1,0.5), fancybox = True, prop={'size': 8})
    
                                    plt.subplot(3,1,3)
                                    plt.plot(total_chp_heat_out[pick_time]*2, label='Heat Export', ls = '-', lw = '0.5', c = c[0])
                                    plt.plot(chp2HD[pick_time]*2, label='CHP to HD', ls = '-', lw = '0.5', c = c[3])
                                    plt.plot(chp2tes[pick_time]*2, label='CHP to Store', ls = '-', lw = '0.5', c = c[4])
                                    plt.title(f"CHP heat {duration}", )
                                    plt.ylabel("Power (kW th)", )
                                    plt.legend(loc = 'center left', bbox_to_anchor=(1,0.5), fancybox = True, prop={'size': 8})
                                    plt.xlabel(f"Time {xaxis}", )
                                    plt.xticks(ticks=x_ticks, labels=x_labels, )
                                                                        
                                    filename = f'{results_location}/{duration}/chp {duration}.png'
                                    plt.savefig(filename, format = 'png',dpi=300, bbox_inches='tight')
                                    fig.clear()
                                    plt.close(fig)       
                                    
                                    
                                    
                                    
                                def plot_test1(pick_time,x_ticks,x_labels,duration,xaxis): 
                                    fig = plt.figure(figsize=(5,3))
                                       
                                    plt.plot(total_chp_heat_out[pick_time]*2, label='Heat Export', ls = '-', lw = '0.5', c = c[0])
                                    plt.plot(chp2HD[pick_time]*2, label='CHP to HD', ls = '-', lw = '0.5', c = c[3])
                                    plt.plot(chp2tes[pick_time]*2, label='CHP to Store', ls = '-', lw = '0.5', c = c[4])
                                    plt.title(f"CHP heat {duration}", )
                                    plt.ylabel("Power (kW th)", )
                                    plt.legend(loc = 'upper left', bbox_to_anchor=(0,-0.15), ncol = 3, fancybox = True, prop={'size': 8})   
                                    plt.xlabel(f"Time {xaxis}", )
                                    plt.xticks(ticks=x_ticks, labels=x_labels, )
                                                                        
                                    filename = f'{results_location}/{duration}/test1 {duration}.png'
                                    plt.savefig(filename, format = 'png',dpi=300, bbox_inches='tight')
                                    fig.clear()
                                    plt.close(fig)    
                                        
                                    
                                def plot_test2(pick_time,x_ticks,x_labels,duration,xaxis): 
                                    fig = plt.figure(figsize=(5,3))
                                    
                                    plt.plot(chp_used[pick_time]*2, label='Fuel', ls = '-', lw = '0.5', c = c[1])
                                    plt.plot(total_chp_heat_out[pick_time]*2, label='Heat Export', ls = '-', lw = '0.5', c = c[0])
                                    plt.plot(chp_elec_out[pick_time]*2, label='Elec Export', ls = '-', lw = '0.5', c = c[2])
                                    plt.title(f"CHP {duration}", )
                                    plt.ylabel("Power (kW)", )
                                    plt.xlabel(f"Time {xaxis}", )
                                    plt.xticks(ticks=x_ticks, labels=x_labels, )
                                    plt.legend(loc = 'upper left', bbox_to_anchor=(0,-0.15), ncol = 3, fancybox = True, prop={'size': 8})   
                                                                            
                                    filename = f'{results_location}/{duration}/test2 {duration}.png'
                                    plt.savefig(filename, format = 'png',dpi=300, bbox_inches='tight')
                                    fig.clear()
                                    plt.close(fig)        
                                        
                                        
                                        
                                        
                                    
                                def plot_heat(pick_time,x_ticks,x_labels,duration,xaxis): 
                                    fig = plt.figure(figsize=(6,5))
                                    
                                    plt.subplot(2,1,1)
                                    plt.plot(DemH[pick_time]*2, label='HD', ls = '-', lw = '0.5', c = c[0])
                                    plt.plot(chp2HD[pick_time]*2, label='CHP', ls = '-', lw = '0.5', c = c[1])
                                    plt.plot(gb2HD[pick_time]*2, label='GB', ls = '-', lw = '0.5', c = c[2])
                                    plt.plot(tes2HD[pick_time]*2, label='Store', ls = '-', lw = '0.5', c = c[3])
                                    plt.plot(gbBH2HD[pick_time]*2, label='GBB', ls = '-', lw = '0.5', c = c[4])
                                    plt.title(f"Heat demand {duration}")
                                    plt.ylabel("Power (kW th)")
                                    plt.xticks(ticks=x_ticks, labels="")
                                    plt.legend(loc = 'center left', bbox_to_anchor=(1,0.5), fancybox = True, prop={'size': 8})
    
                                    plt.subplot(2,1,2)
                                    plt.plot(total_Qc[pick_time]*2, label='Qcharge', ls = '-', lw = '0.5', c = c[0])
                                    plt.plot(total_Qex[pick_time]*2, label='Qextract', ls = '-', lw = '0.5', c = c[1])
                                    plt.plot(Qloss_S_g[pick_time]*2, label='Qloss', ls = '-', lw = '0.5', c = c[2])
                                    plt.title(f"Store {duration}")
                                    plt.ylabel("Power (kW th)")
                                    plt.xticks(ticks=x_ticks, labels=x_labels)
                                    plt.legend(loc = 'center left', bbox_to_anchor=(1,0.5), fancybox = True, prop={'size': 8})
                                    plt.xlabel(f"Time {xaxis}")
                                    
                                    filename = f'{results_location}/{duration}/heat {duration}.png'
                                    plt.savefig(filename, format = 'png',dpi=300, bbox_inches='tight')
                                    fig.clear()
                                    plt.close(fig)                                                                           
                                    
                                def plot_final_node_temp(pick_time,x_ticks,x_labels,duration,xaxis): 
                                    if RLW > 0: # if store exists
                                        fig = plt.figure(figsize=(5,6))
                                        
                                        plt.subplot(3,1,1)
                                        plt.plot(f_gr8_t[pick_time], ls = '-', lw = '0.5')
                                        plt.title(f"Ground Ring 8 (4.5-5.5m) {duration}")
                                        plt.xticks(ticks=x_ticks, labels="")
                                        
                                        plt.subplot(3,1,2)
                                        plt.plot(f_csw_t[pick_time], ls = '-', lw = '0.5')
                                        plt.title(f"Concrete Shaft Wall (3.5-4.5m) {duration}")
                                        plt.ylabel(f"Node tempearture for each layer ({DegC})")
                                        plt.xticks(ticks=x_ticks, labels="")
                                        plt.legend(['L1','L2','L3','L4','L5','L6','L7','L8','L9','L10'],loc='center left', bbox_to_anchor=(1,0.5), fancybox = True, prop={'size': 8})
    
                                        plt.subplot(3,1,3)
                                        plt.plot(f_s_t[pick_time], ls = '-', lw = '0.5')
                                        plt.title(f"Fluid inside shaft (0-3.5m) {duration}")
                                        plt.xticks(ticks=x_ticks, labels=x_labels)
                                        plt.xlabel(f"Time {xaxis}")
                                        
                                        filename = f'{results_location}/{duration}/final_node_temp {duration}.png'
                                        plt.savefig(filename, format = 'png',dpi=300, bbox_inches='tight')
                                        fig.clear()
                                        plt.close(fig)            
                                                                                                                  
                                # plot all graph and enflow csv 
                                def plot_all_graph(start_hour,end_hour,x_ticks,x_labels,duration,xaxis):
                                    plot_f_s_t(pick_time(start_hour,end_hour),x_ticks,x_labels,duration,xaxis) 
                                    plot_f_s_t_half(pick_time(start_hour,end_hour),x_ticks,x_labels,duration,xaxis) 
                                    
                                    plot_chp(pick_time(start_hour,end_hour),x_ticks,x_labels,duration,xaxis) 
                                    plot_heat(pick_time(start_hour,end_hour),x_ticks,x_labels,duration,xaxis) 
                                    # plot_hpeh(pick_time(start_hour,end_hour),x_ticks,x_labels,duration,xaxis) 
                                    # plot_electrical(pick_time(start_hour,end_hour),x_ticks,x_labels,duration,xaxis) 
                                    plot_final_node_temp(pick_time(start_hour,end_hour),x_ticks,x_labels,duration,xaxis)
                                    
                                    plot_test1(pick_time(start_hour,end_hour),x_ticks,x_labels,duration,xaxis)
                                    plot_test2(pick_time(start_hour,end_hour),x_ticks,x_labels,duration,xaxis)
                                    
                                    header = enflow_header
                                    filename = f'{results_location}/{duration}/data {duration}.csv' 
                                    np.savetxt(filename, 
                                    enflow_stack(pick_time(start_hour,end_hour)), delimiter=',', header=', '.join(header), fmt='%f', comments='')   
                                  
                                #%%% plot time 6          
                                def plot_full():
                                    start_hour = 0
                                    end_hour   = nt
                                    year_ticks = []
                                    year_labels = []
                                    Mod_Year = int(nt / 17520) # 2 data/hr * 24hr * 365 days
                                    for year in range (Mod_Year+1):
                                        year_ticks.append(17520 * year)
                                        year_labels.append(str(year))
                                    x_ticks=year_ticks
                                    x_labels=year_labels
                                    duration = 'full'
                                    xaxis = '(years)'
                                    plot_all_graph(start_hour,end_hour,x_ticks,x_labels,duration,xaxis)
                                
                                def plot_year():
                                    start_hour = 17520 * (Mod_Year-1) 
                                    end_hour   = nt
                                    month_ticks  = [    0,  1488,  2832,  4320,  5760,  7248,  8688, 10176, 11664, 13104, 14592, 16032] 
                                    month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'] 
                                    x_ticks=month_ticks
                                    x_labels=month_labels
                                    duration = 'year'
                                    xaxis = '(months)'
                                    plot_all_graph(start_hour,end_hour,x_ticks,x_labels,duration,xaxis)
                                    
                                def plot_summer():
                                    start_hour = 17520 * (Mod_Year-1) + 7248
                                    end_hour   = 17520 * (Mod_Year-1) + 11664
                                    summer_ticks = []
                                    summer_labels = []
                                    total_day = int(len(pick_time(start_hour,end_hour)) / (48*7)) # 2 data/hr * 24hr
                                    for day in range (total_day+1):
                                        summer_ticks.append(day * 48 * 7)
                                        summer_labels.append(str(day))
                                    x_ticks=summer_ticks
                                    x_labels=summer_labels
                                    duration = 'summer'
                                    xaxis = '(weeks)'
                                    plot_all_graph(start_hour,end_hour,x_ticks,x_labels,duration,xaxis)
                                    
                                def plot_winter():
                                    start_hour = 17520 * (Mod_Year-2) + 16032
                                    end_hour   = 17520 * (Mod_Year-2) + 16032 + 2832
                                    winter_ticks = []
                                    winter_labels = []
                                    total_day = int(len(pick_time(start_hour,end_hour)) / (48*7)) # 2 data/hr * 24hr
                                    for day in range (total_day+1):
                                        winter_ticks.append(day * 48 * 7)
                                        winter_labels.append(str(day))
                                    x_ticks=winter_ticks
                                    x_labels=winter_labels
                                    duration = 'winter'
                                    xaxis = '(weeks)'
                                    plot_all_graph(start_hour,end_hour,x_ticks,x_labels,duration,xaxis)
                                    
                                def plot_summer_3d():
                                    duration = "3 days in Summer End of July" 
                                    start_hour = 17520 * (Mod_Year-1) + 8688 + 48 * (28) # 10032
                                    end_hour   = 17520 * (Mod_Year-1) + 8688 + 48 * (28 + 3) # 10176
                                    summer_3d_ticks = []
                                    summer_3d_labels = []
                                    total_day = int(len(pick_time(start_hour,end_hour)) / 48) # 2 data/hr * 24hr
                                    for day in range (total_day+1):
                                        summer_3d_ticks.append(day * 48)
                                        summer_3d_labels.append(str(day))
                                    x_ticks=summer_3d_ticks
                                    x_labels=summer_3d_labels
                                    duration = 'summer3d'
                                    xaxis = '(days)'
                                    plot_all_graph(start_hour,end_hour,x_ticks,x_labels,duration,xaxis)
                                
                                def plot_winter_3d():
                                    start_hour = 17520 * (Mod_Year-1) + 16032 + 48 * (27) # 17328
                                    end_hour   = 17520 * (Mod_Year-1) + 16032 + 48 * (27 + 3) # 17472
                                    winter_3d_ticks = []
                                    winter_3d_labels = []
                                    total_day = int(len(pick_time(start_hour,end_hour)) / 48) # 2 data/hr * 24hr
                                    for day in range (total_day+1):
                                        winter_3d_ticks.append(day * 48)
                                        winter_3d_labels.append(str(day))
                                    x_ticks=winter_3d_ticks
                                    x_labels=winter_3d_labels
                                    duration = 'winter3d'
                                    xaxis = '(days)'
                                    plot_all_graph(start_hour,end_hour,x_ticks,x_labels,duration,xaxis)
                                
                                #%%% plot settings    
                                # plot colour
                                red         = '#c1272d'
                                blue        = '#0000a7'
                                green       = '#008176'
                                purple      = "#ba03af"
                                yellow      = '#eecc16'
                                black       = "#000000"
                                brown       = "#854802"
                                grey        = "#b3b3b3"
                                
                                c = np.array(["#c1272d",
                                "#0000a7",
                                "#008176",
                                "#ba03af",
                                "#eecc16",
                                "#000000",                                
                                "#854802",
                                "#b3b3b3"])
                                
                                # run plot 11*6=66 + csv 6 
                                plot_full()
                                plot_year()
                                plot_summer()
                                plot_winter()
                                plot_summer_3d()
                                plot_winter_3d()
                                
                                #%%% plot ground heating envelope
                                if RLW > 0: # if store exists
                                    your_array = f_n_t
                                    
                                    def calculate_column_averages_with_similar_names_and_rows(array, start_row, end_row):
                                        column_sums = {}
                                        column_counts = {}
                                        # Iterate through headers to find columns with similar names after '_'
                                        headers = header_node
                                        for header in headers:
                                            if '_' in header:
                                                _, suffix = header.split('_')
                                                if suffix not in column_sums:
                                                    column_sums[suffix] = 0
                                                    column_counts[suffix] = 0
                                    
                                        # Iterate through rows within the specified range
                                        for current_row in range(start_row, end_row):
                                            # Iterate through columns with similar names after '_'
                                            for col, header in enumerate(headers):
                                                if '_' in header:
                                                    prefix, suffix = header.split('_')
                                                    if suffix in column_sums:
                                                        try:
                                                            column_sums[suffix] += float(array[current_row][col])
                                                            column_counts[suffix] += 1
                                                        except ValueError:
                                                            pass  # Skip non-numeric values
                                    
                                        # Calculate averages for each column with similar names
                                        column_averages = {}
                                        for suffix, sum in column_sums.items():
                                            count = column_counts[suffix]
                                            column_averages[suffix] = sum / count if count != 0 else 0
                                        return column_averages
                                    
                                    # Example usage:
                                    averages_all = []
                                    for _ in range(Mod_Year):
                                        start_row = 0 + _ * 17519  # Starting row (inclusive)
                                        end_row = (1 + _) * 17519  # Ending row (exclusive)
                                        averages = calculate_column_averages_with_similar_names_and_rows(your_array, start_row, end_row)
                                        averages_all.append(averages)
                                    
                                    # Custom x-axis array (reversed)
                                    x_axis_values = (np.array([256., 128., 64., 32., 16., 8., 4., 2., 1., 0.]) + 3.65)[::-1]
                                    
                                    # Plotting
                                    fig = plt.figure(figsize=(5, 3))
                                    for i, average in enumerate(averages_all):
                                        suffixes = list(average.keys())[::-1]  # Reverse the order of the keys
                                        values = [average[suffix] for suffix in suffixes]
                                        plt.plot(x_axis_values, values, label=f'Year {i + 1}')
                                    
                                    plt.xlabel('Distance from center of MSTES (m)')
                                    plt.ylabel(f'Mean temperature ({DegC})')
                                    plt.title(f'Ground heating envelope over {Mod_Year}yrs for {store_temp}{DegC} Store')
                                    # plt.legend(loc='center left', bbox_to_anchor=(1,0.5), fancybox = True, prop={'size': 8})
                                    plt.legend(loc = 'upper left', bbox_to_anchor=(0,-0.15), ncol = 4, fancybox = True, prop={'size': 8})   

                                    plt.xlim(right=100)
                                    
                                    filename = f'{results_location}/Ghe{Mod_Year}yrs{store_temp}{DegC}.png'
                                    plt.savefig(filename, format='png', dpi=300, bbox_inches='tight')
                                    plt.close(fig)
                                                            
#%%% Combine kpi_last      
if combine_kpi_last == 1:                                
  def combine_csv_files(root_dir, file_name):
    combined_data = pd.DataFrame()
    for subdir, dirs, files in os.walk(root_dir):
      for file in files:
        if file_name in file:
          temp_data = pd.read_csv(os.path.join(subdir, file), encoding='ISO-8859-1')
          combined_data = pd.concat([combined_data, temp_data])
    return combined_data
  root_dir = 'c:/data/steam/'
  file_name = 'kpi_last'  # replace with your file name
  combined_data = combine_csv_files(root_dir, file_name)
  combined_data.to_csv('c:/data/steam/combined.csv', index=False, encoding='ISO-8859-1')
