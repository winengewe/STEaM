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
Dat = np.loadtxt('C:\\Users\cmb22235\OneDrive - University of Strathclyde\Desktop\STEaM WP4 team\MSTES-HP\Energy Flow & MTES\Win 1D Model\en_flows_Win_data.csv', delimiter=',') 
# Dat = np.loadtxt('C:\\Users\cmb22235\OneDrive - University of Strathclyde\Desktop\STEaM WP4 team\Energy Flow & MTES\Win Model\en_flows_Win_data1.csv', delimiter=',') 
# Descriptions for Column A to D == Column [0] to [3] in en_flow_Win_data file is listed below
# A [0] Yearly Coylton GSP Electricity Demand (MWh/0.5h)
# B [1] Yearly Coylton 33kV Wind (MWh/0.5h)
# C [2] Yearly West Whitlawburn Housing Co-operative Heat Demand (kWh/0.5h)
# D [3] Yearly ambient air temp (DegC)

# XTC = np.loadtxt('C:\\Users\cmb22235\OneDrive - University of Strathclyde\Desktop\STEaM WP4 team\MSTES-HP\Energy Flow & MTES\Win 1D Model\mth_geo_1.csv',delimiter=',')
# Geo = np.loadtxt('C:\\Users\cmb22235\OneDrive - University of Strathclyde\Desktop\STEaM WP4 team\MSTES-HP\Energy Flow & MTES\Win 1D Model\mth_geo2_1.csv',delimiter=',')

#%% Fixed variable
Mod_Year = 1 # Modeling years
lifetime = 20 # lifetime of system, years
ts = 1 # timesteps per half-hour of base data
MTS = int(1800/ts) # timestep in seconds per half-hour, 0.5hr*60min*60sec)
nt = int(17520 * ts * Mod_Year) # total number of timesteps (1yr*365days*24hr*2data/hr * model yr)

Tamb = Dat[:,3]
Dat = np.repeat(Dat,ts,0) / ts # Set demand and surplus data for sub-half-hour timesteps
Tamb = np.repeat(Tamb,ts,0) # Ambient temperature for Heat Pump
PeakHD = 4300 # Peak Heat demand, kW
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
deltaT = 20

# Outputs
plot = 1 # '1' generate all results; else only generate store temp and kpi_last 
combine_kpi_last = 0 # '1' combine kpi_last csv file in specific folder for combination of parameters

#%% Sensitivity Inputs
heat_demand_temperatures = [50]  # DegC [50]
capex_factors = [1]  # CAPEX adjustment factor [0.5,1]
ambient_temperatures = [10]  # Heat source temp for Heat Pump, DegC [10,15,20]
store_temperatures = [55]  # Heat store temp, DegC [30,55,70]
min_store_temps_for_HP = [20]  # Min store temp, DegC [10.5,15,20]
ratio_HP_to_EH = [1]  # Heat Pump to Electric Heater ratios [1.5, 1, 0.7, 0.5, 0.3, 0]
water_layer_h = [20]  # TES Water layer heights, m [0,20,40]
fixed_tariff = 0.5 # £/kWh
tariff_mapping = { # (Wind Surplus tariff, Wind Shortfall tariff), £/kWh
1: (0.1, 0.1),  # Mapping for tariff_price = 1, 
2: (0.1, 0.3),  # Mapping for tariff_price = 2, 
3: (0.1, 0.6),  # Mapping for tariff_price = 3, 
4: (0.1, 1.2)   # Mapping for tariff_price = 3, 
}     
tariff_mapping_choice = [2] # select tariff_mapping [1,2,3,4]

#%% Looping
for T_HD in heat_demand_temperatures:
    for CAfactor in capex_factors:
        for T_amb in ambient_temperatures:
            for store_temp in store_temperatures:
                for min_store_temp_HP in min_store_temps_for_HP:
                    if min_store_temp_HP >= store_temp and store_temp != 0:
                        print("Error in min_store_temp_HP: min_store_temp_HP should be less than store_temp.")
                        continue  # Skip the rest of the loop for this configuration
                    for Ratio_HPEH in ratio_HP_to_EH:
                        Size_HP = PeakHD * Ratio_HPEH  # Size of HP from Grid (kW)
                        Size_EH = PeakHD * (1 - Ratio_HPEH) if Ratio_HPEH <= 1 else 0  # Size of EH from Grid (kW)
                        for RLW in water_layer_h:
                            water_height_TS = RLW * (number_layers - 1)  # water height of TS (m)
                            TS_volume = np.pi * radius_TS**2 * water_height_TS  # Volume of the thermal store (m3)
                            for tariff_price in tariff_mapping_choice:  # To iterate over multiple scenarios, include them in the list
                                WS_tariff, NW_tariff = tariff_mapping.get(tariff_price, (None, None)) 
                                
                                #%% Data Initialisation
                                Wind_Gen             = np.zeros((nt,1)) # Wind Gen (kWh e)
                                Elec_Dem             = np.zeros((nt,1)) # Electric Demand (kWh e)
                                WS                   = np.zeros((nt,1)) # Wind Surplus (kWh e)
                                DemH                 = np.zeros((nt,1)) # Heat Demand (kWh th) 
                                
                                eIn_HPtoHD_WS_NoS    = np.zeros((nt,1)) # ElecIn to HP to HD using WS in NoStore (kWh e)
                                Qout_HPtoHD_WS_NoS   = np.zeros((nt,1)) # HeatOut from HP to HD using WS in NoStore (kWh th)
                                eIn_EHtoHD_WS_NoS    = np.zeros((nt,1)) # ElecIn to EH to HD using WS in NoStore (kWh e)
                                Qout_EHtoHD_WS_NoS   = np.zeros((nt,1)) # HeatOut from EH to HD using WS in NoStore (kWh th)
                                total_eIn_HD_WS_NoS  = np.zeros((nt,1)) # Total ElecIn to HD using WS in NoStore (kWh e)
                                total_Qout_HD_WS_NoS = np.zeros((nt,1)) # Total HeatOut to HD using WS in NoStore (kWh th)
                                
                                Qout_StoHD_direct    = np.zeros((nt,1)) # HeatOut from Store to HD directly (kWh th)
                                Qex_StoHD_direct     = np.zeros((nt,1)) # Heat Extract from Store to HD directly (kWh th)
                                
                                eIn_HPtoHD_WS_S      = np.zeros((nt,1)) # ElecIn to HP to HD using WS in Store (kWh e)
                                Qout_HPtoHD_WS_S     = np.zeros((nt,1)) # HeatOut from HP to HD using WS in Store (kWh th)
                                Qex_StoHP_WS         = np.zeros((nt,1)) # Heat Extract from Store to HP using WS (kWh th)
                                eIn_EHtoHD_WS_S      = np.zeros((nt,1)) # ElecIn to EH to HD using WS in Store (kWh e)
                                Qout_EHtoHD_WS_S     = np.zeros((nt,1)) # HeatOut from EH to HD using WS in Store (kWh th)
                                Qex_StoEH_WS         = np.zeros((nt,1)) # Heat Extract from Store to EH using WS (kWh th)
                                
                                eIn_HPtoHD_G_S       = np.zeros((nt,1)) # ElecIn to HP to HD using Non-Wind Grid in Store (kWh e)
                                Qout_HPtoHD_G_S      = np.zeros((nt,1)) # HeatOut from HP to HD using Non-Wind Grid in Store (kWh th)
                                Qex_StoHP_G          = np.zeros((nt,1)) # Heat Extract from Store to HP using Non-Wind Grid (kWh th)
                                eIn_EHtoHD_G_S       = np.zeros((nt,1)) # ElecIn to EH to HD using Non-Wind Grid in Store (kWh e)
                                Qout_EHtoHD_G_S      = np.zeros((nt,1)) # HeatOut from EH to HD using Non-Wind Grid in Store (kWh th)
                                Qex_StoEH_G          = np.zeros((nt,1)) # Heat Extract from Store to EH using G (kWh th)
                                
                                total_eIn_HD_WS_S    = np.zeros((nt,1)) # Total ElecIn to HD using WS in Store (kWh e)
                                total_eIn_HD_G_S     = np.zeros((nt,1)) # Total ElecIn to HD using Non-Wind Grid in Store (kWh e) 
                                total_eIn_HD_S       = np.zeros((nt,1)) # Total ElecIn to HD in Store (kWh e)
                                total_Qout_HD_WS_S   = np.zeros((nt,1)) # Total HeatOut to HD using WS in Store (kWh th)
                                total_Qout_HD_G_S    = np.zeros((nt,1)) # Total HeatOut to HD using Non-Wind Grid in Store (kWh th)
                                total_Qout_HD_S      = np.zeros((nt,1)) # Total HeatOut to HD from Store (kWh th) 
                                total_Qex_S          = np.zeros((nt,1)) # Total Heat Extract from Store (kWh th)
                                
                                eIn_HPtoHD_G_NoS     = np.zeros((nt,1)) # ElecIn to HP to HD using Non-Wind Grid in NoStore (kWh e)
                                Qout_HPtoHD_G_NoS    = np.zeros((nt,1)) # HeatOut from HP to HD using Non-Wind Grid in NoStore (kWh th)
                                eIn_EHtoHD_G_NoS     = np.zeros((nt,1)) # ElecIn to EH to HD using Non-Wind Grid in NoStore (kWh e)
                                Qout_EHtoHD_G_NoS    = np.zeros((nt,1)) # HeatOut from EH to HD using Non-Wind Grid in NoStore (kWh th)
                                total_eIn_HD_G_NoS   = np.zeros((nt,1)) # Total ElecIn to HD using Non-Wind Grid in NoStore (kWh e)
                                total_Qout_HD_G_NoS  = np.zeros((nt,1)) # Total HeatOut to HD using Non-Wind Grid in NoStore (kWh th)
                                total_eIn_HD_NoS     = np.zeros((nt,1)) # Total ElecIn to HD in NoStore (kWh e)
                                total_Qout_HD_NoS    = np.zeros((nt,1)) # Total HeatOut direct to HD in NoStore Period (kWh th)
                                
                                eIn_HP_CS_WS         = np.zeros((nt,1)) # ElecIn to HP to charge Store using WS (kWh e)
                                Qout_HP_CS_WS        = np.zeros((nt,1)) # HeatOut from HP to charge Store using WS (kWh th)
                                eIn_EH_CS_WS         = np.zeros((nt,1)) # ElecIn to EH to charge Store using WS (kWh e)
                                Qout_EH_CS_WS        = np.zeros((nt,1)) # HeatOut from EH to charge Store using WS (kWh th)
                                total_eIn_CS_WS      = np.zeros((nt,1)) # Total ElecIn to charge Store using WS (kWh e)
                                total_Qout_CS_WS     = np.zeros((nt,1)) # Total HeatOut Charge to Store using WS (kWh th)
                                eIn_WS_G             = np.zeros((nt,1)) # Electrcity of WS Residual (kWh e)
                                Un_H                 = np.zeros((nt,1)) # Unmet HD (kWh th)
                                
                                Qloss_S_g            = np.zeros((nt,1)) # Heat Losses from Store to ground (kWh th) 
                                Tariff               = np.zeros((nt,1)) # Wind based Tariff (£/kWh e)
                                total_elec_used_sys  = np.zeros((nt,1)) # Total Elec Used in System (kWh e)
                                Fuel_cost            = np.zeros((nt,1)) # Fuel Cost based on WS tariff (£)
                                
                                total_elec_used_WS   = np.zeros((nt,1)) # Total Elec used in WS periods (kWh e)
                                total_elec_used_G    = np.zeros((nt,1)) # Total Elec used in Non-Wind Grid periods (kWh e)
                                total_elec_used_HP   = np.zeros((nt,1)) # Total Elec used in HP (kWh e)
                                total_elec_used_EH   = np.zeros((nt,1)) # Total Elec used in EH (kWh e)
                                total_Qout_HPtoHD    = np.zeros((nt,1)) # Total HeatOut direct from HP to HD (kWh th)
                                total_Qout_EHtoHD    = np.zeros((nt,1)) # Total HeatOut direct from EH to HD (kWh th)
                                Elec_cost_WS         = np.zeros((nt,1)) # Elec cost in wind surplus periods (£)
                                Elec_cost_G          = np.zeros((nt,1)) # Elec cost in non-wind periods (£)
                                
                                total_eIn_HP         = np.zeros((nt,1)) # Total ElecIn HP (kWh e)
                                total_eIn_EH         = np.zeros((nt,1)) # Total ElecIn EH (kWh e)
                                total_Qout_HP        = np.zeros((nt,1)) # Total HeatOut HP (kWh th)
                                total_Qout_EH        = np.zeros((nt,1)) # Total HeatOut EH (kWh th)
                                
                                OpH = np.zeros((lifetime)) # Yearly operating costs (£)
                                
                                #%% Nodes Temp Initialisation
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
                                    COP_EH = 1 # COP of Electric heater
                                    R_HP = Size_HP / (3600 / MTS) # Residual HP
                                    R_EH = Size_EH / (3600 / MTS) # Residual EH
                                    
                                    #%%% HP + post EH case
                                    if R_HP != 0: # use HP
                                    
                                        #%%%% Priority 1: WS to HP to HD
                                        if Av_WS * R_HP * Res_HD > 0: # if Wind surplus, residual HP and HD exists
                                            T_HP_output = T_HD # Output Temperature for Heat Pump (DegC)
                                            T_HP_source = T_amb # Fixed Ambient heat source temp for heat pump (DegC) or Variable ambient air temp Dat[td,3]                                 
                                            COP_HP = 8.77 - 0.15 * (T_HP_output-T_HP_source) + 0.000734 * (T_HP_output-T_HP_source)**2 # COP of Heat Pump
                                            eIn_HPtoHD_WS_NoS[t,0] = min(Av_WS, R_HP / COP_HP, Res_HD / COP_HP) # ElecIn to HP to HD using WS in NoStore (kWh e)
                                            Qout_HPtoHD_WS_NoS[t,0] = eIn_HPtoHD_WS_NoS[t,0] * COP_HP # HeatOut from HP to HD using WS in NoStore (kWh th)
                                            Av_WS -= eIn_HPtoHD_WS_NoS[t,0] # Residual Wind Surplus (kWh e)
                                            Res_HD -= Qout_HPtoHD_WS_NoS[t,0] # Residual HD (kWh th)
                                            R_HP -= Qout_HPtoHD_WS_NoS[t,0] # Remaining HP Qout (kWh th)
                                        
                                        #%%%% Priority 2: WS to EH to HD
                                        if Av_WS * R_EH * Res_HD > 0: # if Wind surplus, residual EH and HD exists
                                            eIn_EHtoHD_WS_NoS[t,0] = min(Av_WS, R_EH / COP_EH, Res_HD / COP_EH) # ElecIn to EH to HD using WS in NoStore (kWh e)
                                            Qout_EHtoHD_WS_NoS[t,0] = eIn_EHtoHD_WS_NoS[t,0] * COP_EH # HeatOut from EH to HD using WS in NS (kWh th)
                                            Av_WS -= eIn_EHtoHD_WS_NoS[t,0] # Residual Wind Surplus (kWh e)
                                            Res_HD -= Qout_EHtoHD_WS_NoS[t,0] # Residual Heat Demand (kWh th) 
                                            R_EH -= Qout_EHtoHD_WS_NoS[t,0] # Remaining EH Qout (kWh th) 
                                            
                                        #%%%% Priority 3: Store to HD
                                        if RLW * Res_HD > 0: # if store exists and residual HD exists 
                                            
                                            #%%%%% store direct feed HD
                                            if nodes_temp[19] >= T_HD: # if top of Store temp is higher than/equal to HD Temp (DegC) 
                                                Qout_StoHD_direct[t,0] = Res_HD # HeatOut from Store to HD directly (kWh th)
                                                Res_HD -= Qout_StoHD_direct[t,0] # Residual HD (kWh th)
                                                # Qex_StoHD_direct[t,0] = (nodes_temp[19] - (T_HD - 10)) / (T_HD - (T_HD - 10)) * Qout_StoHD_direct[t,0] # Heat Extract from Store to HD directly (kWh th) 
                                                Qex_StoHD_direct[t,0] = Qout_StoHD_direct[t,0] # Heat Extract from Store to HD directly (kWh th) 
                                            
                                            #%%%%% use HP+EH    
                                            elif  Res_HD > 0 and min_store_temp_HP <= nodes_temp[19] < T_HD: # if residual HD exists and min store temp <= top water node temp of Store < HD temp
                                                                                               
                                                #%%%%%% Using WS-HP
                                                if Av_WS * R_HP * Res_HD > 0: # if Wind surplus, residual HP and HD exists
                                                    T_HP_output = T_HD # Output Temperature for HP (DegC) 
                                                    T_HP_source = nodes_temp[19] # Source Temperature for HP (DegC) 
                                                    COP_HP = 8.77 - 0.15 * (T_HP_output-T_HP_source) + 0.000734 * (T_HP_output-T_HP_source)**2 # COP of HP
                                                    eIn_HPtoHD_WS_S[t,0] = min(Av_WS, R_HP / COP_HP, Res_HD / COP_HP) # ElecIn to HP to HD using WS in Store (kWh e)
                                                    Qout_HPtoHD_WS_S[t,0] = eIn_HPtoHD_WS_S[t,0] * COP_HP # HeatOut from HP to HD using WS in Store (kWh th)
                                                    Av_WS -= eIn_HPtoHD_WS_S[t,0] # Residual Wind Surplus (kWh e)
                                                    Res_HD -= Qout_HPtoHD_WS_S[t,0] # Residual HD (kWh th)
                                                    R_HP -= Qout_HPtoHD_WS_S[t,0]  # Remaining HP Qout (kWh th)
                                                    Qex_StoHP_WS[t,0] = Qout_HPtoHD_WS_S[t,0] - eIn_HPtoHD_WS_S[t,0] # Heat Extract from Store to HP using WS (kWh th)
        
                                                #%%%%%% Using WS-EH
                                                if Av_WS * R_EH * Res_HD > 0: # if Wind surplus, residual HP and EH exists
                                                    eIn_EHtoHD_WS_S[t,0] = min(Av_WS, R_EH / COP_EH, Res_HD / COP_EH) # ElecIn to EH to HD using WS in Store (kWh e)
                                                    Qout_EHtoHD_WS_S[t,0] = eIn_EHtoHD_WS_S[t,0] * COP_EH # HeatOut from EH to HD using WS in Store (kWh th)
                                                    Av_WS -= eIn_EHtoHD_WS_S[t,0] # Residual Wind Surplus (kWh e)
                                                    Res_HD -= Qout_EHtoHD_WS_S[t,0] # Residual Heat Demand (kWh th) 
                                                    R_EH -= Qout_EHtoHD_WS_S[t,0] # Remaining EH Qout (kWh th)
                                            
                                                #%%%%%% Using G-HP
                                                if R_HP * Res_HD > 0: # if residual HP and HD exists
                                                    T_HP_output = T_HD # Output Temperature for HP (DegC) 
                                                    T_HP_source = nodes_temp[19] # Source Temperature for HP (DegC) 
                                                    COP_HP = 8.77 - 0.15 * (T_HP_output-T_HP_source) + 0.000734 * (T_HP_output-T_HP_source)**2 # COP of HP                                           
                                                    eIn_HPtoHD_G_S[t,0] = min(R_HP / COP_HP, Res_HD / COP_HP) # ElecIn to HP to HD using Non-Wind Grid in Store (kWh e)
                                                    Qout_HPtoHD_G_S[t,0] = eIn_HPtoHD_G_S[t,0] * COP_HP # HeatOut from HP to HD using Non-Wind Grid in Store (kWh th)
                                                    Res_HD -= Qout_HPtoHD_G_S[t,0] # Residual HD (kWh th)
                                                    R_HP -= Qout_HPtoHD_G_S[t,0] # Remaining HP Qout (kWh th)
                                                    Qex_StoHP_G[t,0] = Qout_HPtoHD_G_S[t,0] - eIn_HPtoHD_G_S[t,0] # Heat Extract from Store to HP using Non-Wind Grid (kWh th)
                                                    
                                                #%%%%%% Using G-EH 
                                                if R_EH * Res_HD > 0: # if residual EH and HD exists
                                                    eIn_EHtoHD_G_S[t,0] = min(R_EH / COP_EH, Res_HD / COP_EH) # ElecIn to EH to HD using Non-Wind Grid in Store (kWh e)
                                                    Qout_EHtoHD_G_S[t,0] = eIn_EHtoHD_G_S[t,0] * COP_EH # HeatOut from EH to HD using Non-Wind Grid in Store (kWh th)
                                                    Res_HD -= Qout_EHtoHD_G_S[t,0] # Residual Heat Demand (kWh th)                                        
                                                    R_EH -= Qout_EHtoHD_G_S[t,0] # Remaining EH Qout (kWh th)
        
                                        #%%%% Priority 4: Non-Wind Grid to HP to HD                                    
                                        if R_HP * Res_HD > 0: # if residual HP and HD exists
                                            T_HP_output = T_HD # Output Temperature for Heat Pump (DegC)
                                            T_HP_source = T_amb # Fixed Ambient heat source temp for heat pump (DegC) or Variable ambient air temp Dat[td,3]                                 
                                            COP_HP = 8.77 - 0.15 * (T_HP_output-T_HP_source) + 0.000734 * (T_HP_output-T_HP_source)**2 # COP of Heat Pump                                
                                            eIn_HPtoHD_G_NoS[t,0] = min(R_HP / COP_HP, Res_HD / COP_HP) # ElecIn to HP to HD using Non-Wind Grid in NoStore (kWh e)
                                            Qout_HPtoHD_G_NoS[t,0] = eIn_HPtoHD_G_NoS[t,0] * COP_HP # HeatOut from HP to HD using Non-Wind Grid in NoStore (kWh th)
                                            Res_HD -= Qout_HPtoHD_G_NoS[t,0] # Residual HD (kWh th)
                                            R_HP -= Qout_HPtoHD_G_NoS[t,0]  # Remaining HP Qout (kWh th)
                                    
                                        #%%%% Priority 5: Non-Wind Grid to EH to HD
                                        if R_EH * Res_HD > 0: # if residual HP and HD exists
                                            eIn_EHtoHD_G_NoS[t,0] = min(R_EH / COP_EH, Res_HD / COP_EH) # ElecIn to EH to HD using Non-Wind Grid in NoStore (kWh e)
                                            Qout_EHtoHD_G_NoS[t,0] = eIn_EHtoHD_G_NoS[t,0] * COP_EH # HeatOut from EH to HD using Non-Wind Grid in NoStore (kWh th)
                                            Res_HD -= Qout_EHtoHD_G_NoS[t,0] # Residual HD (kWh th)
                                            R_EH -= Qout_EHtoHD_G_NoS[t,0]  # Remaining EH Qout (kWh th) 
                                            
                                    #%%% full EH case
                                    elif R_HP == 0: # no HP use
                                    
                                        #%%%% Priority 1: WS to EH to HD
                                        if Av_WS * R_EH * Res_HD > 0: # if Wind surplus, residual EH and HD exists
                                            eIn_EHtoHD_WS_NoS[t,0] = min(Av_WS, R_EH / COP_EH, Res_HD / COP_EH) # ElecIn to EH to HD using WS in NoStore (kWh e)
                                            Qout_EHtoHD_WS_NoS[t,0] = eIn_EHtoHD_WS_NoS[t,0] * COP_EH # HeatOut from EH to HD using WS in NS (kWh th)
                                            Av_WS -= eIn_EHtoHD_WS_NoS[t,0] # Residual Wind Surplus (kWh e)
                                            Res_HD -= Qout_EHtoHD_WS_NoS[t,0] # Residual Heat Demand (kWh th) 
                                            R_EH -= Qout_EHtoHD_WS_NoS[t,0] # Remaining EH Qout (kWh th)                                     
                                    
                                        #%%%% Priority 2: Store to HD
                                        if RLW * Res_HD > 0: # if store exists and residual HD exists
                                        
                                            #%%%%% store direct feed HD
                                            if nodes_temp[19] >= T_HD: # if top of Store temp is higher than/equal to HD Temp (DegC)
                                                Qout_StoHD_direct[t,0] = Res_HD # HeatOut from Store to HD directly (kWh th)
                                                Res_HD -= Qout_StoHD_direct[t,0] # Residual HD (kWh th)
                                                # Qex_StoHD_direct[t,0] = (nodes_temp[19] - (T_HD - 10)) / (T_HD - (T_HD - 10)) * Qout_StoHD_direct[t,0] # Heat Extract from Store to HD directly (kWh th) 
                                                Qex_StoHD_direct[t,0] = Qout_StoHD_direct[t,0] # Heat Extract from Store to HD directly (kWh th) 
        
                                            #%%%%% use EH
                                            elif Res_HD > 0 and T_HD - 10 < nodes_temp[19] < T_HD: # if residual HD exists and if return temp from HD < top node temp < Temp HD  
                                            
                                                #%%%%%% Using WS-EH
                                                if Av_WS * R_EH * Res_HD > 0: # if Wind surplus, residual EH and HD exists
                                                    eIn_EHtoHD_WS_S[t,0] = min(Av_WS, R_EH / COP_EH, Res_HD / COP_EH) # ElecIn to EH to HD using WS in Store (kWh e)
                                                    Qout_EHtoHD_WS_S[t,0] = eIn_EHtoHD_WS_S[t,0] * COP_EH # HeatOut from EH to HD using WS in Store (kWh th)
                                                    Av_WS -= eIn_EHtoHD_WS_S[t,0] # Residual Wind Surplus (kWh e)
                                                    Res_HD -= Qout_EHtoHD_WS_S[t,0] # Residual Heat Demand (kWh th) 
                                                    R_EH -= Qout_EHtoHD_WS_S [t,0] # Remaining EH Qout (kWh th)
                                                    Qex_StoEH_WS[t,0] = (nodes_temp[19] - (T_HD - 10)) / (T_HD - (T_HD - 10)) * Qout_EHtoHD_WS_S[t,0] # Heat extract from Store to EH using WS                                          
                                                
                                                #%%%%%% Using G-EH
                                                if R_EH * Res_HD > 0: # if residual EH and HD exists
                                                    eIn_EHtoHD_G_S[t,0] = min(R_EH / COP_EH, Res_HD / COP_EH) # ElecIn to EH to HD using Non-Wind Grid (kWh e)
                                                    Qout_EHtoHD_G_S[t,0] = eIn_EHtoHD_G_S[t,0] * COP_EH # HeatOut from EH to HD using Non-Wind Grid (kWh th)
                                                    Res_HD -= Qout_EHtoHD_G_S[t,0] # Residual Heat Demand (kWh th) 
                                                    R_EH -= Qout_EHtoHD_G_S[t,0]  # Remaining EH Qout (kWh th)
                                                    Qex_StoEH_G[t,0] = (nodes_temp[19] - (T_HD - 10)) / (T_HD - (T_HD - 10)) * Qout_EHtoHD_G_S[t,0] # Heat extract from Store to EH using G                                          
                                                                                        
                                        #%%%% Priority 3: Non-Wind Grid to EH to HD 
                                        if R_EH * Res_HD > 0: # if residual EH and HD exists
                                            eIn_EHtoHD_G_NoS[t,0] = min(R_EH / COP_EH, Res_HD / COP_EH) # ElecIn to EH to HD using Non-Wind Grid (kWh e)
                                            Qout_EHtoHD_G_NoS[t,0] = eIn_EHtoHD_G_NoS[t,0] * COP_EH # HeatOut from EH to HD using Non-Wind Grid (kWh th)
                                            Res_HD -= Qout_EHtoHD_G_NoS[t,0] # Residual Heat Demand (kWh th) 
                                            R_EH -= Qout_EHtoHD_G_NoS[t,0]  # Remaining EH Qout (kWh th)
                                                                        
                                    #%%% Tariff
                                    # Fixed Tariff
                                    if tariff_select == 1:
                                        tariff = fixed_tariff # £/kWh
                                        Tariff[t,0] = tariff # £/kWh
                                
                                    # Wind-based Tariff          
                                    if tariff_select == 2:
                                        if WS[t,0]> 0: # WS period
                                            tariff = WS_tariff # £/kWh
                                        else: # Non-Wind Period
                                            tariff = NW_tariff # £/kWh
                                        Tariff[t,0] = tariff # £/kWh
                                    
                                    #%%% Charge Heat to Store
                                    if RLW > 0: # if store exist
                                        if nodes_temp[109] < (store_temp-2) and Tariff[t,0] == WS_tariff: # Only charge if bottom store water temp < store heating temp -2 and Wind Surplus exists
                                                                               
                                            #%%%%% Priority 6: WS to HP to Store
                                            if Av_WS * R_HP > 0: # if Wind Surplus and residual HP exists
                                                T_HP_output = store_temp # Charge the Store to specific Store Temp (DegC) 
                                                T_HP_source = T_amb # Fixed Ambient heat source temp for heat pump (DegC) or Variable ambient air temp Dat[td,3]                                 
                                                COP_HP = 8.77 - 0.15 * (T_HP_output-T_HP_source) + 0.000734 * (T_HP_output-T_HP_source)**2 
                                                eIn_HP_CS_WS[t,0] = min(Av_WS, R_HP / COP_HP) # ElecIn to HP to charge Store using WS (kWh e)
                                                Qout_HP_CS_WS[t,0] = eIn_HP_CS_WS[t,0] * COP_HP # HeatOut from HP to charge Store using WS (kWh th)
                                                Av_WS -= eIn_HP_CS_WS[t,0] # Residual Wind Surplus (kWh e)
                                    
                                            #%%%%% Priority 7: WS to EH to Store
                                            if Av_WS * R_EH > 0: # if Wind Surplus and residual EH exists
                                                eIn_EH_CS_WS[t,0] = min(Av_WS, R_EH / COP_EH) # ElecIn to EH to charge Store using WS (kWh e)
                                                Qout_EH_CS_WS[t,0] = eIn_EH_CS_WS[t,0] * COP_EH # HeatOut from EH to charge Store using WS (kWh th)
                                                Av_WS -= eIn_EH_CS_WS[t,0] # Residual Wind Surplus (kWh e) 
                                                                         
                                    #%%% Priority 8: Export or WS Residual
                                    eIn_WS_G[t,0] = Av_WS # Priority 5: Electrcity of WS Residual (kWh e)
                                    Un_H[t,0] = Res_HD # Unmet heat demand (kWh th)  
                                    if Un_H[t,0] > 1e-10:
                                        print(t, Un_H[t,0]) # check unmet heat demand
                                
                                    #%% KPI Calculation
                                    
                                    # Total ElecIn and HeatOut from WS to HD
                                    total_eIn_HD_WS_NoS[t,0] = eIn_HPtoHD_WS_NoS[t,0] + eIn_EHtoHD_WS_NoS[t,0] # Total ElecIn to HD using WS in NS (kWh e)
                                    total_Qout_HD_WS_NoS[t,0] = Qout_HPtoHD_WS_NoS[t,0] + Qout_EHtoHD_WS_NoS[t,0] # Total HeatOut to HD using WS in NS (kWh th)
                                
                                    # Total ElecIn and HeatOut to HD using WS & G and Store
                                    total_eIn_HD_WS_S[t,0] = eIn_HPtoHD_WS_S[t,0] + eIn_EHtoHD_WS_S[t,0] # Total ElecIn to HD using WS in Store (kWh e)
                                    total_eIn_HD_G_S[t,0] = eIn_HPtoHD_G_S[t,0] + eIn_EHtoHD_G_S[t,0] # Total ElecIn to HD using Non-Wind Grid in Store (kWh e)     
                                    total_eIn_HD_S[t,0] = total_eIn_HD_WS_S[t,0] + total_eIn_HD_G_S[t,0] # Total ElecIn to HD in Store (kWh e)        
                                    total_Qout_HD_WS_S[t,0] = Qout_HPtoHD_WS_S[t,0] + Qout_EHtoHD_WS_S[t,0] # Total HeatOut to HD using WS in Store (kWh th)
                                    total_Qout_HD_G_S[t,0] = Qout_HPtoHD_G_S[t,0] + Qout_EHtoHD_G_S[t,0] # Total HeatOut to HD using Non-Wind Grid in Store (kWh th)
                                    total_Qout_HD_S[t,0] = Qout_StoHD_direct[t,0] + total_Qout_HD_WS_S[t,0] + total_Qout_HD_G_S[t,0]  # Total HeatOut to HD from Store (kWh th) 
                                    
                                    # Total Heat Extract from Store
                                    total_Qex_S[t,0] = Qex_StoHD_direct[t,0] + Qex_StoHP_WS[t,0] + Qex_StoHP_G[t,0] + Qex_StoEH_WS[t,0] + Qex_StoEH_G[t,0] # Total Heat Extract from Store (kWh th)
                                
                                    # Total ElecIn and HeatOut from Non-Wind Grid to Heat Demand
                                    total_eIn_HD_G_NoS[t,0] = eIn_HPtoHD_G_NoS[t,0] + eIn_EHtoHD_G_NoS[t,0] # Total ElecIn to HD using Non-Wind Grid in NoStore (kWh e)
                                    total_Qout_HD_G_NoS[t,0] = Qout_HPtoHD_G_NoS[t,0] + Qout_EHtoHD_G_NoS[t,0] # Total HeatOut to HD using Non-Wind Grid in NoStore (kWh th)
                                
                                    # Total ElecIn and HeatOut from WS and Non-Wind Grid to Heat Demand
                                    total_eIn_HD_NoS[t,0] = total_eIn_HD_WS_NoS[t,0] + total_eIn_HD_G_NoS[t,0] # Total ElecIn to HD in NoStore (kWh e)
                                    total_Qout_HD_NoS[t,0] = total_Qout_HD_WS_NoS[t,0] + total_Qout_HD_G_NoS[t,0] # Total HeatOut direct to HD in NoStore Period (kWh th)
                                
                                    # Total ElecIn and Heat Input to charge Store
                                    total_eIn_CS_WS[t,0] = eIn_HP_CS_WS[t,0] + eIn_EH_CS_WS[t,0] # Total ElecIn to charge Store using WS (kWh e)
                                    total_Qout_CS_WS[t,0] = Qout_HP_CS_WS[t,0] + Qout_EH_CS_WS[t,0] # Total HeatOut Charge to Store using WS (kWh th)
                                
                                    # Total Electricity Used System
                                    total_elec_used_sys[t,0] = total_eIn_HD_NoS[t,0] + total_eIn_HD_S[t,0] + total_eIn_CS_WS[t,0] # kWh
                                    total_elec_used_WS[t,0] = total_eIn_HD_WS_NoS[t,0] + total_eIn_HD_WS_S[t,0] + total_eIn_CS_WS[t,0]
                                    total_elec_used_G[t,0] = total_eIn_HD_G_S[t,0] + total_eIn_HD_G_NoS[t,0]
                                    total_elec_used_HP[t,0] = eIn_HPtoHD_WS_NoS[t,0] + eIn_HPtoHD_WS_S[t,0] + eIn_HPtoHD_G_S[t,0] + eIn_HPtoHD_G_NoS[t,0] + eIn_HP_CS_WS[t,0]
                                    total_elec_used_EH[t,0] = eIn_EHtoHD_WS_NoS[t,0] + eIn_EHtoHD_WS_S[t,0] + eIn_EHtoHD_G_S[t,0] + eIn_EHtoHD_G_NoS[t,0] + eIn_EH_CS_WS[t,0]
                                    
                                    # Cost
                                    Fuel_cost[t,0] = Tariff[t,0] * total_elec_used_sys[t,0] # £/yr ususally per year then times lifetime
                                    Elec_cost_WS[t,0] = Tariff[t,0] * total_elec_used_WS[t,0]
                                    Elec_cost_G[t,0] = Tariff[t,0] * total_elec_used_G[t,0]
                                    
                                    # Total HPEH
                                    total_eIn_HP[t,0] = eIn_HPtoHD_WS_NoS[t,0] + eIn_HPtoHD_WS_S[t,0] + eIn_HPtoHD_G_S[t,0] + eIn_HPtoHD_G_NoS[t,0] + eIn_HP_CS_WS[t,0]
                                    total_eIn_EH[t,0] = eIn_EHtoHD_WS_NoS[t,0] + eIn_EHtoHD_WS_S[t,0] + eIn_EHtoHD_G_S[t,0] + eIn_EHtoHD_G_NoS[t,0] + eIn_EH_CS_WS[t,0]
                                    total_Qout_HP[t,0] = Qout_HPtoHD_WS_NoS[t,0] + Qout_HPtoHD_WS_S[t,0] + Qout_HPtoHD_G_S[t,0] + Qout_HPtoHD_G_NoS[t,0] + Qout_HP_CS_WS[t,0]
                                    total_Qout_EH[t,0] = Qout_EHtoHD_WS_NoS[t,0] + Qout_EHtoHD_WS_S[t,0] + Qout_EHtoHD_G_S[t,0] + Qout_EHtoHD_G_NoS[t,0] + Qout_EH_CS_WS[t,0]
                                    total_Qout_HPtoHD[t,0] = Qout_HPtoHD_WS_NoS[t,0] + Qout_HPtoHD_WS_S[t,0] + Qout_HPtoHD_G_S[t,0] + Qout_HPtoHD_G_NoS[t,0] 
                                    total_Qout_EHtoHD[t,0] = Qout_EHtoHD_WS_NoS[t,0] + Qout_EHtoHD_WS_S[t,0] + Qout_EHtoHD_G_S[t,0] + Qout_EHtoHD_G_NoS[t,0]
    
                                    #%% Set configurations for shatstore model
                                    # Extra Charge/Discharge Condition in Store
                                    if RLW > 0: 
                                        if total_Qout_CS_WS[t,0] - total_Qex_S[t,0] > 0.: # Charge > Discharge
                                            charge = total_Qout_CS_WS[t,0] - total_Qex_S[t,0]
                                            discharge = 0.
                                        elif total_Qout_CS_WS[t,0] - total_Qex_S[t,0] < 0.: # Discharge > Charge
                                            charge = 0.
                                            discharge = -(total_Qout_CS_WS[t,0] - total_Qex_S[t,0])
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
                                                            
                                    #%% End of the for loop
                                    # print(tr) # print no. of simulation/ timestep counter
                                    tr = tr + 1 # add no. of simulation
                                    
                                #%% Specific node temperature
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
                            
                                #%% Economic Analysis
                                HeatOpex = Fuel_cost
                                HP_CAPEX = 600 # Heat Capital Expenses (£/kW)
                                
                                if RLW == 0:
                                    TS_CAPEX = 0
                                else:
                                    TS_CAPEX = 7982*TS_volume**-0.483 # (£100/m3) fixed estimated price but should be exponential graph
                                    
                                CAPEX = Size_HP * HP_CAPEX + TS_volume * TS_CAPEX # Total Capital Expenses (£)
                                HPEX = Size_HP * HP_CAPEX 
                                
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
                                # base_location = "/Users/cmb22235/OneDrive - University of Strathclyde/Desktop/STEaM WP4 team/MSTES-HP/Energy Flow & MTES/Results/"
                                base_location = "/Users/cmb22235/OneDrive - University of Strathclyde/Desktop/"
                                folder_name = f'Win_MY{Mod_Year},ST{store_temp},TES{water_height_TS},Tf{int(int(WS_tariff*100))}&{int(NW_tariff*100)}'
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
                                    WS = np.append(WS, WS[int(nt - (nt / Mod_Year)):nt])
                                    DemH = np.append(DemH, DemH[int(nt - (nt / Mod_Year)):nt])
                                    total_elec_used_sys = np.append(total_elec_used_sys, total_elec_used_sys[int(nt - (nt / Mod_Year)):nt])
                                    total_elec_used_WS = np.append(total_elec_used_WS, total_elec_used_WS[int(nt - (nt / Mod_Year)):nt])
                                    total_elec_used_G = np.append(total_elec_used_G, total_elec_used_G[int(nt - (nt / Mod_Year)):nt])
                                    total_elec_used_HP = np.append(total_elec_used_HP, total_elec_used_HP[int(nt - (nt / Mod_Year)):nt])
                                    total_elec_used_EH = np.append(total_elec_used_EH, total_elec_used_EH[int(nt - (nt / Mod_Year)):nt])
                                    total_Qout_HD_NoS = np.append(total_Qout_HD_NoS, total_Qout_HD_NoS[int(nt - (nt / Mod_Year)):nt])
                                    total_Qout_HD_S = np.append(total_Qout_HD_S, total_Qout_HD_S[int(nt - (nt / Mod_Year)):nt])
                                    total_Qout_HPtoHD = np.append(total_Qout_HPtoHD, total_Qout_HPtoHD[int(nt - (nt / Mod_Year)):nt])
                                    total_Qout_EHtoHD = np.append(total_Qout_EHtoHD, total_Qout_EHtoHD[int(nt - (nt / Mod_Year)):nt])
                                    Qout_HP_CS_WS = np.append(Qout_HP_CS_WS, Qout_HP_CS_WS[int(nt - (nt / Mod_Year)):nt])
                                    Qout_EH_CS_WS = np.append(Qout_EH_CS_WS, Qout_EH_CS_WS[int(nt - (nt / Mod_Year)):nt])
                                    Qloss_S_g = np.append(Qloss_S_g, Qloss_S_g[int(nt - (nt / Mod_Year)):nt])
                                    Fuel_cost = np.append(Fuel_cost, Fuel_cost[int(nt - (nt / Mod_Year)):nt])
                                    Elec_cost_WS = np.append(Elec_cost_WS, Elec_cost_WS[int(nt - (nt / Mod_Year)):nt])
                                    Elec_cost_G = np.append(Elec_cost_G, Elec_cost_G[int(nt - (nt / Mod_Year)):nt])
                                    total_eIn_HD_S = np.append(total_eIn_HD_S, total_eIn_HD_S[int(nt - (nt / Mod_Year)):nt])
                                    total_eIn_CS_WS = np.append(total_eIn_CS_WS, total_eIn_CS_WS[int(nt - (nt / Mod_Year)):nt])
                                    total_eIn_HD_NoS = np.append(total_eIn_HD_NoS, total_eIn_HD_NoS[int(nt - (nt / Mod_Year)):nt])
                                    total_Qout_CS_WS = np.append(total_Qout_CS_WS, total_Qout_CS_WS[int(nt - (nt / Mod_Year)):nt])
                                
                                #%%% Kpi for last year and run for many years
                                kpi_header = [ 'TES height (m)',
                                               'Size of HP (MW)',
                                               'Size of EH (MW)',
                                               f'Heat Demand Temperature ({DegC})',
                                               f'Heating Store Temperature ({DegC})',
                                               f'Min Store Temp ({DegC})',
                                               f'Ambient Source Temperature for Heat Pump ({DegC})',
                                               'Tariff for WS period (£/kWh)',
                                               'Tariff for Non-wind period (£/kWh)',
                                               
                                                'Wind Surplus (GWh e)',
                                                'HD (GWh th)',
                                                'Total Elec Used in System (GWh e)',  
                                                'Total Elec used in WS periods (GWh e)',
                                                'Total Elec used in Non-Wind Grid periods (GWh e)',
                                                'Total Elec used in HP (GWh e)',
                                                'Total Elec used in EH (GWh e)',
                                                'Total HeatOut direct to HD in NoStore Period (GWh th)',
                                                'Total HeatOut to HD from Store (GWh th)',
                                                'Total HeatOut direct from HP to HD (GWh th)',
                                                'Total HeatOut direct from EH to HD (GWh th)',
                                                'HeatOut from HP to charge Store using WS (GWh th)',
                                                'HeatOut from EH to charge Store using WS (GWh th)',
                                                'Heat Losses from Store to ground (GWh th)',
                                                'Fuel Cost (£) x (10**6)',
                                                'Elec cost in wind surplus periods (£) x (10**6)',
                                                'Elec cost in non-wind periods (£) x (10**6)',
                                                'FlexPA(HP)',
                                               
                                               'CAPEX (£) x (10**6)',
                                               'Overall COP (kWh th/kWh e)',
                                               'Store COP (kWh th/kWh e)',
                                               'Direct Supply COP (kWh th/kWh e)',
                                               'Proportion of heat via store (%)',
                                               'FLEX(HP) (%)',
                                               'Store heat loss proportion (%)',
                                               'LCOHwd5 (£/kWh)', 
                                               'LCOHwd5Inf (£/kWh)',
                                               'LCOHwd8 (£/kWh)', 
                                               'LCOHwd8Inf (£/kWh)'                                               
                                                ]
                                
                                def kpi_stack_last(pick_time):       
                                    kpi_stack_last = np.column_stack((water_height_TS,
                                        Size_HP/1000,
                                        Size_EH/1000,
                                        T_HD,
                                        store_temp,
                                        min_store_temp_HP,
                                        T_amb,
                                        WS_tariff,
                                        NW_tariff,
                                        
                                        np.sum(WS[pick_time])/(10**6),
                                        np.sum(DemH[pick_time])/(10**6),
                                        np.sum(total_elec_used_sys[pick_time])/(10**6),
                                        np.sum(total_elec_used_WS[pick_time])/(10**6),
                                        np.sum(total_elec_used_G[pick_time])/(10**6),
                                        np.sum(total_elec_used_HP[pick_time])/(10**6),
                                        np.sum(total_elec_used_EH[pick_time])/(10**6),
                                        np.sum(total_Qout_HD_NoS[pick_time])/(10**6),
                                        np.sum(total_Qout_HD_S[pick_time])/(10**6),
                                        np.sum(total_Qout_HPtoHD[pick_time])/(10**6),
                                        np.sum(total_Qout_EHtoHD[pick_time])/(10**6),
                                        np.sum(Qout_HP_CS_WS[pick_time])/(10**6),
                                        np.sum(Qout_EH_CS_WS[pick_time])/(10**6),
                                        np.sum(Qloss_S_g[pick_time])/(10**6),
                                        np.sum(Fuel_cost[pick_time])/(10**6),
                                        np.sum(Elec_cost_WS[pick_time])/(10**6),
                                        np.sum(Elec_cost_G[pick_time])/(10**6),
                                        (np.sum(total_elec_used_WS[pick_time])/(10**6))/lifetime,
                                        
                                        CapI/(10**6),    
                                        ((np.sum(total_Qout_HD_NoS[pick_time])+np.sum(total_Qout_HD_S[pick_time]))/np.sum(total_elec_used_sys[pick_time])),                                
                                        np.sum(total_Qout_HD_S[pick_time])/(np.sum(total_eIn_HD_S[pick_time])+np.sum(total_eIn_CS_WS[pick_time])),
                                        np.sum(total_Qout_HD_NoS[pick_time])/np.sum(total_eIn_HD_NoS[pick_time]),
                                        (np.sum(total_Qout_HD_S[pick_time])/np.sum(DemH[pick_time]))*100,
                                        (np.sum(total_elec_used_WS[pick_time])/np.sum(total_elec_used_sys[pick_time]))*100,
                                        (np.sum(Qloss_S_g[pick_time])/np.sum(total_Qout_CS_WS[pick_time]))*100,
                                        LCOH,
                                        LCOH2,
                                        LCOH3,
                                        LCOH4
                                        ))
                                    return kpi_stack_last
        
                                def kpi_run_last(start_hour,end_hour,duration):
                                    header = kpi_header
                                    filename = f'kpi {duration}.csv'
                                    np.savetxt(f"{kpi_location}/{filename}", 
                                    kpi_stack_last(pick_time(start_hour,end_hour)), delimiter=',', header=', '.join(header), fmt='%f', comments='')
                                    
                                def kpi_last():
                                    start_hour = 0
                                    end_hour   = 17520 * lifetime
                                    duration = 'last'
                                    kpi_run_last(start_hour,end_hour,duration)
                                kpi_last()
    
                                #%%% enflow csv header+stack
                                if plot == 1:
                                    enflow_header = [ 'Wind Gen (kWh e)',
                                                    'Electric Demand (kWh e)',
                                                    'Wind Surplus (kWh e)',
                                                    'Heat Demand (kWh th) ',
                                                    'ElecIn to HP to HD using WS in NoStore (kWh e)',
                                                    'HeatOut from HP to HD using WS in NoStore (kWh th)',
                                                    'ElecIn to EH to HD using WS in NoStore (kWh e)',
                                                    'HeatOut from EH to HD using WS in NS (kWh th)',
                                                    'Total ElecIn to HD using WS in NS (kWh e)',
                                                    'Total HeatOut to HD using WS in NS (kWh th)',
                                                    'HeatOut from Store to HD directly (kWh th)',
                                                    'Heat Extract from Store to HD directly (kWh th)',
                                                    'ElecIn to HP to HD using WS in Store (kWh e)',
                                                    'HeatOut from HP to HD using WS in Store (kWh th)',
                                                    'Heat Extract from Store to HP using WS (kWh th)',
                                                    'ElecIn to EH to HD using WS in Store (kWh e)',
                                                    'HeatOut from EH to HD using WS in Store (kWh th)',
                                                    'ElecIn to HP to HD using Non-Wind Grid in Store (kWh e)',
                                                    'HeatOut from HP to HD using Non-Wind Grid in Store (kWh th)',
                                                    'Heat Extract from Store to HP using Non-Wind Grid (kWh th)',
                                                    'ElecIn to EH to HD using Non-Wind Grid in Store (kWh e)',
                                                    'HeatOut from EH to HD using Non-Wind Grid in Store (kWh th)',
                                                    'Heat Extract from S to EH using WS',
                                                    'Heat Extract from S to EH using G',
                                                    'Total ElecIn to HD using WS in Store (kWh e)',
                                                    'Total ElecIn from Non-Wind Grid from Store to HD (kWh e)',
                                                    'Total ElecIn to HD in Store (kWh e)',
                                                    'Total HeatOut to HD using WS in Store (kWh th)',
                                                    'Total HeatOut to HD using Non-Wind Grid in Store (kWh th)',
                                                    'Total HeatOut from Store to HD directly (kWh th)',
                                                    'Total Heat Extract from Store (kWh th)',
                                                    'ElecIn to HP to HD using Non-Wind Grid in NoStore (kWh e)',
                                                    'HeatOut from HP to HD using Non-Wind Grid in NoStore (kWh th)',
                                                    'ElecIn to EH to HD using Non-Wind Grid in NoStore (kWh e)',
                                                    'HeatOut from EH to HD using Non-Wind Grid in NoStore (kWh th)',
                                                    'Total ElecIn to HD using Non-Wind Grid in NoStore (kWh e)',
                                                    'Total HeatOut to HD using Non-Wind Grid in NoStore (kWh th)',
                                                    'Total ElecIn to HD in NoStore (kWh e)',
                                                    'Total HeatOut to HD in NoStore (kWh th)',
                                                    'ElecIn to HP to charge Store using WS (kWh e)',
                                                    'HeatOut from HP to charge Store using WS (kWh th)',
                                                    'ElecIn to EH to charge Store using WS (kWh e)',
                                                    'HeatOut from EH to charge Store using WS (kWh th)',
                                                    'Total ElecIn to charge Store using WS (kWh e)',
                                                    'Total HeatOut Charge to Store using WS (kWh th)',
                                                    'Electrcity of WS Residual (kWh e)',
                                                    'Unmet HD (kWh th)',
                                                    'Heat Losses from Store to ground (kWh th)',
                                                    'Tariff (£/kWh e)',
                                                    'total_elec_used_sys (kWh e)',
                                                    'Fuel Cost (£)',
                                                    ]
                                    
                                    def enflow_stack(pick_time):
                                        enflow_stack = np.column_stack ((Wind_Gen[pick_time],
                                            Elec_Dem[pick_time],
                                            WS[pick_time],
                                            DemH[pick_time],
                                            eIn_HPtoHD_WS_NoS[pick_time],
                                            Qout_HPtoHD_WS_NoS[pick_time],
                                            eIn_EHtoHD_WS_NoS[pick_time],
                                            Qout_EHtoHD_WS_NoS[pick_time],
                                            total_eIn_HD_WS_NoS[pick_time],
                                            total_Qout_HD_WS_NoS[pick_time],
                                            Qout_StoHD_direct[pick_time],
                                            Qex_StoHD_direct[pick_time],
                                            eIn_HPtoHD_WS_S[pick_time],
                                            Qout_HPtoHD_WS_S[pick_time],
                                            Qex_StoHP_WS[pick_time],
                                            eIn_EHtoHD_WS_S[pick_time],
                                            Qout_EHtoHD_WS_S[pick_time],
                                            eIn_HPtoHD_G_S[pick_time],
                                            Qout_HPtoHD_G_S[pick_time],
                                            Qex_StoHP_G[pick_time],
                                            eIn_EHtoHD_G_S[pick_time],
                                            Qout_EHtoHD_G_S[pick_time],
                                            Qex_StoEH_WS[pick_time],
                                            Qex_StoEH_G[pick_time],
                                            total_eIn_HD_WS_S[pick_time],
                                            total_eIn_HD_G_S[pick_time],
                                            total_eIn_HD_S[pick_time],
                                            total_Qout_HD_WS_S[pick_time],
                                            total_Qout_HD_G_S[pick_time],  
                                            total_Qout_HD_S[pick_time], 
                                            total_Qex_S[pick_time],
                                            eIn_HPtoHD_G_NoS[pick_time],
                                            Qout_HPtoHD_G_NoS[pick_time],
                                            eIn_EHtoHD_G_NoS[pick_time],
                                            Qout_EHtoHD_G_NoS[pick_time],
                                            total_eIn_HD_G_NoS[pick_time],
                                            total_Qout_HD_G_NoS[pick_time],
                                            total_eIn_HD_NoS[pick_time], 
                                            total_Qout_HD_NoS[pick_time], 
                                            eIn_HP_CS_WS[pick_time],
                                            Qout_HP_CS_WS[pick_time],
                                            eIn_EH_CS_WS[pick_time],
                                            Qout_EH_CS_WS[pick_time],
                                            total_eIn_CS_WS[pick_time],
                                            total_Qout_CS_WS[pick_time],
                                            eIn_WS_G[pick_time],
                                            Un_H[pick_time],
                                            Qloss_S_g[pick_time],
                                            Tariff[pick_time],
                                            total_elec_used_sys[pick_time],
                                            Fuel_cost[pick_time],
                                            ))
                                        return enflow_stack
                                   
                                    #%%% plot graphs   
                                    plt.style.use('ggplot')
                                    plt.rcParams.update({'font.size': 6})
                                               
                                    def plot_tariff_elec(pick_time,x_ticks,x_labels,duration,xaxis): 
                                        fig = plt.figure(figsize=(5,3))                                        
                                        
                                        plt.subplot(2,1,1)
                                        plt.plot((Tariff[pick_time]), label='Tariff', ls = '-', lw = '0.5', c = c[5])
                                        plt.title(f"Wind-based Tariff {duration}", )
                                        plt.ylabel("Tariff (£/kW e)", )
                                        # plt.legend(loc = 'center left', bbox_to_anchor=(1,0.5), fancybox = True, prop={'size': 8})
                                        plt.xticks(ticks=x_ticks, labels="")
                                        
                                        plt.subplot(2,1,2)
                                        plt.plot(total_elec_used_sys[pick_time]*2, label='Total', ls = '-', lw = '0.5', c = c[0])
                                        plt.plot(total_elec_used_WS[pick_time]*2, label='Grid Low tariff', ls = '-', lw = '0.5', c = c[1])
                                        plt.plot(total_elec_used_G[pick_time]*2, label='Grid High tariff', ls = '-', lw = '0.5', c = c[2])
                                        plt.plot(total_eIn_CS_WS[pick_time]*2, label='Charge', ls = '-', lw = '0.5', c = c[3])
                                        plt.title(f"Electrical consumption {duration}", )
                                        plt.ylabel("Power (kW e)", )
                                        plt.xlabel(f"Time {xaxis}", )
                                        plt.xticks(ticks=x_ticks, labels=x_labels)
                                        plt.legend(loc = 'center left', bbox_to_anchor=(1,0.5), fancybox = True, prop={'size': 8})

                                        filename = f'{results_location}/{duration}/tariff_elec {duration}.png'
                                        plt.savefig(filename, format = 'png',dpi=300, bbox_inches='tight')
                                        fig.clear()
                                        plt.close(fig) 
                                      
                                    def plot_f_s_t(pick_time,x_ticks,x_labels,duration,xaxis): 
                                        if RLW > 0: # if store exists
                                            
                                            fig = plt.figure(figsize=(5,3))
                                            plt.plot(f_s_t[pick_time], ls = '-', lw = '0.5')
                                            plt.title(f"Water Temp for {store_temp}{DegC} Store ({water_height_TS}m)")
                                            plt.xticks(ticks=x_ticks, labels=x_labels)
                                            plt.xlabel(f"Time {xaxis}")
                                            plt.ylabel(f"Node tempearture for each layer ({DegC})")
                                            # plt.legend(['L1','L2','L3','L4','L5','L6','L7','L8','L9','L10'],loc='center left', bbox_to_anchor=(0.5,-0.5), fancybox = True, prop={'size': 8})
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
                                        
                                    def plot_wind_energy(pick_time,x_ticks,x_labels,duration,xaxis): 
                                        fig = plt.figure(figsize=(6,5))
                                        
                                        plt.subplot(3,1,1)
                                        plt.plot(Wind_Gen[pick_time]*2, label='Wind energy', ls = '-', lw = '0.5', c = c[0])
                                        plt.plot(Elec_Dem[pick_time]*2, label='Electric demand', ls = '-', lw = '0.5', c = c[1])
                                        plt.plot(WS[pick_time]*2, label='Wind surplus', ls = '-', lw = '0.5', c = c[2])
                                        plt.plot(eIn_WS_G[pick_time]*2, label='Wind residual', ls = '-', lw = '0.5', c = c[3])
                                        plt.title(f"Wind energy {duration}", )
                                        plt.ylabel("Power (kW e)", )
                                        plt.xticks(ticks=x_ticks, labels="", )
                                        plt.legend(loc = 'center left', bbox_to_anchor=(1,0.5), fancybox = True, prop={'size': 8})
    
                                        plt.subplot(3,1,2)
                                        plt.plot(total_elec_used_WS[pick_time]*2, label='Total', ls = '-', lw = '0.5', c = c[0])
                                        plt.plot(total_eIn_CS_WS[pick_time]*2, label='Charge Store', ls = '-', lw = '0.5', c = c[1])
                                        plt.plot(total_eIn_HD_WS_NoS[pick_time]*2, label='HD w NoStore', ls = '-', lw = '0.5', c = c[2])
                                        plt.plot(total_eIn_HD_WS_S[pick_time]*2, label='HD w Store', ls = '-', lw = '0.5', c = c[3])
                                        plt.title(f"Electrical consumption using Wind surplus {duration}", )
                                        plt.ylabel("Power (kW e)", )
                                        plt.xticks(ticks=x_ticks, labels="", )
                                        plt.legend(loc = 'center left', bbox_to_anchor=(1,0.5), fancybox = True, prop={'size': 8})
    
                                        plt.subplot(3,1,3)
                                        plt.plot((Tariff[pick_time]), label='Wind tariff', ls = '-', lw = '0.5', c = c[0])
                                        plt.title(f"Tariff {duration}", )
                                        plt.ylabel("Tariff (£/kWh e)", )
                                        plt.legend(loc = 'center left', bbox_to_anchor=(1,0.5), fancybox = True, prop={'size': 8})
                                        
                                        plt.xlabel(f"Time {xaxis}", )
                                        plt.xticks(ticks=x_ticks, labels=x_labels, )
                                                                            
                                        filename = f'{results_location}/{duration}/wind_energy {duration}.png'
                                        plt.savefig(filename, format = 'png',dpi=300, bbox_inches='tight')
                                        fig.clear()
                                        plt.close(fig)           
                                        
                                    def plot_heat(pick_time,x_ticks,x_labels,duration,xaxis): 
                                        fig = plt.figure(figsize=(6,5))
                                        
                                        plt.subplot(2,1,1)
                                        plt.plot(DemH[pick_time]*2, label='Heat demand', ls = '-', lw = '0.5', c = c[0])
                                        plt.plot(total_Qout_HD_WS_NoS[pick_time]*2, label='Qout w NoStore in WS', ls = '-', lw = '0.5', c = c[1])
                                        plt.plot(total_Qout_HD_S[pick_time]*2, label='Qout w Store', ls = '-', lw = '0.5', c = c[2])
                                        plt.plot(total_Qout_HD_G_NoS[pick_time]*2, label='Qout w NoStore in NW', ls = '-', lw = '0.5', c = c[3])
                                        plt.title(f"Heat demand {duration}")
                                        plt.ylabel("Power (kW th)")
                                        plt.xticks(ticks=x_ticks, labels="")
                                        plt.legend(loc = 'center left', bbox_to_anchor=(1,0.5), fancybox = True, prop={'size': 8})
    
                                        plt.subplot(2,1,2)
                                        plt.plot(total_Qout_CS_WS[pick_time]*2, label='Qcharge', ls = '-', lw = '0.5', c = c[0])
                                        plt.plot(total_Qex_S[pick_time]*2, label='Qextract', ls = '-', lw = '0.5', c = c[1])
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
                                        
                                    def plot_hpeh(pick_time,x_ticks,x_labels,duration,xaxis): 
                                        fig = plt.figure(figsize=(7,9))
                                        
                                        plt.subplot(4,1,1)
                                        plt.plot(total_eIn_HP[pick_time]*2, label='Total', ls = '-', lw = '0.5', c = c[0])
                                        plt.plot(eIn_HPtoHD_WS_NoS[pick_time]*2, label='HD w NoStore in WS', ls = '-', lw = '0.5', c = c[1])
                                        plt.plot(eIn_HPtoHD_WS_S[pick_time]*2, label='HD w Store in WS', ls = '-', lw = '0.5', c = c[2])
                                        plt.plot(eIn_HPtoHD_G_S[pick_time]*2, label='HD w Store in NW', ls = '-', lw = '0.5', c = c[3])
                                        plt.plot(eIn_HPtoHD_G_NoS[pick_time]*2, label='HD w NoStore in NW', ls = '-', lw = '0.5', c = c[4])
                                        plt.plot(eIn_HP_CS_WS[pick_time]*2, label='Charge store', ls = '-', lw = '0.5', c = c[5])
                                        plt.title(f"Heat pump electrical consumption {duration}")
                                        plt.ylabel("Power (kW e)")
                                        plt.xticks(ticks=x_ticks, labels="")
                                        
                                        plt.subplot(4,1,2)
                                        plt.plot(total_Qout_HP[pick_time]*2, label='Total', ls = '-', lw = '0.5', c = c[0])
                                        plt.plot(Qout_HPtoHD_WS_NoS[pick_time]*2, label='HD w NoStore in WS', ls = '-', lw = '0.5', c = c[1])
                                        plt.plot(Qout_HPtoHD_WS_S[pick_time]*2, label='HD w Store in WS', ls = '-', lw = '0.5', c = c[2])
                                        plt.plot(Qout_HPtoHD_G_S[pick_time]*2, label='HD w Store in NW', ls = '-', lw = '0.5', c = c[3])
                                        plt.plot(Qout_HPtoHD_G_NoS[pick_time]*2, label='HD w NoStore in NW', ls = '-', lw = '0.5', c = c[4])
                                        plt.plot(Qout_HP_CS_WS[pick_time]*2, label='Charge store', ls = '-', lw = '0.5', c = c[5])
                                        plt.title(f"Heat pump heat output {duration}")
                                        plt.ylabel("Power (kW th)")
                                        plt.xticks(ticks=x_ticks, labels="")
                                        
                                        plt.subplot(4,1,3)
                                        plt.plot(total_eIn_EH[pick_time]*2, label='Total', ls = '-', lw = '0.5', c = c[0])
                                        plt.plot(eIn_EHtoHD_WS_NoS[pick_time]*2, label='HD w NoStore in WS', ls = '-', lw = '0.5', c = c[1])
                                        plt.plot(eIn_EHtoHD_WS_S[pick_time]*2, label='HD w Store in WS', ls = '-', lw = '0.5', c = c[2])
                                        plt.plot(eIn_EHtoHD_G_S[pick_time]*2, label='HD w Store in NW', ls = '-', lw = '0.5', c = c[3])
                                        plt.plot(eIn_EHtoHD_G_NoS[pick_time]*2, label='HD w NoStore in NW', ls = '-', lw = '0.5', c = c[4])
                                        plt.plot(eIn_EH_CS_WS[pick_time]*2, label='Charge store', ls = '-', lw = '0.5', c = c[5])
                                        plt.title(f"Electric heater electrical consumption {duration}")
                                        plt.ylabel("Power (kW e)")
                                        plt.xticks(ticks=x_ticks, labels="")
                                        
                                        plt.subplot(4,1,4)
                                        plt.plot(total_Qout_EH[pick_time]*2, label='Total', ls = '-', lw = '0.5', c = c[0])
                                        plt.plot(Qout_EHtoHD_WS_NoS[pick_time]*2, label='HD w NoStore in WS', ls = '-', lw = '0.5', c = c[1])
                                        plt.plot(Qout_EHtoHD_WS_S[pick_time]*2, label='HD w Store in WS', ls = '-', lw = '0.5', c = c[2])
                                        plt.plot(Qout_EHtoHD_G_S[pick_time]*2, label='HD w Store in NW', ls = '-', lw = '0.5', c = c[3])
                                        plt.plot(Qout_EHtoHD_G_NoS[pick_time]*2, label='HD w NoStore in NW', ls = '-', lw = '0.5', c = c[4])
                                        plt.plot(Qout_EH_CS_WS[pick_time]*2, label='Charge store', ls = '-', lw = '0.5', c = c[5])
                                        plt.title(f"Electric heater heat output {duration}")
                                        plt.ylabel("Power (kW th)")
                                        plt.xticks(ticks=x_ticks, labels=x_labels)
                                        plt.xlabel(f"Time {xaxis}")
                                        plt.legend(loc = 'upper left', bbox_to_anchor=(0,-0.25), ncol = 3, fancybox = True, prop={'size': 8})                                      
                                        
                                        filename = f'{results_location}/{duration}/hpeh {duration}.png'
                                        plt.savefig(filename, format = 'png',dpi=300, bbox_inches='tight')
                                        fig.clear()
                                        plt.close(fig)                                             
                                        
                                    def plot_electrical(pick_time,x_ticks,x_labels,duration,xaxis): 
                                        fig = plt.figure(figsize=(5,3))
                                        
                                        # plt.subplot(2,1,1)
                                        # plt.plot(total_elec_used_sys[pick_time]*2, label='Total electricity consumption', ls = '-', lw = '0.5', c = c[0])
                                        plt.plot(total_elec_used_WS[pick_time]*2, label='Wind surplus', ls = '-', lw = '0.5', c = c[1])
                                        plt.plot(total_elec_used_G[pick_time]*2, label='Wind shortfall', ls = '-', lw = '0.5', c = c[2])
                                        plt.plot(total_eIn_CS_WS[pick_time]*2, label='Charge store', ls = '-', lw = '0.5', c = c[3])
                                        plt.title(f"Electrical consumption {duration}")
                                        plt.ylabel("Power (kW e)")
                                        # plt.xticks(ticks=x_ticks, labels="")
                                        plt.xticks(ticks=x_ticks, labels=x_labels)

                                        # plt.legend(loc = 'center left', bbox_to_anchor=(1,0.5), fancybox = True, prop={'size': 8})
                                        plt.legend(loc = 'upper left', bbox_to_anchor=(0,-0.15), ncol = 3, fancybox = True, prop={'size': 8})   
    
                                        # plt.subplot(2,1,2)
                                        # plt.plot(total_eIn_HD_G_NoS[pick_time]*2, label='HD w NoStore', ls = '-', lw = '0.5', c = c[0])
                                        # plt.plot(total_eIn_HD_G_S[pick_time]*2, label='HD w Store', ls = '-', lw = '0.5', c = c[1])
                                        # plt.title(f"Electrical consumption using Non-Wind grid {duration}")
                                        # plt.ylabel("Power (kW e)")
                                        # plt.xticks(ticks=x_ticks, labels=x_labels)
                                        # # plt.legend(loc = 'center left', bbox_to_anchor=(1,0.5), fancybox = True, prop={'size': 8})
                                        # plt.legend(loc = 'upper left', bbox_to_anchor=(0,-0.25), ncol = 3, fancybox = True, prop={'size': 8})   

                                        plt.xlabel(f"Time {xaxis}")
                                        
                                        filename = f'{results_location}/{duration}/electrical {duration}.png'
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
                                        plot_tariff_elec(pick_time(start_hour,end_hour),x_ticks,x_labels,duration,xaxis) 
                                        plot_f_s_t(pick_time(start_hour,end_hour),x_ticks,x_labels,duration,xaxis) 
                                        plot_f_s_t_half(pick_time(start_hour,end_hour),x_ticks,x_labels,duration,xaxis) 
                                        plot_wind_energy(pick_time(start_hour,end_hour),x_ticks,x_labels,duration,xaxis) 
                                        plot_heat(pick_time(start_hour,end_hour),x_ticks,x_labels,duration,xaxis) 
                                        plot_hpeh(pick_time(start_hour,end_hour),x_ticks,x_labels,duration,xaxis) 
                                        plot_electrical(pick_time(start_hour,end_hour),x_ticks,x_labels,duration,xaxis) 
                                        plot_final_node_temp(pick_time(start_hour,end_hour),x_ticks,x_labels,duration,xaxis)
                                        
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
                                    
                                    #%%% Bar plot monthly
                                    
                                    # Define the number of days in each month (non-leap year)
                                    days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
                                    
                                    # Initialize an empty list to store the monthly data
                                    DemH_monthly_data = []
                                    total_Qout_HD_WS_NoS_monthly_data = []
                                    total_Qout_HD_S_monthly_data = []
                                    total_Qout_HD_G_NoS_monthly_data = []
                                    
                                    total_eIn_HD_NoS_monthly_data = []
                                    total_eIn_HD_S_monthly_data = []
                                    total_eIn_CS_WS_monthly_data = []
                                    total_eIn_HD_WS_NoS_monthly_data = []
                                    total_eIn_HD_WS_S_monthly_data = []
                                    total_eIn_HD_G_NoS_monthly_data = []
                                    total_eIn_HD_G_S_monthly_data = []
                                    
                                    total_Qout_CS_WS_monthly_data = []
                                    total_Qex_S_monthly_data = []
                                    Qout_StoHD_direct_monthly_data = []
                                    Qloss_S_g_monthly_data = []
                                    
                                    WS_monthly_data = []
                                    
                                    # For each month, slice the corresponding number of days & append to the list
                                    start = 17520 * (Mod_Year-1) # last year only
                                    for days in days_in_month:
                                        end = start + days * 48 # no. of data
                                        
                                        DemH_monthly_data.append(DemH[start:end])
                                        total_Qout_HD_WS_NoS_monthly_data.append(total_Qout_HD_WS_NoS[start:end])
                                        total_Qout_HD_S_monthly_data.append(total_Qout_HD_S[start:end])
                                        total_Qout_HD_G_NoS_monthly_data.append(total_Qout_HD_G_NoS[start:end])
                                        
                                        total_eIn_HD_NoS_monthly_data.append(total_eIn_HD_NoS[start:end])
                                        total_eIn_HD_S_monthly_data.append(total_eIn_HD_S[start:end])
                                        total_eIn_CS_WS_monthly_data.append(total_eIn_CS_WS[start:end])
                                        total_eIn_HD_WS_NoS_monthly_data.append(total_eIn_HD_WS_NoS[start:end])
                                        total_eIn_HD_WS_S_monthly_data.append(total_eIn_HD_WS_S[start:end])
                                        total_eIn_HD_G_NoS_monthly_data.append(total_eIn_HD_G_NoS[start:end])
                                        total_eIn_HD_G_S_monthly_data.append(total_eIn_HD_G_S[start:end])
                                        
                                        total_Qout_CS_WS_monthly_data.append(total_Qout_CS_WS[start:end])
                                        total_Qex_S_monthly_data.append(total_Qex_S[start:end])
                                        Qout_StoHD_direct_monthly_data.append(Qout_StoHD_direct[start:end])
                                        Qloss_S_g_monthly_data.append(Qloss_S_g[start:end])
                                        
                                        WS_monthly_data.append(WS[start:end])
                                        
                                        start = end
                                    
                                    # Calculate the sum of data in each month
                                    DemH_monthly_sum = np.array([np.sum(month) for month in DemH_monthly_data])
                                    total_Qout_HD_WS_NoS_monthly_sum = np.array([np.sum(month) for month in total_Qout_HD_WS_NoS_monthly_data])
                                    total_Qout_HD_S_monthly_sum = np.array([np.sum(month) for month in total_Qout_HD_S_monthly_data])
                                    total_Qout_HD_G_NoS_monthly_sum = np.array([np.sum(month) for month in total_Qout_HD_G_NoS_monthly_data])
                                    
                                    total_eIn_HD_NoS_monthly_sum = np.array([np.sum(month) for month in total_eIn_HD_NoS_monthly_data])
                                    total_eIn_HD_S_monthly_sum = np.array([np.sum(month) for month in total_eIn_HD_S_monthly_data])
                                    total_eIn_CS_WS_monthly_sum = np.array([np.sum(month) for month in total_eIn_CS_WS_monthly_data])
                                    total_eIn_HD_WS_NoS_monthly_sum = np.array([np.sum(month) for month in total_eIn_HD_WS_NoS_monthly_data])
                                    total_eIn_HD_WS_S_monthly_sum = np.array([np.sum(month) for month in total_eIn_HD_WS_S_monthly_data])
                                    total_eIn_HD_G_NoS_monthly_sum = np.array([np.sum(month) for month in total_eIn_HD_G_NoS_monthly_data])
                                    total_eIn_HD_G_S_monthly_sum = np.array([np.sum(month) for month in total_eIn_HD_G_S_monthly_data])
                                    Overall_Elec = total_eIn_HD_NoS_monthly_sum+total_eIn_HD_S_monthly_sum+total_eIn_CS_WS_monthly_sum+total_eIn_HD_WS_NoS_monthly_sum+total_eIn_HD_WS_S_monthly_sum+total_eIn_HD_G_NoS_monthly_sum+total_eIn_HD_G_S_monthly_sum
                                    
                                    total_Qout_CS_WS_monthly_sum = np.array([np.sum(month) for month in total_Qout_CS_WS_monthly_data])
                                    total_Qex_S_monthly_sum = np.array([np.sum(month) for month in total_Qex_S_monthly_data])
                                    Qout_StoHD_direct_monthly_sum = np.array([np.sum(month) for month in Qout_StoHD_direct_monthly_data])
                                    Qloss_S_g_monthly_sum = np.array([np.sum(month) for month in Qloss_S_g_monthly_data])
                                    
                                    WS_monthly_sum = np.array([np.sum(month) for month in WS_monthly_data])
                                    
                                    # Create month label
                                    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 
                                              'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                                    
                                    # Create a bar plot
                                    fig = plt.figure()
                                    plt.bar(months, total_Qout_HD_WS_NoS_monthly_sum/(10**6), color = c[0])
                                    plt.bar(months, total_Qout_HD_S_monthly_sum/(10**6), bottom = total_Qout_HD_WS_NoS_monthly_sum/(10**6), color = c[1])
                                    plt.bar(months, total_Qout_HD_G_NoS_monthly_sum/(10**6), bottom = total_Qout_HD_WS_NoS_monthly_sum/(10**6) + total_Qout_HD_S_monthly_sum/(10**6), color = c[2])
                                    plt.bar(months, DemH_monthly_sum, width=0.25, color = c[7])
                                    plt.title('HD')
                                    plt.xlabel('Month')
                                    plt.ylabel("Energy (GWh th)")
                                    plt.legend(["HeatOut to HD using WS in NoStore","HeatOut to HD from Store","HeatOut to HD using Non-Wind Grid in NoStore","Heat Demand"],
                                                loc = 'upper center', bbox_to_anchor=(0.5, -0.15), ncol = 2, prop={'size': 8}); plt.tight_layout()
                                    
                                    plt.savefig(f'{results_location}/HD.png',
                                                format = 'png', dpi=300, bbox_inches='tight')
                                    fig.clear()
                                    plt.close(fig)
                                    
                                    fig = plt.figure()
                                    plt.bar(months, total_eIn_HD_NoS_monthly_sum/(10**6), color = c[0])
                                    plt.bar(months, total_eIn_HD_S_monthly_sum/(10**6), bottom = total_eIn_HD_NoS_monthly_sum/(10**6), color = c[1])
                                    plt.bar(months, total_eIn_CS_WS_monthly_sum/(10**6), bottom = total_eIn_HD_NoS_monthly_sum/(10**6) + total_eIn_HD_S_monthly_sum/(10**6), color = c[2])
                                    plt.bar(months, total_eIn_HD_WS_NoS_monthly_sum/(10**6), bottom = total_eIn_HD_NoS_monthly_sum/(10**6) + total_eIn_HD_S_monthly_sum/(10**6) + total_eIn_CS_WS_monthly_sum/(10**6), color = c[3])
                                    plt.bar(months, total_eIn_HD_WS_S_monthly_sum/(10**6), bottom = total_eIn_HD_NoS_monthly_sum/(10**6) + total_eIn_HD_S_monthly_sum/(10**6) + total_eIn_CS_WS_monthly_sum/(10**6) + total_eIn_HD_WS_NoS_monthly_sum/(10**6), color = c[4])
                                    plt.bar(months, total_eIn_HD_G_NoS_monthly_sum/(10**6), bottom = total_eIn_HD_NoS_monthly_sum/(10**6) + total_eIn_HD_S_monthly_sum/(10**6) + total_eIn_CS_WS_monthly_sum/(10**6) + total_eIn_HD_WS_NoS_monthly_sum/(10**6) + total_eIn_HD_WS_S_monthly_sum/(10**6), color = c[5])
                                    plt.bar(months, total_eIn_HD_G_S_monthly_sum/(10**6), bottom = total_eIn_HD_NoS_monthly_sum/(10**6) + total_eIn_HD_S_monthly_sum/(10**6) + total_eIn_CS_WS_monthly_sum/(10**6) + total_eIn_HD_WS_NoS_monthly_sum/(10**6) + total_eIn_HD_WS_S_monthly_sum/(10**6) + total_eIn_HD_G_NoS_monthly_sum/(10**6), color = c[6])
                                    plt.bar(months, Overall_Elec/(10**6), width=0.25, color = c[7])
                                    plt.title('Overall Elec Used')
                                    plt.xlabel('Month')
                                    plt.ylabel("Energy (GWh e)")
                                    plt.legend(["ElecIn to HD in NoStore","ElecIn to HD in Store","ElecIn to charge Store using WS","ElecIn to HD using WS in NoStore","ElecIn to HD using WS in Store",
                                                "ElecIn to HD using Non-Wind Grid in NoStore","ElecIn to HD using Non-Wind Grid in Store","Overall Elec"],
                                                loc = 'upper center', bbox_to_anchor=(0.5, -0.15), ncol = 2, prop={'size': 8}); plt.tight_layout()
                                    plt.savefig(f'{results_location}/Elec.png',
                                                format = 'png', dpi=300, bbox_inches='tight')
                                    fig.clear()
                                    plt.close(fig)
                                    
                                    x = np.arange(len(months))
                                    width = 0.3
                                    fig = plt.figure()
                                    plt.bar(x-width, total_Qout_CS_WS_monthly_sum/(10**6), width, color = c[0])
                                    plt.bar(x, total_Qex_S_monthly_sum/(10**6), width, color = c[1])
                                    plt.bar(x, Qout_StoHD_direct_monthly_sum/(10**6), width, color = c[7])
                                    plt.bar(x+width, Qloss_S_g_monthly_sum/(10**6), width, color = c[2])
                                    plt.title('Store')
                                    plt.xlabel('Month')
                                    plt.xticks(x, labels=months)
                                    plt.ylabel("Energy (GWh th)")
                                    plt.legend(["HeatOut Charge to Store using WS","Heat Extract from Store","HeatOut from Store to HD directly","Heat Losses from Store to ground"],
                                                loc = 'upper center', bbox_to_anchor=(0.5, -0.15), ncol = 2, prop={'size': 8}); plt.tight_layout()
                                    plt.savefig(f'{results_location}/store.png',
                                                format = 'png', dpi=300, bbox_inches='tight')
                                    fig.clear()
                                    plt.close(fig)       
                                    
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
                                    
#%% Combine kpi_last      
if combine_kpi_last == 1:                                
  def combine_csv_files(root_dir, file_name):
    combined_data = pd.DataFrame()
    for subdir, dirs, files in os.walk(root_dir):
      for file in files:
        if file_name in file:
          temp_data = pd.read_csv(os.path.join(subdir, file), encoding='ISO-8859-1')
          combined_data = pd.concat([combined_data, temp_data])
    return combined_data
  root_dir = '/Users/cmb22235/OneDrive - University of Strathclyde/Desktop/STEaM WP4 team/Energy Flow & MTES/Results'
  file_name = 'kpi last'  # replace with your file name
  combined_data = combine_csv_files(root_dir, file_name)
  combined_data.to_csv('combined.csv', index=False, encoding='ISO-8859-1')
