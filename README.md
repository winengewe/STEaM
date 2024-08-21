# STEaM
GigaWatt-Hour Subsurface Thermal Energy storAge: Engineered structures and legacy Mine shafts: STEaM

# Objectives

To investigate the potential for Mine Shaft Thermal Energy Storage (MSTES) to deliver low-cost renewable district heating and support balancing of the future renewable electricity grid. 

The approach taken has been:

•	Key knowledge was extracted from the relevant literature.
•	A case study and scenarios were defined.
•	Concepts were developed for MSTES integration, and KPIs proposed.
•	A model of MSTES integration into district heating and renewable grid was developed.
•	The model and case study were used to deliver insights into potential performance. 
•	The viability of MSTES in a future renewable grid was discussed. 

# Description

shaftstore_5a.py is a new thermal model of the mineshaft was developed to evaluate temperatures of the stored water, mineshaft structure, and surrounding geology. The nodal model comprises multiple connected horizontal nodes representing water, shaft walls, and surrounding geology. The internal thermal model of the shaft is the PyLESA multi-node TES tank model [1] representing an open-loop system where hot water is added to the top layer and cooler water removed from the bottom layer to be heated etc. This model defines a series of differential equations for dT/dt for each layer that are solved using ODE solvers. In the MSTES model the standard tank heat loss parameters are replaced by heat exchanges with the concrete wall of the shaft and the surrounding geology using the same multi-ring concept as the CaRM (Capacity Resistance) model [2].

[1] Lyden A, Flett G, Tuohy PG. PyLESA: A Python modelling tool for planning-level Local, integrated, and smart Energy Systems Analysis. SoftwareX 2021;14:100699. https://doi.org/10.1016/j.softx.2021.100699.

[2] Katsura T, Higashitani T, Fang Y, Sakata Y, Nagano K, Akai H, et al. A New Simulation Model for Vertical Spiral Ground Heat Exchangers Combining Cylindrical Source Model and Capacity Resistance Model. Energies 2020;13:1339. https://doi.org/10.3390/en13061339.

Two separate scenarios were then investigated:
1.	Integration of MSTES into a HP supplied district heating system where the grid support would be running preferentially during wind surplus periods to consume wind that would be curtailed and avoiding running in periods of wind shortfall.
2.	Integration of MSTES into a CHP supplied district heating system where the grid support would be through running the CHP preferentially during periods of wind shortfall and avoiding exports in periods of wind surplus to reduce wind curtailment.

Both scenarios were assessed from ‘feasibility’, ‘flexibility’ and ‘financial’ perspectives. Feasibility considered the practicality of MSTES as useful storage including efficiency, temperatures in the shaft and surrounding geology etc. The flexibility provision of the store includes the size of flexible load, the duration of electrical load shift etc. For the HP supplied system % and quantity of electricity imported during wind surplus periods would be indicators of grid support, for the system CHP the % and quantity of electricity exported during periods of wind shortfall. Financial performance of the system is captured in levelised cost of heat (LCOH) with fuel costs based on imports (minus revenue from exports exported for the CHP case), plus capital costs with and without the MSTES integration with the base system, are allocated across the heat supplied to the end consumer.

MSTES-CHP.py is MineShaft Thermal Energy Storage - Combined Heat and Power Python file

MSTES-HP.py is MineShaft Thermal Energy Storage - Heat Pump Python file

# How to use

1) Download the all the files.
2) Open the XXX.py file in an Integrated Development Environment(IDE) for Python (etc. Spyder, Pycharm, Vscode).
3) Open 'shaftstore_5a.py' to modify the code for the thermal store.
4) Open either 'MSTES-CHP.py' or 'MSTES-HP.py' for CHP or HP integration energy system.
5) Change the directory of the files in the code for the inputs and outputs.
6) Change the variables and parameters according to your system design.
7) Run the code.
8) If you want to combine all the data in csv., open the 'combine_csv.py' and it will generate a 'combined.csv' file.
9) Check the results (graphs in png. format and data in csv. format) in the 'results' folder.
