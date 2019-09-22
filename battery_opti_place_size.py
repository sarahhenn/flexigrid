# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 15:34:57 2019

@author: she
"""

# import extern functions
import numpy as np
import gurobipy as gp
import pickle
import time
import matplotlib.pyplot as plt
import pandas as pd
import pandapower as pp
import pandapower.networks as nw
import pandapower.plotting as plot
from pandapower.plotting.simple_plot_bat import simple_plot_bat
from pandapower.plotting.plotly import simple_plotly
# import own function
import python.clustering_medoid as clustering


# set parameters 
building_type = "EFH"       # EFH, ZFH, MFH_6WE, MFH_10WE, MFH_15WE
building_age  = "2005"      # 1960, 1980, 2005 
emission_year = "2017"      # 2017, 2030, 2050 
# TODO: implement mixed shares of buildings
# TODO: adjust emission factors regarding to national weather conditions
 
# TODO: load data for useable roofarea per building type
# TODO: calculate PV injection within the model, if possible set PV area as MILP-variable     
#useable_roofarea  = 0.30    #Default value: 0.25

# set options
options =   {"dhw_electric": True, # define if dhw is provided electrically
            "P_pv": 10.0, # installed peak PV power
            "eta_inverter": 0.97, # PV inverter efficiency
            "show_grid_plots": False, # show gridplots before and after optimization
            "filename_results": "results/" + building_type + "_" + \
                                                   building_age + ".pkl"
            }

# build dictionary with technical data
batData =   {"Cap_min":0.0,
             "Cap_max":150.0,
             "etaCh": 0.95,
             "etaDis": 0.95,
             "selfDis":0.0,
             "pc_ratio": 1.0,
             "c_inv": 800.0, # price for battery [€/kW]
             "c_om_rel": 0.05, # percentual share for operation and maintenance costs
             "lifetime": 15 
             }

# build dictionary with prices
prices =   {"elec_energy": 0.278, # electricity base price [€/a]
            "elec_base": 150.0, # electricity price [€/kWh]
            "sell": 0.1018, # feed-in-tariff: EEG (10/2020) for PV plants up to 10 kW
            }

# build dictionary with further economical data
eco     =   {"t_calc": 15, # calculation period
             "rate": 0.05, # interest rate 
             "infl": 0.02, # inflation rate
             "elec_prChange": 0.0 # future increase in electricity price
            }
   
eco["q"]            = 1 + eco["rate"]
# compute capital recovery factor (crf) and price-dynamic cash value factor (b)
eco["crf"]          = ((eco["q"] ** eco["t_calc"] * eco["rate"]) / (eco["q"] ** eco["t_calc"] - 1)) 
eco["b"]            = {}                       
eco["b"]["elec"]    = ((1 - (eco["elec_prChange"] / eco["q"]) ** eco["t_calc"]) / (eco["q"] - eco["elec_prChange"]))
eco["b"]["infl"]    = ((1 - (eco["infl"] / eco["q"]) ** eco["t_calc"]) / (eco["q"] - eco["infl"]))

# determine residual value for battery     
T_n = batData["lifetime"]
T   = eco["t_calc"]        
n   = int(T/T_n)
r   = eco["infl"]
q   = eco["q"]     
rval = (sum((r/q)**(n*T_n) for n in range(0,n+1)) - ((r**(n*T_n) * ((n+1)*T_n - T)) / (T_n * q**T)))
                     
#%% data import

#determine the optimization folder in which all input data and results are placed
operationFolder="D:\\git\\flexigrid"
#the input data is always in this source folder
sourceFolder=operationFolder+"\\input"

raw_inputs = {} 

raw_inputs["heat"]  = np.maximum(0, np.loadtxt(sourceFolder+"\\Typgebäude\\"+building_type+"\\"+building_age+"\\heat.csv") / 1000) 
raw_inputs["dhw"]  = np.maximum(0, np.loadtxt(sourceFolder+"\\Typgebäude\\"+building_type+"\\"+building_age+"\\dhw.csv") / 1000) 
raw_inputs["electricity"]  = np.maximum(0, np.loadtxt(sourceFolder+"\\Typgebäude\\"+building_type+"\\"+building_age+"\\electricity.csv") / 1000) 
raw_inputs["solar_roof"]  = np.maximum(0, np.loadtxt(sourceFolder+"\\Typgebäude\\"+building_type+"\\"+building_age+"\\solar_roof.csv") / 1000)       
raw_inputs["temperature"] = np.loadtxt(sourceFolder+"\\Typgebäude\\"+building_type+"\\"+building_age+"\\temperature.csv")

emi_input = pd.read_csv(sourceFolder+"\\emission_factor_"+emission_year+".csv", header=0, usecols=[2])
raw_inputs["co2"] = np.zeros([8760])    
for t in range (0, 8760):
    i=t*4
    raw_inputs["co2"][t]= np.mean(emi_input[i:(i+4)])


#%% data clustering 
    
inputs_clustering = np.array([raw_inputs["heat"], 
                              raw_inputs["dhw"],
                              raw_inputs["electricity"],
                              raw_inputs["solar_roof"],
                              raw_inputs["temperature"],
                              raw_inputs["co2"]])

number_clusters = 12
(inputs, nc, z) = clustering.cluster(inputs_clustering, 
                                     number_clusters=number_clusters,
                                     norm=2,
                                     mip_gap=0.0,
                                     weights=[1,1,1,1,0,1])


# Determine time steps per day
len_day = int(inputs_clustering.shape[1] / 365)

clustered = {}
clustered["heat"]        = inputs[0]
clustered["dhw"]         = inputs[1]
clustered["electricity"] = inputs[2]
clustered["solar_irrad"] = inputs[3]
clustered["temperature"] = inputs[4]
clustered["co2"]         = inputs[5]
clustered["weights"]     = nc
clustered["z"]           = z
        
#%% set and calculate building energy system data, as well as load and injection profiles

# calculate parameters for load and generation
if options ["dhw_electric"]:
    powerElec = clustered["electricity"] + clustered["dhw"]
else:
    powerElec = clustered["electricity"]

# calculate PV injection
pv_data = pd.read_excel (sourceFolder+"\\pv_info.xlsx")

i_NOCT = pv_data["i_NOCT"][0] # [kW/m²]
T_NOCT = pv_data["T_NOCT"][0] # [°C] 
P_NOCT = pv_data["P_NOCT"][0] # [kW]     
gamma = pv_data["gamma"][0]   # [%/kW]
area_nom = pv_data["area"][0] # [m²]
P_nom = pv_data["P_nom"][0] # [kW]
eta_inverter = pv_data["eta_inverter"][0] # [-]

# Interpolate cell temperature.
# Without solar irradiation, the cell temperature has to be equal to the ambient temperature. 
# At NOCT irradiation, the cell's temperature has to be equal to t_NOCT
T_cell = (clustered["temperature"] + clustered["solar_irrad"] / i_NOCT * (T_NOCT - clustered["temperature"]))
eta_NOCT = P_NOCT / (area_nom * i_NOCT)
# Compute electrical efficiency of the cell
eta_el   = eta_NOCT * (1 +  gamma / 100 * (T_cell - T_NOCT))

# compute generated power
powerGen = options["P_pv"] *(area_nom/P_nom) * eta_el * eta_inverter * clustered["solar_irrad"]

#build dictionary with misc data
days = [i for i in range(number_clusters)]
timesteps = [i for i in range(len_day)]
dt = 1

#%% create network

# load example net (IEEE 9 buses)
'''
typical kerber grids:   landnetz_freileitung_1(), landnetz_freileitung_2(), landnetz_kabel_1(), landnetz_kabel_2(),
                        dorfnetz(), vorstadtnetz_kabel_1(), vorstadtnetz_kabel_2()
    -> create network with nw.create_kerber_name
                        
extreme kerber grids:   landnetz_freileitung(), landnetz_kabel(), landnetz_freileitung_trafo(), landnetz_kabel_trafo(), 
                        dorfnetz(), dorfnetz_trafo(), 
                        vorstadtnetz_1(), vorstadtnetz_2(), vorstadtnetz_trafo_1(), vorstadtnetz_trafo_2()
    -> create network with nw.kb_extrem_name   
            
'''
#net = nw.create_kerber_landnetz_freileitung_2()
net = nw.create_kerber_dorfnetz()

if options["show_grid_plots"]:
# simple plot of net with existing geocoordinates or generated artificial geocoordinates
    plot.simple_plot(net, show_plot=True)
    plot.show()

#%% extract node and line information from pandas-network

# specify grid nodes for whole grid and trafo; choose and allocate load, injection and battery nodes
nodes = {}

nodes["grid"] = net.bus.index.to_numpy()
nodes["trafo"] = net.trafo['lv_bus'].to_numpy()
nodes["load"] = net.load['bus'].to_numpy()
#nodeInj = net.load['bus'].to_numpy()
nodes["bat"] = net.load['bus'].to_numpy()

gridnodes = list(nodes["grid"])

# extract existing lines 
nodeLines = []
for i in range(len(net.line['from_bus'])):
    nodeLines.append((net.line['from_bus'][i],net.line['to_bus'][i]))
nodeLines = gp.tuplelist(nodeLines)

# extract maximal current for lines
# multiply with 400 V to get maximal power in kW          
powerLine_max = {}
for [n,m] in nodeLines:
    powerLine_max[n,m] = (net.line['max_i_ka'][nodeLines.index((n,m))])*400
    
# extract battery nodes and define technical data for them
capBat_max = {}
capBat_min = {}

for n in gridnodes:
    if n in nodes["bat"]:
        capBat_max[n] = batData["Cap_max"]
        capBat_min[n] = batData["Cap_min"]

# attach plug-in loads and PV generatrion to building nodes
# TODO: do the same with EV loads!?
powerPlug = {}
powerPV = {}
for n in gridnodes:
    for d in days:
        for t in timesteps:
            if n in nodes["load"]:
                powerPlug[n,d,t] = powerElec[d,t]
                powerPV[n,d,t] = powerGen[d,t]
            else:
                powerPlug[n,d,t] = np.zeros_like(powerElec[d,t])
                powerPV[n,d,t] = np.zeros_like(powerGen[d,t])
        

#%% optimization model

print("start modelling")
model = gp.Model("Optimal Battery Placement and Sizing")
#%% economic variables

# initiate cost variables: there are cost-variables for investment, operation & maintenance, 
# demand costs (electricity, fuel costs) and fix costs resulting from base prices 
c_inv  = model.addVars(gridnodes, vtype="C", name="c_inv")      
c_om   = model.addVars(gridnodes, vtype="C", name="c_om")            
c_dem  = model.addVars(gridnodes, vtype="C", name="c_dem")         
c_fix  = model.addVars(gridnodes, vtype="C", name="c_fix")  

# revenues and subsidies                
revenue = model.addVars(gridnodes, vtype="C", name="revenue_")

# variables for total node costs, total costs and total emissions
c_node = model.addVars(gridnodes, vtype="C", name="c_total", lb= -gp.GRB.INFINITY)
c_total = model.addVar(vtype="C", name="c_total", lb= -gp.GRB.INFINITY)
emission_node = model.addVars(gridnodes, vtype="C", name= "CO2_emission", lb= -gp.GRB.INFINITY) 
emission = model.addVar(vtype="C", name= "CO2_emission", lb= -gp.GRB.INFINITY)  

#%% technical variables

# add grid variables to model

# set trafo bounds due to technichal limits
trafo_min = float(net.trafo.sn_mva*(-1000.))
trafo_max = float(net.trafo.sn_mva*1000.)
powerTrafo = model.addVars(days,timesteps, vtype="C", lb=trafo_min, ub=trafo_max, name="powerTrafo_"+str(t))
        
# set line bounds due to technical limits                             
powerLine = model.addVars(nodeLines,days,timesteps, vtype="C", lb=-10000, name="powerLine_")

# add bat variables to model
# x = model.addVars(dridnodes, vtype="B", name="bat_existance_"+str(n))      
# y = model.addVars(gridnodes, days, timesteps, vtype="B", name= "activation_charge_"+str(n)+str(t))
capacity = model.addVars(gridnodes, vtype="C", name="Cap_"+str(n))
SOC = model.addVars(gridnodes, days, timesteps, vtype="C", name="SOC_"+str(n)+str(d)+str(t))
SOC_init = model.addVars(gridnodes, days, vtype="C", name="SOC_init_"+str(n)+str(d))
powerCh = model.addVars(gridnodes, days, timesteps, vtype="C", name="powerLoad_"+str(n)+str(t))
powerDis = model.addVars(gridnodes, days, timesteps, vtype="C", name="powerFeed_"+str(n)+str(t))

# define auxilary variables to compute resulting load and injection per node
powerLoad = model.addVars(gridnodes, days, timesteps, vtype="C", name="powerLoad_"+str(n)+str(t))
powerInj = model.addVars(gridnodes, days, timesteps, vtype="C", name="powerInj_"+str(n)+str(t))
powerInjPV = model.addVars(gridnodes, days, timesteps, vtype="C", name="powerInjPV_"+str(n)+str(t))
powerInjBat = model.addVars(gridnodes, days, timesteps, vtype="C", name="powerInjBat_"+str(n)+str(t))
powerUsePV = model.addVars(gridnodes, days, timesteps, vtype="C", name="powerUseBat_"+str(n)+str(t))
powerUseBat = model.addVars(gridnodes, days, timesteps, vtype="C", name="powerUseBat_"+str(n)+str(t))
    
model.update()

#%% define constraints

## economical constraints

model.addConstr(c_total ==  (c_inv.sum('*') 
                            + c_om.sum('*') 
                            + c_dem.sum('*') 
                            + c_fix.sum('*')
                            - revenue.sum('*')))
     
# compute annual investment costs
model.addConstrs((c_inv[n] == eco["crf"] * rval * batData["c_inv"] * capacity[n]
                    for n in gridnodes), name="investment_costs"+str(n))

# compute annual operation and maintenance costs
model.addConstrs((c_om[n] == eco["b"]["infl"] * batData["c_om_rel"] * c_inv[n]
                    for n in gridnodes), name="maintenance_costs"+str(n))

# compute annual demand related costs
el_total_node = {}
for n in gridnodes:
    el_total_node[n] = (dt * sum(clustered["weights"][d] * sum(powerLoad[n,d,t] 
                        for t in timesteps) for d in days) * dt)
    
model.addConstrs((c_dem[n] == eco["crf"] * eco["b"]["infl"] * el_total_node[n] * prices["elec_energy"]
                    for n in gridnodes), name="demand_costs"+str(n))

# compute annual fix costs for electricity
model.addConstrs((c_fix[n] == prices["elec_base"] for n in gridnodes), name="fix_costs"+str(n))

# compute annual revenues for electricity feed-in
model.addConstrs((revenue[n] == prices["sell"] for n in gridnodes), name="revenues"+str(n))

# annual electricity demand per node
el_total_node = {}
for n in gridnodes:
    el_total_node[n] = (dt * sum(clustered["weights"][d] * sum(powerLoad[n,d,t] 
                        for t in timesteps) for d in days) * dt)
            
 
#%% grid constraints
# set energy balance for all nodes
for n in gridnodes:
    for d in days:
        for t in timesteps:
            if n in nodes["trafo"]:
            
                model.addConstr(powerLine.sum(n,'*',d,t) - powerLine.sum('*',n,d,t) == 
                                powerTrafo[d,t], name="node balance_"+str(n)+str(t))

            else:
                model.addConstr(powerLine.sum(n,'*',d,t) - powerLine.sum('*',n,d,t) == 
                                powerInj[n,d,t] - powerLoad[n,d,t], name="node balance_"+str(n))

# set line limits
# TODO: check if it's better to set line limits like this or to set lb/ub of variable to min/max values      
for [n,m] in nodeLines:
        for d in days:
            for t in timesteps: 
            
                model.addConstr(powerLine[n,m,d,t] <= powerLine_max[n,m], name="line power max_"+str(n)+str(m)+str(t))
                model.addConstr(powerLine[n,m,d,t] >= (-1)*powerLine_max[n,m], name="line power min_"+str(n)+str(m)+str(t))
            
            
#%% battery constraints

# binary variables x/y needed? don't think so right  now -> build LP

# maximum power is defined by power/capacity ratio       
for n in gridnodes:
    for d in days:                   
        for t in timesteps:
            if n in nodes["bat"]:
                
                model.addConstr(powerCh[n,d,t]  >= 0, name="min power_"+str(n)+str(t))
                model.addConstr(powerCh[n,d,t]  <= capacity[n]*batData["pc_ratio"], name="max power_"+str(n)+str(t))
                model.addConstr(powerDis[n,d,t] >= 0, name="min power_"+str(n)+str(t))
                model.addConstr(powerDis[n,d,t] <= capacity[n]*batData["pc_ratio"], name="max power_"+str(n)+str(t))

# set limitations for battery capacity
for n in gridnodes:
    if n in nodes["bat"]:

        model.addConstr(capacity[n] <= capBat_max[n], name="Battery_capacity_max")
        model.addConstr(capacity[n] >= capBat_min[n], name="Battery_capacity_min")
        
        model.addConstrs((capacity[n] >= SOC_init[n,d] for d in days), name="Battery_capacity_SOC_init")
        model.addConstrs((capacity[n] >= SOC[n,d,t] for d in days for t in timesteps), name="Battery_capacity_SOC")
        
    else:
        
        model.addConstr(capacity[n] == 0, name="Battery_capacity_max")
                
# SOC repetitions: SOC at the end of typeday == SOC at the beginning of typeday
for n in gridnodes:   
    for d in days:   
        
        model.addConstr(SOC_init[n,d] == SOC[n,d,len(timesteps)-1],
                                                       name="repetitions_" +str(d))

for n in gridnodes:   
    for d in days:         
        for t in timesteps:
            if t == 0:
                SOC_previous = SOC_init[n,d]
            else:
                SOC_previous = SOC[n,d,t-1]
        
            model.addConstr(SOC[n,d,t] == SOC_previous 
                        + (dt * (powerCh[n,d,t] * batData["etaCh"] - powerDis[n,d,t]/batData["etaDis"])) 
                        - batData["selfDis"]*dt*SOC_previous, name="storage balance_"+str(n)+str(t))
            
#%% energy balances for every node

# split injected power in power from PV and power from battery
model.addConstrs((powerInj[n,d,t] == powerInjPV[n,d,t] + powerInjBat[n,d,t] 
                  for n in gridnodes for d in days for t in timesteps), name="powerInj"+str(n)+str(d)+str(t))

# split power from PV generation in injected and used power
model.addConstrs((powerInjPV[n,d,t] == powerPV[n,d,t] - powerUsePV[n,d,t] 
                  for n in gridnodes for d in days for t in timesteps), name="powerInjPV"+str(n)+str(d)+str(t))

# split battery discharging power in injected and used power
model.addConstrs((powerDis[n,d,t] == powerInjBat[n,d,t] + powerUseBat[n,d,t] 
                  for n in gridnodes for d in days for t in timesteps), name="powerInj_UseBat"+str(n)+str(d)+str(t))

# node energy balance
model.addConstrs((powerPlug[n,d,t] + powerCh[n,d,t] == 
                  powerLoad[n,d,t] + powerUsePV[n,d,t] + powerUseBat[n,d,t] 
                  for n in gridnodes for d in days for t in timesteps), name="powerInj_UseBat"+str(n)+str(d)+str(t))         
            
#%% start optimization

# set objective function

model.setObjective(sum(sum(powerTrafo[d,t]*clustered["co2"][d,t] for t in timesteps) for d in days), gp.GRB.MINIMIZE)                

# adgust gurobi settings
model.Params.TimeLimit = 25

model.Params.MIPGap = 0.02
model.Params.NumericFocus = 3
model.Params.MIPFocus = 3
model.Params.Aggregate = 1

model.optimize()

if model.status==gp.GRB.Status.INFEASIBLE:
    model.computeIIS()
    f=open('errorfile.txt','w')
    f.write('\nThe following constraint(s) cannot be satisfied:\n')
    for c in model.getConstrs():
        if c.IISConstr:
            f.write('%s' % c.constrName)
            f.write('\n')
    f.close()


#%% retrieve results

# grid results    
res_powerTrafo = {}
res_powerTrafo = np.array([[powerTrafo[d,t].X for t in timesteps]for d in days])
    
res_powerLine = {}
for [n,m] in nodeLines:
    res_powerLine[n,m] = np.array([[powerLine[n,m,d,t].X for t in timesteps] for d in days])    

# battery operation results
res_capacity = {}
res_powerCh = {}
res_powerDis = {}
res_SOC = {}
res_SOC_init = {}
for n in gridnodes:
    res_capacity = np.array([capacity[n].X for n in gridnodes])
    res_powerCh[n] = np.array([[powerCh[n,d,t].X for t in timesteps] for d in days])
    res_powerDis[n] = np.array([[powerDis[n,d,t].X for t in timesteps] for d in days])
    res_SOC[n] = np.array([[SOC[n,d,t].X for t in timesteps] for d in days])
    res_SOC_init[n] = np.array([SOC_init[n,d].X for d in days])

# node energy management results
    
res_powerLoad = {}
res_powerInj = {}
res_powerInjPV = {}
res_powerInjBat = {}
res_powerUsePV = {}
res_powerUseBat = {}

for n in gridnodes:
    res_powerLoad[n] = np.array([[powerLoad[n,d,t].X for t in timesteps] for d in days])
    res_powerInj[n] = np.array([[powerInj[n,d,t].X for t in timesteps] for d in days])
    res_powerInjPV[n] = np.array([[powerInjPV[n,d,t].X for t in timesteps] for d in days])
    res_powerInjBat[n] = np.array([[powerInjBat[n,d,t].X for t in timesteps] for d in days])
    res_powerUsePV[n] = np.array([[powerUsePV[n,d,t].X for t in timesteps] for d in days])
    res_powerUseBat[n] = np.array([[powerUseBat[n,d,t].X for t in timesteps] for d in days])

# save results 
with open(options["filename_results"], "wb") as fout:
    pickle.dump(model.ObjVal, fout, pickle.HIGHEST_PROTOCOL)
    pickle.dump(model.Runtime, fout, pickle.HIGHEST_PROTOCOL)  
    pickle.dump(model.MIPGap, fout, pickle.HIGHEST_PROTOCOL)
    pickle.dump(res_powerTrafo, fout, pickle.HIGHEST_PROTOCOL)
    pickle.dump(res_powerLine, fout, pickle.HIGHEST_PROTOCOL)
    pickle.dump(res_capacity, fout, pickle.HIGHEST_PROTOCOL)
    pickle.dump(res_powerCh, fout, pickle.HIGHEST_PROTOCOL)
    pickle.dump(res_powerDis, fout, pickle.HIGHEST_PROTOCOL)
    pickle.dump(res_SOC, fout, pickle.HIGHEST_PROTOCOL)
    pickle.dump(res_SOC_init, fout, pickle.HIGHEST_PROTOCOL)
    pickle.dump(res_powerLoad, fout, pickle.HIGHEST_PROTOCOL)
    pickle.dump(res_powerInj, fout, pickle.HIGHEST_PROTOCOL)
    pickle.dump(res_powerInjPV, fout, pickle.HIGHEST_PROTOCOL)
    pickle.dump(res_powerInjBat, fout, pickle.HIGHEST_PROTOCOL)
    pickle.dump(res_powerUsePV, fout, pickle.HIGHEST_PROTOCOL)
    pickle.dump(res_powerUseBat, fout, pickle.HIGHEST_PROTOCOL)
    pickle.dump(res_c_om, fout, pickle.HIGHEST_PROTOCOL)
    pickle.dump(res_c_dem, fout, pickle.HIGHEST_PROTOCOL)
    pickle.dump(res_c_fix, fout, pickle.HIGHEST_PROTOCOL)
    pickle.dump(res_c_total, fout, pickle.HIGHEST_PROTOCOL)
    pickle.dump(res_rev, fout, pickle.HIGHEST_PROTOCOL)
    pickle.dump(res_emission, fout, pickle.HIGHEST_PROTOCOL)  

#%% plot grid with batteries highlighted
if options["show_grid_plots"]:
    
    bat_ex = np.zeros(len(gridnodes))
    for n in gridnodes:
        if res_capacity[n] >0:
            bat_ex[n] = 1
    
    netx=net
    netx['bat']=pd.DataFrame(bat_ex, columns=['ex'])
    simple_plot_bat(netx, show_plot=True, bus_color='b', bat_color='r')


#    return (res_x, res_power,res_soc, model.ObjVal,model.MIPGap,model.Runtime,model.ObjBound)