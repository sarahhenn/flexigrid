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
import matplotlib
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
options =   {# define if dhw is provided electrically
            "dhw_electric": True }


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

#build dictionary with batData
batData =   {"Pmin":0.0,
             "Pmax":10.0,
             "SOCmin":0.1,
             "SOCmax":0.9,
             "energyContent":10.0,
             "etaCh":0.95,
             "etaDis":0.95,
             "selfDis":0.0, 
             "init":0.5}


# calculate parameters for load and generation
if options ["dhw_electric"]:
    powerLoad = clustered["electricity"] + clustered["dhw"]
else:
    powerLoad = clustered["electricity"]
# TODO: calculate PV generation
powerGen = np.zeros_like(powerLoad)

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
net = nw.create_kerber_landnetz_freileitung_2()

# simple plot of net with existing geocoordinates or generated artificial geocoordinates
plot.simple_plot(net, show_plot=True)

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
powerBat_max = {}
powerBat_min = {}
SOC_max = {}
SOC_min = {}
SOC_init = {}
for n in gridnodes:
    if n in nodes["bat"]:
        powerBat_max[n] = batData["Pmax"]
        powerBat_min[n] = batData["Pmin"]
        SOC_max[n] = batData["SOCmax"]
        SOC_min[n] = batData["SOCmin"]
        SOC_init[n] = batData["init"]
    else:
        powerBat_max[n] = 0.0
        powerBat_min[n] = 0.0
        SOC_max[n] = 0.0
        SOC_min[n] = 0.0
        SOC_init[n] = 0.0

# attach plug-in loads and PV generatrion to building nodes
# TODO: do the same with EV loads!?
powerPlug = {}
powerPV = {}
for n in gridnodes:
    for d in days:
        for t in timesteps:
            if n in nodes["load"]:
                powerPlug[n,d,t] = powerLoad[d,t]
                powerPV[n,d,t] = powerGen[d,t]
            else:
                powerPlug[n,d,t] = np.zeros_like(powerLoad[d,t])
                powerPV[n,d,t] = np.zeros_like(powerGen[d,t])
        

#%% optimization model

print("start modelling")
model = gp.Model("Optimal Battery Placement and Sizing")

# initiate grid variables
powerTrafo = {}
powerLine = {}

# initiate bat variables
#x = {} # Battery existance
#y = {} # Battery activity -> is maybe needed later do avoid simultaneous charging and discharging
SOC = {} # Battery state of charge
powerCh = {} # Power load battery
powerDis = {} #Power feed-in battery

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
SOC = model.addVars(gridnodes, days, timesteps, vtype="C", name="SOC_"+str(n)+str(t))
powerCh = model.addVars(gridnodes, days, timesteps, vtype="C", name="powerLoad_"+str(n)+str(t))
powerDis = model.addVars(gridnodes, days, timesteps, vtype="C", name="powerFeed_"+str(n)+str(t))
    
model.update()

#%% grid optimization

#TODO: stopped here
# set energy balance for all nodes
for n in gridnodes:
    for d in days:
        for t in timesteps:
            if n in nodes["trafo"]:
            
                model.addConstr(powerLine.sum(n,'*',d,t) - powerLine.sum('*',n,d,t) == 
                                powerTrafo[d,t], name="node balance_"+str(n)+str(t))

            else:
                model.addConstr(powerLine.sum(n,'*',d,t) - powerLine.sum('*',n,d,t) == 
                                powerPV[n,d,t] - powerPlug[n,d,t] - powerCh[n,d,t] + powerDis[n,d,t], name="node balance_"+str(n))



# set line limits
# TODO: check if it's better to set line limits like this or to set lb/ub of variable to min/max values      
for [n,m] in nodeLines:
        for d in days:
            for t in timesteps: 
            
                model.addConstr(powerLine[n,m,d,t] <= powerLine_max[n,m], name="line power max_"+str(n)+str(m)+str(t))
                model.addConstr(powerLine[n,m,d,t] >= (-1)*powerLine_max[n,m], name="line power min_"+str(n)+str(m)+str(t))
            
            
#%% battery optimization

# binary variables x/y needed? don't think so right  now -> build LP

for n in gridnodes:
    for d in days:                   
        for t in timesteps:
            if n in nodes["bat"]:
                
                model.addConstr(powerCh[n,d,t]  >= powerBat_min[n], name="min power_"+str(n)+str(t))
                model.addConstr(powerCh[n,d,t]  <= powerBat_max[n], name="max power_"+str(n)+str(t))
                model.addConstr(powerDis[n,d,t] >= powerBat_min[n], name="min power_"+str(n)+str(t))
                model.addConstr(powerDis[n,d,t] <= powerBat_max[n], name="max power_"+str(n)+str(t))

for n in gridnodes:
    for d in days:
        for t in timesteps:
            model.addConstr(SOC[n,d,t] <= SOC_max[n], name="max sto cap_"+str(n)+str(t))
            model.addConstr(SOC[n,d,t] >= SOC_min[n], name="min sto cap_"+str(n)+str(t))

for n in gridnodes:   
    for d in days:         
        for t in timesteps:
            if t == 0:
                SOC_previous = SOC_init[n]
            else:
                SOC_previous = SOC[n,d,t-1]
        
            model.addConstr(SOC[n,d,t] == (SOC_previous 
                        + (dt * (powerCh[n,d,t] * batData["etaCh"] - powerDis[n,d,t]/batData["etaDis"])) / batData["energyContent"]) 
                        - batData["selfDis"]*dt*SOC_previous, name="storage balance_"+str(n)+str(t))
                
            if t == (len(timesteps)-1):
                model.addConstr(SOC[n,d,t] == SOC_init[n])

## set objective function

model.setObjective(sum(sum(powerTrafo[d,t]*clustered["co2"][d,t] for t in timesteps) for d in days), gp.GRB.MINIMIZE)                

# adgust gurobi settings
model.Params.TimeLimit = 60
    
model.setParam('MIPGap',0.01)    
model.setParam('MIPGapAbs',2)
model.setParam('Threads',1) 
model.setParam('OutputFlag',0)
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
    
res_powerLine = {}
for [n,m] in nodeLines:
    res_powerLine[n,m] = np.array([[powerLine[n,m,d,t].X for t in timesteps] for d in days])    

res_powerCh = {}
res_powerDis = {}
res_soc = {}
for n in gridnodes:
    res_powerCh[n] = np.array([[powerCh[n,d,t].X for t in timesteps] for d in days])
    res_powerDis = np.array([[powerDis[n,d,t].X for t in timesteps] for d in days])
    res_soc = np.array([[SOC[n,d,t].X for t in timesteps] for d in days])

res_powerTrafo = {}
res_powerTrafo = np.array([[powerTrafo[d,t].X for t in timesteps]for d in days])


#%% plot grid with batteries highlighted
res_Ch = {}
for n in gridnodes:
    res_Ch[n] = sum(sum(res_powerCh[n][d,t] for t in timesteps) for d in days)

bat_ex = np.zeros(len(gridnodes))
for n in gridnodes:
    if res_Ch[n] >0:
        bat_ex[n] = 1

netx=net
netx['bat']=pd.DataFrame(bat_ex, columns=['ex'])
simple_plot_bat(netx, show_plot=True, bus_color='b', bat_color='r')


#    return (res_x, res_power,res_soc, model.ObjVal,model.MIPGap,model.Runtime,model.ObjBound)