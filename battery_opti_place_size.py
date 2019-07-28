# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 15:34:57 2019

@author: she
"""

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
from os.path import join as pjoin

#determine the optimization folder in which all input data and results are placed
operationFolder="D:\\git\\flexigrid"
#the input data is always in this source folder
sourceFolder=operationFolder+"\\input"

p_dem=np.loadtxt(sourceFolder+"\\p_dem.txt")
p_gen=np.loadtxt(sourceFolder+"\\p_gen.txt")
# TODO: convert generation to negative number
p_gen = np.abs(p_gen)
#co2dyn=np.loadtxt(sourceFolder+"\\co2dyn.txt")
#co2dyn=co2dyn[:,1]
co2 = pd.concat([pd.read_csv(pjoin(sourceFolder, 'emission_factor_2017.csv'), header=0, usecols=[1,2], names=['marg_2017','mix_2017']), 
                 pd.read_csv(pjoin(sourceFolder, 'emission_factor_TM2030.csv'), header=0, usecols=[1,2], names=['marg_2030','mix_2030']),
                 pd.read_csv(pjoin(sourceFolder, 'emission_factor_TM2050.csv'), header=0, usecols=[1,2], names=['marg_2050','mix_2050']),
                 ], axis=1)
#co2dyn=[1.,1.,0.,1.,0.,0.,0.,1.,0.,1.]

#weatherData=np.loadtxt(sourceFolder+"\\weatherData.txt")

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

#%% set input parameter

#build dictionary with misc data
timesteps = 100
misc={"time steps":timesteps, "dt":0.25}

# specify grid nodes
nodeGrid = net.bus.index.to_numpy() # all grid nodes
nodeTrafo = net.trafo['lv_bus'].to_numpy() # trafo
nodeLoad = net.load['bus'].to_numpy() # here: buildings
#nodeInj = net.load['bus'].to_numpy() # here: buildings
nodeBat = net.load['bus'].to_numpy() # here: buildings
nodeLines = []
for i in range(len(net.line['from_bus'])):
    nodeLines.append([net.line['from_bus'][i],net.line['to_bus'][i]])

# extract maximal current for lines
# multiply with 400 V to get maximal power in kW
powerLine_max = np.zeros((len(nodeGrid),len(nodeGrid)))
for k in range(len(nodeGrid)):
    for l in range(len(nodeGrid)):
        if [k,l] in nodeLines:
            powerLine_max[k,l] = (net.line['max_i_ka'][nodeLines.index([k,l])])*400


# specify bat data
P_min_kW = 0.0
P_max_kW = 10.0
SOC_min = 0.1 
SOC_max = 0.9
etaCh = 0.95
etaDis = 0.95
# TODO: sizes (and power? as variables
energyContent_kWh = 10.0
init = 0.5
selfDis = 0.0

#build dictionary with batData
batData={"Pmin":P_min_kW,"Pmax":P_max_kW,"SOCmin":SOC_min,"SOCmax":SOC_max,"energyContent":energyContent_kWh,"etaCh":etaCh,"etaDis":etaDis,"selfDis":selfDis, "init":init}

p_gen = p_gen[15000:(timesteps+15000),:]/1000
p_dem = p_dem[15000:(timesteps+15000),:]/1000

# parameters for load and generation
powerLoad = np.zeros((len(nodeGrid),timesteps))
powerGen = np.zeros((len(nodeGrid),timesteps))
#TODO: Matrizen invertieren

m = 0
for n in range(len(nodeGrid)):
    if n in nodeLoad:
        powerLoad[n,:] = p_dem[:,m]
        m=m+1
        
m = 0
for n in range(len(nodeGrid)):
    if n in nodeLoad:
        powerGen[n,:] = p_gen[:,m] 
        m=m+1

# =============================================================================
# m = 0
# for n in range(len(nodeGrid)):
#     if n in nodeLoad:
#         powerLoad[n,:] = 1 
#         m=m+1
# =============================================================================

#%% optimization model

print("start modelling")
model = gp.Model("Optimal Battery Placement and Sizing")

# initiate grid variables
powerTrafo = {}
powerLine = {}

# initiate bat variables
x = {} # Battery existance
y = {} # Battery activity
SOC = {} # Battery state of charge
powerCh = {} # Power load battery
powerDis = {} #Power feed-in battery

# add grid variables to model
# set trafo bounds due to technichal limits
for t in range(misc["time steps"]):
    powerTrafo[t] = model.addVar(vtype="C", lb=(net.trafo.sn_mva*(-1000.)), ub=(net.trafo.sn_mva*1000.), name="powerTrafo_"+str(t))
        
# set line bounds due to technical limits
#TODO!!! klappt theoretisch, aber auf 0 spalte/zeile achten!  
for t in range(misc["time steps"]):          
    for k in range(len(nodeGrid)):
        for l in range(len(nodeGrid)):
            powerLine[k,l,t] = model.addVar(vtype="C", lb=-10000, name="powerLine_"+str(k)+"_"+str(l))

        
# add variables to model    
for n in range(len(nodeGrid)):
    for t in range(misc["time steps"]):
        
        x[n] = model.addVar(vtype="B", name="bat_existance_"+str(n))
        y[n,t] = model.addVar(vtype="B", name= "activation_charge_"+str(n)+str(t))
        
        SOC[n,t] = model.addVar(vtype="C", name="SOC_"+str(n)+str(t))
        powerCh[n,t] = model.addVar(vtype="C", name="powerLoad_"+str(n)+str(t))
        powerDis[n,t] = model.addVar(vtype="C", name="powerFeed_"+str(n)+str(t))
    
model.update()

#%% grid optimization

# set energy balance for all nodes
for n in range(len(nodeGrid)):
    for t in range(misc["time steps"]):
        if n in nodeTrafo:
            model.addConstr(sum(powerLine[n,m,t] for m in nodeGrid)- sum(powerLine[m,n,t] for m in nodeGrid) == powerTrafo[t], name="node balance_"+str(n)+str(m)+str(t))
        else:
            model.addConstr(sum(powerLine[n,m,t] for m in nodeGrid)- sum(powerLine[m,n,t] for m in nodeGrid) == 
                            powerGen[n,t] - powerLoad[n,t] - powerCh[n,t] + powerDis[n,t], name="node balance_"+str(n)+str(m)+str(t))

print("halfway there")

#TODO!!! klappt theoretisch, aber auf 0 spalte/zeile achten!             
for t in range(misc["time steps"]): 
    for k in range(len(nodeGrid)):
        for l in range(len(nodeGrid)):
            
            model.addConstr(powerLine[k,l,t] <= powerLine_max[k,l], name="line power max_"+str(k)+str(l)+str(t))
            model.addConstr(powerLine[k,l,t] >= (-1)*powerLine_max[k,l], name="line power min_"+str(k)+str(l)+str(t))
            
            
#%% battery optimization

for n in range(len(nodeGrid)):                    
    for t in range(misc["time steps"]):
        
        model.addConstr(powerCh[n,t]  >= x[n] * y[n,t] * batData["Pmin"], name="min power_"+str(n)+str(t))
        model.addConstr(powerCh[n,t]  <= x[n] * y[n,t] * batData["Pmax"], name="max power_"+str(n)+str(t))
        model.addConstr(powerDis[n,t] >= x[n] * (1-y[n,t]) * batData["Pmin"], name="min power_"+str(n)+str(t))
        model.addConstr(powerDis[n,t] <= x[n] * (1-y[n,t]) * batData["Pmax"], name="max power_"+str(n)+str(t))
    
        
#TODO: später an Entscheidungsvariable für Speicherexistenz knüpfen 
for n in range(len(nodeGrid)):
    if n not in nodeBat:
        model.addConstr(x[n] == 0, name="battery_existance_"+str(n))

for n in range(len(nodeGrid)):
    for t in range(misc["time steps"]):
        model.addConstr(SOC[n,t] <= x[n]*batData["SOCmax"], name="max sto cap_"+str(n)+str(t))
        model.addConstr(SOC[n,t] >= x[n]*batData["SOCmin"], name="min sto cap_"+str(n)+str(t))

for n in range(len(nodeGrid)):            
    for t in range(misc["time steps"]):
        if t == 0:
            SOC_previous = batData["init"]*x[n]
        else:
            SOC_previous = SOC[n,t-1]
        
        model.addConstr(SOC[n,t] == (SOC_previous 
                        + (misc["dt"] * (powerCh[n,t] * batData["etaCh"] - powerDis[n,t]/batData["etaDis"])) / batData["energyContent"]) 
                        - batData["selfDis"]*misc["dt"]*SOC_previous, name="storage balance_"+str(n)+str(t))
        if t == (timesteps-1):
            model.addConstr(SOC[n,t] == batData["init"]*x[n])

## set objective function
#model.setObjective(sum((x[n,t] for t in range(misc["time steps"]) for n in nodeGrid)), gp.GRB.MINIMIZE)
model.setObjective(sum(powerTrafo[t]*co2.mix_2050[t] for t in range(misc["time steps"])), gp.GRB.MINIMIZE)                
#model.setObjective(sum(x[n] for n in nodeGrid), gp.GRB.MAXIMIZE)

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


powerLineSum = np.array([[sum(powerLine[n,m,t].X for t in range(misc["time steps"])) for m in range(len(nodeGrid))] for n in range(len(nodeGrid))])    
    
res_x = np.array([x[n].X for n in range(len(nodeGrid))])
res_powerCh = np.array([[powerCh[n,t].X for t in range(misc["time steps"])] for n in range(len(nodeGrid))])
res_powerDis = np.array([[powerDis[n,t].X for t in range(misc["time steps"])] for n in range(len(nodeGrid))])
res_soc = np.array([[SOC[n,t].X for t in range(misc["time steps"])] for n in range(len(nodeGrid))])
res_powerTrafo = np.array([powerTrafo[t].X for t in range(misc["time steps"])])

netx=net
netx['bat']=pd.DataFrame(res_x, columns=['ex'])
simple_plot_bat(netx, show_plot=True, bus_color='b', bat_color='r')

#    return (res_x, res_power,res_soc, model.ObjVal,model.MIPGap,model.Runtime,model.ObjBound)