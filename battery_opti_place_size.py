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
import pandapower as pp
import pandapower.networks as nw
import pandapower.plotting as plot

#%% create network

# load example net (IEEE 9 buses)
net = nw.create_kerber_landnetz_kabel_1()
# simple plot of net with existing geocoordinates or generated artificial geocoordinates
plot.simple_plot(net, show_plot=True)


#%% set input parameter

#build dictionary with misc data
timesteps = 10
misc={"time steps":timesteps, "dt":1.0}

# specify grid nodes
nodeGrid = net.bus.index.to_numpy() # all grid nodes
nodeTrafo = net.trafo['lv_bus'].to_numpy() # trafo
nodeLoad = net.load['bus'].to_numpy() # here: buildings
nodeInj = net.load['bus'].to_numpy() # here: buildings
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
energyContent_kWh = 10.0
init = 0.5
selfDis = 0.1

#build dictionary with batData
batData={"Pmin":P_min_kW,"Pmax":P_max_kW,"SOCmin":SOC_min,"SOCmax":SOC_max,"energyContent":energyContent_kWh,"etaCh":etaCh,"etaDis":etaDis,"selfDis":selfDis, "init":init}

# parameters for load and generation
powerLoad = np.zeros((len(nodeGrid),timesteps))
powerGen = np.zeros((len(nodeGrid),timesteps))
#TODO: einladen der Gebäude-Lasten und der PV-Generation

powerGen[5,5] = 60

#%% optimization model

model = gp.Model("Optimal Battery Placement and Sizing")

#model.setObjective(sum(((co2[t]+hou["congSignal"][t]) * (powerLoad[t]-powerFeed[t])) for t in range(misc["time steps"])),
#                               gp.GRB.MINIMIZE)   

# initiate grid variables
powerTrafo = {}
powerLine = {}

# initiate bat variables
x = {} # Battery existance
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
        x[n] = model.addVar(vtype="B", name="activation_charge_"+str(n))
        SOC[n,t] = model.addVar(vtype="C", name="SOC_"+str(n)+str(t))
        powerCh[n,t] = model.addVar(vtype="C", name="powerLoad_"+str(n)+str(t))
        powerDis[n,t] = model.addVar(vtype="C", name="powerFeed_"+str(n)+str(t))
    
model.update()

#%% grid optimization

# set energy balance for all nodes
for n in range(len(nodeGrid)):
    for t in range(misc["time steps"]):
        if n in nodeTrafo:
            model.addConstr(sum(powerLine[n,m,t] for m in nodeGrid)- sum(powerLine[m,n,t] for m in nodeGrid) == powerTrafo[t], name="node balance_"+str(n)+str(t))
        else:
            #TODO: ggf. hier mit Kantenbilanz verrechnen. 
            model.addConstr(sum(powerLine[n,m,t] for m in nodeGrid)- sum(powerLine[m,n,t] for m in nodeGrid) == 
                            powerGen[n,t] - powerLoad[n,t] - powerCh[n,t] + powerDis[n,t], name="node balance_"+str(n)+str(t))


#TODO!!! klappt theoretisch, aber auf 0 spalte/zeile achten!            
for t in range(misc["time steps"]): 
    for k in range(len(nodeGrid)):
        for l in range(len(nodeGrid)):
                if [k,l] in nodeLines:
                    model.addConstr(powerLine[k,l,t] <= powerLine_max[k,l], name="line power max_"+str(k)+str(l)+str(t))
                    model.addConstr(powerLine[k,l,t] >= (-1)*powerLine_max[k,l], name="line power min_"+str(k)+str(l)+str(t))
                else:
                    model.addConstr(powerLine[k,l,t] == 0, name="line power max_"+str(k)+str(l)+str(t))
                

#%% battery optimization

for n in range(len(nodeGrid)):                    
    for t in range(misc["time steps"]):
        model.addConstr(powerCh[n,t] >= x[n] * batData["Pmin"], name="min power_"+str(n)+str(t))
        model.addConstr(powerCh[n,t] <= x[n] * batData["Pmax"], name="max power_"+str(n)+str(t))
        model.addConstr(powerDis[n,t] >= x[n] * batData["Pmin"], name="min power_"+str(n)+str(t))
        model.addConstr(powerDis[n,t] <= x[n] * batData["Pmax"], name="max power_"+str(n)+str(t))
        
#TODO: später an Entscheidungsvariable für Speicherexistenz knüpfen 
for n in range(len(nodeGrid)):
    if n not in nodeBat:
        model.addConstr(x[n] == 0, name="battery_existance_"+str(n))

for n in range(len(nodeGrid)):
    for t in range(misc["time steps"]):
        model.addConstr(SOC[n,t] <= batData["SOCmax"], name="max sto cap_"+str(n)+str(t))
        model.addConstr(SOC[n,t] >= batData["SOCmin"], name="min sto cap_"+str(n)+str(t))

for n in range(len(nodeGrid)):            
    for t in range(misc["time steps"]):
        if t == 0:
            SOC_previous = batData["init"]
        else:
            SOC_previous = SOC[n,t-1]
        
        model.addConstr(SOC[n,t] == (SOC_previous +
                                            (misc["dt"] * (powerCh[n,t] * batData["etaCh"] - powerDis[n,t]/batData["etaDis"])) / batData["energyContent"]) - batData["selfDis"]*misc["dt"]*SOC_previous, 
                                            name="storage balance_"+str(n)+str(t))        

#model.setObjective(sum((x[n,t] for t in range(misc["time steps"]) for n in nodeGrid)), gp.GRB.MINIMIZE)
#model.setObjective(sum(powerTrafo[t] for t in range(misc["time steps"])), gp.GRB.MAXIMIZE)                
model.setObjective(sum(x[n] for n in nodeGrid), gp.GRB.MINIMIZE)

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
    
    
res_x = np.array([x[n].X for n in range(len(nodeGrid))])
res_powerCh = np.array([[powerCh[n,t].X for t in range(misc["time steps"])] for n in range(len(nodeGrid))])
res_powerDis = np.array([[powerDis[n,t].X for t in range(misc["time steps"])] for n in range(len(nodeGrid))])
res_soc = np.array([[SOC[n,t].X for t in range(misc["time steps"])] for n in range(len(nodeGrid))])
res_powerTrafo = np.array([powerTrafo[t].X for t in range(misc["time steps"])])

#    return (res_x, res_power,res_soc, model.ObjVal,model.MIPGap,model.Runtime,model.ObjBound)