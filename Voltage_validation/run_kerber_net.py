# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 15:34:57 2019

@author: she
"""

# import extern functions
import pandapower.networks as nw
#import pandapower.plotting as plot
#from pandapower.plotting.simple_plot_bat import simple_plot_bat
import gurobipy as gp
import numpy as np

#%% create network

#net = nw.create_kerber_landnetz_freileitung_1()
#net = nw.create_kerber_landnetz_freileitung_2()
#net = nw.create_kerber_landnetz_kabel_1()
#net = nw.create_kerber_landnetz_kabel_2()
#net = nw.create_kerber_dorfnetz()
#net = nw.create_kerber_vorstadtnetz_kabel_1()
#net = nw.create_kerber_vorstadtnetz_kabel_2()

#net = nw.kb_extrem_landnetz_freileitung()
#net = nw.kb_extrem_landnetz_kabel()
#net = nw.kb_extrem_landnetz_freileitung_trafo()
#net = nw.kb_extrem_landnetz_kabel_trafo()
#net = nw.kb_extrem_dorfnetz()
#net = nw.kb_extrem_dorfnetz_trafo()
#net = nw.kb_extrem_vorstadtnetz_1()
#net = nw.kb_extrem_vorstadtnetz_2()
#net = nw.kb_extrem_vorstadtnetz_trafo_1()
net = nw.kb_extrem_vorstadtnetz_trafo_2()

nodes = {}
nodes["grid"] = net.bus.index.to_numpy()

nodeLines = []
for i in range(len(net.line['from_bus'])):
    nodeLines.append((net.line['from_bus'][i],net.line['to_bus'][i]))
nodeLines = gp.tuplelist(nodeLines)

gridnodes = list(net.bus.index.to_numpy())

lines = list(net.line.index.to_numpy())

res_r_arr = []
res_x_arr = []
res_per_line = []

for n in lines:
    res_r_arr.append(net.line.r_ohm_per_km[n])
    res_x_arr.append(net.line.x_ohm_per_km[n])
    res_per_line.append(net.line.length_km[n]*((net.line.r_ohm_per_km[n]**2 + net.line.x_ohm_per_km[n]**2)**(1/2)))    
#    max_res_single = np.amax(res_per_line)    
#        #maximum des arrays
  
first_in_line = []
lastload = []

for [n,m] in nodeLines:
    if n == 1:
        first_in_line.append (m)
        if m == 2:
            pass
        else:
            lastload.append (m-1)
               
lineLength = {}
lineCurrent = {}
lineRes = {}
lineRes_r = {}
lineRes_x = {}    

for [n,m] in nodeLines:
    lineLength[n,m] = net.line.length_km[m-2]
    lineCurrent[n,m] = net.line.max_i_ka[m-2]
    lineRes_r[n,m] = net.line.r_ohm_per_km[m-2]
    lineRes_x[n,m] = net.line.x_ohm_per_km[m-2]
    lineRes[n,m] = (net.line.r_ohm_per_km[m-2]**2 + net.line.x_ohm_per_km[m-2]**2)**(1/2)

for [n,m] in nodeLines:  
    max_r = np.max(lineRes[n,m])
    min_r = np.min(lineRes[n,m])

line_to = {}
res_to = {}
line_to[0] = 0
line_to[1] = 0
res_to[0] = 0
res_to[1] = 0
res_to_list = []
res_to_list.append(0)
i_max = {}
delta_u_max = {}
U_until = {}
U_until[0]=0
U_until[1]=0

for [n,m] in nodeLines:
    line_to[m] = lineLength[n,m] + line_to[n]
    res_to[m] = lineRes[n,m] + res_to [n]
    res_to_list.append(lineRes[n,m] + res_to [n])
    i_max[n,m]=(net.line['max_i_ka'][nodeLines.index((n,m))])
#max_res_sum = np.amax(res_to_list)
    
for [n,m] in nodeLines:
    delta_u_max[n,m]=i_max[n,m]*lineRes[n,m]
    U_until[m] = U_until[n] + delta_u_max[n,m]
 
      
#print("Trafoleistung: "+str(net.trafo.sn_mva*1000))

#return(net, nodes, nodeLines, lineLength, lineCurrent, res_per_line, lineRes_r, lineRes_x, lineRes)
