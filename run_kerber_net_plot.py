# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 15:34:57 2019

@author: she
"""

# import extern functions
import numpy as np
import pickle
import pandas as pd
import pandapower as pp
import pandapower.networks as nw
import pandapower.plotting as plot
from pandapower.plotting.simple_plot_bat import simple_plot_bat
import gurobipy as gp

# import own function
import python.clustering_medoid as clustering
import python.parse_inputs as pik
import python.grid_optimization as opti
#import python.grid_optimization_master as opti2
import python.read_basic as reader


# set parameters 
building_type = "EFH"       # EFH, ZFH, MFH_6WE, MFH_10WE, MFH_15WE
building_age  = "2005"      # 1960, 1980, 2005 
emission_year = "2017"      # 2017, 2030, 2050 


# set options
options =   {"show_grid_plots": True}


#%% create network

# load example net (IEEE 9 buses)

#net = nw.create_kerber_landnetz_freileitung_1()
#net = nw.create_kerber_landnetz_freileitung_2()
#net = nw.create_kerber_landnetz_kabel_1()
#net = nw.create_kerber_landnetz_kabel_2()
net = nw.create_kerber_dorfnetz()
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
#net = nw.kb_extrem_vorstadtnetz_trafo_2()

## implement line_length
## vgl bzw. maximum, loads und line_length

#######nodes["grid"] = net.bus.index.to_numpy()

nodeLines = []
for i in range(len(net.line['from_bus'])):
    nodeLines.append((net.line['from_bus'][i],net.line['to_bus'][i]))
nodeLines = gp.tuplelist(nodeLines)

lines = list(net.line.index.to_numpy())

res_r_arr = []
res_x_arr = []
max_res = []

for n in lines:
    res_r_arr.append(net.line.r_ohm_per_km[n])
    res_x_arr.append(net.line.x_ohm_per_km[n])
    max_res.append(net.line.length_km[n]*((net.line.r_ohm_per_km[n]**2 + net.line.x_ohm_per_km[n]**2)**(1/2)))
    #max_res = np.array(net.line.length_km[n]*((net.line.r_ohm_per_km[n]**2 + net.line.x_ohm_per_km[n]**2)**(1/2)))
    max_res_value = np.amax(max_res)    
    #maximum des arrays

first_in_line = []
lastload = []

for [n,m] in nodeLines:
    if n == 1:
        first_in_line.append (m)
        if m == 2:
            pass
        else:
            lastload.append (m-1)
    
        


if options["show_grid_plots"]:
# simple plot of net with existing geocoordinates or generated artificial geocoordinates
    plot.simple_plot(net, show_plot=True)

