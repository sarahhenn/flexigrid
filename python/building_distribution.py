# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 17:18:29 2019

@author: Chrissi
"""
import numpy as np
import gurobipy as gp
import math

def round_to_int(n):
    return math.floor(n + 0.5) 

def allocate (net, options): #inputvariablen
    
    ratio = {}
    ratio["pv"] = 0.3
    ratio["hp"] = 0.2
    ratio["ev"] = 0.05
    
    nodes = {}

    nodes["grid"] = net.bus.index.to_numpy()
    nodes["trafo"] = net.trafo['lv_bus'].to_numpy()
    nodes["load"] = net.load['bus'].to_numpy()
    nodes["supply"] = nodes["grid"]  
    nodes["supply"] = np.delete(nodes["supply"],nodes["load"])   #Reihenfolge wichtig !
    nodes["supply"] = np.delete(nodes["supply"],[0,1]) 
    
    gridnodes = list(net.bus.index.to_numpy())
    
    nodeLines = []
    for i in range(len(net.line['from_bus'])):
        nodeLines.append((net.line['from_bus'][i],net.line['to_bus'][i]))
    nodeLines = gp.tuplelist(nodeLines)
    
    num_of_loads = len(nodes["load"])
    
    num_of_branches = 0
    for n,m in nodeLines:
        if n == 1:
            num_of_branches = num_of_branches + 1
    
   
    loads_per_branch = {}
    loads_per_branch["pv"] = round_to_int (ratio["pv"]* num_of_loads/num_of_branches)
    
    
    
    
    
    
    
    return (num_of_branches, num_of_loads, loads_per_branch)
