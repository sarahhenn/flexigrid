# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 17:18:29 2019

@author: Chrissi
"""
import numpy as np
import gurobipy as gp
import math
import random
import xlrd
import xlsxwriter
import pickle
#import operator

#%%
def round_to_int(n):
    return math.floor(n + 0.5) 

def remove_n_minimums(d, n):
    for i in range(n):
        min_key = min(d.keys(), key=lambda k: d[k])
        del d[min_key]


def remove_n_maximums(d, n):
    for i in range(n):
        max_key = max(d.keys(), key=lambda k: d[k])
        del d[max_key]

def remove_n_randoms(d, n):         ## TO DO? change to choose_n_randoms and add writing in excel
    for i in range(n):
        random_key = random.choice(list(d))
        del d[random_key]
    #write. etc siehe unten
    

#%% Main Function of allocation
def allocate (net, options, names, district_options, distributionFolder, randomfile, ev_file): 
    
    ratio = {}
    ratio["pv"] = district_options["pv"]
    ratio["mfh"] = district_options["mfh"]
    ratio["hp"] = district_options["hp"]
    #ratio["tes"] = district_options["tes"]
    ratio["ev"] = district_options["ev"]
    
    nodes = {}
    
    nodes["grid"] = net.bus.index.to_numpy()
    nodes["trafo"] = net.trafo['lv_bus'].to_numpy()
    nodes["load"] = net.load['bus'].to_numpy()
    nodes["supply"] = nodes["grid"]
    nodes["supply"] = np.delete(nodes["supply"],nodes["load"])
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
            num_of_branches += 1
    
    loads_per_net_with = {}
    loads_per_net_with["pv"] = round_to_int (ratio["pv"]* num_of_loads)
    loads_per_net_with["mfh"] = round_to_int (ratio["mfh"]* num_of_loads)
    loads_per_net_with["hp"] = round_to_int (ratio["hp"]* num_of_loads)
    #loads_per_net_with["tes"] = round_to_int (ratio["tes"]* num_of_loads)
    loads_per_net_with["ev"] = round_to_int (ratio["ev"]* num_of_loads)
            
    j=0
    loads_per_branch = {}
    for n,m in nodeLines:
        if n == 1:
            j+=1
            loads_per_branch["branch_" + str(j)] = []            
        elif m in nodes["load"]:
            loads_per_branch["branch_" + str(j)].append(m) 
               
    lineLength = {}
    for [n,m] in nodeLines:
        lineLength[n,m] = net.line.length_km[m-2]

#    line_to = {}
#    line_to["loads"] = [0]
#    line_to["nodes"] = [0]
#    for [n,m] in nodeLines:
#        line_to["nodes"][m] = lineLength[n,m] + line_to["nodes"][n]
    
    
    line_to_node = {}
    line_to_node[1] = 0
    for [n,m] in nodeLines:
        line_to_node[m] = lineLength[n,m] + line_to_node[n]
    
    line_to_load = {}
    for [n,m] in nodeLines:
        if m in nodes["load"]:
            line_to_load[m] = lineLength[n,m] + line_to_node[n]    
    
    loads_with={}
    loads_with["pv"] = dict(line_to_load)
    loads_with["mfh"] = dict(line_to_load)
    loads_with["hp"] = dict(line_to_load)
    #loads_with["tes"] = dict(line_to_load)
    loads_with["ev"] = {} #dict(line_to_load)
    
    #sorted_lineLength = sorted(line_to.items(), key=operator.itemgetter(1))
    #sorted_lineLength = np.delete(sorted_lineLength,[0,0])
    
    #%% system allocation
    l_pv = num_of_loads - loads_per_net_with["pv"]
    l_mfh = num_of_loads - loads_per_net_with["mfh"]
    l_hp = num_of_loads - loads_per_net_with["hp"]
    #l_tes = num_of_loads - loads_per_net_with["tes"]
    l_ev = num_of_loads - loads_per_net_with["ev"]
    
    if district_options["case"] == "best":
        print("Best case grid")
        remove_n_minimums(loads_with["pv"], l_pv)
        remove_n_minimums(loads_with["mfh"], l_mfh) ## ???
        remove_n_minimums(loads_with["hp"], l_hp)   ## aus pv auswählen????
        #remove_n_minimums(loads_with["tes"], l_tes) ## raus?
        if l_ev == 0:
            pass
        elif l_ev == num_of_loads:
            loads_with["ev"] = {}
        else:
            #remove_n_minimums(loads_with["ev"], l_ev) ### nur platzhalter!!
            try:    # checks if random distribution for selected composition exists
                f = open(distributionFolder + "\\" + "ev_best_" + randomfile)
                f.close()
                ev_book  = xlrd.open_workbook(distributionFolder + "\\" + "ev_best_" + randomfile) # directory?
                sheet = ev_book.sheet_by_name("Sheet1")
                for row in [1,sheet.nrows]:
                    loads_with["ev"] = sheet.cell_value(row, 0)
            except FileNotFoundError:
                print("Optimal bat placement for selected distribution not generated. Please load distribution without EV first.")
            #pass
    elif district_options["case"] == "worst":
        print("Worst case grid")
        remove_n_maximums(loads_with["pv"], l_pv)
        remove_n_minimums(loads_with["mfh"], l_mfh)
        remove_n_minimums(loads_with["hp"], l_hp)
        #remove_n_minimums(loads_with["tes"], l_tes)  ## raus?
        remove_n_minimums(loads_with["ev"], l_ev)
    elif district_options["case"] == "random":
        try:    # checks if random distribution for selected composition exists
            f = open(distributionFolder + "\\" + randomfile + ".xlsx")
            f.close()
            print("Loading random distribution")            
            random_book  = xlrd.open_workbook(distributionFolder + "\\" + randomfile) # directory?
            sheet = random_book.sheet_by_name("Sheet1")
            for row in [1,sheet.nrows]:
                loads_with["pv"] = sheet.cell_value(row, 0)
                loads_with["mfh"] = sheet.cell_value(row, 0)
                loads_with["hp"] = sheet.cell_value(row, 0)
                #loads_with["tes"] = sheet.cell_value(row, 0)
                loads_with["ev"] = sheet.cell_value(row, 0)
              # if distribution doesn't exist, it is being created      
        except FileNotFoundError:
            print('Random district is generated')
            remove_n_randoms(loads_with["pv"], l_pv)
            remove_n_randoms(loads_with["mfh"], l_mfh)
            remove_n_randoms(loads_with["hp"], l_hp)
            #remove_n_randoms(loads_with["tes"], l_tes)
            remove_n_randoms(loads_with["ev"], l_ev)
            random_book = xlsxwriter.Workbook(distributionFolder + "\\" + randomfile)
            sheet = random_book.add_worksheet()
            col = -1
            for key in loads_with.keys():
                row = 0
                col +=1
                sheet.write(row, col, key)
                for item in loads_with[key]:
                    row += 1
                    sheet.write(row, col, item)
            random_book.close()      
    else: 
        print("Error: Select case for building distribution")
    
    loads_with["efh"] = dict(line_to_load)
    for n in gridnodes:
        if n in loads_with["mfh"]:
               del loads_with["efh"][n]     
        #max_key = max(loads_with["efh"].keys(), key=lambda k: loads_with["efh"][k])
#        del loads_with["efh"][key]
    
#    loads_with["efh"] = np.delete(list(loads_with["efh"]), list(loads_with["mfh"]))
    
    #%% return results
    with open(names["building_results"], "wb") as fout:
        pickle.dump(loads_with, fout, pickle.HIGHEST_PROTOCOL)              #01
        pickle.dump(line_to_load, fout, pickle.HIGHEST_PROTOCOL)            #02
        pickle.dump(loads_per_branch, fout, pickle.HIGHEST_PROTOCOL)        #03
        pickle.dump(loads_per_net_with, fout, pickle.HIGHEST_PROTOCOL)      #04
#        pickle.dump(loads_per_branch, fout, pickle.HIGHEST_PROTOCOL)        #05
#        pickle.dump(loads_per_branch, fout, pickle.HIGHEST_PROTOCOL)        #06
#        pickle.dump(loads_per_branch, fout, pickle.HIGHEST_PROTOCOL)        #07
#        pickle.dump(loads_per_branch, fout, pickle.HIGHEST_PROTOCOL)        #08
    

    
    return (loads_with)#num_of_branches, num_of_loads, line_to_load, loads_with)
