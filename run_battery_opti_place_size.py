# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 15:34:57 2019

@author: she
"""

# import extern functions
import numpy as np
import math
import pickle
import pandas as pd
import pandapower as pp
import pandapower.networks as nw
import pandapower.plotting as plot
#from pandapower.plotting.plotly import simple_plotly
#import pandapower.plotting.plotly.simple_plotly 
#from pandapower.plotting import *
from pandapower.plotting.simple_plot_bat import simple_plot_bat
#from pandapower.plotting.simple_plot import simple_plot


# import own function
import python.clustering_medoid as clustering
import python.parse_inputs as pik
#import python.grid_optimization_2nd_building as opti
import python.grid_optimization as opti
#import python.grid_optimization_master as opti2
import python.building_distribution as dist 
import python.read_basic as reader


#%% set parameters 
building_type = "EFH"       # EFH, ZFH, MFH_6WE, MFH_10WE, MFH_15WE
building_type2 = "MFH_6WE"  # second building type, query "mfh"
building_age  = "2005"      # 1960, 1980, 2005 
emission_year = "2017"      # 2017, 2030, 2050 
# District parameters
# second "option" for district coices, as floats
district_options = {"mfh" : 0.33,                  # ratio of MFH to EFH in %
                    "pv" : 0.2,                    # ratio in %
                    "hp" : 0.2,                    # ratio in %
                    "ev" : 0.1,                    # ratio in %
                    "case" : "random"               # "best", "worst" and "random"
                    }

# set options
options =   {"static_emissions": True,   # True: calculation with static emissions, 
                                         # False: calculation with timevariant emissions
             "rev_emissions": True,      # True: emissions revenues for feed-in
                                         # False: no emissions revenues for feed-in
             "dhw_electric": True,       # define if dhw is provided decentrally by electricity
             "P_pv": 20.0,               # installed peak PV power
             "hp_mode": "energy_opt",    # choose between "off" (no hp) and "energy_opt" and "grid_opt"
             "T_VL": 35,                 # choose between 35 and 55 "Vorlauftemperatur" 
             "alpha_th": 0.8,            # relative size of heat pump (between 0 and 1)
             "beta_th": 0.5,             # relative size of thermal energy storage (between 0 and 1)
             "EV_mode": "on_demand",         # choose between "off" (no EVs), "on_demand", "grid_reactive" and "bi_directional"
             "show_grid_plots" : True,   # show gridplots before and after optimization
             "phi" : 25.842              # 
            }

mfh = str(math.floor(district_options["mfh"]*100))  # ratio of MFH to EFH in %
pv = str(math.floor(district_options["pv"]*100))    # ratio in %
hp = str(math.floor(district_options["hp"]*100))    # ratio in %
ev = str(math.floor(district_options["ev"]*100))    # ratio in %
case = district_options["case"]                     # Case: best, worst, random
tes = str(options["beta_th"])                            # TES-size

#%% data import

#determine the optimization folder in which all input data and results are placed
operationFolder = "C:\\Users\\Chrissi\\Git\\Flexigrid"
'''
Hier ist dein operationFolder noch abgelegt ;-) 
'''
#operationFolder="D:\\git\\flexigrid"       
#the input data is always in this source folder
sourceFolder = operationFolder + "\\input"
distributionFolder =  operationFolder + "\\distribution"

raw_inputs = {} 

raw_inputs["heat"]  = np.maximum(0, np.loadtxt(sourceFolder+"\\Typgebäude\\"+building_type+"\\"+building_age+"\\heat.csv") / 1000) 
raw_inputs["dhw"]  = np.maximum(0, np.loadtxt(sourceFolder+"\\Typgebäude\\"+building_type+"\\"+building_age+"\\dhw.csv") / 1000) 
raw_inputs["electricity"]  = np.maximum(0, np.loadtxt(sourceFolder+"\\Typgebäude\\"+building_type+"\\"+building_age+"\\electricity.csv") / 1000) 
raw_inputs["solar_roof"]  = np.maximum(0, np.loadtxt(sourceFolder+"\\Typgebäude\\"+building_type+"\\"+building_age+"\\solar_roof.csv") / 1000)       
raw_inputs["temperature"] = np.loadtxt(sourceFolder+"\\Typgebäude\\"+building_type+"\\"+building_age+"\\temperature.csv")

raw_inputs["heat2"]  = np.maximum(0, np.loadtxt(sourceFolder+"\\Typgebäude\\"+building_type2+"\\"+building_age+"\\heat.csv") / 1000) 
raw_inputs["dhw2"]  = np.maximum(0, np.loadtxt(sourceFolder+"\\Typgebäude\\"+building_type2+"\\"+building_age+"\\dhw.csv") / 1000) 
raw_inputs["electricity2"]  = np.maximum(0, np.loadtxt(sourceFolder+"\\Typgebäude\\"+building_type2+"\\"+building_age+"\\electricity.csv") / 1000) 
raw_inputs["solar_roof2"]  = np.maximum(0, np.loadtxt(sourceFolder+"\\Typgebäude\\"+building_type2+"\\"+building_age+"\\solar_roof.csv") / 1000)       
raw_inputs["temperature2"] = np.loadtxt(sourceFolder+"\\Typgebäude\\"+building_type2+"\\"+building_age+"\\temperature.csv")

emi_input = pd.read_csv(sourceFolder+"\\emission_factor_"+emission_year+".csv", header=0, usecols=[2])
raw_inputs["co2_dyn"] = np.zeros([8760])    
for t in range (0, 8760):   
    i=t*4
    raw_inputs["co2_dyn"][t]= np.mean(emi_input[i:(i+4)])

#%% data clustering 
    
#inputs_clustering = np.array([raw_inputs["heat"], 
#                              raw_inputs["dhw"],
#                              raw_inputs["electricity"],
#                              raw_inputs["solar_roof"],
#                              raw_inputs["temperature"],
#                              raw_inputs["co2_dyn"]])

inputs_clustering = np.array([raw_inputs["heat"],                     #0 
                              raw_inputs["heat2"],                    #1
                              raw_inputs["dhw"],                      #2
                              raw_inputs["dhw2"],                     #3
                              raw_inputs["electricity"],              #4
                              raw_inputs["electricity2"],             #5
                              raw_inputs["solar_roof"],               #6
                              raw_inputs["solar_roof2"],              #7
                              raw_inputs["temperature"],              #8
                              raw_inputs["temperature2"],             #9
                              raw_inputs["co2_dyn"]])                 #10
    
number_clusters = 12
#(inputs, nc, z) = clustering.cluster(inputs_clustering, 
#                                     number_clusters=number_clusters,
#                                     norm=2,
#                                     mip_gap=0.0,
#                                     weights=[1,1,1,1,0,1,1])

(inputs, nc, z) = clustering.cluster(inputs_clustering, 
                                        number_clusters=number_clusters,
                                        norm=2,
                                        mip_gap=0.0,
                                        weights=[1,1,2,2,2,2,2,2,1,1,2]) 


# Determine time steps per day
len_day = int(inputs_clustering.shape[1] / 365)

clustered = {}
#clustered["heat"]           = inputs[0]
#clustered["dhw"]            = inputs[1]
#clustered["electricity"]    = inputs[2]
#clustered["solar_irrad"]    = inputs[3]
#clustered["temperature"]    = inputs[4]
#clustered["co2_dyn"]        = inputs[5]
#clustered["co2_stat"]       = np.zeros_like(clustered["co2_dyn"])
#clustered["co2_stat"][:,:]  = np.mean(raw_inputs["co2_dyn"])
#clustered["weights"]        = nc
#clustered["z"]              = z

clustered = {}
clustered["heat"]           = inputs[0]
clustered["dhw"]            = inputs[2]
clustered["electricity"]    = inputs[4]
clustered["solar_irrad"]    = inputs[6]
clustered["temperature"]    = inputs[8]

clustered["heat2"]           = inputs[1]
clustered["dhw2"]            = inputs[3]
clustered["electricity2"]    = inputs[5]
clustered["solar_irrad2"]    = inputs[7]
clustered["temperature2"]    = inputs[9]

clustered["co2_dyn"]        = inputs[10]
clustered["co2_stat"]       = np.zeros_like(clustered["co2_dyn"])
clustered["co2_stat"][:,:]  = np.mean(raw_inputs["co2_dyn"])
clustered["weights"]        = nc
clustered["z"]              = z

#%% load devices, econmoics, etc.
   
devs = pik.read_devices(timesteps           = len_day, 
                        days                = number_clusters,
                        temperature_ambient = clustered["temperature"],
                        solar_irradiation   = clustered["solar_irrad"],
                        days_per_cluster    = clustered["weights"])

(eco, params, devs) = pik.read_economics(devs)
params    = pik.compute_parameters(params, number_clusters, len_day)

#%% create network

# load example net (IEEE 9 buses)
'''
typical kerber grids:   landnetz_freileitung_1(), 
                        landnetz_freileitung_2(),  
                        landnetz_kabel_1(), 
                        landnetz_kabel_2(),
                        dorfnetz(), 
                        vorstadtnetz_kabel_1(), 
                        vorstadtnetz_kabel_2()
    -> create network with nw.create_kerber_name
                        
extreme kerber grids:   landnetz_freileitung(), 
                        landnetz_kabel(), 
                        landnetz_freileitung_trafo(), 
                        landnetz_kabel_trafo(), 
                        dorfnetz(), 
                        dorfnetz_trafo(), 
                        vorstadtnetz_1(), 
                        vorstadtnetz_2(), 
                        vorstadtnetz_trafo_1(), 
                        vorstadtnetz_trafo_2()
    -> create network with nw.kb_extrem_name   
            
'''

#net = nw.create_kerber_landnetz_freileitung_1()
#net.name = "landnetz_freileitungl_1"
net = nw.create_kerber_landnetz_freileitung_2()
net.name = "landnetz_freileitung_2"
#net = nw.create_kerber_landnetz_kabel_1()
#net.name = "landnetz_kabel_1"
#net = nw.create_kerber_landnetz_kabel_2()
#net.name = "landnetz_kabel_2"
#net = nw.create_kerber_dorfnetz()
#net.name = "dorfnetz"
#net = nw.create_kerber_vorstadtnetz_kabel_1()
#net.name = "vorstadtnetz_kabel_1"
#net = nw.create_kerber_vorstadtnetz_kabel_2()
#net.name = "vorstadtnetz_kabel_2"
#
#net = nw.kb_extrem_landnetz_freileitung()
#net.name = "ex_landnetz_freileitung"
#net = nw.kb_extrem_landnetz_kabel()
#net.name = "ex_landnetz_kabel_"
#net = nw.kb_extrem_landnetz_freileitung_trafo()
#net.name = "ex_landnetz_freileitung_trafo"
#net = nw.kb_extrem_landnetz_kabel_trafo()
#net.name = "ex_landnetz_kabel_trafo"
#net = nw.kb_extrem_dorfnetz()
#net.name = "ex_dorfnetz"
#net = nw.kb_extrem_dorfnetz_trafo()
#net.name = "ex_dorfnetz_trafo"
#net = nw.kb_extrem_vorstadtnetz_1()
#net.name = "ex_vorstadtnetz_1"
#net = nw.kb_extrem_vorstadtnetz_2()
#net.name = "ex_vorstadtnetz_2"
#net = nw.kb_extrem_vorstadtnetz_trafo_1()
#net.name = "ex_vorstadtnetz_trafo_1"
#net = nw.kb_extrem_vorstadtnetz_trafo_2()
#net.name = "ex_vorstadtnetz_trafo_2"

if options["dhw_electric"]:
    filename = net.name + "_" + building_type + "_" + building_age + "_" + building_type2 + "_mfh" + mfh + "_pv" + pv + "_hp"  + hp + "tes" + tes + "_dhwe" + "_ev" + ev + "_" + case
    ev_file = net.name + "_mfh" + mfh + "_pv" + pv + "_hp"  + hp + "tes" + tes + "_dhwe" + ".xlsx"
else:
    filename = net.name + "_" + building_type + "_" + building_age + "_" + building_type2 + "_mfh" + mfh + "_pv" + pv + "_hp"  + hp + "tes" + tes + "_ev" + ev + "_" + case
    ev_file = net.name + "_mfh" + mfh + "_pv" + pv + "_hp"  + hp + "tes" + tes + ".xlsx"

randomfile = net.name + "_mfh" + mfh + "_pv" + pv + "_hp"  + hp + "_ev" + ev + ".xlsx" 


if options["show_grid_plots"]:
# simple plot of net with existing geocoordinates or generated artificial geocoordinates
    plot.simple_plot(net, show_plot=True)


names =   {"filename_results": "results/" + filename + ".pkl",
            "building_results": "results/" + "dist_" + filename + ".pkl"
            }

#%% find distribution for various building types

(load_with) = dist.allocate(net, options, names, district_options, distributionFolder, randomfile, ev_file)

#%% Store clustered input parameters
    
filename_input = "results/inputs_" + building_type + "_" + building_age + ".pkl"
with open(filename_input, "wb") as f_in:
    pickle.dump(clustered, f_in, pickle.HIGHEST_PROTOCOL)

#%% Define dummy parameters, options and start optimization
emi_max = 1000000000
(costs, emission, gridnodes) = opti.compute(emi_max, net, eco, devs, clustered, params, options, district_options, names, load_with, randomfile, ev_file, distributionFolder)

outputs = reader.read_results("results/" + filename, "results/" + "dist_" + filename)

#%% plot grid with batteries highlighted

if options["show_grid_plots"]:
    
    bat_ex = np.zeros(len(outputs["nodes"]["grid"]))
    for n in outputs["nodes"]["grid"]:
        if outputs["res_capacity"][n] > 0:
            bat_ex[n] = 1
    
    netx=net
    netx['bat']=pd.DataFrame(bat_ex, columns=['ex'])
    simple_plot_bat(netx, show_plot=True, bus_color='b', bat_color='r')
    


