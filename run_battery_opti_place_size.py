# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 15:34:57 2019

@author: she
"""
        


# import extern functions
import numpy as np
import pickle
import pandas as pd
import os
import tempfile
import pandapower as pp
import pandapower.networks as nw
import pandapower.plotting as plot
from pandapower.plotting.simple_plot_bat import simple_plot_bat


# import own function
import python.clustering_medoid as clustering
import python.parse_inputs as pik
import python.grid_optimization as opti
import python.read_basic as reader
import timeloop_flexigrid as loop
import python.flexigrid_plotting_results as plot_res

# set parameters 
building_type = "EFH"       # EFH, ZFH, MFH_6WE, MFH_10WE, MFH_15WE
building_age  = "2005"      # 1960, 1980, 2005 
emission_year = "2017"      # 2017, 2030, 2050 

# TODO: implement mixed shares of buildings
# TODO: adjust emission factors regarding to national weather conditions
 
# TODO: load data for useable roofarea per building type
# TODO: PV area as MILP-variable??     
#useable_roofarea  = 0.30    #Default value: 0.25

# set options
options =   {"static_emissions": False,  # True: calculation with static emissions,
                                        # False: calculation with timevariant emissions
            "rev_emissions": False,      # True: emissions revenues for feed-in
                                        # False: no emissions revenues for feed-in
            "dhw_electric": True,       # define if dhw is provided decentrally by electricity
            "P_pv": 20.00,              # installed peak PV power
            "with_hp": True,            # usage of heat pumps
            "hp_mode": "grid_opt",    # choose between "energy_opt" and "grid_opt"
            "T_VL": 35,                 # choose between 35 and 55 "Vorlauftemperatur" 
            "alpha_th": 0.8,            # relative size of heat pump (between 0 and 1)
            "beta_th": 0.2,             # relative size of thermal energy storage (between 0 and 1)
            "show_grid_plots": False,   # show gridplots before and after optimization
            
            "filename_results": "results/" + building_type + "_" + \
                                                   building_age + ".pkl",
            "apc_while_voltage_violation": True,    #True: uses apc, when voltage violations occur
                                                    #False: does not use apc, when voltage violations occur
            "bat_ch_while_voltage_violation": True, #True: increases battery charge, when voltage violations occur
                                                    #False: does not increase battery charge, when voltage violations occur
             #note: power costs for heatpump can only be separated for a cost calculation for sum(nodes), not for grid in total!
             #select your cost function calculation through objective function in grid_optimization
            "heatpump_seperated_costs": True,       #True: Heatpumps power costs: 18.56 ct/kWh (apart from other power users)
                                                    #False: Heatpump power costs: 27.8 ct/kWh (equal to other power users)
             "allow_apc_opti": True                #True: Curtailment allowed to be set in optimization
                                                    #False: Curtailment only through additional constraint
            }
                     
#%% data import

#determine the optimization folder in which all input data and results are placed
operationFolder="C:\\users\\flori\\pycharmprojects\\flexigrid"
#the input data is always in this source folder
sourceFolder=operationFolder+"\\input"

raw_inputs = {} 

raw_inputs["heat"]  = np.maximum(0, np.loadtxt(sourceFolder+"\\Typgebäude\\"+building_type+"\\"+building_age+"\\heat.csv") / 1000) 
raw_inputs["dhw"]  = np.maximum(0, np.loadtxt(sourceFolder+"\\Typgebäude\\"+building_type+"\\"+building_age+"\\dhw.csv") / 1000) 
raw_inputs["electricity"]  = np.maximum(0, np.loadtxt(sourceFolder+"\\Typgebäude\\"+building_type+"\\"+building_age+"\\electricity.csv") / 1000) 
raw_inputs["solar_roof"]  = np.maximum(0, np.loadtxt(sourceFolder+"\\Typgebäude\\"+building_type+"\\"+building_age+"\\solar_roof.csv") / 1000)       
raw_inputs["temperature"] = np.loadtxt(sourceFolder+"\\Typgebäude\\"+building_type+"\\"+building_age+"\\temperature.csv")

emi_input = pd.read_csv(sourceFolder+"\\emission_factor_"+emission_year+".csv", header=0, usecols=[2])
raw_inputs["co2_dyn"] = np.zeros([8760])    
for t in range (0, 8760):
    i=t*4
    raw_inputs["co2_dyn"][t]= np.mean(emi_input[i:(i+4)])

#%% data clustering 
    
inputs_clustering = np.array([raw_inputs["heat"], 
                              raw_inputs["dhw"],
                              raw_inputs["electricity"],
                              raw_inputs["solar_roof"],
                              raw_inputs["temperature"],
                              raw_inputs["co2_dyn"]])

number_clusters = 3
(inputs, nc, z) = clustering.cluster(inputs_clustering, 
                                     number_clusters=number_clusters,
                                     norm=2,
                                     mip_gap=0.0,
                                     weights=[1,2,2,2,1,2])


# Determine time steps per day
len_day = int(inputs_clustering.shape[1] / 365)

clustered = {}
clustered["heat"]           = inputs[0]
clustered["dhw"]            = inputs[1]
clustered["electricity"]    = inputs[2]
clustered["solar_irrad"]    = inputs[3]
clustered["temperature"]    = inputs[4]
clustered["co2_dyn"]        = inputs[5]
clustered["co2_stat"]       = np.zeros_like(clustered["co2_dyn"])
clustered["co2_stat"][:,:]  = np.mean(raw_inputs["co2_dyn"])
clustered["weights"]        = nc
clustered["z"]               = z

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
typical kerber grids:   landnetz_freileitung_1(), landnetz_freileitung_2(), landnetz_kabel_1(), landnetz_kabel_2(),
                        dorfnetz(), vorstadtnetz_kabel_1(), vorstadtnetz_kabel_2()
    -> create network with nw.create_kerber_name
                        
extreme kerber grids:   landnetz_freileitung(), landnetz_kabel(), landnetz_freileitung_trafo(), landnetz_kabel_trafo(), 
                        dorfnetz(), dorfnetz_trafo(), 
                        vorstadtnetz_1(), vorstadtnetz_2(), vorstadtnetz_trafo_1(), vorstadtnetz_trafo_2()
    -> create network with nw.kb_extrem_name   
            
'''
#net = nw.create_kerber_landnetz_freileitung_2()
#net = nw.create_kerber_vorstadtnetz_kabel_2()
net = nw.create_kerber_landnetz_freileitung_1()

if options["show_grid_plots"]:
# simple plot of net with existing geocoordinates or generated artificial geocoordinates
    plot.simple_plot(net, show_plot=True)

    #%% Store clustered input parameters
    
filename = "results/inputs_" + building_type + "_" + building_age + ".pkl"
with open(filename, "wb") as f_in:
    pickle.dump(clustered, f_in, pickle.HIGHEST_PROTOCOL)

# specify grid nodes for whole grid and trafo; choose and allocate load, injection and battery nodes
# draw parameters from pandapower network
nodes = {}

nodes["grid"] = net.bus.index.to_numpy()
nodes["trafo"] = net.trafo['lv_bus'].to_numpy()
nodes["load"] = net.load['bus'].to_numpy()
nodes["bat"] = net.load['bus'].to_numpy()

#define gridnodes, days and timesteps
gridnodes = list(nodes["grid"])
days        = [i for i in range(params["days"])]
timesteps   = [i for i in range(params["time_steps"])]


# solution_found as continuos variable for while loop
solution_found = []
for d in days:
    solution_found.append(False)
boolean_loop = True
# constraint_apc models APC, gets reduced from 1 to 0 in iteration steps with range 0.1
constraint_apc = {}
# constraint_bat models forced battery charging and discharging. Gets raised and reduced in case of voltage violation
constraint_batCh = {}
constraint_batDis = {}
# create array to flag whether values are critical for powerflow. If not critical: 0, if critical: 1
critical_flag = {}
iteration_counter = 0
for n in gridnodes:
    for d in days:
        for t in timesteps:
            critical_flag[n, d, t] = 0
            constraint_apc[n, d] = 0
            constraint_batCh[n, d, t] = 0
            constraint_batDis[n, d, t] = 0

while boolean_loop:

    print("")
    print("!!! Iteration counter is currently at " +str(iteration_counter) + "!!!")
    print("")
    #run DC-optimization
    (costs, emission, timesteps, days, powInjRet, powSubtrRet, gridnodes, res_exBat, powChPrev, powDisPrev) = opti.compute(net, nodes, gridnodes, days, timesteps, eco, devs, clustered,params, options, constraint_apc, constraint_batCh, constraint_batDis,critical_flag)

    outputs = reader.read_results(building_type + "_" + building_age, options)
    # %% plot grid with batteries highlighted
    if options["show_grid_plots"]:

        bat_ex = np.zeros(len(outputs["nodes"]["grid"]))
        for n in outputs["nodes"]["grid"]:
            if outputs["res_capacity"][n] > 0:
                bat_ex[n] = 1

        netx = net
        netx['bat'] = pd.DataFrame(bat_ex, columns=['ex'])
        simple_plot_bat(netx, show_plot=True, bus_color='b', bat_color='r')

    # run AC-Powerflow-Solver
    (output_dir, critical_flag, solution_found,vm_pu_total) = loop.run_timeloop(net, timesteps, days, powInjRet, powSubtrRet,gridnodes, critical_flag, solution_found)
    #vm_pu_total_array = np.array([[[vm_pu_total[n, d, t] for t in timesteps] for d in days] for n in gridnodes])
    for d in days:
        if (solution_found[d] == False):

            print("Additional constrains have to be set for day" +str(d))
            if options["apc_while_voltage_violation"]:
                if options["bat_ch_while_voltage_violation"]:
                    print("You selected both apc and additional battery charge in case of voltage violations.")
                    for n in gridnodes:
                        if (any(critical_flag[n, d, t] == 1 for t in timesteps)):
                            if(any(vm_pu_total[n,d,t] < 0.96 for t in timesteps)):
                                pass
                            elif(any(vm_pu_total[n,d,t] > 1.04 for t in timesteps)):
                                constraint_apc[n,d] += 0.1
                                if(constraint_apc[n,d] >= 1):
                                    print("You have reached the maximal amount of curtailment!")
                                    print("Will set curtailment to 100 Percent automatically.")
                                    constraint_apc[n,d] = 1
                            else:
                                pass
                            for t in timesteps:
                                if(critical_flag[n,d,t] == 1):
                                    if (vm_pu_total[n,d,t] < 0.96):
                                        constraint_batCh[n,d,t] = 0.1 * powChPrev[n,d,t]
                                        if(constraint_batCh[n,d,t] <= 0):
                                            # if battery didnt get charged anyway, battery discharge is forced to be bigger than 0
                                            constraint_batCh[n,d,t] = 0
                                            constraint_batDis[n,d,t] = 1.1 * powDisPrev[n,d,t]
                                            if(powDisPrev == 0):
                                                constraint_batDis[n,d,t] = 2
                                    elif (vm_pu_total[n,d,t] > 1.04):
                                        constraint_batCh[n, d, t] = 1.1 * powChPrev[n,d,t]



                else:
                    print("You selected only apc in case of voltage violations.")
                    for n in gridnodes:
                        if (any(critical_flag[n, d, t] == 1 for t in timesteps)):
                            if (any(vm_pu_total[n,d,t] < 0.96 for t in timesteps)):
                                print("Only apc will not fix any voltage issues, because the load is too high on day" +str(d))
                            elif (any(vm_pu_total[n,d,t] > 1.04 for t in timesteps)):
                                constraint_apc[n, d] += 0.1
                                if (constraint_apc[n, d] >= 1):
                                    print("You have reached the maximal amount of curtailment!")
                                    print("Will set curtailment to 100 Percent automatically.")
                                    constraint_apc[n, d] = 1

            elif (options["bat_ch_while_voltage_violation"] == True and options["apc_while_voltage_violation"] == False):
                print("You selected only additional battery charge in case of voltage violations.")
                for n in gridnodes:
                    for t in timesteps:
                        if (critical_flag[n, d, t] == 1):
                            if (vm_pu_total[n,d,t] < 0.96):
                                constraint_batCh[n, d, t] = 0.1 * powChPrev[n,d,t]
                                if (constraint_batCh[n, d, t] <= 0):
                                    # if battery didnt get charged anyway, battery discharge is forced to be bigger than 0
                                    constraint_batCh[n, d, t] = 0
                                    constraint_batDis[n, d, t] = 1.1 * powDisPrev[n,d,t]
                                    if (powDisPrev == 0):
                                        constraint_batDis[n, d, t] = 2
                            elif (vm_pu_total[n,d,t] > 1.04):
                                constraint_batCh[n, d, t] = 1.1 * powChPrev[n,d,t]

            elif (options["bat_ch_while_voltage_violation"] == False and options["apc_while_voltage_violation"] == False):
                print("Error: You did not select any measure in case of voltage violations!")


        if(solution_found[d] == True):
                print("Solution was successfully found for day" + str(d))

    iteration_counter += 1


    if all(solution_found[d] == True for d in days):
        print("Congratulations! Your optimization and loadflow calculation has been successfully finished after " + str(iteration_counter) + "iteration steps!")
        break

print("this is the end")
#plot_res.plot_results(outputs, days, gridnodes, timesteps)