# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 15:34:57 2019

@author: she
"""

# import extern functions
import shutil
import numpy as np
import pickle
import pandas as pd
import os
import tempfile
import time
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


# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

#!!!!!!!!!!!!!!!!!!!!!! HIER 15 TAGE
# get time from start to calculate duration of the program
t1 = int(time.time())

# set parameters
building_type = "EFH"  # EFH, ZFH, MFH_6WE, MFH_10WE, MFH_15WE
building_age = "2005"  # 1960, 1980, 2005
emission_year = "2017"  # 2017, 2030, 2050

# TODO: implement mixed shares of buildings
# TODO: adjust emission factors regarding to national weather conditions

# TODO: load data for useable roofarea per building type
# TODO: PV area as MILP-variable??
# useable_roofarea  = 0.30    #Default value: 0.25

# set options
options = {"static_emissions": False,  # True: calculation with static emissions,
           # False: calculation with timevariant emissions
           "rev_emissions": False,  # True: emissions revenues for feed-in
           # False: no emissions revenues for feed-in
           "dhw_electric": True,  # define if dhw is provided decentrally by electricity
           "P_pv": 10.00,  # installed peak PV power
           "with_hp": True,  # usage of heat pumps
           "hp_mode": "grid_opt",  # choose between "energy_opt" and "grid_opt"
           "T_VL": 35,  # choose between 35 and 55 "Vorlauftemperatur"
           "alpha_th": 1,  # relative size of heat pump (between 0 and 1)
           "beta_th": 0.417,  # relative size of thermal energy storage (between 0 and 1)
           "show_grid_plots": False,  # show gridplots before and after optimization

           "filename_results": "results/" + building_type + "_" + \
                               building_age + "Typtage_15.pkl",
           "filename_inputs": "results/inputs_" + building_type + "_" + \
                              building_age + "Typtage_15.pkl",
           "apc_while_voltage_violation": False,  # True: uses apc, when voltage violations occur
           # False: does not use apc, when voltage violations occur
           "cut_Inj/Subtr_while_voltage_violation": True,  # True: cuts Inj or Subtr, when voltage violations occur
           # depends automatically on the fact, whether voltage is too high or too low
           # note: power costs for heatpump can only be separated for a cost calculation for sum(nodes), not for grid in total!
           # select your cost function calculation through objective function in grid_optimization
           "heatpump_seperated_costs": True,  # True: Heatpumps power costs: 18.56 ct/kWh (apart from other power users)
           # False: Heatpump power costs: 27.8 ct/kWh (equal to other power users)
           "allow_apc_opti": True,  # True: Curtailment allowed to be set in optimization
           # False: Curtailment only through additional constraint
           "change_value_node_violation_abs": 2
           # specify, for how much the absolute values of inj and subtr should change in case of voltage violations
           }

# %% data import

# determine the optimization folder in which all input data and results are placed
operationFolder = "C:\\users\\flori\\pycharmprojects\\flexigrid"
# the input data is always in this source folder
sourceFolder = operationFolder + "\\input"

raw_inputs = {}

raw_inputs["heat"] = np.maximum(0, np.loadtxt(
    sourceFolder + "\\Typgebäude\\" + building_type + "\\" + building_age + "\\heat.csv") / 1000)
raw_inputs["dhw"] = np.maximum(0, np.loadtxt(
    sourceFolder + "\\Typgebäude\\" + building_type + "\\" + building_age + "\\dhw.csv") / 1000)
raw_inputs["electricity"] = np.maximum(0, np.loadtxt(
    sourceFolder + "\\Typgebäude\\" + building_type + "\\" + building_age + "\\electricity.csv") / 1000)
raw_inputs["solar_roof"] = np.maximum(0, np.loadtxt(
    sourceFolder + "\\Typgebäude\\" + building_type + "\\" + building_age + "\\solar_roof.csv") / 1000)
raw_inputs["temperature"] = np.loadtxt(
    sourceFolder + "\\Typgebäude\\" + building_type + "\\" + building_age + "\\temperature.csv")

emi_input = pd.read_csv(sourceFolder + "\\emission_factor_" + emission_year + ".csv", header=0, usecols=[2])
raw_inputs["co2_dyn"] = np.zeros([8760])
for t in range(0, 8760):
    i = t * 4
    raw_inputs["co2_dyn"][t] = np.mean(emi_input[i:(i + 4)])

# %% data clustering

inputs_clustering = np.array([raw_inputs["heat"],
                              raw_inputs["dhw"],
                              raw_inputs["electricity"],
                              raw_inputs["solar_roof"],
                              raw_inputs["temperature"],
                              raw_inputs["co2_dyn"]])

number_clusters = 15
(inputs, nc, z) = clustering.cluster(inputs_clustering,
                                     number_clusters=number_clusters,
                                     norm=2,
                                     mip_gap=0.0,
                                     weights=[1, 2, 2, 2, 1, 2])

# Determine time steps per day
len_day = int(inputs_clustering.shape[1] / 365)

clustered = {}
clustered["heat"] = inputs[0]
clustered["dhw"] = inputs[1]
clustered["electricity"] = inputs[2]
clustered["solar_irrad"] = inputs[3]
clustered["temperature"] = inputs[4]
clustered["co2_dyn"] = inputs[5]
clustered["co2_stat"] = np.zeros_like(clustered["co2_dyn"])
clustered["co2_stat"][:, :] = np.mean(raw_inputs["co2_dyn"])
clustered["weights"] = nc
clustered["z"] = z

# %% load devices, econmoics, etc.

devs = pik.read_devices(timesteps=len_day,
                        days=number_clusters,
                        temperature_ambient=clustered["temperature"],
                        solar_irradiation=clustered["solar_irrad"],
                        days_per_cluster=clustered["weights"])

(eco, params, devs) = pik.read_economics(devs)
params = pik.compute_parameters(params, number_clusters, len_day)

# %% create network

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
net_name = "dorfnetz"
# net = nw.create_kerber_+net_name
fkt_name = "create_kerber_" + net_name
fkt = getattr(nw, fkt_name)
net = fkt()
# net = nw.create_kerber_vorstadtnetz_kabel_2()
# net = nw.create_kerber_landnetz_kabel_2()

if options["show_grid_plots"]:
    # simple plot of net with existing geocoordinates or generated artificial geocoordinates
    plot.simple_plot(net, show_plot=True)

# %% Store clustered input parameters

with open(options["filename_inputs"], "wb") as f_in:
    pickle.dump(clustered, f_in, pickle.HIGHEST_PROTOCOL)

# specify grid nodes for whole grid and trafo; choose and allocate load, injection and battery nodes
# draw parameters from pandapower network
nodes = {}

nodes["grid"] = net.bus.index.to_numpy()
nodes["trafo"] = net.trafo['lv_bus'].to_numpy()
nodes["load"] = net.load['bus'].to_numpy()
nodes["bat"] = net.load['bus'].to_numpy()

# define sgens for net in order to be able to include gen values in timeloop
nodesload = list(nodes["load"])
for n in nodesload:
    pp.create_sgen(net, n, p_mw=0)

# define gridnodes, days and timesteps
gridnodes = list(nodes["grid"])
days = [i for i in range(params["days"])]
timesteps = [i for i in range(params["time_steps"])]

# solution_found as continuos variable for while loop
solution_found = []
for d in days:
    solution_found.append(False)
boolean_loop = True
# constraint_apc models APC, gets reduced from 1 to 0 in iteration steps with range 0.1
constraint_apc = {}
# constraint for Injection and Subtraction. Inj gets cut when voltage is too high, Subtr gets cut when voltage is too low
constraint_InjMin = {}
constraint_SubtrMin = {}
constraint_InjMax = {}
constraint_SubtrMax = {}
# create array to flag whether values are critical for powerflow. If not critical: 0, if critical: 1
critical_flag = {}
iteration_counter = 0
# introduce boolean to state infeasability
infeasability = False

change_value = options["change_value_node_violation_abs"]

for n in gridnodes:
    for d in days:
        for t in timesteps:
            critical_flag[n, d, t] = 0
            constraint_apc[n, d] = 0
            constraint_InjMin[n, d, t] = 0
            constraint_SubtrMin[n, d, t] = 0
            constraint_InjMax[n, d, t] = 10000
            constraint_SubtrMax[n, d, t] = 10000

while boolean_loop:
    print("")
    print("!!! Iteration counter is currently at " + str(iteration_counter) + "!!!")
    print("")
    """for d in days:
            output_dir = os.path.join(tempfile.gettempdir(), "time_series_example" + str(d))
            shutil.rmtree(output_dir)"""

    # run DC-optimization
    (
    costs_grid, emissions_grid, timesteps, days, powInjRet, powSubtrRet, gridnodes, res_exBat, powInjPrev, powSubtrPrev,
    emissions_nodes, costs_nodes, objective_function) = opti.compute(net, nodes, gridnodes, days, timesteps, eco, devs,
                                                                     clustered, params, options, constraint_apc,
                                                                     constraint_InjMin, constraint_SubtrMin,
                                                                     constraint_InjMax, constraint_SubtrMax,
                                                                     critical_flag)

    outputs = reader.read_results(options)
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
    (output_dir, critical_flag, solution_found, vm_pu_total) = loop.run_timeloop(fkt, timesteps, days, powInjRet,
                                                                                 powSubtrRet, gridnodes, critical_flag,
                                                                                 solution_found)
    # vm_pu_total_array = np.array([[[vm_pu_total[n, d, t] for t in timesteps] for d in days] for n in gridnodes])
    print("zwischenstop")
    for d in days:
        if (solution_found[d] == False):

            print("Additional constrains have to be set for day" + str(d))
            if options["apc_while_voltage_violation"]:
                if options["cut_Inj/Subtr_while_voltage_violation"]:
                    print("You selected both apc and Inj/Subtr cutting in case of voltage violations.")
                    for n in gridnodes:
                        if (any(critical_flag[n, d, t] == 1 for t in timesteps)):
                            if (any(vm_pu_total[n, d, t] < 0.96 for t in timesteps)):
                                pass
                            elif (any(vm_pu_total[n, d, t] > 1.04 for t in timesteps)):
                                constraint_apc[n, d] += 0.1
                                if (constraint_apc[n, d] >= 1):
                                    print("You have reached the maximal amount of curtailment!")
                                    print("Will set curtailment to 100 Percent automatically.")
                                    constraint_apc[n, d] = 1
                            else:
                                pass
                            for t in timesteps:
                                if (critical_flag[n, d, t] == 1):
                                    if (vm_pu_total[n, d, t] < 0.96):
                                        # relative Lösung wirft Problem der Spannungsweiterleitung auf
                                        constraint_SubtrMax[n, d, t] = 0.90 * powSubtrPrev[n, d, t]
                                        """# absolute Regelung:
                                        if (powSubtrPrev[n, d, t] < change_value):
                                            constraint_SubtrMax[n, d, t] = 0
                                            print("Subtraction already set to 0 for node" +str(n) + " and timestep" +str(t))
                                            print("Raising Injection now!")
                                            constraint_InjMin[n,d,t] += change_value
                                        else:
                                            constraint_SubtrMax[n,d,t] = powSubtrPrev[n,d,t] - change_value"""

                                    elif (vm_pu_total[n, d, t] > 1.04):
                                        constraint_InjMax[n, d, t] = 0.90 * powInjPrev[n, d, t]
                                        """
                                        #absolute änderung
                                        if (powInjPrev[n,d,t] < change_value):
                                            constraint_InjMax[n,d,t] = 0
                                            print("Injection already set to 0 for node" +str(n) + " and timestep" +str(t))
                                            print("Raising Subtraction now!")
                                            constraint_SubtrMin[n,d,t] += change_value
                                        else:
                                            constraint_InjMax[n,d,t] = powInjPrev[n,d,t] - change_value"""

                else:
                    print("You selected only apc in case of voltage violations.")
                    for n in gridnodes:
                        if (any(critical_flag[n, d, t] == 1 for t in timesteps)):
                            if (any(vm_pu_total[n, d, t] < 0.96 for t in timesteps)):
                                print(
                                    "Only apc will not fix any voltage issues, because the load is too high on day" + str(
                                        d))
                                infeasability = True
                            elif (any(vm_pu_total[n, d, t] > 1.04 for t in timesteps)):
                                constraint_apc[n, d] += 0.1
                                if (constraint_apc[n, d] >= 1):
                                    print("You have reached the maximal amount of curtailment!")
                                    print("Will set curtailment to 100 Percent automatically.")
                                    constraint_apc[n, d] = 1
                                    print(
                                        "Since you only selected apc, it has reached 100 Percent and you haven't found a solution, the problem appears to be infeasable for these settings!")
                                    infeasability = True

            elif (options["cut_Inj/Subtr_while_voltage_violation"] == True and options[
                "apc_while_voltage_violation"] == False):
                print("You selected only Inj/Subtr cutting in case of voltage violations.")
                for n in gridnodes:
                    for t in timesteps:
                        if (critical_flag[n, d, t] == 1):
                            if (vm_pu_total[n, d, t] < 0.96):
                                constraint_SubtrMax[n, d, t] = 0.90 * powSubtrPrev[n, d, t]
                                """# absolute Regelung:
                                if (powSubtrPrev[n, d, t] < change_value):
                                    constraint_SubtrMax[n, d, t] = 0
                                    print("Subtraction already set to 0 for node" +str(n) + " and timestep" +str(t))
                                    print("Raising Injection now!")
                                    constraint_InjMin[n,d,t] += change_value
                                else:
                                    constraint_SubtrMax[n,d,t] = powSubtrPrev[n,d,t] - change_value"""

                            elif (vm_pu_total[n, d, t] > 1.04):
                                constraint_InjMax[n, d, t] = 0.90 * powInjPrev[n, d, t]
                                """
                                #absolute Regelung
                                if (powInjPrev[n, d, t] < change_value):
                                    constraint_InjMax[n, d, t] = 0
                                    print("Injection already set to 0 for node" + str(n) + " and timestep" + str(t))
                                    print("Raising Subtraction now!")
                                    constraint_SubtrMin[n, d, t] += change_value
                                else:
                                    constraint_InjMax[n, d, t] = powInjPrev[n, d, t] - change_value"""

            elif (options["cut_Inj/Subtr_while_voltage_violation"] == False and options[
                "apc_while_voltage_violation"] == False):
                print("Error: You did not select any measure in case of voltage violations!")
                infeasability = True

        if (solution_found[d] == True):
            print("Solution was successfully found for day" + str(d))

    if infeasability:
        print("Error: Model appears to be infeasable for the selected settings!")
        print("Reasons are stated above.")
        break

    iteration_counter += 1

    if all(solution_found[d] == True for d in days):
        print("Congratulations! Your optimization and loadflow calculation has been successfully finished after " + str(
            iteration_counter - 1) + " iteration steps!")
        break

t2 = int(time.time())
duration_program = t1 - t2

print("this is the end")
# plot_res.plot_results(outputs, days, gridnodes, timesteps)
print("")
print("")
print("objective value für 15 Typtage:")
print(objective_function)
print("duration_program für 15 Typtage")
print(duration_program)
print("")
print("")
print("")

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

#!!!!!!!!!!!!!!!!!!!!!! HIER 14 TAGE
# get time from start to calculate duration of the program
t1 = int(time.time())

# set parameters
building_type = "EFH"  # EFH, ZFH, MFH_6WE, MFH_10WE, MFH_15WE
building_age = "2005"  # 1960, 1980, 2005
emission_year = "2017"  # 2017, 2030, 2050

# TODO: implement mixed shares of buildings
# TODO: adjust emission factors regarding to national weather conditions

# TODO: load data for useable roofarea per building type
# TODO: PV area as MILP-variable??
# useable_roofarea  = 0.30    #Default value: 0.25

# set options
options = {"static_emissions": False,  # True: calculation with static emissions,
           # False: calculation with timevariant emissions
           "rev_emissions": False,  # True: emissions revenues for feed-in
           # False: no emissions revenues for feed-in
           "dhw_electric": True,  # define if dhw is provided decentrally by electricity
           "P_pv": 10.00,  # installed peak PV power
           "with_hp": True,  # usage of heat pumps
           "hp_mode": "grid_opt",  # choose between "energy_opt" and "grid_opt"
           "T_VL": 35,  # choose between 35 and 55 "Vorlauftemperatur"
           "alpha_th": 1,  # relative size of heat pump (between 0 and 1)
           "beta_th": 0.417,  # relative size of thermal energy storage (between 0 and 1)
           "show_grid_plots": False,  # show gridplots before and after optimization

           "filename_results": "results/" + building_type + "_" + \
                               building_age + "Typtage_14.pkl",
           "filename_inputs": "results/inputs_" + building_type + "_" + \
                              building_age + "Typtage_14.pkl",
           "apc_while_voltage_violation": False,  # True: uses apc, when voltage violations occur
           # False: does not use apc, when voltage violations occur
           "cut_Inj/Subtr_while_voltage_violation": True,  # True: cuts Inj or Subtr, when voltage violations occur
           # depends automatically on the fact, whether voltage is too high or too low
           # note: power costs for heatpump can only be separated for a cost calculation for sum(nodes), not for grid in total!
           # select your cost function calculation through objective function in grid_optimization
           "heatpump_seperated_costs": True,  # True: Heatpumps power costs: 18.56 ct/kWh (apart from other power users)
           # False: Heatpump power costs: 27.8 ct/kWh (equal to other power users)
           "allow_apc_opti": True,  # True: Curtailment allowed to be set in optimization
           # False: Curtailment only through additional constraint
           "change_value_node_violation_abs": 2
           # specify, for how much the absolute values of inj and subtr should change in case of voltage violations
           }

# %% data import

# determine the optimization folder in which all input data and results are placed
operationFolder = "C:\\users\\flori\\pycharmprojects\\flexigrid"
# the input data is always in this source folder
sourceFolder = operationFolder + "\\input"

raw_inputs = {}

raw_inputs["heat"] = np.maximum(0, np.loadtxt(
    sourceFolder + "\\Typgebäude\\" + building_type + "\\" + building_age + "\\heat.csv") / 1000)
raw_inputs["dhw"] = np.maximum(0, np.loadtxt(
    sourceFolder + "\\Typgebäude\\" + building_type + "\\" + building_age + "\\dhw.csv") / 1000)
raw_inputs["electricity"] = np.maximum(0, np.loadtxt(
    sourceFolder + "\\Typgebäude\\" + building_type + "\\" + building_age + "\\electricity.csv") / 1000)
raw_inputs["solar_roof"] = np.maximum(0, np.loadtxt(
    sourceFolder + "\\Typgebäude\\" + building_type + "\\" + building_age + "\\solar_roof.csv") / 1000)
raw_inputs["temperature"] = np.loadtxt(
    sourceFolder + "\\Typgebäude\\" + building_type + "\\" + building_age + "\\temperature.csv")

emi_input = pd.read_csv(sourceFolder + "\\emission_factor_" + emission_year + ".csv", header=0, usecols=[2])
raw_inputs["co2_dyn"] = np.zeros([8760])
for t in range(0, 8760):
    i = t * 4
    raw_inputs["co2_dyn"][t] = np.mean(emi_input[i:(i + 4)])

# %% data clustering

inputs_clustering = np.array([raw_inputs["heat"],
                              raw_inputs["dhw"],
                              raw_inputs["electricity"],
                              raw_inputs["solar_roof"],
                              raw_inputs["temperature"],
                              raw_inputs["co2_dyn"]])

number_clusters = 14
(inputs, nc, z) = clustering.cluster(inputs_clustering,
                                     number_clusters=number_clusters,
                                     norm=2,
                                     mip_gap=0.0,
                                     weights=[1, 2, 2, 2, 1, 2])

# Determine time steps per day
len_day = int(inputs_clustering.shape[1] / 365)

clustered = {}
clustered["heat"] = inputs[0]
clustered["dhw"] = inputs[1]
clustered["electricity"] = inputs[2]
clustered["solar_irrad"] = inputs[3]
clustered["temperature"] = inputs[4]
clustered["co2_dyn"] = inputs[5]
clustered["co2_stat"] = np.zeros_like(clustered["co2_dyn"])
clustered["co2_stat"][:, :] = np.mean(raw_inputs["co2_dyn"])
clustered["weights"] = nc
clustered["z"] = z

# %% load devices, econmoics, etc.

devs = pik.read_devices(timesteps=len_day,
                        days=number_clusters,
                        temperature_ambient=clustered["temperature"],
                        solar_irradiation=clustered["solar_irrad"],
                        days_per_cluster=clustered["weights"])

(eco, params, devs) = pik.read_economics(devs)
params = pik.compute_parameters(params, number_clusters, len_day)

# %% create network

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
net_name = "dorfnetz"
# net = nw.create_kerber_+net_name
fkt_name = "create_kerber_" + net_name
fkt = getattr(nw, fkt_name)
net = fkt()
# net = nw.create_kerber_vorstadtnetz_kabel_2()
# net = nw.create_kerber_landnetz_kabel_2()

if options["show_grid_plots"]:
    # simple plot of net with existing geocoordinates or generated artificial geocoordinates
    plot.simple_plot(net, show_plot=True)

# %% Store clustered input parameters

with open(options["filename_inputs"], "wb") as f_in:
    pickle.dump(clustered, f_in, pickle.HIGHEST_PROTOCOL)

# specify grid nodes for whole grid and trafo; choose and allocate load, injection and battery nodes
# draw parameters from pandapower network
nodes = {}

nodes["grid"] = net.bus.index.to_numpy()
nodes["trafo"] = net.trafo['lv_bus'].to_numpy()
nodes["load"] = net.load['bus'].to_numpy()
nodes["bat"] = net.load['bus'].to_numpy()

# define sgens for net in order to be able to include gen values in timeloop
nodesload = list(nodes["load"])
for n in nodesload:
    pp.create_sgen(net, n, p_mw=0)

# define gridnodes, days and timesteps
gridnodes = list(nodes["grid"])
days = [i for i in range(params["days"])]
timesteps = [i for i in range(params["time_steps"])]

# solution_found as continuos variable for while loop
solution_found = []
for d in days:
    solution_found.append(False)
boolean_loop = True
# constraint_apc models APC, gets reduced from 1 to 0 in iteration steps with range 0.1
constraint_apc = {}
# constraint for Injection and Subtraction. Inj gets cut when voltage is too high, Subtr gets cut when voltage is too low
constraint_InjMin = {}
constraint_SubtrMin = {}
constraint_InjMax = {}
constraint_SubtrMax = {}
# create array to flag whether values are critical for powerflow. If not critical: 0, if critical: 1
critical_flag = {}
iteration_counter = 0
# introduce boolean to state infeasability
infeasability = False

change_value = options["change_value_node_violation_abs"]

for n in gridnodes:
    for d in days:
        for t in timesteps:
            critical_flag[n, d, t] = 0
            constraint_apc[n, d] = 0
            constraint_InjMin[n, d, t] = 0
            constraint_SubtrMin[n, d, t] = 0
            constraint_InjMax[n, d, t] = 10000
            constraint_SubtrMax[n, d, t] = 10000

while boolean_loop:
    print("")
    print("!!! Iteration counter is currently at " + str(iteration_counter) + "!!!")
    print("")
    """for d in days:
            output_dir = os.path.join(tempfile.gettempdir(), "time_series_example" + str(d))
            shutil.rmtree(output_dir)"""

    # run DC-optimization
    (
    costs_grid, emissions_grid, timesteps, days, powInjRet, powSubtrRet, gridnodes, res_exBat, powInjPrev, powSubtrPrev,
    emissions_nodes, costs_nodes, objective_function) = opti.compute(net, nodes, gridnodes, days, timesteps, eco, devs,
                                                                     clustered, params, options, constraint_apc,
                                                                     constraint_InjMin, constraint_SubtrMin,
                                                                     constraint_InjMax, constraint_SubtrMax,
                                                                     critical_flag)

    outputs = reader.read_results(options)
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
    (output_dir, critical_flag, solution_found, vm_pu_total) = loop.run_timeloop(fkt, timesteps, days, powInjRet,
                                                                                 powSubtrRet, gridnodes, critical_flag,
                                                                                 solution_found)
    # vm_pu_total_array = np.array([[[vm_pu_total[n, d, t] for t in timesteps] for d in days] for n in gridnodes])
    print("zwischenstop")
    for d in days:
        if (solution_found[d] == False):

            print("Additional constrains have to be set for day" + str(d))
            if options["apc_while_voltage_violation"]:
                if options["cut_Inj/Subtr_while_voltage_violation"]:
                    print("You selected both apc and Inj/Subtr cutting in case of voltage violations.")
                    for n in gridnodes:
                        if (any(critical_flag[n, d, t] == 1 for t in timesteps)):
                            if (any(vm_pu_total[n, d, t] < 0.96 for t in timesteps)):
                                pass
                            elif (any(vm_pu_total[n, d, t] > 1.04 for t in timesteps)):
                                constraint_apc[n, d] += 0.1
                                if (constraint_apc[n, d] >= 1):
                                    print("You have reached the maximal amount of curtailment!")
                                    print("Will set curtailment to 100 Percent automatically.")
                                    constraint_apc[n, d] = 1
                            else:
                                pass
                            for t in timesteps:
                                if (critical_flag[n, d, t] == 1):
                                    if (vm_pu_total[n, d, t] < 0.96):
                                        # relative Lösung wirft Problem der Spannungsweiterleitung auf
                                        constraint_SubtrMax[n, d, t] = 0.90 * powSubtrPrev[n, d, t]
                                        """# absolute Regelung:
                                        if (powSubtrPrev[n, d, t] < change_value):
                                            constraint_SubtrMax[n, d, t] = 0
                                            print("Subtraction already set to 0 for node" +str(n) + " and timestep" +str(t))
                                            print("Raising Injection now!")
                                            constraint_InjMin[n,d,t] += change_value
                                        else:
                                            constraint_SubtrMax[n,d,t] = powSubtrPrev[n,d,t] - change_value"""

                                    elif (vm_pu_total[n, d, t] > 1.04):
                                        constraint_InjMax[n, d, t] = 0.90 * powInjPrev[n, d, t]
                                        """
                                        #absolute änderung
                                        if (powInjPrev[n,d,t] < change_value):
                                            constraint_InjMax[n,d,t] = 0
                                            print("Injection already set to 0 for node" +str(n) + " and timestep" +str(t))
                                            print("Raising Subtraction now!")
                                            constraint_SubtrMin[n,d,t] += change_value
                                        else:
                                            constraint_InjMax[n,d,t] = powInjPrev[n,d,t] - change_value"""

                else:
                    print("You selected only apc in case of voltage violations.")
                    for n in gridnodes:
                        if (any(critical_flag[n, d, t] == 1 for t in timesteps)):
                            if (any(vm_pu_total[n, d, t] < 0.96 for t in timesteps)):
                                print(
                                    "Only apc will not fix any voltage issues, because the load is too high on day" + str(
                                        d))
                                infeasability = True
                            elif (any(vm_pu_total[n, d, t] > 1.04 for t in timesteps)):
                                constraint_apc[n, d] += 0.1
                                if (constraint_apc[n, d] >= 1):
                                    print("You have reached the maximal amount of curtailment!")
                                    print("Will set curtailment to 100 Percent automatically.")
                                    constraint_apc[n, d] = 1
                                    print(
                                        "Since you only selected apc, it has reached 100 Percent and you haven't found a solution, the problem appears to be infeasable for these settings!")
                                    infeasability = True

            elif (options["cut_Inj/Subtr_while_voltage_violation"] == True and options[
                "apc_while_voltage_violation"] == False):
                print("You selected only Inj/Subtr cutting in case of voltage violations.")
                for n in gridnodes:
                    for t in timesteps:
                        if (critical_flag[n, d, t] == 1):
                            if (vm_pu_total[n, d, t] < 0.96):
                                constraint_SubtrMax[n, d, t] = 0.90 * powSubtrPrev[n, d, t]
                                """# absolute Regelung:
                                if (powSubtrPrev[n, d, t] < change_value):
                                    constraint_SubtrMax[n, d, t] = 0
                                    print("Subtraction already set to 0 for node" +str(n) + " and timestep" +str(t))
                                    print("Raising Injection now!")
                                    constraint_InjMin[n,d,t] += change_value
                                else:
                                    constraint_SubtrMax[n,d,t] = powSubtrPrev[n,d,t] - change_value"""

                            elif (vm_pu_total[n, d, t] > 1.04):
                                constraint_InjMax[n, d, t] = 0.90 * powInjPrev[n, d, t]
                                """
                                #absolute Regelung
                                if (powInjPrev[n, d, t] < change_value):
                                    constraint_InjMax[n, d, t] = 0
                                    print("Injection already set to 0 for node" + str(n) + " and timestep" + str(t))
                                    print("Raising Subtraction now!")
                                    constraint_SubtrMin[n, d, t] += change_value
                                else:
                                    constraint_InjMax[n, d, t] = powInjPrev[n, d, t] - change_value"""

            elif (options["cut_Inj/Subtr_while_voltage_violation"] == False and options[
                "apc_while_voltage_violation"] == False):
                print("Error: You did not select any measure in case of voltage violations!")
                infeasability = True

        if (solution_found[d] == True):
            print("Solution was successfully found for day" + str(d))

    if infeasability:
        print("Error: Model appears to be infeasable for the selected settings!")
        print("Reasons are stated above.")
        break

    iteration_counter += 1

    if all(solution_found[d] == True for d in days):
        print("Congratulations! Your optimization and loadflow calculation has been successfully finished after " + str(
            iteration_counter - 1) + " iteration steps!")
        break

t2 = int(time.time())
duration_program = t1 - t2

print("this is the end")
# plot_res.plot_results(outputs, days, gridnodes, timesteps)
print("")
print("")
print("objective value für 14 Typtage:")
print(objective_function)
print("duration_program für 14 Typtage")
print(duration_program)
print("")
print("")
print("")

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

#!!!!!!!!!!!!!!!!!!!!!! HIER 13 TAGE
# get time from start to calculate duration of the program
t1 = int(time.time())

# set parameters
building_type = "EFH"  # EFH, ZFH, MFH_6WE, MFH_10WE, MFH_15WE
building_age = "2005"  # 1960, 1980, 2005
emission_year = "2017"  # 2017, 2030, 2050

# TODO: implement mixed shares of buildings
# TODO: adjust emission factors regarding to national weather conditions

# TODO: load data for useable roofarea per building type
# TODO: PV area as MILP-variable??
# useable_roofarea  = 0.30    #Default value: 0.25

# set options
options = {"static_emissions": False,  # True: calculation with static emissions,
           # False: calculation with timevariant emissions
           "rev_emissions": False,  # True: emissions revenues for feed-in
           # False: no emissions revenues for feed-in
           "dhw_electric": True,  # define if dhw is provided decentrally by electricity
           "P_pv": 10.00,  # installed peak PV power
           "with_hp": True,  # usage of heat pumps
           "hp_mode": "grid_opt",  # choose between "energy_opt" and "grid_opt"
           "T_VL": 35,  # choose between 35 and 55 "Vorlauftemperatur"
           "alpha_th": 1,  # relative size of heat pump (between 0 and 1)
           "beta_th": 0.417,  # relative size of thermal energy storage (between 0 and 1)
           "show_grid_plots": False,  # show gridplots before and after optimization

           "filename_results": "results/" + building_type + "_" + \
                               building_age + "Typtage_13.pkl",
           "filename_inputs": "results/inputs_" + building_type + "_" + \
                              building_age + "Typtage_13.pkl",
           "apc_while_voltage_violation": False,  # True: uses apc, when voltage violations occur
           # False: does not use apc, when voltage violations occur
           "cut_Inj/Subtr_while_voltage_violation": True,  # True: cuts Inj or Subtr, when voltage violations occur
           # depends automatically on the fact, whether voltage is too high or too low
           # note: power costs for heatpump can only be separated for a cost calculation for sum(nodes), not for grid in total!
           # select your cost function calculation through objective function in grid_optimization
           "heatpump_seperated_costs": True,  # True: Heatpumps power costs: 18.56 ct/kWh (apart from other power users)
           # False: Heatpump power costs: 27.8 ct/kWh (equal to other power users)
           "allow_apc_opti": True,  # True: Curtailment allowed to be set in optimization
           # False: Curtailment only through additional constraint
           "change_value_node_violation_abs": 2
           # specify, for how much the absolute values of inj and subtr should change in case of voltage violations
           }

# %% data import

# determine the optimization folder in which all input data and results are placed
operationFolder = "C:\\users\\flori\\pycharmprojects\\flexigrid"
# the input data is always in this source folder
sourceFolder = operationFolder + "\\input"

raw_inputs = {}

raw_inputs["heat"] = np.maximum(0, np.loadtxt(
    sourceFolder + "\\Typgebäude\\" + building_type + "\\" + building_age + "\\heat.csv") / 1000)
raw_inputs["dhw"] = np.maximum(0, np.loadtxt(
    sourceFolder + "\\Typgebäude\\" + building_type + "\\" + building_age + "\\dhw.csv") / 1000)
raw_inputs["electricity"] = np.maximum(0, np.loadtxt(
    sourceFolder + "\\Typgebäude\\" + building_type + "\\" + building_age + "\\electricity.csv") / 1000)
raw_inputs["solar_roof"] = np.maximum(0, np.loadtxt(
    sourceFolder + "\\Typgebäude\\" + building_type + "\\" + building_age + "\\solar_roof.csv") / 1000)
raw_inputs["temperature"] = np.loadtxt(
    sourceFolder + "\\Typgebäude\\" + building_type + "\\" + building_age + "\\temperature.csv")

emi_input = pd.read_csv(sourceFolder + "\\emission_factor_" + emission_year + ".csv", header=0, usecols=[2])
raw_inputs["co2_dyn"] = np.zeros([8760])
for t in range(0, 8760):
    i = t * 4
    raw_inputs["co2_dyn"][t] = np.mean(emi_input[i:(i + 4)])

# %% data clustering

inputs_clustering = np.array([raw_inputs["heat"],
                              raw_inputs["dhw"],
                              raw_inputs["electricity"],
                              raw_inputs["solar_roof"],
                              raw_inputs["temperature"],
                              raw_inputs["co2_dyn"]])

number_clusters = 13
(inputs, nc, z) = clustering.cluster(inputs_clustering,
                                     number_clusters=number_clusters,
                                     norm=2,
                                     mip_gap=0.0,
                                     weights=[1, 2, 2, 2, 1, 2])

# Determine time steps per day
len_day = int(inputs_clustering.shape[1] / 365)

clustered = {}
clustered["heat"] = inputs[0]
clustered["dhw"] = inputs[1]
clustered["electricity"] = inputs[2]
clustered["solar_irrad"] = inputs[3]
clustered["temperature"] = inputs[4]
clustered["co2_dyn"] = inputs[5]
clustered["co2_stat"] = np.zeros_like(clustered["co2_dyn"])
clustered["co2_stat"][:, :] = np.mean(raw_inputs["co2_dyn"])
clustered["weights"] = nc
clustered["z"] = z

# %% load devices, econmoics, etc.

devs = pik.read_devices(timesteps=len_day,
                        days=number_clusters,
                        temperature_ambient=clustered["temperature"],
                        solar_irradiation=clustered["solar_irrad"],
                        days_per_cluster=clustered["weights"])

(eco, params, devs) = pik.read_economics(devs)
params = pik.compute_parameters(params, number_clusters, len_day)

# %% create network

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
net_name = "dorfnetz"
# net = nw.create_kerber_+net_name
fkt_name = "create_kerber_" + net_name
fkt = getattr(nw, fkt_name)
net = fkt()
# net = nw.create_kerber_vorstadtnetz_kabel_2()
# net = nw.create_kerber_landnetz_kabel_2()

if options["show_grid_plots"]:
    # simple plot of net with existing geocoordinates or generated artificial geocoordinates
    plot.simple_plot(net, show_plot=True)

# %% Store clustered input parameters

with open(options["filename_inputs"], "wb") as f_in:
    pickle.dump(clustered, f_in, pickle.HIGHEST_PROTOCOL)

# specify grid nodes for whole grid and trafo; choose and allocate load, injection and battery nodes
# draw parameters from pandapower network
nodes = {}

nodes["grid"] = net.bus.index.to_numpy()
nodes["trafo"] = net.trafo['lv_bus'].to_numpy()
nodes["load"] = net.load['bus'].to_numpy()
nodes["bat"] = net.load['bus'].to_numpy()

# define sgens for net in order to be able to include gen values in timeloop
nodesload = list(nodes["load"])
for n in nodesload:
    pp.create_sgen(net, n, p_mw=0)

# define gridnodes, days and timesteps
gridnodes = list(nodes["grid"])
days = [i for i in range(params["days"])]
timesteps = [i for i in range(params["time_steps"])]

# solution_found as continuos variable for while loop
solution_found = []
for d in days:
    solution_found.append(False)
boolean_loop = True
# constraint_apc models APC, gets reduced from 1 to 0 in iteration steps with range 0.1
constraint_apc = {}
# constraint for Injection and Subtraction. Inj gets cut when voltage is too high, Subtr gets cut when voltage is too low
constraint_InjMin = {}
constraint_SubtrMin = {}
constraint_InjMax = {}
constraint_SubtrMax = {}
# create array to flag whether values are critical for powerflow. If not critical: 0, if critical: 1
critical_flag = {}
iteration_counter = 0
# introduce boolean to state infeasability
infeasability = False

change_value = options["change_value_node_violation_abs"]

for n in gridnodes:
    for d in days:
        for t in timesteps:
            critical_flag[n, d, t] = 0
            constraint_apc[n, d] = 0
            constraint_InjMin[n, d, t] = 0
            constraint_SubtrMin[n, d, t] = 0
            constraint_InjMax[n, d, t] = 10000
            constraint_SubtrMax[n, d, t] = 10000

while boolean_loop:
    print("")
    print("!!! Iteration counter is currently at " + str(iteration_counter) + "!!!")
    print("")
    """for d in days:
            output_dir = os.path.join(tempfile.gettempdir(), "time_series_example" + str(d))
            shutil.rmtree(output_dir)"""

    # run DC-optimization
    (
    costs_grid, emissions_grid, timesteps, days, powInjRet, powSubtrRet, gridnodes, res_exBat, powInjPrev, powSubtrPrev,
    emissions_nodes, costs_nodes, objective_function) = opti.compute(net, nodes, gridnodes, days, timesteps, eco, devs,
                                                                     clustered, params, options, constraint_apc,
                                                                     constraint_InjMin, constraint_SubtrMin,
                                                                     constraint_InjMax, constraint_SubtrMax,
                                                                     critical_flag)

    outputs = reader.read_results(options)
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
    (output_dir, critical_flag, solution_found, vm_pu_total) = loop.run_timeloop(fkt, timesteps, days, powInjRet,
                                                                                 powSubtrRet, gridnodes, critical_flag,
                                                                                 solution_found)
    # vm_pu_total_array = np.array([[[vm_pu_total[n, d, t] for t in timesteps] for d in days] for n in gridnodes])
    print("zwischenstop")
    for d in days:
        if (solution_found[d] == False):

            print("Additional constrains have to be set for day" + str(d))
            if options["apc_while_voltage_violation"]:
                if options["cut_Inj/Subtr_while_voltage_violation"]:
                    print("You selected both apc and Inj/Subtr cutting in case of voltage violations.")
                    for n in gridnodes:
                        if (any(critical_flag[n, d, t] == 1 for t in timesteps)):
                            if (any(vm_pu_total[n, d, t] < 0.96 for t in timesteps)):
                                pass
                            elif (any(vm_pu_total[n, d, t] > 1.04 for t in timesteps)):
                                constraint_apc[n, d] += 0.1
                                if (constraint_apc[n, d] >= 1):
                                    print("You have reached the maximal amount of curtailment!")
                                    print("Will set curtailment to 100 Percent automatically.")
                                    constraint_apc[n, d] = 1
                            else:
                                pass
                            for t in timesteps:
                                if (critical_flag[n, d, t] == 1):
                                    if (vm_pu_total[n, d, t] < 0.96):
                                        # relative Lösung wirft Problem der Spannungsweiterleitung auf
                                        constraint_SubtrMax[n, d, t] = 0.90 * powSubtrPrev[n, d, t]
                                        """# absolute Regelung:
                                        if (powSubtrPrev[n, d, t] < change_value):
                                            constraint_SubtrMax[n, d, t] = 0
                                            print("Subtraction already set to 0 for node" +str(n) + " and timestep" +str(t))
                                            print("Raising Injection now!")
                                            constraint_InjMin[n,d,t] += change_value
                                        else:
                                            constraint_SubtrMax[n,d,t] = powSubtrPrev[n,d,t] - change_value"""

                                    elif (vm_pu_total[n, d, t] > 1.04):
                                        constraint_InjMax[n, d, t] = 0.90 * powInjPrev[n, d, t]
                                        """
                                        #absolute änderung
                                        if (powInjPrev[n,d,t] < change_value):
                                            constraint_InjMax[n,d,t] = 0
                                            print("Injection already set to 0 for node" +str(n) + " and timestep" +str(t))
                                            print("Raising Subtraction now!")
                                            constraint_SubtrMin[n,d,t] += change_value
                                        else:
                                            constraint_InjMax[n,d,t] = powInjPrev[n,d,t] - change_value"""

                else:
                    print("You selected only apc in case of voltage violations.")
                    for n in gridnodes:
                        if (any(critical_flag[n, d, t] == 1 for t in timesteps)):
                            if (any(vm_pu_total[n, d, t] < 0.96 for t in timesteps)):
                                print(
                                    "Only apc will not fix any voltage issues, because the load is too high on day" + str(
                                        d))
                                infeasability = True
                            elif (any(vm_pu_total[n, d, t] > 1.04 for t in timesteps)):
                                constraint_apc[n, d] += 0.1
                                if (constraint_apc[n, d] >= 1):
                                    print("You have reached the maximal amount of curtailment!")
                                    print("Will set curtailment to 100 Percent automatically.")
                                    constraint_apc[n, d] = 1
                                    print(
                                        "Since you only selected apc, it has reached 100 Percent and you haven't found a solution, the problem appears to be infeasable for these settings!")
                                    infeasability = True

            elif (options["cut_Inj/Subtr_while_voltage_violation"] == True and options[
                "apc_while_voltage_violation"] == False):
                print("You selected only Inj/Subtr cutting in case of voltage violations.")
                for n in gridnodes:
                    for t in timesteps:
                        if (critical_flag[n, d, t] == 1):
                            if (vm_pu_total[n, d, t] < 0.96):
                                constraint_SubtrMax[n, d, t] = 0.90 * powSubtrPrev[n, d, t]
                                """# absolute Regelung:
                                if (powSubtrPrev[n, d, t] < change_value):
                                    constraint_SubtrMax[n, d, t] = 0
                                    print("Subtraction already set to 0 for node" +str(n) + " and timestep" +str(t))
                                    print("Raising Injection now!")
                                    constraint_InjMin[n,d,t] += change_value
                                else:
                                    constraint_SubtrMax[n,d,t] = powSubtrPrev[n,d,t] - change_value"""

                            elif (vm_pu_total[n, d, t] > 1.04):
                                constraint_InjMax[n, d, t] = 0.90 * powInjPrev[n, d, t]
                                """
                                #absolute Regelung
                                if (powInjPrev[n, d, t] < change_value):
                                    constraint_InjMax[n, d, t] = 0
                                    print("Injection already set to 0 for node" + str(n) + " and timestep" + str(t))
                                    print("Raising Subtraction now!")
                                    constraint_SubtrMin[n, d, t] += change_value
                                else:
                                    constraint_InjMax[n, d, t] = powInjPrev[n, d, t] - change_value"""

            elif (options["cut_Inj/Subtr_while_voltage_violation"] == False and options[
                "apc_while_voltage_violation"] == False):
                print("Error: You did not select any measure in case of voltage violations!")
                infeasability = True

        if (solution_found[d] == True):
            print("Solution was successfully found for day" + str(d))

    if infeasability:
        print("Error: Model appears to be infeasable for the selected settings!")
        print("Reasons are stated above.")
        break

    iteration_counter += 1

    if all(solution_found[d] == True for d in days):
        print("Congratulations! Your optimization and loadflow calculation has been successfully finished after " + str(
            iteration_counter - 1) + " iteration steps!")
        break

t2 = int(time.time())
duration_program = t1 - t2

print("this is the end")
# plot_res.plot_results(outputs, days, gridnodes, timesteps)
print("")
print("")
print("objective value für 13 Typtage:")
print(objective_function)
print("duration_program für 13 Typtage")
print(duration_program)
print("")
print("")
print("")

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

#!!!!!!!!!!!!!!!!!!!!!! HIER 12 TAGE
# get time from start to calculate duration of the program
t1 = int(time.time())

# set parameters
building_type = "EFH"  # EFH, ZFH, MFH_6WE, MFH_10WE, MFH_15WE
building_age = "2005"  # 1960, 1980, 2005
emission_year = "2017"  # 2017, 2030, 2050

# TODO: implement mixed shares of buildings
# TODO: adjust emission factors regarding to national weather conditions

# TODO: load data for useable roofarea per building type
# TODO: PV area as MILP-variable??
# useable_roofarea  = 0.30    #Default value: 0.25

# set options
options = {"static_emissions": False,  # True: calculation with static emissions,
           # False: calculation with timevariant emissions
           "rev_emissions": False,  # True: emissions revenues for feed-in
           # False: no emissions revenues for feed-in
           "dhw_electric": True,  # define if dhw is provided decentrally by electricity
           "P_pv": 10.00,  # installed peak PV power
           "with_hp": True,  # usage of heat pumps
           "hp_mode": "grid_opt",  # choose between "energy_opt" and "grid_opt"
           "T_VL": 35,  # choose between 35 and 55 "Vorlauftemperatur"
           "alpha_th": 1,  # relative size of heat pump (between 0 and 1)
           "beta_th": 0.417,  # relative size of thermal energy storage (between 0 and 1)
           "show_grid_plots": False,  # show gridplots before and after optimization

           "filename_results": "results/" + building_type + "_" + \
                               building_age + "Typtage_12.pkl",
           "filename_inputs": "results/inputs_" + building_type + "_" + \
                              building_age + "Typtage_12.pkl",
           "apc_while_voltage_violation": False,  # True: uses apc, when voltage violations occur
           # False: does not use apc, when voltage violations occur
           "cut_Inj/Subtr_while_voltage_violation": True,  # True: cuts Inj or Subtr, when voltage violations occur
           # depends automatically on the fact, whether voltage is too high or too low
           # note: power costs for heatpump can only be separated for a cost calculation for sum(nodes), not for grid in total!
           # select your cost function calculation through objective function in grid_optimization
           "heatpump_seperated_costs": True,  # True: Heatpumps power costs: 18.56 ct/kWh (apart from other power users)
           # False: Heatpump power costs: 27.8 ct/kWh (equal to other power users)
           "allow_apc_opti": True,  # True: Curtailment allowed to be set in optimization
           # False: Curtailment only through additional constraint
           "change_value_node_violation_abs": 2
           # specify, for how much the absolute values of inj and subtr should change in case of voltage violations
           }

# %% data import

# determine the optimization folder in which all input data and results are placed
operationFolder = "C:\\users\\flori\\pycharmprojects\\flexigrid"
# the input data is always in this source folder
sourceFolder = operationFolder + "\\input"

raw_inputs = {}

raw_inputs["heat"] = np.maximum(0, np.loadtxt(
    sourceFolder + "\\Typgebäude\\" + building_type + "\\" + building_age + "\\heat.csv") / 1000)
raw_inputs["dhw"] = np.maximum(0, np.loadtxt(
    sourceFolder + "\\Typgebäude\\" + building_type + "\\" + building_age + "\\dhw.csv") / 1000)
raw_inputs["electricity"] = np.maximum(0, np.loadtxt(
    sourceFolder + "\\Typgebäude\\" + building_type + "\\" + building_age + "\\electricity.csv") / 1000)
raw_inputs["solar_roof"] = np.maximum(0, np.loadtxt(
    sourceFolder + "\\Typgebäude\\" + building_type + "\\" + building_age + "\\solar_roof.csv") / 1000)
raw_inputs["temperature"] = np.loadtxt(
    sourceFolder + "\\Typgebäude\\" + building_type + "\\" + building_age + "\\temperature.csv")

emi_input = pd.read_csv(sourceFolder + "\\emission_factor_" + emission_year + ".csv", header=0, usecols=[2])
raw_inputs["co2_dyn"] = np.zeros([8760])
for t in range(0, 8760):
    i = t * 4
    raw_inputs["co2_dyn"][t] = np.mean(emi_input[i:(i + 4)])

# %% data clustering

inputs_clustering = np.array([raw_inputs["heat"],
                              raw_inputs["dhw"],
                              raw_inputs["electricity"],
                              raw_inputs["solar_roof"],
                              raw_inputs["temperature"],
                              raw_inputs["co2_dyn"]])

number_clusters = 12
(inputs, nc, z) = clustering.cluster(inputs_clustering,
                                     number_clusters=number_clusters,
                                     norm=2,
                                     mip_gap=0.0,
                                     weights=[1, 2, 2, 2, 1, 2])

# Determine time steps per day
len_day = int(inputs_clustering.shape[1] / 365)

clustered = {}
clustered["heat"] = inputs[0]
clustered["dhw"] = inputs[1]
clustered["electricity"] = inputs[2]
clustered["solar_irrad"] = inputs[3]
clustered["temperature"] = inputs[4]
clustered["co2_dyn"] = inputs[5]
clustered["co2_stat"] = np.zeros_like(clustered["co2_dyn"])
clustered["co2_stat"][:, :] = np.mean(raw_inputs["co2_dyn"])
clustered["weights"] = nc
clustered["z"] = z

# %% load devices, econmoics, etc.

devs = pik.read_devices(timesteps=len_day,
                        days=number_clusters,
                        temperature_ambient=clustered["temperature"],
                        solar_irradiation=clustered["solar_irrad"],
                        days_per_cluster=clustered["weights"])

(eco, params, devs) = pik.read_economics(devs)
params = pik.compute_parameters(params, number_clusters, len_day)

# %% create network

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
net_name = "dorfnetz"
# net = nw.create_kerber_+net_name
fkt_name = "create_kerber_" + net_name
fkt = getattr(nw, fkt_name)
net = fkt()
# net = nw.create_kerber_vorstadtnetz_kabel_2()
# net = nw.create_kerber_landnetz_kabel_2()

if options["show_grid_plots"]:
    # simple plot of net with existing geocoordinates or generated artificial geocoordinates
    plot.simple_plot(net, show_plot=True)

# %% Store clustered input parameters

with open(options["filename_inputs"], "wb") as f_in:
    pickle.dump(clustered, f_in, pickle.HIGHEST_PROTOCOL)

# specify grid nodes for whole grid and trafo; choose and allocate load, injection and battery nodes
# draw parameters from pandapower network
nodes = {}

nodes["grid"] = net.bus.index.to_numpy()
nodes["trafo"] = net.trafo['lv_bus'].to_numpy()
nodes["load"] = net.load['bus'].to_numpy()
nodes["bat"] = net.load['bus'].to_numpy()

# define sgens for net in order to be able to include gen values in timeloop
nodesload = list(nodes["load"])
for n in nodesload:
    pp.create_sgen(net, n, p_mw=0)

# define gridnodes, days and timesteps
gridnodes = list(nodes["grid"])
days = [i for i in range(params["days"])]
timesteps = [i for i in range(params["time_steps"])]

# solution_found as continuos variable for while loop
solution_found = []
for d in days:
    solution_found.append(False)
boolean_loop = True
# constraint_apc models APC, gets reduced from 1 to 0 in iteration steps with range 0.1
constraint_apc = {}
# constraint for Injection and Subtraction. Inj gets cut when voltage is too high, Subtr gets cut when voltage is too low
constraint_InjMin = {}
constraint_SubtrMin = {}
constraint_InjMax = {}
constraint_SubtrMax = {}
# create array to flag whether values are critical for powerflow. If not critical: 0, if critical: 1
critical_flag = {}
iteration_counter = 0
# introduce boolean to state infeasability
infeasability = False

change_value = options["change_value_node_violation_abs"]

for n in gridnodes:
    for d in days:
        for t in timesteps:
            critical_flag[n, d, t] = 0
            constraint_apc[n, d] = 0
            constraint_InjMin[n, d, t] = 0
            constraint_SubtrMin[n, d, t] = 0
            constraint_InjMax[n, d, t] = 10000
            constraint_SubtrMax[n, d, t] = 10000

while boolean_loop:
    print("")
    print("!!! Iteration counter is currently at " + str(iteration_counter) + "!!!")
    print("")
    """for d in days:
            output_dir = os.path.join(tempfile.gettempdir(), "time_series_example" + str(d))
            shutil.rmtree(output_dir)"""

    # run DC-optimization
    (
    costs_grid, emissions_grid, timesteps, days, powInjRet, powSubtrRet, gridnodes, res_exBat, powInjPrev, powSubtrPrev,
    emissions_nodes, costs_nodes, objective_function) = opti.compute(net, nodes, gridnodes, days, timesteps, eco, devs,
                                                                     clustered, params, options, constraint_apc,
                                                                     constraint_InjMin, constraint_SubtrMin,
                                                                     constraint_InjMax, constraint_SubtrMax,
                                                                     critical_flag)

    outputs = reader.read_results(options)
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
    (output_dir, critical_flag, solution_found, vm_pu_total) = loop.run_timeloop(fkt, timesteps, days, powInjRet,
                                                                                 powSubtrRet, gridnodes, critical_flag,
                                                                                 solution_found)
    # vm_pu_total_array = np.array([[[vm_pu_total[n, d, t] for t in timesteps] for d in days] for n in gridnodes])
    print("zwischenstop")
    for d in days:
        if (solution_found[d] == False):

            print("Additional constrains have to be set for day" + str(d))
            if options["apc_while_voltage_violation"]:
                if options["cut_Inj/Subtr_while_voltage_violation"]:
                    print("You selected both apc and Inj/Subtr cutting in case of voltage violations.")
                    for n in gridnodes:
                        if (any(critical_flag[n, d, t] == 1 for t in timesteps)):
                            if (any(vm_pu_total[n, d, t] < 0.96 for t in timesteps)):
                                pass
                            elif (any(vm_pu_total[n, d, t] > 1.04 for t in timesteps)):
                                constraint_apc[n, d] += 0.1
                                if (constraint_apc[n, d] >= 1):
                                    print("You have reached the maximal amount of curtailment!")
                                    print("Will set curtailment to 100 Percent automatically.")
                                    constraint_apc[n, d] = 1
                            else:
                                pass
                            for t in timesteps:
                                if (critical_flag[n, d, t] == 1):
                                    if (vm_pu_total[n, d, t] < 0.96):
                                        # relative Lösung wirft Problem der Spannungsweiterleitung auf
                                        constraint_SubtrMax[n, d, t] = 0.90 * powSubtrPrev[n, d, t]
                                        """# absolute Regelung:
                                        if (powSubtrPrev[n, d, t] < change_value):
                                            constraint_SubtrMax[n, d, t] = 0
                                            print("Subtraction already set to 0 for node" +str(n) + " and timestep" +str(t))
                                            print("Raising Injection now!")
                                            constraint_InjMin[n,d,t] += change_value
                                        else:
                                            constraint_SubtrMax[n,d,t] = powSubtrPrev[n,d,t] - change_value"""

                                    elif (vm_pu_total[n, d, t] > 1.04):
                                        constraint_InjMax[n, d, t] = 0.90 * powInjPrev[n, d, t]
                                        """
                                        #absolute änderung
                                        if (powInjPrev[n,d,t] < change_value):
                                            constraint_InjMax[n,d,t] = 0
                                            print("Injection already set to 0 for node" +str(n) + " and timestep" +str(t))
                                            print("Raising Subtraction now!")
                                            constraint_SubtrMin[n,d,t] += change_value
                                        else:
                                            constraint_InjMax[n,d,t] = powInjPrev[n,d,t] - change_value"""

                else:
                    print("You selected only apc in case of voltage violations.")
                    for n in gridnodes:
                        if (any(critical_flag[n, d, t] == 1 for t in timesteps)):
                            if (any(vm_pu_total[n, d, t] < 0.96 for t in timesteps)):
                                print(
                                    "Only apc will not fix any voltage issues, because the load is too high on day" + str(
                                        d))
                                infeasability = True
                            elif (any(vm_pu_total[n, d, t] > 1.04 for t in timesteps)):
                                constraint_apc[n, d] += 0.1
                                if (constraint_apc[n, d] >= 1):
                                    print("You have reached the maximal amount of curtailment!")
                                    print("Will set curtailment to 100 Percent automatically.")
                                    constraint_apc[n, d] = 1
                                    print(
                                        "Since you only selected apc, it has reached 100 Percent and you haven't found a solution, the problem appears to be infeasable for these settings!")
                                    infeasability = True

            elif (options["cut_Inj/Subtr_while_voltage_violation"] == True and options[
                "apc_while_voltage_violation"] == False):
                print("You selected only Inj/Subtr cutting in case of voltage violations.")
                for n in gridnodes:
                    for t in timesteps:
                        if (critical_flag[n, d, t] == 1):
                            if (vm_pu_total[n, d, t] < 0.96):
                                constraint_SubtrMax[n, d, t] = 0.90 * powSubtrPrev[n, d, t]
                                """# absolute Regelung:
                                if (powSubtrPrev[n, d, t] < change_value):
                                    constraint_SubtrMax[n, d, t] = 0
                                    print("Subtraction already set to 0 for node" +str(n) + " and timestep" +str(t))
                                    print("Raising Injection now!")
                                    constraint_InjMin[n,d,t] += change_value
                                else:
                                    constraint_SubtrMax[n,d,t] = powSubtrPrev[n,d,t] - change_value"""

                            elif (vm_pu_total[n, d, t] > 1.04):
                                constraint_InjMax[n, d, t] = 0.90 * powInjPrev[n, d, t]
                                """
                                #absolute Regelung
                                if (powInjPrev[n, d, t] < change_value):
                                    constraint_InjMax[n, d, t] = 0
                                    print("Injection already set to 0 for node" + str(n) + " and timestep" + str(t))
                                    print("Raising Subtraction now!")
                                    constraint_SubtrMin[n, d, t] += change_value
                                else:
                                    constraint_InjMax[n, d, t] = powInjPrev[n, d, t] - change_value"""

            elif (options["cut_Inj/Subtr_while_voltage_violation"] == False and options[
                "apc_while_voltage_violation"] == False):
                print("Error: You did not select any measure in case of voltage violations!")
                infeasability = True

        if (solution_found[d] == True):
            print("Solution was successfully found for day" + str(d))

    if infeasability:
        print("Error: Model appears to be infeasable for the selected settings!")
        print("Reasons are stated above.")
        break

    iteration_counter += 1

    if all(solution_found[d] == True for d in days):
        print("Congratulations! Your optimization and loadflow calculation has been successfully finished after " + str(
            iteration_counter - 1) + " iteration steps!")
        break

t2 = int(time.time())
duration_program = t1 - t2

print("this is the end")
# plot_res.plot_results(outputs, days, gridnodes, timesteps)
print("")
print("")
print("objective value für 12 Typtage:")
print(objective_function)
print("duration_program für 12 Typtage")
print(duration_program)
print("")
print("")
print("")

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

#!!!!!!!!!!!!!!!!!!!!!! HIER 11 TAGE
# get time from start to calculate duration of the program
t1 = int(time.time())

# set parameters
building_type = "EFH"  # EFH, ZFH, MFH_6WE, MFH_10WE, MFH_15WE
building_age = "2005"  # 1960, 1980, 2005
emission_year = "2017"  # 2017, 2030, 2050

# TODO: implement mixed shares of buildings
# TODO: adjust emission factors regarding to national weather conditions

# TODO: load data for useable roofarea per building type
# TODO: PV area as MILP-variable??
# useable_roofarea  = 0.30    #Default value: 0.25

# set options
options = {"static_emissions": False,  # True: calculation with static emissions,
           # False: calculation with timevariant emissions
           "rev_emissions": False,  # True: emissions revenues for feed-in
           # False: no emissions revenues for feed-in
           "dhw_electric": True,  # define if dhw is provided decentrally by electricity
           "P_pv": 10.00,  # installed peak PV power
           "with_hp": True,  # usage of heat pumps
           "hp_mode": "grid_opt",  # choose between "energy_opt" and "grid_opt"
           "T_VL": 35,  # choose between 35 and 55 "Vorlauftemperatur"
           "alpha_th": 1,  # relative size of heat pump (between 0 and 1)
           "beta_th": 0.417,  # relative size of thermal energy storage (between 0 and 1)
           "show_grid_plots": False,  # show gridplots before and after optimization

           "filename_results": "results/" + building_type + "_" + \
                               building_age + "Typtage_11.pkl",
           "filename_inputs": "results/inputs_" + building_type + "_" + \
                              building_age + "Typtage_11.pkl",
           "apc_while_voltage_violation": False,  # True: uses apc, when voltage violations occur
           # False: does not use apc, when voltage violations occur
           "cut_Inj/Subtr_while_voltage_violation": True,  # True: cuts Inj or Subtr, when voltage violations occur
           # depends automatically on the fact, whether voltage is too high or too low
           # note: power costs for heatpump can only be separated for a cost calculation for sum(nodes), not for grid in total!
           # select your cost function calculation through objective function in grid_optimization
           "heatpump_seperated_costs": True,  # True: Heatpumps power costs: 18.56 ct/kWh (apart from other power users)
           # False: Heatpump power costs: 27.8 ct/kWh (equal to other power users)
           "allow_apc_opti": True,  # True: Curtailment allowed to be set in optimization
           # False: Curtailment only through additional constraint
           "change_value_node_violation_abs": 2
           # specify, for how much the absolute values of inj and subtr should change in case of voltage violations
           }

# %% data import

# determine the optimization folder in which all input data and results are placed
operationFolder = "C:\\users\\flori\\pycharmprojects\\flexigrid"
# the input data is always in this source folder
sourceFolder = operationFolder + "\\input"

raw_inputs = {}

raw_inputs["heat"] = np.maximum(0, np.loadtxt(
    sourceFolder + "\\Typgebäude\\" + building_type + "\\" + building_age + "\\heat.csv") / 1000)
raw_inputs["dhw"] = np.maximum(0, np.loadtxt(
    sourceFolder + "\\Typgebäude\\" + building_type + "\\" + building_age + "\\dhw.csv") / 1000)
raw_inputs["electricity"] = np.maximum(0, np.loadtxt(
    sourceFolder + "\\Typgebäude\\" + building_type + "\\" + building_age + "\\electricity.csv") / 1000)
raw_inputs["solar_roof"] = np.maximum(0, np.loadtxt(
    sourceFolder + "\\Typgebäude\\" + building_type + "\\" + building_age + "\\solar_roof.csv") / 1000)
raw_inputs["temperature"] = np.loadtxt(
    sourceFolder + "\\Typgebäude\\" + building_type + "\\" + building_age + "\\temperature.csv")

emi_input = pd.read_csv(sourceFolder + "\\emission_factor_" + emission_year + ".csv", header=0, usecols=[2])
raw_inputs["co2_dyn"] = np.zeros([8760])
for t in range(0, 8760):
    i = t * 4
    raw_inputs["co2_dyn"][t] = np.mean(emi_input[i:(i + 4)])

# %% data clustering

inputs_clustering = np.array([raw_inputs["heat"],
                              raw_inputs["dhw"],
                              raw_inputs["electricity"],
                              raw_inputs["solar_roof"],
                              raw_inputs["temperature"],
                              raw_inputs["co2_dyn"]])

number_clusters = 11
(inputs, nc, z) = clustering.cluster(inputs_clustering,
                                     number_clusters=number_clusters,
                                     norm=2,
                                     mip_gap=0.0,
                                     weights=[1, 2, 2, 2, 1, 2])

# Determine time steps per day
len_day = int(inputs_clustering.shape[1] / 365)

clustered = {}
clustered["heat"] = inputs[0]
clustered["dhw"] = inputs[1]
clustered["electricity"] = inputs[2]
clustered["solar_irrad"] = inputs[3]
clustered["temperature"] = inputs[4]
clustered["co2_dyn"] = inputs[5]
clustered["co2_stat"] = np.zeros_like(clustered["co2_dyn"])
clustered["co2_stat"][:, :] = np.mean(raw_inputs["co2_dyn"])
clustered["weights"] = nc
clustered["z"] = z

# %% load devices, econmoics, etc.

devs = pik.read_devices(timesteps=len_day,
                        days=number_clusters,
                        temperature_ambient=clustered["temperature"],
                        solar_irradiation=clustered["solar_irrad"],
                        days_per_cluster=clustered["weights"])

(eco, params, devs) = pik.read_economics(devs)
params = pik.compute_parameters(params, number_clusters, len_day)

# %% create network

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
net_name = "dorfnetz"
# net = nw.create_kerber_+net_name
fkt_name = "create_kerber_" + net_name
fkt = getattr(nw, fkt_name)
net = fkt()
# net = nw.create_kerber_vorstadtnetz_kabel_2()
# net = nw.create_kerber_landnetz_kabel_2()

if options["show_grid_plots"]:
    # simple plot of net with existing geocoordinates or generated artificial geocoordinates
    plot.simple_plot(net, show_plot=True)

# %% Store clustered input parameters

with open(options["filename_inputs"], "wb") as f_in:
    pickle.dump(clustered, f_in, pickle.HIGHEST_PROTOCOL)

# specify grid nodes for whole grid and trafo; choose and allocate load, injection and battery nodes
# draw parameters from pandapower network
nodes = {}

nodes["grid"] = net.bus.index.to_numpy()
nodes["trafo"] = net.trafo['lv_bus'].to_numpy()
nodes["load"] = net.load['bus'].to_numpy()
nodes["bat"] = net.load['bus'].to_numpy()

# define sgens for net in order to be able to include gen values in timeloop
nodesload = list(nodes["load"])
for n in nodesload:
    pp.create_sgen(net, n, p_mw=0)

# define gridnodes, days and timesteps
gridnodes = list(nodes["grid"])
days = [i for i in range(params["days"])]
timesteps = [i for i in range(params["time_steps"])]

# solution_found as continuos variable for while loop
solution_found = []
for d in days:
    solution_found.append(False)
boolean_loop = True
# constraint_apc models APC, gets reduced from 1 to 0 in iteration steps with range 0.1
constraint_apc = {}
# constraint for Injection and Subtraction. Inj gets cut when voltage is too high, Subtr gets cut when voltage is too low
constraint_InjMin = {}
constraint_SubtrMin = {}
constraint_InjMax = {}
constraint_SubtrMax = {}
# create array to flag whether values are critical for powerflow. If not critical: 0, if critical: 1
critical_flag = {}
iteration_counter = 0
# introduce boolean to state infeasability
infeasability = False

change_value = options["change_value_node_violation_abs"]

for n in gridnodes:
    for d in days:
        for t in timesteps:
            critical_flag[n, d, t] = 0
            constraint_apc[n, d] = 0
            constraint_InjMin[n, d, t] = 0
            constraint_SubtrMin[n, d, t] = 0
            constraint_InjMax[n, d, t] = 10000
            constraint_SubtrMax[n, d, t] = 10000

while boolean_loop:
    print("")
    print("!!! Iteration counter is currently at " + str(iteration_counter) + "!!!")
    print("")
    """for d in days:
            output_dir = os.path.join(tempfile.gettempdir(), "time_series_example" + str(d))
            shutil.rmtree(output_dir)"""

    # run DC-optimization
    (
    costs_grid, emissions_grid, timesteps, days, powInjRet, powSubtrRet, gridnodes, res_exBat, powInjPrev, powSubtrPrev,
    emissions_nodes, costs_nodes, objective_function) = opti.compute(net, nodes, gridnodes, days, timesteps, eco, devs,
                                                                     clustered, params, options, constraint_apc,
                                                                     constraint_InjMin, constraint_SubtrMin,
                                                                     constraint_InjMax, constraint_SubtrMax,
                                                                     critical_flag)

    outputs = reader.read_results(options)
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
    (output_dir, critical_flag, solution_found, vm_pu_total) = loop.run_timeloop(fkt, timesteps, days, powInjRet,
                                                                                 powSubtrRet, gridnodes, critical_flag,
                                                                                 solution_found)
    # vm_pu_total_array = np.array([[[vm_pu_total[n, d, t] for t in timesteps] for d in days] for n in gridnodes])
    print("zwischenstop")
    for d in days:
        if (solution_found[d] == False):

            print("Additional constrains have to be set for day" + str(d))
            if options["apc_while_voltage_violation"]:
                if options["cut_Inj/Subtr_while_voltage_violation"]:
                    print("You selected both apc and Inj/Subtr cutting in case of voltage violations.")
                    for n in gridnodes:
                        if (any(critical_flag[n, d, t] == 1 for t in timesteps)):
                            if (any(vm_pu_total[n, d, t] < 0.96 for t in timesteps)):
                                pass
                            elif (any(vm_pu_total[n, d, t] > 1.04 for t in timesteps)):
                                constraint_apc[n, d] += 0.1
                                if (constraint_apc[n, d] >= 1):
                                    print("You have reached the maximal amount of curtailment!")
                                    print("Will set curtailment to 100 Percent automatically.")
                                    constraint_apc[n, d] = 1
                            else:
                                pass
                            for t in timesteps:
                                if (critical_flag[n, d, t] == 1):
                                    if (vm_pu_total[n, d, t] < 0.96):
                                        # relative Lösung wirft Problem der Spannungsweiterleitung auf
                                        constraint_SubtrMax[n, d, t] = 0.90 * powSubtrPrev[n, d, t]
                                        """# absolute Regelung:
                                        if (powSubtrPrev[n, d, t] < change_value):
                                            constraint_SubtrMax[n, d, t] = 0
                                            print("Subtraction already set to 0 for node" +str(n) + " and timestep" +str(t))
                                            print("Raising Injection now!")
                                            constraint_InjMin[n,d,t] += change_value
                                        else:
                                            constraint_SubtrMax[n,d,t] = powSubtrPrev[n,d,t] - change_value"""

                                    elif (vm_pu_total[n, d, t] > 1.04):
                                        constraint_InjMax[n, d, t] = 0.90 * powInjPrev[n, d, t]
                                        """
                                        #absolute änderung
                                        if (powInjPrev[n,d,t] < change_value):
                                            constraint_InjMax[n,d,t] = 0
                                            print("Injection already set to 0 for node" +str(n) + " and timestep" +str(t))
                                            print("Raising Subtraction now!")
                                            constraint_SubtrMin[n,d,t] += change_value
                                        else:
                                            constraint_InjMax[n,d,t] = powInjPrev[n,d,t] - change_value"""

                else:
                    print("You selected only apc in case of voltage violations.")
                    for n in gridnodes:
                        if (any(critical_flag[n, d, t] == 1 for t in timesteps)):
                            if (any(vm_pu_total[n, d, t] < 0.96 for t in timesteps)):
                                print(
                                    "Only apc will not fix any voltage issues, because the load is too high on day" + str(
                                        d))
                                infeasability = True
                            elif (any(vm_pu_total[n, d, t] > 1.04 for t in timesteps)):
                                constraint_apc[n, d] += 0.1
                                if (constraint_apc[n, d] >= 1):
                                    print("You have reached the maximal amount of curtailment!")
                                    print("Will set curtailment to 100 Percent automatically.")
                                    constraint_apc[n, d] = 1
                                    print(
                                        "Since you only selected apc, it has reached 100 Percent and you haven't found a solution, the problem appears to be infeasable for these settings!")
                                    infeasability = True

            elif (options["cut_Inj/Subtr_while_voltage_violation"] == True and options[
                "apc_while_voltage_violation"] == False):
                print("You selected only Inj/Subtr cutting in case of voltage violations.")
                for n in gridnodes:
                    for t in timesteps:
                        if (critical_flag[n, d, t] == 1):
                            if (vm_pu_total[n, d, t] < 0.96):
                                constraint_SubtrMax[n, d, t] = 0.90 * powSubtrPrev[n, d, t]
                                """# absolute Regelung:
                                if (powSubtrPrev[n, d, t] < change_value):
                                    constraint_SubtrMax[n, d, t] = 0
                                    print("Subtraction already set to 0 for node" +str(n) + " and timestep" +str(t))
                                    print("Raising Injection now!")
                                    constraint_InjMin[n,d,t] += change_value
                                else:
                                    constraint_SubtrMax[n,d,t] = powSubtrPrev[n,d,t] - change_value"""

                            elif (vm_pu_total[n, d, t] > 1.04):
                                constraint_InjMax[n, d, t] = 0.90 * powInjPrev[n, d, t]
                                """
                                #absolute Regelung
                                if (powInjPrev[n, d, t] < change_value):
                                    constraint_InjMax[n, d, t] = 0
                                    print("Injection already set to 0 for node" + str(n) + " and timestep" + str(t))
                                    print("Raising Subtraction now!")
                                    constraint_SubtrMin[n, d, t] += change_value
                                else:
                                    constraint_InjMax[n, d, t] = powInjPrev[n, d, t] - change_value"""

            elif (options["cut_Inj/Subtr_while_voltage_violation"] == False and options[
                "apc_while_voltage_violation"] == False):
                print("Error: You did not select any measure in case of voltage violations!")
                infeasability = True

        if (solution_found[d] == True):
            print("Solution was successfully found for day" + str(d))

    if infeasability:
        print("Error: Model appears to be infeasable for the selected settings!")
        print("Reasons are stated above.")
        break

    iteration_counter += 1

    if all(solution_found[d] == True for d in days):
        print("Congratulations! Your optimization and loadflow calculation has been successfully finished after " + str(
            iteration_counter - 1) + " iteration steps!")
        break

t2 = int(time.time())
duration_program = t1 - t2

print("this is the end")
# plot_res.plot_results(outputs, days, gridnodes, timesteps)
print("")
print("")
print("objective value für 11 Typtage:")
print(objective_function)
print("duration_program für 11 Typtage")
print(duration_program)
print("")
print("")
print("")

# !!!!!!!!!!!!!!!