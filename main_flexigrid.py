# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 15:34:57 2019

@author: she
"""

# import extern functions
import numpy as np
import pickle
import pandas as pd
import time
import pandapower.plotting as plot
from pandapower.plotting.simple_plot_bat import simple_plot_bat
from pandapower.plotting import pf_res_plotly
#just give net to this function and it will plot the voltages as well!

# import own function
import python.read_basic as reader
import python.flexigrid_plotting_results as plot_res
from settings import options, building_age, building_type, emission_year, number_clusters, operationFolder, sourceFolder, net, fkt
import python.data_preparation as prep
import python.evaluation_methods as evaluations

# get time from start to calculate duration of the program
t1 = int(time.time())

# import data from excel files
raw_inputs = prep.data_import(sourceFolder,building_type,building_age,emission_year)

# use EBC clustering algorithm (with settings)
clustered, len_day = prep.clustering_import(raw_inputs)

# import other needed dicts
(eco, params, devs) = prep.others_import(clustered, len_day)

if options["show_simple_plots"]:
# simple plot of net with existing geocoordinates or generated artificial geocoordinates
    plot.simple_plot(net, show_plot=True)

if options["show_detailed_plots"]:
    pf_res_plotly(net)

#%% Store clustered input parameters

with open(options["filename_inputs"], "wb") as f_in:
    pickle.dump(clustered, f_in, pickle.HIGHEST_PROTOCOL)

# modulate pandapower grid
(gridnodes, days, timesteps, nodes) = prep.grid_modulation(params)

# create a dict with everything the evaluations need for processing
all_dict_evaluation = {net, options, days, gridnodes, timesteps, nodes, eco, devs, params,clustered,fkt,building_type,building_age}

if options["run_which_opti"] == "normal":
    objective_function = evaluations.normal_evaluation(**all_dict_evaluation)
elif options["run_which_opti"] == "pareto_oNB":
    objective_function = evaluations.pareto_evaluation_oNB(**all_dict_evaluation)
elif options["run_which_opti"] == "pareto_mNB":
    objective_function = evaluations.pareto_evaluation_mNB(**all_dict_evaluation)
else:
    print("ERROR:"
          "You did not select a valid evaluation method!"
          "Adjust options in settings script(run_which_opti) and try again!")

outputs = reader.read_results(options)
# %% plot grid with batteries highlighted
if options["show_simple_plots"]:

    bat_ex = np.zeros(len(outputs["nodes"]["grid"]))
    for n in outputs["nodes"]["grid"]:
        if outputs["res_capacity"][n] > 0:
            bat_ex[n] = 1

    netx = net
    netx['bat'] = pd.DataFrame(bat_ex, columns=['ex'])
    simple_plot_bat(netx, show_plot=True, bus_color='b', bat_color='r')

t2 = int(time.time())
duration_program = t1 - t2

print("this is the end")
#plot_res.plot_results(outputs, days, gridnodes, timesteps)
print("")
print("")
print("objective value für 30 Typtage:")
print(objective_function)
print("duration_program für 30 Typtage")
print(duration_program)
print("")