import os
import tempfile

import pandapower as pp
import pandas as pd
import numpy as np
import pandapower.networks as nw

from pandapower.plotting import pf_res_plotly
from pandapower.timeseries.data_sources.frame_data import DFData
from pandapower.timeseries.output_writer import OutputWriter
from pandapower.timeseries.run_time_series import run_timeseries
from pandapower.control.controller.const_control import ConstControl

def timeseries_each_day(output_dir, net, timesteps, d, powInjRet, powSubtrRet, gridnodes, nodesload):

    #create new variable, because pandapower only uses time_steps in stead of timesteps
    time_steps = range(0,24)

    # nicht sicher, ob das hier rein mus oder nicht
    pp.set_user_pf_options(net, init_vm_pu= "flat", init_va_degree = "dc", calculate_voltage_angles = True)


    #retrieve data source
    profilesInj, profilesSubtr, dsInj, dsSubtr = retrieve_data_source(timesteps, d, powInjRet, powSubtrRet, gridnodes, nodesload)

    """#create controllers (to control P values of the load and the gen, which are now combined as positive and negative values in one dataframe)
    #create one controller for every node.
    create_controllers(net, dsInj, dsSubtr, dsTotal, gridnodes)"""

    create_controllers(net,dsInj,dsSubtr, profilesInj, profilesSubtr)

    #the output writer with the desired results to be stored to files
    ow = create_output_writer(net, time_steps, output_dir=output_dir)

    #the main time series function
    run_timeseries(net, time_steps, continue_on_divergence=True, output_writer=ow)

    return ow

def retrieve_data_source(timesteps, d, powInjRet, powSubtrRet, gridnodes, nodesload):

    #initialize new arrays the size of wanted array
    powInjDay = np.zeros((len(gridnodes),len(timesteps)))
    powSubtrDay = np.zeros((len(gridnodes),len(timesteps)))

    #new array without axis d (inside for loop for d)
    for n in gridnodes:
        for t in timesteps:
            powInjDay[n,t] = powInjRet[d,n,t]
            powSubtrDay[n,t] = powSubtrRet[d,n,t]

    profilesPreInj = pd.DataFrame(powInjDay)
    profilesPreSubtr = pd.DataFrame(powSubtrDay)

    #transpose DataFrame to fit the standard layout of given DataFrame. Afterwards columns are nodes, rows are timesteps
    profilesInj = profilesPreInj.transpose()
    profilesSubtr = profilesPreSubtr.transpose()

    # remove columns if no load or gen is connected, to be synced to net.load and net.sgen (so controller works properly)
    for n in profilesSubtr.columns:
        if n in nodesload:
            pass
        else:
            profilesSubtr = profilesSubtr.drop(n, axis=1)

    for n in profilesInj.columns:
        if n in nodesload:
            pass
        else:
            profilesInj = profilesInj.drop(n, axis=1)

    #split up profiles in gen(injection) and load(subtraction) profiles, to properly insert them in 2 const_controllers
    dsInj = DFData(profilesInj)
    dsSubtr = DFData(profilesSubtr)

    return profilesInj, profilesSubtr, dsInj, dsSubtr

def create_controllers(net, dsInj, dsSubtr, profilesInj, profilesSubtr):

    #Listenlösung, geht sehr schnell, aber die voltages variieren nicht mehr...
    ConstControl(net, element= 'load', variable= 'p_mw', element_index=net.load.index, data_source=dsSubtr, profile_name=dsSubtr.df.columns)
    ConstControl(net, element= 'sgen', variable= 'p_mw', element_index=net.sgen.index, data_source=dsInj, profile_name=dsInj.df.columns)

def create_output_writer(net, time_steps, output_dir):

    ow = OutputWriter(net, time_steps, output_path=output_dir, output_file_type=".json")
    #these variables are saved to the harddisk after / during the time series loop
    ow.log_variable('res_load', 'p_mw')
    ow.log_variable('res_bus', 'vm_pu')
    ow.log_variable('res_line', 'loading_percent')
    ow.log_variable('res_line', 'i_ka')
    ow.log_variable('res_sgen', 'p_mw')

    return ow

    #execution follows:

def run_timeloop(fkt, timesteps, days, powInjRet, powSubtrRet, gridnodes,critical_flag,solution_found):

    net = fkt()
    nodes = {}

    nodes["grid"] = net.bus.index.to_numpy()
    nodes["trafo"] = net.trafo['lv_bus'].to_numpy()
    nodes["load"] = net.load['bus'].to_numpy()
    nodes["bat"] = net.load['bus'].to_numpy()

    # define sgens for net in order to be able to include gen values in timeloop
    nodesload = list(nodes["load"])
    for n in nodesload:
        pp.create_sgen(net, n, p_mw=0)

    vm_pu_total = {}
    for n in gridnodes:
        for d in days:
            for t in timesteps:
                vm_pu_total[n,d,t] = 0

    for d in days:
        output_dir = os.path.join(tempfile.gettempdir(), "time_series_example" + str(d))
        print("Results can be found in your local temp folder: {}".format(output_dir))
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        ow = timeseries_each_day(output_dir, net, timesteps, d, powInjRet, powSubtrRet, gridnodes, nodesload)

        # read out json files for voltages and return time and place of violation
        vm_pu_file = os.path.join(output_dir, "res_bus", "vm_pu.json")
        vm_pu = pd.read_json(vm_pu_file, convert_axes=True)
        # sort dataframe to get timesteps(rows) in the correct order
        vm_pu = vm_pu.sort_index(axis=0)
        # vm_pu was creating keyerror, so changed into array
        vm_pu_final = vm_pu.values

        for n in gridnodes:
            for t in timesteps:

                if(vm_pu_final[t,n] < 0.96 or vm_pu_final[t,n] > 1.04):
                    if(n in nodes["bat"]):
                        critical_flag[n,d,t] = 1
                        solution_found[d] = False
                        print("voltage violation found for node "+str(n)+" and timestep "+str(t))
                else:
                    critical_flag[n,d,t] = 0

        if(all((vm_pu_final[t,n] >= 0.96 and vm_pu_final[t,n] <= 1.04) for n in gridnodes for t in timesteps)) == True:
            solution_found[d] = True
            print("solution was found for day" +str(d))

        for n in gridnodes:
            for t in timesteps:
                vm_pu_total[n,d,t] = vm_pu_final[t,n]

        pf_res_plotly(net)
        print("haha")

    vm_pu_total = np.array([[[vm_pu_total[n,d,t] for t in timesteps] for d in days] for n in gridnodes])
    """ow.remove_output_variable('res_load', 'p_mw')
    ow.remove_output_variable('res_bus', 'vm_pu')
    ow.remove_output_variable('res_line', 'loading_percent')
    ow.remove_output_variable('res_line', 'i_ka')
    ow.remove_output_variable('res_sgen', 'p_mw')"""

    return output_dir,critical_flag,solution_found, vm_pu_total