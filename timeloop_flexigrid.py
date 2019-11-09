import os
import tempfile

import pandapower as pp
import pandas as pd
import numpy as np

from pandapower.timeseries.data_sources.frame_data import DFData
from pandapower.timeseries.output_writer import OutputWriter
from pandapower.timeseries.run_time_series import run_timeseries
from pandapower.control.controller.const_control import ConstControl

def timeseries_each_day(output_dir, net, timesteps, d, powInjRet, powSubtrRet, gridnodes):

    #create new variable, because pandapower only uses time_steps in stead of timesteps
    time_steps = timesteps

    #retrieve data source
    profilesSubtr, profilesInj, dsInj, dsSubtr = retrieve_data_source(timesteps, d, powInjRet, powSubtrRet, gridnodes)

    #create controllers (to control P values of the load and the gen, which are now combined as positive and negative values in one dataframe)
    #create one controller for every node.
    for n in gridnodes:
        create_controllers(net, dsInj, dsSubtr,n)

    #the output writer with the desired results to be stored to files
    ow = create_output_writer(net, timesteps, output_dir=output_dir)

    #the main time series function
    run_timeseries(net, time_steps, output_writer=ow, continue_on_divergence=True)
    #pp.diagnostic(net)

def retrieve_data_source(timesteps, d, powInjRet, powSubtrRet, gridnodes):

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

    #split up profiles in gen(injection) and load(subtraction) profiles, to properly insert them in 2 const_controllers
    dsInj = DFData(profilesInj)
    dsSubtr = DFData(profilesSubtr)

    return profilesInj, profilesSubtr, dsInj, dsSubtr

def create_controllers(net, dsInj, dsSubtr, n):

    ConstControl(net, element='sgen',variable='p_mw',element_index=net.sgen.index,data_source=dsInj, profile_name=[n])
    ConstControl(net, element='load', variable='p_mw', element_index=net.load.index,data_source=dsSubtr, profile_name=[n])

    """create the output writer. Instead of saving the whole net (which would take a lot of time), we extract only pre defined outputs.
        In this case we:

        save the results ro "../timeseries/tests/outputs"
        write the results to ".xls" excel files. (Possible are: .json , .p, . csv"
        log the variables "p_mw" from "res_load", "vm_pu" from "res_bus" and two res_line values."""

def create_output_writer(net, timesteps, output_dir):
    time_steps = timesteps
    ow = OutputWriter(net, time_steps, output_path=output_dir, output_file_type=".json")
    #these variables are saved to the harddisk after / during the time series loop
    #ow.log_variable('res_load', 'p_mw')
    ow.log_variable('res_bus', 'vm_pu')
    ow.log_variable('res_line', 'loading_percent')
    ow.log_variable('res_line', 'i_ka')
    return ow


    #execution follows:

def run_timeloop(net, timesteps, days, powInjRet, powSubtrRet, gridnodes,critical_flag,solution_found):

    nodes = {}

    nodes["grid"] = net.bus.index.to_numpy()
    nodes["trafo"] = net.trafo['lv_bus'].to_numpy()
    nodes["load"] = net.load['bus'].to_numpy()
    nodes["bat"] = net.load['bus'].to_numpy()

    for d in days:
        output_dir = os.path.join(tempfile.gettempdir(), "time_series_example" + str(d))
        print("Results can be found in your local temp folder: {}".format(output_dir))
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        timeseries_each_day(output_dir, net, timesteps, d, powInjRet, powSubtrRet, gridnodes)

        # read out json files for voltages and return time and place of violation
        vm_pu_file = os.path.join(output_dir, "res_bus", "vm_pu.json")
        vm_pu = pd.read_json(vm_pu_file, convert_axes=True)
        # sort dataframe to get timesteps(rows) in the correct order
        vm_pu = vm_pu.sort_index(axis=0)
        # vm_pu was creating keyerror, so changed into array
        vm_pu_final = vm_pu.values

        for n in gridnodes:
            for t in timesteps:

                #TODO: [t,n] okay oder andersherum?
                if(vm_pu_final[t,n] < 0.96 or vm_pu_final[t,n] > 1.04):
                    if(n in nodes["bat"]):
                        critical_flag[n,d,t] = 1
                        solution_found[d] = False
                        print("voltage violation found for node "+str(n)+" and timestep "+str(t))
                else:
                    pass

        if((vm_pu_final[t,n] >= 0.96 and vm_pu_final[t,n] <= 1.04) for n in gridnodes for t in timesteps):
            solution_found[d] = True


    return output_dir,critical_flag,solution_found