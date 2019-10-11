import os
import tempfile

import pandapower as pp
import pandas as pd
import numpy as np

from pandapower.timeseries.data_sources.frame_data import DFData
from pandapower.timeseries.output_writer import OutputWriter
from pandapower.timeseries.run_time_series import run_timeseries
from pandapower.control.controller.const_control import ConstControl

#kann ich jeden Tag über diese Funktion einzeln laufen lassen, indem ich in der rundatei davon vielleicht das mache für alle Tage?
def timeseries_each_day(output_dir, net, timesteps, d, powChRet, powDisRet, powPVRet, powPlugRet, gridnodes):

    #create new variable, because pandapower only uses time_steps in stead of timesteps
    time_steps = timesteps

    #retrieve data source
    profilesLoad, profilesGen, dsLoad, dsGen = retrieve_data_source(timesteps, d, powChRet, powDisRet, powPVRet, powPlugRet, gridnodes)

    #create controllers (to control P values of the load and the gen, which are now combined as positive and negative values in one dataframe)
    #create one controller for every node.
    for n in gridnodes:
        create_controllers(net, dsLoad, dsGen,n)

    #the output writer with the desired results to be stored to files
    ow = create_output_writer(net, timesteps, output_dir=output_dir)

    #the main time series function
    run_timeseries(net, time_steps, output_writer=ow, continue_on_divergence=True)

def retrieve_data_source(timesteps, d, powChRet, powDisRet, powPVRet, powPlugRet, gridnodes):

    #calculate new dicts for load and gens with DC-OPF values
    genDay = {}
    loadDay = {}
    for t in timesteps:
        for n in gridnodes:
            genDay[n,t] = powPVRet[n,d,t] + powDisRet[(9,11,23)][n,d,t]
            loadDay[n,t] = powPlugRet[n,d,t] + powChRet[(9,11,23)][n,d,t]

    #transforming dict into an array to write into a dataframe
    genDayArray = np.array([[genDay[n,t] for t in timesteps] for n in gridnodes])
    loadDayArray = np.array([[loadDay[n,t] for t in timesteps] for n in gridnodes])

    profilesPreLoad = pd.DataFrame(loadDayArray)
    profilesPreGen = pd.DataFrame(genDayArray)

    #transpose DataFrame to fit the standard layout of given DataFrame. Afterwards columns are nodes, rows are timesteps
    profilesLoad = profilesPreLoad.transpose()
    profilesGen = profilesPreGen.transpose()

    #split up profiles in gen and load profiles, to properly insert them in 2 const_controllers
    dsLoad = DFData(profilesLoad)
    dsGen = DFData(profilesGen)

    return profilesLoad, profilesGen, dsLoad, dsGen

def create_controllers(net, dsLoad, dsGen, n):

    ConstControl(net, element='load',variable='p_mw',element_index=net.load.index,data_source=dsLoad, profile_name=[n])
    ConstControl(net, element='sgen', variable='p_mw', element_index=net.sgen.index,data_source=dsGen, profile_name=[n])

    """create the output writer. Instead of saving the whole net (which would take a lot of time), we extract only pre defined outputs.
        In this case we:

        save the results ro "../timeseries/tests/outputs"
        write the results to ".xls" excel files. (Possible are: .json , .p, . csv"
        log the variables "p_mw" from "res_load", "vm_pu" from "res_bus" and two res_line values."""

def create_output_writer(net, timesteps, output_dir):
    time_steps = timesteps
    ow = OutputWriter(net, time_steps, output_path=output_dir, output_file_type=".xls")
    #these variables are saved to the harddisk after / during the time series loop
    ow.log_variable('res_load', 'p_mw')
    ow.log_variable('res_bus', 'vm_pu')
    ow.log_variable('res_line', 'loading_percent')
    ow.log_variable('res_line', 'i_ka')
    return ow


    #execution follows:

def run_timeloop(net, timesteps, days, powChRet, powDisRet, powPVRet, powPlugRet, gridnodes):

    for d in days:
        output_dir = os.path.join(tempfile.gettempdir(), "time_series_example" + str(d))
        print("Results can be found in your local temp folder: {}".format(output_dir))
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        timeseries_each_day(output_dir, net, timesteps, d, powChRet, powDisRet, powPVRet, powPlugRet, gridnodes)