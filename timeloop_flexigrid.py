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

    #retrieve data source
    profilesLoad, profilesGen, dsLoad, dsGen = retrieve_data_source(timesteps, d, powChRet, powDisRet, powPVRet, powPlugRet, gridnodes)

    #create controllers (to control P values of the load and the gen, which are now combined as positive and negative values in one dataframe)
    create_controllers(net, dsLoad, dsGen)

    #the output writer with the desired results to be stored to files
    #ow = create_output_writer(net, time_steps, output_dir=output_dir)

    #the main time series function
    #run_timeseries(net, time_steps, output_writer=ow)

def retrieve_data_source(timesteps, d, powChRet, powDisRet, powPVRet, powPlugRet, gridnodes):

    #create new two-dimensional lists for the cumulated gen and load
    #positive value means gen, negative value equals load
    powDay = {}
    #try with just using one genDay, because whole timloop depends on d value
    """genDayOne = {}
    genDayTwo = {}
    genDayThree = {}
    genDayFour = {}
    genDayFive = {}
    genDaySix = {}
    genDaySeven = {}
    genDayEight = {}
    genDayNine = {}
    genDayTen = {}
    genDayEleven = {}
    genDayTwelve = {}"""

    for t in timesteps:
        for n in gridnodes:
            powDay[n,t] = powPVRet[n,d,t] - powPlugRet[n,d,t] + powDisRet[(9,11,23)][n,d,t] - powChRet[(9,11,23)][n,d,t]

    #split power in load and gen for Const_Controller
    genDay = {}
    loadDay = {}
    
    for t in timesteps:
        for n in gridnodes:
            if powDay[n,t] < 0:
                loadDay[n,t] = powDay[n,t] * -1
                genDay[n,t] = 0
            elif powDay[n,t] >= 0:
                loadDay[n,t] = 0
                genDay[n,t] = powDay[n,t]

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

def create_controllers(net, dsLoad, dsGen):

    ConstControl(net, element='load',variable='p_mw',element_index=[np.where(0)],data_source=dsLoad, profile_name=["load1_p"])
    ConstControl(net, element='sgen', variable='p_mw', element_index=[np.where(0)],data_source=dsGen, profile_name=["sgen1_p"])
    print("stop")

    #create the output writer. Instead of saving the whole net (which would take a lot of time), we extract only pre defined outputs.
    #In this case we:

     #   save the results ro "../timeseries/tests/outputs"
      #  write the results to ".xls" excel files. (Possible are: .json , .p, . csv"
       # log the variables "p_mw" from "res_load", "vm_pu" from "res_bus" and two res_line values.

def create_output_writer(net, time_steps, output_dir):
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