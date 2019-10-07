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
def timeseries_each_day(output_dir, net, timesteps, d, powerCh, powerDis, powerPV, powerPlug, gridnodes):

    #retrieve data source
    profiles, ds = retrieve_data_source(timesteps, d, powerCh, powerDis, powerPV, powerPlug, gridnodes)

    #create controllers (to control P values of the load and the gen)
    create_controllers(net, ds)

    #the output writer with the desired results to be stored to files
    ow = create_output_writer(net, time_steps, output_dir=output_dir)

    #the main time series function
    run_timeseries(net, time_steps, output_writer=ow)

def retrieve_data_source(timesteps, d, powerPV, powerPlug, powerDis, powerCh, gridnodes):

    profiles = pd.DataFrame()

    # create new two-dimensional lists for the cumulated gen and load
    # positive value means gen, negative value equals load
    genDayOne = {}
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
    genDayTwelve = {}
    for t in timesteps:
        for n in gridnodes:
            genDayOne[n, t] = powerPV[n, 0, t] - powerPlug[n, 0, t] + powerDis[n, 0, t] - powerCh[n, 0, t]

    print(genDayOne)
    for t in timesteps:
        for n in gridnodes:
            genDayTwo[n, t] = powerPV[n, 1, t] - powerPlug[n, 1, t] + powerDis[n, 1, t].X - powerCh[n, 1, t].X
    for t in timesteps:
        for n in gridnodes:
            genDayThree[n, t] = powerPV[n, 2, t] - powerPlug[n, 2, t] + powerDis[n, 2, t].X - powerCh[n, 2, t].X
    for t in timesteps:
        for n in gridnodes:
            genDayFour[n, t] = powerPV[n, 3, t] - powerPlug[n, 3, t] + powerDis[n, 3, t].X - powerCh[n, 3, t].X
    for t in timesteps:
        for n in gridnodes:
            genDayFive[n, t] = powerPV[n, 4, t] - powerPlug[n, 4, t] + powerDis[n, 4, t].X - powerCh[n, 4, t].X
    for t in timesteps:
        for n in gridnodes:
            genDaySix[n, t] = powerPV[n, 5, t] - powerPlug[n, 5, t] + powerDis[n, 5, t].X - powerCh[n, 5, t].X
    for t in timesteps:
        for n in gridnodes:
            genDaySeven[n, t] = powerPV[n, 6, t] - powerPlug[n, 6, t] + powerDis[n, 6, t].X - powerCh[n, 6, t].X
    for t in timesteps:
        for n in gridnodes:
            genDayEight[n, t] = powerPV[n, 7, t] - powerPlug[n, 7, t] + powerDis[n, 7, t].X - powerCh[n, 7, t].X
    for t in timesteps:
        for n in gridnodes:
            genDayNine[n, t] = powerPV[n, 8, t] - powerPlug[n, 8, t] + powerDis[n, 8, t].X - powerCh[n, 8, t].X
    for t in timesteps:
        for n in gridnodes:
            genDayTen[n, t] = powerPV[n, 9, t] - powerPlug[n, 9, t] + powerDis[n, 9, t].X - powerCh[n, 9, t].X
    for t in timesteps:
        for n in gridnodes:
            genDayEleven[n, t] = powerPV[n, 10, t] - powerPlug[n, 10, t] + powerDis[n, 10, t].X - powerCh[n, 10, t].X
    for t in timesteps:
        for n in gridnodes:
            genDayTwelve[n, t] = powerPV[n, 11, t] - powerPlug[n, 11, t] + powerDis[n, 11, t].X - powerCh[n, 11, t].X

    print(genDayTwelve)
    print("hihi")
    #profiles['load1_p'] =
    #profiles['sgen1_p'] =

    ds = DFData(profiles)

    return profiles, ds

def create_controllers(net, ds):
    ConstControl(net, element='load', variable='p_mw', element_index=[0],
                 data_source=ds, profile_name=["load1_p"])
    ConstControl(net, element='sgen', variable='p_mw', element_index=[0],
                 data_source=ds, profile_name=["sgen1_p"])

"""create the output writer. Instead of saving the whole net (which would take a lot of time), we extract only pre defined outputs.
    In this case we: 
        
        save the results ro "../timeseries/tests/outputs"
        write the results to ".xls" excel files. (Possible are: .json , .p, . csv" 
        log the variables "p_mw" from "res_load", "vm_pu" from "res_bus" and two res_line values. 
"""

def create_output_writer(net, time_steps, output_dir):
    ow = OutputWriter(net, time_steps, output_path=output_dir, output_file_type=".xls")
    #these variables are saved to the harddisk after / during the time series loop
    ow.log_variable('res_load', 'p_mw')
    ow.log_variable('res_bus', 'vm_pu')
    ow.log_variable('res_line', 'loading_percent')
    ow.log_variable('res_line', 'i_ka')
    return ow

#execution follows:

def run_timeloop(net, timesteps, days, powerCh, powerDis, powerPV, powerPlug, gridnodes):
    for d in days:
        output_dir = os.path.join(tempfile.gettempdir(), "time_series_example" + str(d))
        print("Results can be found in your local temp folder: {}".format(output_dir))
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        timeseries_each_day(output_dir, net, timesteps, d, powerCh, powerDis, powerPV, powerPlug, gridnodes)