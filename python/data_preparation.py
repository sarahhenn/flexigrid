"""
created on Thursday 06/02/2020
@ author: fpo

file for preparation of data, in order to shorten the run file

"""
# python libraries
import numpy as np
import pandas as pd
import pandapower as pp

# import own functions
from settings import options, building_age, building_type, emission_year, number_clusters, operationFolder, sourceFolder, net
import python.clustering_medoid as clustering
import python.parse_inputs as pik

def data_import(sourceFolder,building_type,building_age,emission_year):
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

    # emissions factor from excel file
    emi_input = pd.read_csv(sourceFolder + "\\emission_factor_" + emission_year + ".csv", header=0, usecols=[2])
    raw_inputs["co2_dyn"] = np.zeros([8760])
    for t in range(0, 8760):
        # hier sagen, ob stündlich oder viertelstündlich i=t*4 ist stündlich
        i = t * 4
        raw_inputs["co2_dyn"][t] = np.mean(emi_input[i:(i + 4)])

    # power generation costs from excel file
    cost_input = pd.read_csv(sourceFolder + "\\data\\generation_cost_EL" + emission_year + ".csv", header=0,
                             usecols=[2])
    raw_inputs["elcost_dyn"] = np.zeros([8760])
    for t in range(0, 8760):
        # hier sagen, ob stündlich oder viertelstündlich i=t*4 ist stündlich
        i = t * 4
        raw_inputs["elcost_dyn"][t] = np.mean(cost_input[i:(i + 4)])

    return raw_inputs

def clustering_import(raw_inputs):

    inputs_clustering = np.array([raw_inputs["heat"],
                                  raw_inputs["dhw"],
                                  raw_inputs["electricity"],
                                  raw_inputs["solar_roof"],
                                  raw_inputs["temperature"],
                                  raw_inputs["co2_dyn"],
                                  raw_inputs["elcost_dyn"]])

    (inputs, nc, z) = clustering.cluster(inputs_clustering,
                                         number_clusters=number_clusters,
                                         norm=2,
                                         mip_gap=0.0,
                                         weights=[1, 2, 2, 2, 1, 2, 2])

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
    clustered["elcost_dyn"] = inputs[6]
    clustered["elcost_stat"] = np.zeros_like(clustered["elcost_dyn"])
    clustered["elcost_stat"][:, :] = np.mean(raw_inputs["elcost_dyn"])
    clustered["weights"] = nc
    clustered["z"] = z

    return(clustered, len_day)

# load devices, economics and parameters
def others_import(clustered, len_day):

    devs = pik.read_devices(timesteps=len_day,
                            days=number_clusters,
                            temperature_ambient=clustered["temperature"],
                            solar_irradiation=clustered["solar_irrad"],
                            days_per_cluster=clustered["weights"])

    (eco, params, devs) = pik.read_economics(devs)
    params = pik.compute_parameters(params, number_clusters, len_day)

    return(eco, params, devs)

def grid_modulation(params):
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

    return(gridnodes, days, timesteps, nodes)
