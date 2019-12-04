# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 15:34:57 2019

@author: she
"""

# import extern functions
from __future__ import division
import gurobipy as gp
import numpy as np
import pickle
import pandas as pd
import random
import xlsxwriter
import math

# import own function
import python.hpopt_energy as hpopt

#%% Start:

def compute(net, eco, devs, clustered, params, options, names, loads_with, randomfile, ev_file, distributionFolder):
    """
    Compute the optimal building energy system consisting of pre-defined 
    devices (devs) for a given building. Furthermore the program can choose
    between different restructuring measures for selected building parts.
    The optimization target can either be the econmic optimum or the ecological
    optimum. 
    The program takes hereby several german subsidy programs for buildings into
    account. 
    
    Parameters
    ----------
    eco : dict
        - b : price-dynamic cash value factor
        - crf : capital recovery factor
        - el : electricity prices
        - energy_tax : additional tax on gas price
        - gas : gas prices
        - inst_costs : installation costs for different devices
        - pel : pellet prices
        - prChange : price changes
        - price_sell_el : price for sold electricity (chp / pv)
        - price_sell_eeg : feed in tariff according to EEG
        - rate : interest rate        
        - q : interest rate + 1
        - t_calc : calculation time
        - tax : value added tax
        
    devs : dict
        - bat : Batteries
        - boiler : Boilers
        - chp : CHP units
        - eh : Electrical heaters
        - hp_air : air heat pump
        - hp_geo: geothermal heat pump
        - pel: Pellet boiler
        - pv : Photovoltaic cells
        - stc : Solar thermal collectors
        - tes : Thermal energy storage systems
        
    clustered : dict
        - heat: Heat load profile
        - dhw : Domestic hot water load profile
        - electricity : Electricity load profile
        - solar_roof : solar irradiation on rooftop
        - temperature : Ambient temperature
        - co2_dyn : time variant co2 factor
        - co2_stat: static co2 factor
        - weights : Weight factors from the clustering algorithm
        
    params : dict
        - c_w : heat capacity of water
        - days : quantity of clustered days
        - dt : time step length (h)
        - rho_w : density of water
        - time_steps : time steps per day
        
    options : dict
        -      
        
    """
    
    # extract parameters
    dt          = params["dt"]
    timesteps   = [i for i in range(params["time_steps"])]
    days        = [i for i in range(params["days"])]   
    
#%% extract node information from pandas-network
    
    # specify grid nodes for whole grid and trafo; choose and allocate load, injection and battery nodes
    nodes = {}
    
    nodes["grid"] = net.bus.index.to_numpy()
    nodes["trafo"] = net.trafo['lv_bus'].to_numpy()
    nodes["load"] = net.load['bus'].to_numpy()
    #nodeInj = net.load['bus'].to_numpy()
    nodes["supply"] = nodes["grid"]
    nodes["each"] = nodes["grid"]
    nodes["each"] = np.delete(nodes["each"],[0,1])
    nodes["supply"] = np.delete(nodes["supply"],nodes["load"])   #Reihenfolge wichtig !
    nodes["supply"] = np.delete(nodes["supply"],[0,1])           #Reihenfolge wichtig !
    
    nodes["bat"] = net.load['bus'].to_numpy()   #Batterien an Loads zugelassen    
  #  nodes["bat"] = nodes["supply"]   #Batterien an Verzweigungen zugelassen
  #  nodes["bat"] = nodes["each"]     #Batterien an allen Knoten nach dem Trafo zugelassen
    
    gridnodes = list(nodes["grid"]) 
    
#%% set and calculate building energy system data, as well as load and injection profiles

    # calculate PV injection
    eta_inverter = 0.97
    # compute generated power
    powerGen = options["P_pv"] * (devs["pv"]["area_mean"]/devs["pv"]["p_nom"]) * devs["pv"]["eta_el"] * eta_inverter * clustered["solar_irrad"] 
    
    # extract modulation level of heatpump
    mod_lvl = devs["hp_air"]["mod_lvl"]
    # extract COP-table for given heat flow temperature
    cop = devs["hp_air"]["cop_w"+str(options["T_VL"])]
    
    # calculate thermal nominal hp capacity according to (Stinner, 2017)
    for n in nodes["load"]:
        if n in loads_with["efh"]:
            if options ["dhw_electric"]: ###changer
                capa_hp_th = options["alpha_th"] * np.max(clustered["heat"] + clustered["dhw"]) 
            else: 
                capa_hp_th = options["alpha_th"] * np.max(clustered["heat"]) 
        else:
            if options ["dhw_electric"]:
                capa_hp_th = options["alpha_th"] * np.max(clustered["heat2"] + clustered["dhw2"]) 
            else: 
                capa_hp_th = options["alpha_th"] * np.max(clustered["heat2"]) 
    
    # electrical nominal hp capacity
    if options ["T_VL"] == 35:
        capa_hp = capa_hp_th/devs["hp_air"]["cop_a-7w35"]
    elif options ["T_VL"] == 55:
        capa_hp = capa_hp_th/devs["hp_air"]["cop_a-7w55"]
        
    # calculate tes capacity according to (Stinner, 2017)
    for n in nodes["load"]:
        if n in loads_with["efh"]: ###changer
            if options ["dhw_electric"]: 
                capa_tes = options["beta_th"] * sum(clustered["weights"][d] * sum((clustered["heat"][d,t] + clustered["dhw"][d,t]) for t in timesteps) for d in days) * dt / sum(clustered["weights"])
            else: 
                capa_tes = options["beta_th"] * sum(clustered["weights"][d] * sum(clustered["heat"][d,t]  for t in timesteps) for d in days) * dt / sum(clustered["weights"])
        else:
            if options ["dhw_electric"]:
                capa_tes = options["beta_th"] * sum(clustered["weights"][d] * sum((clustered["heat2"][d,t] + clustered["dhw2"][d,t]) for t in timesteps) for d in days) * dt / sum(clustered["weights"])
            else: 
                capa_tes = options["beta_th"] * sum(clustered["weights"][d] * sum(clustered["heat2"][d,t]  for t in timesteps) for d in days) * dt / sum(clustered["weights"])
            
    if options["hp_mode"] == "energy_opt":
        
        (res_actHP, res_powerHP, res_powerEH, res_SOC_tes, res_SOC_init_tes, res_ch_tes, res_dch_tes, res_heatHP, res_heatEH) = hpopt.optimize(options, params, clustered, devs, capa_hp, capa_tes)
        
        res_powerHPGrid = {} 
        res_powerHPPV = {}
        res_powerHPBat = {}
        res_powerEHGrid = {}
        res_powerEHPV = {}
        res_powerEHBat = {}
        
        for n in nodes["load"]:
            if n in loads_with["efh"]:  ###changer
                if options ["dhw_electric"]:
                    powerElec = clustered["electricity"] + clustered["dhw"] + res_powerHP + res_powerEH
                else: 
                    powerElec = clustered["electricity"] + res_powerHP + res_powerEH
            else:
                if options ["dhw_electric"]: 
                    powerElec = clustered["electricity2"] + clustered["dhw2"] + res_powerHP + res_powerEH
                else: 
                    powerElec = clustered["electricity2"] + res_powerHP + res_powerEH

    else:
        
        res_actHP = {} 
        res_powerHP = {}
        res_powerEH = {}
        res_SOC_tes = {} 
        res_SOC_init_tes = {}
        res_ch_tes = {}
        res_dch_tes = {}
        res_heatHP = {}
        res_heatEH = {}
        res_powerHPGrid = {} 
        res_powerHPPV = {}
        res_powerHPBat = {}
        res_powerEHGrid = {}
        res_powerEHPV = {}
        res_powerEHBat = {}
        
        for n in nodes["load"]:
            if n in loads_with["efh"]: ###changer
                if options ["dhw_electric"]: 
                    powerElec = clustered["electricity"] + clustered["dhw"] 
                else:
                    powerElec = clustered["electricity"] 
            else:
                if options ["dhw_electric"]:
                    powerElec = clustered["electricity2"] + clustered["dhw2"] 
                else:
                    powerElec = clustered["electricity2"] 
  
#%% extract line information from pandas-network

    
    # set nominal Voltage for grid and Voltage bounds for nodes
    U_nominal = net.trafo.vn_lv_kv[0]*1000 
    voltNode_max         = U_nominal*1.04
    voltNode_min         = U_nominal*0.96
    
    
    # extract existing lines 
    nodeLines = []
    for i in range(len(net.line['from_bus'])):
        nodeLines.append((net.line['from_bus'][i],net.line['to_bus'][i]))
    nodeLines = gp.tuplelist(nodeLines)
    
    # extract maximal current for lines
    # multiply with 400 V to get maximal power in kW      
    powerLine_max = {}
    for [n,m] in nodeLines:
        powerLine_max[n,m] = (net.line['max_i_ka'][nodeLines.index((n,m))])*U_nominal
    
    # maximal and minimal voltage in kV as difference from rated voltage
    lineLength = {}
    specRes_r = {}
    specRes_x = {}
    specRes_ap = {}
    res_r = {}
    res_x = {}
    
    for [n,m] in nodeLines:
        lineLength[n,m]     = net.line['length_km'][nodeLines.index((n,m))]
        specRes_r[n,m]      = net.line['r_ohm_per_km'][nodeLines.index((n,m))]
        specRes_x[n,m]      = net.line['x_ohm_per_km'][nodeLines.index((n,m))]
        specRes_ap[n,m]     = (specRes_r[n,m] * specRes_x[n,m]) 
        res_r[n,m]          = specRes_r[n,m]*lineLength[n,m]
        res_x[n,m]          = specRes_x[n,m]*lineLength[n,m]
   
#%% insert EVs
    # set maximal power for ev charge and discharge
    ev_max = 3.6 
    
    if options["EV_mode"] == "off":
        pass
    else:
        # build realistic random daily demands in kWh
        demEV = {}
        for n in gridnodes:
            for d in days:
                if n in loads_with["ev"]:
                    demEV[n,d] = random.uniform(2.5,4.7)
                else: 
                    demEV[n,d] = 0
        
        # build random availability matrix
        availEVday = {}
        availEV = {}
        for n in gridnodes:
            for d in days:
                if n in loads_with["ev"]:
                    availEVday[n,d] = random.randrange(15,20)
                else: 
                    availEVday[n,d] = 0
                for t in timesteps:
                    if t >= availEVday[n,d]:
                        availEV[n,d,t] = 1
                    else:
                       availEV[n,d,t] = 0 

    
#%% define battery settings, final gens and loads    
        
    # extract battery nodes and define technical data for them
    capBat_max = {}
    capBat_min = {}
    chBat_max = {}
    chBat_min = {}
    
    for n in gridnodes:
        if n in nodes["bat"]:
            capBat_max[n] = devs["bat"]["cap_max"]
            capBat_min[n] = devs["bat"]["cap_min"]
            chBat_max[n] = devs["bat"]["cap_max"]
            chBat_min[n] = devs["bat"]["cap_min"]
    
    # attach plug-in loads and PV generatrion to building nodes
    # TODO: do the same with EV loads!?
    
    ##former: selected systems for every load
#    powerPlug = {}
#    powerPV = {}
#    heatload = {}
#    dhwload = {}
#    for n in gridnodes:
#        for d in days:
#            for t in timesteps:
#                if n in nodes["load"]:                        
#                    heatload[n,d,t] = clustered["heat"][d,t]
#                    dhwload[n,d,t] = clustered["dhw"][d,t]
#                    powerPlug[n,d,t] = powerElec[d,t]
#                    powerPV[n,d,t] = powerGen[d,t]
#                else:
#                    heatload[n,d,t] = np.zeros_like(clustered["heat"][d,t])
#                    dhwload[n,d,t] = np.zeros_like(clustered["dhw"][d,t])
#                    powerPlug[n,d,t] = np.zeros_like(powerElec[d,t])
#                    powerPV[n,d,t] = np.zeros_like(powerGen[d,t])

     ## system distribution with efh/mfh
    powerPlug = {}
    powerPV = {}
    heatload = {}
    dhwload = {}
    for n in gridnodes:
        for d in days:
            for t in timesteps:
                if n in nodes["load"]:                                      
                    powerPlug[n,d,t] = powerElec[d,t]
                else:                                                       
                    powerPlug[n,d,t] = np.zeros_like(powerElec[d,t])
                if n in loads_with["pv"]:                                   
                    powerPV[n,d,t] = powerGen[d,t]
                else:                                                       
                    powerPV[n,d,t] = np.zeros_like(powerGen[d,t])                
                if n in loads_with["hp"]:                                                       
                    if n in loads_with["efh"]:                              
                        heatload[n,d,t] = clustered["heat"][d,t]
                        dhwload[n,d,t] = clustered["dhw"][d,t]
                    else:                                                   
                        heatload[n,d,t] = clustered["heat2"][d,t]
                        dhwload[n,d,t] = clustered["dhw2"][d,t]
                else:                                                       
                    heatload[n,d,t] = np.zeros_like(clustered["heat"][d,t])
                    dhwload[n,d,t] = np.zeros_like(clustered["dhw"][d,t])


#%% optimization model

    print("start modelling")
    model = gp.Model("Optimal Battery Placement and Sizing")
    
    #%% economic variables
    
    # initiate cost variables: there are cost-variables for investment, operation & maintenance, 
    # demand costs (electricity, fuel costs) and fix costs resulting from base prices
    # moreover there are revenues from feed in
    
    # for every node
    c_inv  = model.addVars(gridnodes, vtype="C", name="c_inv")      
    c_om   = model.addVars(gridnodes, vtype="C", name="c_om")            
    c_dem  = model.addVars(gridnodes, vtype="C", name="c_dem")         
    c_fix  = model.addVars(gridnodes, vtype="C", name="c_fix")                 
    revenues = model.addVars(gridnodes, vtype="C", name="revenue")
    
    # for the whole grid
    c_dem_grid  = model.addVar(vtype="C", name="c_dem_grid")                          
    revenues_grid = model.addVar(vtype="C", name="revenue_grid")
    
    # variables for total node costs, total costs and total emissions
    c_total_nodes = model.addVar(vtype="C", name="c_total", lb= -gp.GRB.INFINITY)
    c_total_grid = model.addVar(vtype="C", name="c_total", lb= -gp.GRB.INFINITY)
    emission_nodes = model.addVars(gridnodes, vtype="C", name= "CO2_emission", lb= -gp.GRB.INFINITY) 
    emission_grid = model.addVar(vtype="C", name= "CO2_emission", lb= -gp.GRB.INFINITY)  
    
    #%% technical variables
    
    # add grid variables to model
    
    # set trafo bounds due to technichal limits
    trafo_max = float(net.trafo.sn_mva*1000.)
    powerTrafoLoad = model.addVars(days,timesteps, vtype="C", lb=0, ub=trafo_max, name="powerTrafo_"+str(t))
    powerTrafoInj = model.addVars(days,timesteps, vtype="C", lb=0, ub=trafo_max, name="powerTrafo_"+str(t))
    
    # activation variable for trafo load
    yTrafo = model.addVars(days,timesteps, vtype="B", name="yTrafo_"+str(t))
    
    # set line bounds due to technical limits                             
    powerLine = model.addVars(nodeLines,days,timesteps, vtype="C", lb=-10000, name="powerLine_")
    voltLine = model.addVars(nodeLines,days,timesteps, vtype="C", lb=-10000, name="voltLine_")
    voltNode = model.addVars(gridnodes, days, timesteps, vtype="C", lb=voltNode_min, ub=voltNode_max, name="voltNode_")

    # EV variables
    ev_load = model.addVars(gridnodes, days, timesteps, vtype="C", lb=0, ub=ev_max, name="ev_Load_"+str(n)+str(t))
    ev_inj = model.addVars(gridnodes, days, timesteps, vtype="C", lb=0, ub=ev_max, name="ev_inj_"+str(n)+str(t))
    
    # add bat variables to model
    x_bat = model.addVars(gridnodes, vtype="B", name="bat_existance_"+str(n))      
    y_bat = model.addVars(gridnodes, days, timesteps, vtype="B", name= "activation_charge_"+str(n)+str(t))
    capacity = model.addVars(gridnodes, vtype="C", name="Cap_"+str(n))
    SOC = model.addVars(gridnodes, days, timesteps, vtype="C", name="SOC_"+str(n)+str(d)+str(t))
    SOC_init = model.addVars(gridnodes, days, vtype="C", name="SOC_init_"+str(n)+str(d))
    powerCh = model.addVars(gridnodes, days, timesteps, vtype="C", name="powerLoad_"+str(n)+str(t))
    powerDis = model.addVars(gridnodes, days, timesteps, vtype="C", name="powerFeed_"+str(n)+str(t))
    
    # define auxilary variables to compute resulting load and injection per node
    powerLoad = model.addVars(gridnodes, days, timesteps, vtype="C", name="powerLoad_"+str(n)+str(t))
    powerInj = model.addVars(gridnodes, days, timesteps, vtype="C", name="powerInj_"+str(n)+str(t))
    powerInjPV = model.addVars(gridnodes, days, timesteps, vtype="C", name="powerInjPV_"+str(n)+str(t))
    powerInjBat = model.addVars(gridnodes, days, timesteps, vtype="C", name="powerInjBat_"+str(n)+str(t))
    powerUsePV = model.addVars(gridnodes, days, timesteps, vtype="C", name="powerUseBat_"+str(n)+str(t))
    powerUseBat = model.addVars(gridnodes, days, timesteps, vtype="C", name="powerUseBat_"+str(n)+str(t))
    
    if options["hp_mode"] == "grid_opt":
        # heatpump operation (important to realize modulation level)
        y_hp = model.addVars(gridnodes, days, timesteps, vtype="B",  name="y_hp")
        # heatpump heat and power
        heat_hp = model.addVars(gridnodes, days, timesteps, vtype="C",  name="Q_hp")
        power_hp = model.addVars(gridnodes, days, timesteps, vtype="C", name="P_hp")
    
        # electrical auxiliary heater
        heat_eh = model.addVars(gridnodes, days, timesteps, vtype="C",  name="Q_eh")
        power_eh = model.addVars(gridnodes, days, timesteps, vtype="C", name="P_eh")
    
        # tes variables for charging, discharging, SoC and initial SoC per typeday
        ch_tes = model.addVars(gridnodes, days, timesteps, vtype="C", name="ch_tes")
        dch_tes = model.addVars(gridnodes, days, timesteps,vtype="C", name="dch_tes")
        soc_tes = model.addVars(gridnodes, days, timesteps,vtype="C", name="soc_tes")
        soc_init_tes = model.addVars(gridnodes, days, vtype="C", name="soc_init_tes")
        
        # heatpump auxilary variables for energy balances
        powerHPGrid = model.addVars(gridnodes, days, timesteps, vtype="C",  name="powerHPGrid") 
        powerHPPV = model.addVars(gridnodes, days, timesteps, vtype="C",  name="powerHPPV")
        powerHPBat = model.addVars(gridnodes, days, timesteps, vtype="C",  name="powerHPBat")

        powerEHGrid = model.addVars(gridnodes, days, timesteps, vtype="C",  name="powerEHGrid") 
        powerEHPV = model.addVars(gridnodes, days, timesteps, vtype="C",  name="powerEHPV")
        powerEHBat = model.addVars(gridnodes, days, timesteps, vtype="C",  name="powerEHBat")
    
    else: 
        pass
    
    model.update()
    
    #%% define constraints
    
    #%% economical constraints
    
    model.addConstr(c_total_nodes == c_inv.sum('*') + c_om.sum('*') + c_fix.sum('*')
                                     + c_dem.sum('*') - revenues.sum('*'), name="total_costs")
                                     
    
    model.addConstr(c_total_grid == c_inv.sum('*') + c_om.sum('*') + c_fix.sum('*')
                                    + c_dem_grid - revenues_grid, name="total_costs_grid")
                                     
         
    # compute annual investment costs per load node
    model.addConstrs((c_inv[n] == eco["crf"] * devs["bat"]["rval"] * 
                      (x_bat[n]*devs["bat"]["c_inv_fix"] + capacity[n]*devs["bat"]["c_inv_var"])
                      for n in gridnodes), name="investment_costs"+str(n))
    
    # compute annual operation and maintenance costs per load node
    model.addConstrs((c_om[n] == eco["b"]["infl"] * devs["bat"]["c_om_rel"] * c_inv[n]
                        for n in gridnodes), name="maintenance_costs"+str(n))
    
    # compute annual fix costs for electricity per load node
    model.addConstrs((c_fix[n] == eco["el"]["el_sta"]["fix"][0] for n in nodes["load"]), name="fix_costs"+str(n))
    
    # compute annual demand related costs load node
    Load_total_node = {}
    for n in gridnodes:
        Load_total_node[n] = (sum(clustered["weights"][d] * sum(powerLoad[n,d,t]  ###changer
                            for t in timesteps) for d in days) * dt)
        
    model.addConstrs((c_dem[n] == eco["crf"] * eco["b"]["el"] * Load_total_node[n] * eco["el"]["el_sta"]["var"][0]
                        for n in gridnodes), name="demand_costs"+str(n))
    
    # compute annual demand related costs per grid
    Load_total_grid = (sum(clustered["weights"][d] * sum(powerTrafoLoad[d,t]  ###changer
                            for t in timesteps) for d in days) * dt)
        
    model.addConstr(c_dem_grid == 
                     eco["crf"] * eco["b"]["el"] * Load_total_grid * eco["el"]["el_sta"]["var"][0], 
                     name="demand_costs_grid")
    
    # compute annual revenues for electricity feed-in per node
    # here: it's assumed that revenues are generated only for PV power
    InjPV_total_node = {}
    for n in gridnodes:
        InjPV_total_node[n] = (sum(clustered["weights"][d] * sum(powerInjPV[n,d,t]  ###changer
                            for t in timesteps) for d in days) * dt)
    
    Inj_total_node = {}
    for n in gridnodes: ###changer
        Inj_total_node[n] = (sum(clustered["weights"][d] * sum(powerInjPV[n,d,t] + powerInjBat[n,d,t] 
                            for t in timesteps) for d in days) * dt)
    
    model.addConstrs((revenues[n] == eco["crf"] * eco["b"]["infl"] * InjPV_total_node[n] * eco["price_sell_eeg"] 
                        for n in gridnodes), name="revenues"+str(n))
    
    # compute annual revenues for electricity feed-in per node
    # here: it's assumed that revenues are generated for all injections to the higher level grid
    Inj_total_grid = (sum(clustered["weights"][d] * sum(powerTrafoInj[d,t] ###changer
                            for t in timesteps) for d in days) * dt)
    
    model.addConstr(revenues_grid == 
                     eco["crf"] * eco["b"]["infl"] * Inj_total_grid * eco["price_sell_eeg"], 
                     name="revenues"+str(n))            
    
    #%% ecological constraints
    
    if options["static_emissions"]:
        # compute annual emissions and emission revenues
        # for single nodes
        emissions_Load_nodes = {}
        emissions_Inj_nodes = {}
        for n in gridnodes: ###changer 
            emissions_Load_nodes[n] = (sum(clustered["weights"][d] * sum((powerLoad[n,d,t] *clustered["co2_stat"][d,t]) 
                                        for t in timesteps) for d in days) * dt)
                        ###changer
            emissions_Inj_nodes[n] = (sum(clustered["weights"][d] * sum((powerInj[n,d,t] *clustered["co2_stat"][d,t]) 
                                        for t in timesteps) for d in days) * dt)
            
        # for total grid     ###changer
        emissions_Load_grid = (sum(clustered["weights"][d] * sum((powerTrafoLoad[d,t] *clustered["co2_stat"][d,t]) 
                                    for t in timesteps) for d in days) * dt)
                            ###changer
        emissions_Inj_grid = (sum(clustered["weights"][d] * sum((powerTrafoInj[d,t] *clustered["co2_stat"][d,t]) 
                                    for t in timesteps) for d in days) * dt)
        
        # calculate emissions with static CO2-factor and revenues
        if options["rev_emissions"]:
            model.addConstrs((emission_nodes[n] == Load_total_node[n] - emissions_Inj_nodes[n]
                            for n in gridnodes), name= "emissions_stat_rev_node"+str(n)) 
            model.addConstr(emission_grid == emissions_Load_grid - emissions_Inj_grid,
                            name= "emissions_stat_rev_grid")
            
        # calculate emissions with static CO2-factor without revenues
        else:
            model.addConstrs((emission_nodes[n] == Load_total_node[n]
                             for n in gridnodes), name= "emissions_stat_rev_node"+str(n))     
            model.addConstr(emission_grid == emissions_Load_grid,
                            name= "emissions_stat_rev_grid")      
    
    else:
        # compute annual emissions and emission revenues
        # for single nodes
        emissions_Load_nodes = {}
        emissions_Inj_nodes = {}
        for n in gridnodes:   ###changer
            emissions_Load_nodes[n] = (sum(clustered["weights"][d] * sum((powerLoad[n,d,t] *clustered["co2_dyn"][d,t]) 
                                        for t in timesteps) for d in days) * dt)
                            ###changer
            emissions_Inj_nodes[n] = (sum(clustered["weights"][d] * sum((powerInj[n,d,t] *clustered["co2_dyn"][d,t]) 
                                        for t in timesteps) for d in days) * dt)
            
        # for total grid        ###changer
        emissions_Load_grid = (sum(clustered["weights"][d] * sum((powerTrafoLoad[d,t] *clustered["co2_dyn"][d,t]) 
                                    for t in timesteps) for d in days) * dt)
                                ###changer
        emissions_Inj_grid = (sum(clustered["weights"][d] * sum((powerTrafoInj[d,t] *clustered["co2_dyn"][d,t]) 
                                    for t in timesteps) for d in days) * dt)
        
        # calculate emissions with timevariant CO2-factor and revenues
        if options["rev_emissions"]:
            model.addConstrs((emission_nodes[n] == Load_total_node[n] - emissions_Inj_nodes[n]
                             for n in gridnodes), name= "emissions_dyn_rev_node") 
            model.addConstr(emission_grid == emissions_Load_grid - emissions_Inj_grid,
                            name= "emissions_dyn_rev_grid")
            
        # calculate emissions with timevariant CO2-factor without revenues
        else:
            model.addConstrs((emission_nodes[n] == Load_total_node[n]
                             for n in gridnodes), name= "emissions_dyn_rev_node")     
            model.addConstr(emission_grid == emissions_Load_grid,
                            name= "emissions_dyn_rev_grid")
    
     
    #%% grid constraints
    
    # prevent trafo from simulataneous load and injection
    model.addConstrs((powerTrafoLoad[d,t] <= yTrafo[d,t] * trafo_max for d in days for t in timesteps), 
                     name="trafoLoad_activation")
    model.addConstrs((powerTrafoInj[d,t] <= (1 - yTrafo[d,t]) * trafo_max for d in days for t in timesteps),
                     name="trafoInj_activation")
    
    # set energy balance for all nodes
    for n in gridnodes:
        for d in days:
            for t in timesteps:
                if n in nodes["trafo"]:
                
                    model.addConstr(powerLine.sum(n,'*',d,t) - powerLine.sum('*',n,d,t) == 
                                    powerTrafoLoad[d,t] - powerTrafoInj[d,t], name="node balance_"+str(n))
                    
                    model.addConstr(powerInj[n,d,t] == 0)
                    model.addConstr(powerLoad[n,d,t] == 0)
    
                else:
                    model.addConstr(powerLine.sum(n,'*',d,t) - powerLine.sum('*',n,d,t) == 
                                    powerInj[n,d,t] - powerLoad[n,d,t], name="node balance_"+str(n))
    
    # set Load over Trafo = 0 to simulate autarky
#    model.addConstrs(powerTrafoLoad[d,t] == 0 for d in days for t in timesteps)   
            
    # set line limits    
    for [n,m] in nodeLines:
        for d in days:
            for t in timesteps: 
                
                    model.addConstr(powerLine[n,m,d,t] <= powerLine_max[n,m], name="line power max_"+str(n)+str(m)+str(t))
                    model.addConstr(powerLine[n,m,d,t] >= (-1)*powerLine_max[n,m], name="line power min_"+str(n)+str(m)+str(t))
                    #model.addConstr(voltLine[n,m,d,t] == (powerLine[n,m,d,t] * lineLength[n,m] * specRes_ap[n,m] / U_nominal))
  
    for d in days:
        for t in timesteps:
            model.addConstr(voltNode[1,d,t] == U_nominal)  
    
    sin_phi = math.sin(options["phi"]) 
    cos_phi = math.cos(options["phi"])        
    for [n,m] in nodeLines:
        for d in days:
            for t in timesteps:
                    #model.addConstr(voltNode[m,d,t] == voltNode[n,d,t] - voltLine[n,m,d,t], name="node voltage_")
                    model.addConstr(voltNode[m,d,t] == voltNode[n,d,t] - 2*(res_r[n,m]*powerLine[n,m,d,t] * cos_phi + res_x[n,m] * powerLine[n,m,d,t] * sin_phi), name="node voltage_")
                    
    #%% EV Constraints 
    
    if options["EV_mode"] == "off":
#        pass  # infeasable when demEV and load/inj set = 0
        for n in gridnodes:
            for d in days:
                for t in timesteps: 
                    model.addConstr(ev_load[n,d,t] == 0)
                    model.addConstr(ev_inj[n,d,t] == 0)  
    else:
        for n in gridnodes:
            for d in days:
                if n in loads_with["ev"]:
                        model.addConstr(demEV[n,d] == sum(ev_load[n,d,'*'] - ev_inj[n,d,'*']))
                else:
                    model.addConstr(demEV[n,d] == 0)
                    for t in timesteps: 
                        model.addConstr(ev_load[n,d,t] == 0)
                        model.addConstr(ev_inj[n,d,t] == 0)
        
        for n in gridnodes:
            for d in days:
                for t in timesteps:
                    # on demand loading
                    if options["EV_mode"] == "on_demand":
                        
                        pass
                    # grid reactive loading
                    if options["EV_mode"] == "grid_reactive":
                        
                        pass
                    # bi-directional loading
                    if options["EV_mode"] == "bi-directional":
                        if not n in loads_with["ev"]:
                            model.addConstr(ev_inj[n,d,t] == 0)
                    else:
                        model.addConstr(ev_inj[n,d,t] == 0)  
    
                    
                    #if options["EV_mode"] == "bi-directional":
                    #    pass
                    
                
   #%% battery constraints
    
    # Battery can be switched on only if it has been purchased       
    for n in gridnodes:
        model.addConstr(params["time_steps"]*params["days"]*x_bat[n] >= sum(sum(y_bat[n,d,t] for t in timesteps) for d in days),  name="Activation_bat")
    
    # maximum power is defined by power/capacity ratio       
    for n in gridnodes:
        for d in days:                   
            for t in timesteps:
                if n in nodes["bat"]:
                    
                    model.addConstr(powerCh[n,d,t]  <= devs["bat"]["P_ch_fix"] + capacity[n]*devs["bat"]["P_ch_var"], 
                                    name="max power_"+str(n)+str(t))
                    model.addConstr(powerCh[n,d,t]  <= y_bat[n,d,t]*chBat_max[n], name="bigM_power_"+str(n)+str(t))
                    model.addConstr(powerDis[n,d,t] <= devs["bat"]["P_dch_fix"] + capacity[n]*devs["bat"]["P_dch_var"], 
                                    name="max power_"+str(n)+str(t))
                    model.addConstr(powerDis[n,d,t]  <= (1-y_bat[n,d,t])*chBat_max[n], name="bigM_power_"+str(n)+str(t))
    
    # set limitations for battery capacity
    for n in gridnodes:
        if n in nodes["bat"]:
    
            model.addConstr(capacity[n] <= x_bat[n]*capBat_max[n], name="Battery_capacity_max")
            model.addConstr(capacity[n] >= x_bat[n]*capBat_min[n], name="Battery_capacity_min")
            
            model.addConstrs((capacity[n] >= SOC_init[n,d] for d in days), name="Battery_capacity_SOC_init")
            model.addConstrs((capacity[n] >= SOC[n,d,t] for d in days for t in timesteps), name="Battery_capacity_SOC")
            
        else:
            
            model.addConstr(capacity[n] == 0, name="Battery_capacity_max")
                    
    # SOC repetitions: SOC at the end of typeday == SOC at the beginning of typeday
    for n in gridnodes:   
        for d in days:   
            
            model.addConstr(SOC_init[n,d] == SOC[n,d,len(timesteps)-1],
                                                           name="repetitions_" +str(d))
    
    for n in gridnodes:   
        for d in days:         
            for t in timesteps:
                if t == 0:
                    SOC_previous = SOC_init[n,d]
                else:
                    SOC_previous = SOC[n,d,t-1]
            
                model.addConstr(SOC[n,d,t] == SOC_previous 
                            + (dt * (powerCh[n,d,t] * devs["bat"]["eta"] - powerDis[n,d,t]/devs["bat"]["eta"])) 
                            - devs["bat"]["k_loss"]*dt*SOC_previous, name="storage balance_"+str(n)+str(t))
          
    #%% heat pump and tes constraints 
    
    if options["hp_mode"] == "grid_opt":
        # heatpump power and heat 
        model.addConstrs((power_hp[n,d,t] <= (y_hp[n,d,t] * capa_hp) for n in gridnodes for d in days for t in timesteps), name="Max_power_hp")
                        
        model.addConstrs((power_hp[n,d,t] >= (y_hp[n,d,t] * capa_hp * mod_lvl) for n in gridnodes for d in days for t in timesteps), name="Min_power_hp")
        
        model.addConstrs((power_hp[n,d,t] == heat_hp[n,d,t] / cop[d,t] for n in gridnodes for d in days for t in timesteps), name="heat_power_coupling")
            
        # electric heater power and heat    
        model.addConstrs((power_eh[n,d,t] == heat_eh[n,d,t]/devs["eh"]["eta"] for n in gridnodes for d in days for t in timesteps), name="heat_power_eh")    
               
        # tes state of charge
        # initial SOC per typeday
        model.addConstrs((soc_init_tes[n,d] <= capa_tes for n in gridnodes for d in days), name="SOC_init_tes")
        # SOC limit for every timestep            
        model.addConstrs((soc_tes[n,d,t] <= capa_tes for n in gridnodes for d in days for t in timesteps), name="SOC_tes")
        # SOC repetitions >> SOC at the end of the day has to be SOC at the beginning of this day
        model.addConstrs((soc_init_tes[n, d] == soc_tes[n, d,params["time_steps"]-1] for n in gridnodes for d in days), name="repetitions_tes")
                         
            
        #k_loss = devs["tes"]["k_loss"]
        k_loss = 0
            
        for n in gridnodes: 
            for d in days:
                for t in timesteps:
                    if t == 0:
                        soc_prev = soc_init_tes[n,d]
                    else:
                        soc_prev = soc_tes[n,d,t-1]
            
                    model.addConstr(soc_tes[n,d,t] == (1 - k_loss) * soc_prev + dt * (ch_tes[n,d,t]*devs["tes"]["eta_ch"]  -  dch_tes[n,d,t]/devs["tes"]["eta_dch"]), name="Storage_balance_tes")
#                    model.addConstr(soc_tes[n,d,t] ==  (soc_prev + ch_tes[n,d,t] - dch_tes[n,d,t]), name="Storage_balance_tes")
 
                    model.addConstr(ch_tes[n,d,t]  == (heat_hp[n,d,t] + heat_eh[n,d,t]), name="Thermal_max_charge_tes")
    
        # differentiation for dhw-heating: either electric or via heating system
        if options["dhw_electric"]:
            model.addConstrs((dch_tes[n,d,t] == heatload[n,d,t] for n in gridnodes for d in days for t in timesteps), name="Thermal_max_discharge_tes")            
        else:     
            model.addConstrs((dch_tes[n,d,t] == (heatload[n,d,t] + dhwload[n,d,t]) for n in gridnodes for d in days for t in timesteps), name="Thermal_max_discharge_tes")
    
                
    #%% energy balances for every node
        
    # split injected power in power from PV and power from battery
    model.addConstrs((powerInj[n,d,t] == powerInjPV[n,d,t] + powerInjBat[n,d,t] 
                      for n in gridnodes for d in days for t in timesteps), name="powerInj"+str(n)+str(d)+str(t))
        
    if options["hp_mode"] == "grid_opt":
            
        # split power from PV generation in injected and used power
        model.addConstrs((powerInjPV[n,d,t] == powerPV[n,d,t] - powerUsePV[n,d,t] - powerHPPV[n,d,t] - powerEHPV[n,d,t]
                            for n in gridnodes for d in days for t in timesteps), name="powerInjPV"+str(n)+str(d)+str(t))
    
        # split battery discharging power in injected and used power
        model.addConstrs((powerDis[n,d,t] == powerInjBat[n,d,t] + powerUseBat[n,d,t] + powerHPBat[n,d,t] + powerEHBat[n,d,t]
                          for n in gridnodes for d in days for t in timesteps), name="powerInj_UseBat"+str(n)+str(d)+str(t))

        # node energy balance   ##EV added
#        model.addConstrs((powerPlug[n,d,t] + powerCh[n,d,t] + powerHPGrid[n,d,t] + powerEHGrid[n,d,t] == 
#                          powerLoad[n,d,t] + powerUsePV[n,d,t] + powerUseBat[n,d,t] 
#                          + powerHPPV[n,d,t] + powerHPBat[n,d,t] + powerEHPV[n,d,t] + powerEHBat[n,d,t]
#                          for n in gridnodes for d in days for t in timesteps), name="powerInj_UseBat"+str(n)+str(d)+str(t))         
        model.addConstrs((powerPlug[n,d,t] + powerCh[n,d,t] + powerHPGrid[n,d,t] + powerEHGrid[n,d,t] + ev_load[n,d,t] == 
                          powerLoad[n,d,t] + powerUsePV[n,d,t] + powerUseBat[n,d,t] 
                          + powerHPPV[n,d,t] + powerHPBat[n,d,t] + powerEHPV[n,d,t] + powerEHBat[n,d,t] #+ ev_inj[n,d,t]
                          for n in gridnodes for d in days for t in timesteps), name="powerInj_UseBat"+str(n)+str(d)+str(t))         

        model.addConstrs((power_hp[n,d,t] == 
                          powerHPGrid[n,d,t] + powerHPPV[n,d,t] + powerHPBat[n,d,t] 
                          for n in gridnodes for d in days for t in timesteps), name="powerInj_UseBat"+str(n)+str(d)+str(t))       
        
        model.addConstrs((power_eh[n,d,t] == 
                          powerEHGrid[n,d,t] + powerEHPV[n,d,t] + powerEHBat[n,d,t] 
                          for n in gridnodes for d in days for t in timesteps), name="powerInj_UseBat"+str(n)+str(d)+str(t))       
        
    else:
        # split power from PV generation in injected and used power
        model.addConstrs((powerInjPV[n,d,t] == powerPV[n,d,t] - powerUsePV[n,d,t] 
                          for n in gridnodes for d in days for t in timesteps), name="powerInjPV"+str(n)+str(d)+str(t))
    
        # split battery discharging power in injected and used power
        model.addConstrs((powerDis[n,d,t] == powerInjBat[n,d,t] + powerUseBat[n,d,t] 
                          for n in gridnodes for d in days for t in timesteps), name="powerInj_UseBat"+str(n)+str(d)+str(t))
    
        # node energy balance ##EV added
#        model.addConstrs((powerPlug[n,d,t] + powerCh[n,d,t] == 
#                          powerLoad[n,d,t] + powerUsePV[n,d,t] + powerUseBat[n,d,t] 
#                          for n in gridnodes for d in days for t in timesteps), name="powerInj_UseBat"+str(n)+str(d)+str(t))         
        model.addConstrs((powerPlug[n,d,t] + powerCh[n,d,t] + ev_load[n,d,t] == 
                          powerLoad[n,d,t] + powerUsePV[n,d,t] + powerUseBat[n,d,t] #+ ev_inj[n,d,t]
                          for n in gridnodes for d in days for t in timesteps), name="powerInj_UseBat"+str(n)+str(d)+str(t))         


        
    #%% start optimization
    
    # set objective function             
    #model.setObjective(XYZ, gp.GRB.MINIMIZE)
    #model.setObjective(XYZ, gp.GRB.MAXIMIZE)
    
        # =============================================================================
        #     model.setObjective(sum(sum((powerTrafoLoad[d,t]-powerTrafoInj[d,t])*clustered["co2_dyn"][d,t] 
        #                                 for t in timesteps) for d in days), gp.GRB.MINIMIZE)    
        # =============================================================================
    # Maximize net present Value
#    a = 50 
#    rbf = ((eco["q"]**a)-1)/((eco["q"]**a)*(eco["q"]-1))
#    model.setObjective(-sum(c_inv[n] for n in gridnodes) + rbf * (revenues_grid - c_total_grid), gp.GRB.MAXIMIZE)
    
    # Minimize costs per year
    model.setObjective(c_total_grid, gp.GRB.MINIMIZE)
    
    # Minimize emissions
    #model.setObjective(sum(emission_nodes[n] for n in gridnodes), gp.GRB.MINIMIZE)
    
    # Minimize power over Trafo (both directions)
    #model.setObjective(sum((powerTrafoLoad[d,t] - powerTrafoInj[d,t]) for d in days for t in timesteps), gp.GRB.MINIMIZE)
    
    # Minimize Power over Trafo into MV-grid
    #model.setObjective(sum((powerTrafoInj[d,t]) for d in days for t in timesteps), gp.GRB.MINIMIZE)
#    powerInj[n,d,t] - powerLoad[n,d,t]
    
    # adgust gurobi settings
    model.Params.TimeLimit = 50    
    model.Params.MIPGap = 0.00
    model.Params.NumericFocus = 3
    model.Params.MIPFocus = 3
    model.Params.Aggregate = 1
    model.Params.DualReductions = 0
    
    model.write("debug.lp")
    model.optimize()
    
    print('Optimization ended with status %d' % model.status)
    
    if model.status==gp.GRB.Status.INFEASIBLE:
        model.computeIIS()
        f=open('errorfile.txt','w')
        f.write('\nThe following constraint(s) cannot be satisfied:\n')
        for c in model.getConstrs():
            if c.IISConstr:
                f.write('%s' % c.constrName)
                f.write('\n')
        f.close()


    #%% retrieve results
    
    # grid results    
    res_powerTrafoLoad = np.array([[powerTrafoLoad[d,t].X for t in timesteps]for d in days])
    res_powerTrafoInj = np.array([[powerTrafoInj[d,t].X for t in timesteps]for d in days])
        
    res_powerLine = {}
    for [n,m] in nodeLines:
        res_powerLine[n,m] = np.array([[powerLine[n,m,d,t].X for t in timesteps] for d in days])    
        
    res_voltLine = {}
    for [n,m] in nodeLines:
        res_voltLine[n,m] = np.array([[voltLine[n,m,d,t].X for t in timesteps] for d in days])
                
    res_voltNode = {}
    for n in gridnodes:
        res_voltNode[n] = np.array([[voltNode[n,d,t].X for t in timesteps] for d in days])
    
    #EV results
    res_ev_load = {}
    res_ev_inj = {}
    for n in gridnodes:
        res_ev_load[n] = np.array([[ev_load[n,d,t].X for t in timesteps] for d in days])
        res_ev_inj[n] = np.array([[ev_inj[n,d,t].X for t in timesteps] for d in days])

    
    # battery operation results
    res_capacity = {}
    res_powerCh = {}
    res_powerDis = {}
    res_SOC = {}
    res_SOC_init = {}
    for n in gridnodes:
        res_capacity = np.array([capacity[n].X for n in gridnodes])
        res_powerCh[n] = np.array([[powerCh[n,d,t].X for t in timesteps] for d in days])
        res_powerDis[n] = np.array([[powerDis[n,d,t].X for t in timesteps] for d in days])
        res_SOC[n] = np.array([[SOC[n,d,t].X for t in timesteps] for d in days])
        res_SOC_init[n] = np.array([SOC_init[n,d].X for d in days])
    
    # node energy management results    
    res_powerLoad = {}
    res_powerInj = {}
    res_powerInjPV = {}
    res_powerInjBat = {}
    res_powerUsePV = {}
    res_powerUseBat = {}
    
    res_powerPV = {}
    res_powerPlug = {} 
    
    for n in gridnodes:
        res_powerLoad[n] = np.array([[powerLoad[n,d,t].X for t in timesteps] for d in days])
        res_powerInj[n] = np.array([[powerInj[n,d,t].X for t in timesteps] for d in days])
        res_powerInjPV[n] = np.array([[powerInjPV[n,d,t].X for t in timesteps] for d in days])
        res_powerInjBat[n] = np.array([[powerInjBat[n,d,t].X for t in timesteps] for d in days])
        res_powerUsePV[n] = np.array([[powerUsePV[n,d,t].X for t in timesteps] for d in days])
        res_powerUseBat[n] = np.array([[powerUseBat[n,d,t].X for t in timesteps] for d in days])
    
        res_powerPV[n] = np.array([[powerPV[n,d,t] for t in timesteps] for d in days])
        res_powerPlug[n] = np.array([[powerPlug[n,d,t] for t in timesteps] for d in days])
    
    # economical and ecological results
    res_c_inv = np.array([c_inv[n].X for n in gridnodes])
    res_c_om = np.array([c_om[n].X for n in gridnodes])
    res_c_dem = np.array([c_dem[n].X for n in gridnodes])
    res_c_fix = np.array([c_fix[n].X for n in gridnodes])
    res_rev = np.array([revenues[n].X for n in gridnodes])
    
    res_c_dem_grid = c_dem_grid.X
    res_rev_grid = revenues_grid.X
    
    # comupte annual costs per node
    res_c_node = res_c_inv + res_c_om + res_c_dem + res_c_fix - res_rev
        
    res_c_total_nodes = c_total_nodes.X
    res_c_total_grid = c_total_grid.X
    res_emission_nodes = np.array([emission_nodes[n].X for n in gridnodes])
    res_emission_grid = emission_grid.X
    
    # compute energy for hp and eh
    if options["hp_mode"] == "grid_opt":
    
        for n in gridnodes:
            
            res_actHP[n] = np.array([[y_hp[n,d,t].X for t in timesteps] for d in days])
            res_powerHP[n] = np.array([[power_hp[n,d,t].X for t in timesteps] for d in days])
            res_powerEH[n] = np.array([[power_eh[n,d,t].X for t in timesteps] for d in days])
            res_heatHP[n] = np.array([[heat_hp[n,d,t].X for t in timesteps] for d in days])
            res_heatEH[n] = np.array([[heat_eh[n,d,t].X for t in timesteps] for d in days])
            
            res_SOC_tes[n] = np.array([[soc_tes[n,d,t].X for t in timesteps] for d in days]) 
            res_SOC_init_tes[n] = np.array([soc_init_tes[n,d].X for d in days])
            res_ch_tes[n] = np.array([[ch_tes[n,d,t].X for t in timesteps] for d in days]) 
            res_dch_tes[n] = np.array([[dch_tes[n,d,t].X for t in timesteps] for d in days]) 

            res_powerHPGrid[n] = np.array([[powerHPGrid[n,d,t].X for t in timesteps] for d in days])  
            res_powerHPPV[n] = np.array([[powerHPPV[n,d,t].X for t in timesteps] for d in days]) 
            res_powerHPBat[n] = np.array([[powerHPBat[n,d,t].X for t in timesteps] for d in days]) 
            res_powerEHGrid[n] = np.array([[powerEHGrid[n,d,t].X for t in timesteps] for d in days]) 
            res_powerEHPV[n] = np.array([[powerEHPV[n,d,t].X for t in timesteps] for d in days]) 
            res_powerEHBat[n] = np.array([[powerEHBat[n,d,t].X for t in timesteps] for d in days]) 
       
    else: 
        pass
    
    res_exBat = {}
    res_actBat = {}
    
    for n in gridnodes:           
        res_exBat[n] = x_bat[n].X
        res_actBat[n] = np.array([[y_bat[n,d,t].X for t in timesteps] for d in days])

    # writing the nodes with a battery into an xlsx file
    bool_bat = {}
    i = 0
    for n in gridnodes:
        if res_capacity[n] >0:
            bool_bat[i] = n
            i += 1        
    ev_book = xlsxwriter.Workbook(distributionFolder + "\\" + "ev_best_" +  ev_file)
    sheet = ev_book.add_worksheet()
    row = 0    
    for b in bool_bat:
        sheet.write(row, 0, bool_bat[b])
        row += 1
    ev_book.close()
    
#    i = 0
#    for n in gridnodes:
#        if res_capacity[n] >0:
#            bool_bat[i] = [n, res_capacity[n]]
#            i += 1        

            
    # save results 
    with open(names["filename_results"], "wb") as fout:
        pickle.dump(model.ObjVal, fout, pickle.HIGHEST_PROTOCOL)            #01
        pickle.dump(model.Runtime, fout, pickle.HIGHEST_PROTOCOL)           #02  
        pickle.dump(model.MIPGap, fout, pickle.HIGHEST_PROTOCOL)            #03
        pickle.dump(res_powerTrafoLoad, fout, pickle.HIGHEST_PROTOCOL)      #04
        pickle.dump(res_powerTrafoInj, fout, pickle.HIGHEST_PROTOCOL)       #05
        pickle.dump(res_powerLine, fout, pickle.HIGHEST_PROTOCOL)           #06
        pickle.dump(res_capacity, fout, pickle.HIGHEST_PROTOCOL)            #07
        pickle.dump(res_powerCh, fout, pickle.HIGHEST_PROTOCOL)             #08
        pickle.dump(res_powerDis, fout, pickle.HIGHEST_PROTOCOL)            #09
        pickle.dump(res_SOC, fout, pickle.HIGHEST_PROTOCOL)                 #10
        pickle.dump(res_SOC_init, fout, pickle.HIGHEST_PROTOCOL)            #11
        pickle.dump(res_powerLoad, fout, pickle.HIGHEST_PROTOCOL)           #12
        pickle.dump(res_powerInj, fout, pickle.HIGHEST_PROTOCOL)            #13
        pickle.dump(res_powerInjPV, fout, pickle.HIGHEST_PROTOCOL)          #14
        pickle.dump(res_powerInjBat, fout, pickle.HIGHEST_PROTOCOL)         #15
        pickle.dump(res_powerUsePV, fout, pickle.HIGHEST_PROTOCOL)          #16
        pickle.dump(res_powerUseBat, fout, pickle.HIGHEST_PROTOCOL)         #17
        pickle.dump(res_powerPV, fout, pickle.HIGHEST_PROTOCOL)             #18
        pickle.dump(res_powerPlug, fout, pickle.HIGHEST_PROTOCOL)           #19
        pickle.dump(res_c_inv, fout, pickle.HIGHEST_PROTOCOL)               #20
        pickle.dump(res_c_om, fout, pickle.HIGHEST_PROTOCOL)                #21
        pickle.dump(res_c_dem, fout, pickle.HIGHEST_PROTOCOL)               #22
        pickle.dump(res_c_fix, fout, pickle.HIGHEST_PROTOCOL)               #23
        pickle.dump(res_rev, fout, pickle.HIGHEST_PROTOCOL)                 #24
        pickle.dump(res_c_dem_grid, fout, pickle.HIGHEST_PROTOCOL)          #25
        pickle.dump(res_rev_grid, fout, pickle.HIGHEST_PROTOCOL)            #26
        pickle.dump(res_c_node, fout, pickle.HIGHEST_PROTOCOL)              #27
        pickle.dump(res_c_total_nodes, fout, pickle.HIGHEST_PROTOCOL)       #28
        pickle.dump(res_c_total_grid, fout, pickle.HIGHEST_PROTOCOL)        #29
        pickle.dump(res_emission_nodes, fout, pickle.HIGHEST_PROTOCOL)      #30
        pickle.dump(res_emission_grid, fout, pickle.HIGHEST_PROTOCOL)       #31
        pickle.dump(nodes, fout, pickle.HIGHEST_PROTOCOL)                   #32
        pickle.dump(nodeLines, fout, pickle.HIGHEST_PROTOCOL)               #33
        pickle.dump(res_actHP, fout, pickle.HIGHEST_PROTOCOL)               #34
        pickle.dump(res_powerHP, fout, pickle.HIGHEST_PROTOCOL)             #35
        pickle.dump(res_powerEH, fout, pickle.HIGHEST_PROTOCOL)             #36
        pickle.dump(res_SOC_tes, fout, pickle.HIGHEST_PROTOCOL)             #37
        pickle.dump(res_SOC_init_tes, fout, pickle.HIGHEST_PROTOCOL)        #
        pickle.dump(res_ch_tes, fout, pickle.HIGHEST_PROTOCOL)              #38
        pickle.dump(res_dch_tes, fout, pickle.HIGHEST_PROTOCOL)             #39
        pickle.dump(res_heatHP, fout, pickle.HIGHEST_PROTOCOL)              #40
        pickle.dump(res_heatEH, fout, pickle.HIGHEST_PROTOCOL)              #41
        pickle.dump(res_voltLine, fout, pickle.HIGHEST_PROTOCOL)            #42
        pickle.dump(res_voltNode, fout, pickle.HIGHEST_PROTOCOL)            #43
        pickle.dump(res_powerHPGrid, fout, pickle.HIGHEST_PROTOCOL)         #44
        pickle.dump(res_powerHPPV, fout, pickle.HIGHEST_PROTOCOL)           #45
        pickle.dump(res_powerHPBat, fout, pickle.HIGHEST_PROTOCOL)          #46
        pickle.dump(res_powerEHGrid, fout, pickle.HIGHEST_PROTOCOL)         #47
        pickle.dump(res_powerEHPV, fout, pickle.HIGHEST_PROTOCOL)           #48
        pickle.dump(res_powerEHBat, fout, pickle.HIGHEST_PROTOCOL)          #49
        pickle.dump(res_exBat, fout, pickle.HIGHEST_PROTOCOL)               #50
        pickle.dump(res_actBat, fout, pickle.HIGHEST_PROTOCOL)              #51
        pickle.dump(res_ev_load, fout, pickle.HIGHEST_PROTOCOL)
        pickle.dump(res_ev_inj, fout, pickle.HIGHEST_PROTOCOL)
        
        return (res_c_total_grid, res_emission_grid, bool_bat)