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

# import own function
import python.hpopt_energy as hpopt

#%% Start:

def compute(net, nodes, gridnodes, days, timesteps, eco, devs, clustered, params, options, batData, constraint_apc, constraint_bat, critical_flag):
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

    #create matrix size [d,t] with value 1 for each cell
    
    # PV efficiency
    eta_inverter = 0.97
    # compute generated power
    powerPV = options["P_pv"] *(devs["pv"]["area_mean"]/devs["pv"]["p_nom"]) * devs["pv"]["eta_el"] * eta_inverter * clustered["solar_irrad"]
    # define max powerPV for curtailment (clustered["solar_irrad"] is a value between 0 and 1)
    powerPVMax = options["P_pv"] * eta_inverter

    # calculate thermal nominal hp capacity according to (Stinner, 2017)
    if options ["dhw_electric"]:
        capa_hp_th = options["alpha_th"] * np.max(clustered["heat"] + clustered["dhw"])
    else: 
        capa_hp_th = options["alpha_th"] * np.max(clustered["heat"])
    
    # electrical nominal hp capacity
    if options ["T_VL"] == 35:
        capa_hp = capa_hp_th/devs["hp_air"]["cop_a-7w35"]
    elif options ["T_VL"] == 55:
        capa_hp = capa_hp_th/devs["hp_air"]["cop_a-7w55"]
        
    # calculate tes capacity according to (Stinner, 2017)
    if options ["dhw_electric"]:
        capa_tes = options["beta_th"] * sum(clustered["weights"][d] * sum(clustered["heat"][d,t] + clustered["dhw"][d,t] for t in timesteps) for d in days) * dt / sum(clustered["weights"])
    else:
        capa_tes = options["beta_th"] * sum(clustered["weights"][d] * sum((clustered["heat"][d,t]) for t in timesteps) for d in days) * dt / sum(clustered["weights"])
    
    if options["hp_mode"] == "energy_opt":
        
        (res_actHP, res_powerHP, res_powerEH, res_SOC_tes, res_ch_tes, res_dch_tes, res_heatHP, res_heatEH) = hpopt.optimize(options, params, clustered, devs, capa_hp, capa_tes)
        
        if options ["dhw_electric"]:
            powerElec = clustered["electricity"] + clustered["dhw"] + res_powerHP + res_powerEH
        else:
            powerElec = clustered["electricity"] + res_powerHP + res_powerEH
    else:
        
        if options ["dhw_electric"]:
            powerElec = clustered["electricity"] + clustered["dhw"]
        else:
            powerElec = clustered["electricity"]
    
#%% extract node and line information from pandas-network
    
    # extract existing lines 
    nodeLines = []
    for i in range(len(net.line['from_bus'])):
        nodeLines.append((net.line['from_bus'][i],net.line['to_bus'][i]))
    nodeLines = gp.tuplelist(nodeLines)
    
    # extract maximal current for lines
    # multiply with 400 V to get maximal power in kW
    # powerLine is positive, if the power flows into the net, out of the trafo ('from bus n to bus m'), trafo is bus 0
    powerLine_max = {}
    for [n,m] in nodeLines:
        powerLine_max[n,m] = (net.line['max_i_ka'][nodeLines.index((n,m))])*400
        
    # extract battery nodes and define technical data for them
    # data is linearized out of devices file
    capBat_max = {}
    capBat_min = {}
    
    for n in gridnodes:
        if n in nodes["bat"]:
            capBat_max[n] = devs["bat"]["cap_max"]
            capBat_min[n] = devs["bat"]["cap_min"]

    # attach plug-in loads and PV generatrion to building nodes
    # assume that PV is the only generated electricity
    # TODO: do the same with EV loads!?
    powerLoad = {}
    powerGen = {}
    for n in gridnodes:
        for t in timesteps:
            for d in days:
                if n in nodes["load"]:
                    powerLoad[n,d,t] = powerElec[d,t]
                    powerGen[n,d,t] = powerPV[d,t]
                else:
                    powerLoad[n,d,t] = np.zeros_like(powerElec[d,t])
                    powerGen[n,d,t] = np.zeros_like(powerPV[d,t])

    powerGenReal = {}
    powerGenRealMax = {}
    for n in gridnodes:
        for t in timesteps:
            for d in days:
                if n in nodes["load"]:
                    powerGenRealMax[n,d,t] = constraint_apc[n,d,t] * powerPVMax
                    if(powerGen[n,d,t] > powerGenRealMax[n,d,t]):
                        powerGenReal[n,d,t] = powerGenRealMax[n,d,t]
                    else:
                        powerGenReal[n,d,t] = powerGen[n,d,t]
                else:
                    powerGenRealMax[n,d,t] = 0
                    powerGenReal[n,d,t] = 0

    powerGenReal_array = np.array([[[powerGenReal[n,d,t] for n in gridnodes]for d in days] for t in timesteps])
    powerGen_array = np.array([[[powerGen[n,d,t] for n in gridnodes]for d in days] for t in timesteps])

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
    revenues_grid = model.addVar(vtype="C", name="revenues_grid")
    
    # variables for total node costs, total costs and total emissions
    c_total_nodes = model.addVar(vtype="C", name="c_total_nodes", lb= -gp.GRB.INFINITY)
    c_total_grid = model.addVar(vtype="C", name="c_total_grid", lb= -gp.GRB.INFINITY)
    emission_nodes = model.addVars(gridnodes, vtype="C", name= "CO2_emission", lb= -gp.GRB.INFINITY) 
    emission_grid = model.addVar(vtype="C", name= "CO2_emission", lb= -gp.GRB.INFINITY)  
    
    #%% technical variables
    
    # add grid variables to model

    # set trafo bounds due to technichal limits
    trafo_max = float(net.trafo.sn_mva*1000.)
    powerTrafoLoad = model.addVars(days,timesteps, vtype="C", ub=trafo_max, name="powerTrafoLoad_"+str(t))
    powerTrafoInj = model.addVars(days, timesteps, vtype="C", ub=trafo_max, name="powerTrafoInj_"+str(t))
    
    # activation variable for trafo load
    yTrafo = model.addVars(days, timesteps, vtype="B", name="yTrafo_"+str(t))
    
    # set powerLine. Remember: can be positive or negative. Positive in direction away from trafo into net
    powerLine = model.addVars(nodeLines,days, timesteps, vtype="C", lb=-10000, name="powerLine_")
    
    # add bat variables to model
    # y = model.addVars(gridnodes, days, timesteps, vtype="B", name= "activation_charge_"+str(n)+str(t))
    capacity = model.addVars(gridnodes, vtype="C", name="Cap_"+str(n))
    SOC = model.addVars(gridnodes,days, timesteps, vtype="C", name="SOC_"+str(n)+str(d)+str(t))
    SOC_init = model.addVars(gridnodes, days, vtype="C", name="SOC_init_"+str(n)+str(d))
    powerCh = model.addVars(gridnodes,days,timesteps, vtype="C", name="powerCh_"+str(n)+str(t))
    powerDis = model.addVars(gridnodes,days,timesteps, vtype="C", name="powerDis_"+str(n)+str(t))
    # activation variable for bat load
    yBat = model.addVars(gridnodes,days,timesteps, vtype="B", name="yBat"+str(t))

    #add Active Power Curailment variables to model
    """APC100 = model.addVars(gridnodes,days, vtype="B", name="APC100_"+str(n)+str(d)+str(t))
    APC30 = model.addVars(gridnodes,days,vtype="B", name="APC30_"+str(n)+str(d)+str(t))
    APC0 = model.addVars(gridnodes,days, vtype="B", name="APC0_"+str(n)+str(d)+str(t))
    powerGenRealMax = model.addVars(gridnodes, days, vtype="C", name="powerGenRealMax"+str(n)+str(d))
    powerGenReal = model.addVars(gridnodes, days, timesteps, vtype="C", name="powerGenReal"+str(n)+str(d)+str(t))"""

    # power flowing from net into complex of bat, house and pv
    powerSubtr = model.addVars(gridnodes,days,timesteps, vtype="C", name="powerSubtr_"+str(n)+str(t))
    # power flowing from complex of bat, house and pv back into net
    powerInj = model.addVars(gridnodes,days,timesteps, vtype="C", name="powerInj_"+str(n)+str(t))

    # all variables inside the complex of house, bat and data
    powerUsePV = model.addVars(gridnodes,days, timesteps, vtype="C", name="powerUsePV_"+str(n)+str(d)+str(t))
    powerPVChBat = model.addVars(gridnodes,days,timesteps, vtype="C", name="powerPVChBat_"+str(n)+str(d)+str(t))
    powerInjPV = model.addVars(gridnodes,days,timesteps, vtype="C", name="powerInjPV"+str(n)+str(d)+str(t))
    powerNetLoad = model.addVars(gridnodes,days,timesteps, vtype="C", name="powerNetLoad_"+str(n)+str(d)+str(t))
    powerNetChBat = model.addVars(gridnodes,days,timesteps, vtype="C", name="powerNetChBat_"+str(n)+str(d)+str(t))
    powerNetDisBat = model.addVars(gridnodes,days,timesteps, vtype="C", name="powerNetDisBat_"+str(n)+str(d)+str(t))
    powerUseBat = model.addVars(gridnodes,days,timesteps, vtype="C", name="powerUseBat_"+str(n)+str(d)+str(t))
        
    model.update()
    
    #%% define constraints
    
    #%% economical constraints
    #fpo: TODO: have to be worked through
    
    model.addConstr(c_total_nodes == c_inv.sum('*') + c_om.sum('*') + c_fix.sum('*')
                                     + c_dem.sum('*') - revenues.sum('*'), name="total_costs")
                                     
    
    model.addConstr(c_total_grid == c_inv.sum('*') + c_om.sum('*') + c_fix.sum('*')
                                    + c_dem_grid - revenues_grid, name="total_costs_grid")
                                     
         
    # compute annual investment costs per load node
    model.addConstrs((c_inv[n] == eco["crf"] * devs["bat"]["rval"] * batData["c_inv"] * capacity[n]
                        for n in gridnodes), name="investment_costs"+str(n))
    
    # compute annual operation and maintenance costs per load node
    model.addConstrs((c_om[n] == eco["b"]["infl"] * batData["c_om_rel"] * c_inv[n]
                        for n in gridnodes), name="maintenance_costs"+str(n))
    
    # compute annual fix costs for electricity per load node
    model.addConstrs((c_fix[n] == eco["el"]["el_sta"]["fix"][0] for n in nodes["load"]), name="fix_costs"+str(n))
    
    # compute annual demand related costs load node
    Load_total_node = {}
    for n in gridnodes:
        Load_total_node[n] = (sum(clustered["weights"][d] * sum(powerSubtr[n,d,t]
                            for t in timesteps) for d in days) * dt)
        
    model.addConstrs((c_dem[n] == eco["crf"] * eco["b"]["el"] * Load_total_node[n] * eco["el"]["el_sta"]["var"][0]
                        for n in gridnodes), name="demand_costs"+str(n))
    
    # compute annual demand related costs per grid
    Load_total_grid = (sum(clustered["weights"][d] * sum(powerTrafoLoad[d,t]
                            for t in timesteps) for d in days) * dt)
        
    model.addConstr(c_dem_grid == 
                     eco["crf"] * eco["b"]["el"] * Load_total_grid * eco["el"]["el_sta"]["var"][0], 
                     name="demand_costs_grid")
    
    # compute annual revenues for electricity feed-in per node
    # here: it's assumed that revenues are generated only for PV power
    InjPV_total_node = {}
    for n in gridnodes:
        InjPV_total_node[n] = (sum(clustered["weights"][d] * sum(powerInjPV[n,d,t]
                            for t in timesteps) for d in days) * dt)
    
    Inj_total_node = {}
    for n in gridnodes:
        Inj_total_node[n] = (sum(clustered["weights"][d] * sum(powerInj[n,d,t]
                            for t in timesteps) for d in days) * dt)

    #fpo: TODO: i think this is missing the possibility of buying electricity and selling it at another time.
    model.addConstrs((revenues[n] == eco["crf"] * eco["b"]["infl"] * InjPV_total_node[n] * eco["price_sell_eeg"] 
                        for n in gridnodes), name="revenues"+str(n))
    
    # compute annual revenues for electricity feed-in per node
    # here: it's assumed that revenues are generated for all injections to the higher level grid
    Inj_total_grid = (sum(clustered["weights"][d] * sum(powerTrafoInj[d,t]
                            for t in timesteps) for d in days) * dt)

    #fpo: TODO: price sell eeg has to be changed aswell, if possibility of charging bat out of net is open
    model.addConstr(revenues_grid == 
                     eco["crf"] * eco["b"]["infl"] * Inj_total_grid * eco["price_sell_eeg"], 
                     name="revenues"+str(n))            
    
    #%% ecological constraints
    
    if options["static_emissions"]:
        # compute annual emissions and emission revenues
        # for single nodes
        emissions_Load_nodes = {}
        emissions_Inj_nodes = {}
        for n in gridnodes:
            emissions_Load_nodes[n] = (sum(clustered["weights"][d] * sum((powerSubtr[n,d,t] *clustered["co2_stat"][d,t])
                                        for t in timesteps) for d in days) * dt)
            emissions_Inj_nodes[n] = (sum(clustered["weights"][d] * sum((powerInj[n,d,t] *clustered["co2_stat"][d,t])
                                        for t in timesteps) for d in days) * dt)
            
        # for total grid
        emissions_Load_grid = (sum(clustered["weights"][d] * sum((powerTrafoLoad[d,t] *clustered["co2_stat"][d,t])
                                    for t in timesteps) for d in days) * dt)
        emissions_Inj_grid = (sum(clustered["weights"][d] * sum((powerTrafoInj[d,t] *clustered["co2_stat"][d,t])
                                    for t in timesteps) for d in days) * dt)
        
        # calculate emissions with static CO2-factor and revenues
        if options["rev_emissions"]:
            model.addConstrs((emission_nodes[n] == emissions_Load_nodes[n] - emissions_Inj_nodes[n]
                            for n in gridnodes), name= "emissions_stat_rev_node"+str(n)) 
            model.addConstr(emission_grid == emissions_Load_grid - emissions_Inj_grid,
                            name= "emissions_stat_rev_grid")
            
        # calculate emissions with static CO2-factor without revenues
        else:
            model.addConstrs((emission_nodes[n] == emissions_Load_nodes[n]
                             for n in gridnodes), name= "emissions_stat_rev_node"+str(n))     
            model.addConstr(emission_grid == emissions_Load_grid,
                            name= "emissions_stat_rev_grid")      
    
    else:
        # compute annual emissions and emission revenues
        # for single nodes
        emissions_Load_nodes = {}
        emissions_Inj_nodes = {}
        for n in gridnodes:
            emissions_Load_nodes[n] = (sum(clustered["weights"][d] * sum((powerSubtr[n,d,t] *clustered["co2_dyn"][d,t])
                                        for t in timesteps) for d in days) * dt)
            emissions_Inj_nodes[n] = (sum(clustered["weights"][d] * sum((powerInj[n,d,t] *clustered["co2_dyn"][d,t])
                                        for t in timesteps) for d in days) * dt)
            
        # for total grid
        emissions_Load_grid = (sum(clustered["weights"][d] * sum((powerTrafoLoad[d,t] *clustered["co2_dyn"][d,t])
                                    for t in timesteps) for d in days) * dt)
        emissions_Inj_grid = (sum(clustered["weights"][d] * sum((powerTrafoInj[d,t] *clustered["co2_dyn"][d,t])
                                    for t in timesteps) for d in days) * dt)
        
        # calculate emissions with timevariant CO2-factor and revenues
        if options["rev_emissions"]:
            model.addConstrs((emission_nodes[n] == emissions_Load_nodes[n] - emissions_Inj_nodes[n]
                             for n in gridnodes), name= "emissions_dyn_rev_node") 
            model.addConstr(emission_grid == emissions_Load_grid - emissions_Inj_grid,
                            name= "emissions_dyn_rev_grid")
            
        # calculate emissions with timevariant CO2-factor without revenues
        else:
            model.addConstrs((emission_nodes[n] == emissions_Load_nodes[n]
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
                    #energy balance around trafo
                    model.addConstr(powerLine.sum(n,'*',d,t) - powerLine.sum('*',n,d,t) ==
                                    powerTrafoLoad[d,t] - powerTrafoInj[d,t], name="node_balance_trafo_"+str(n))

                    model.addConstr(powerInj[n,d,t] == 0)
                    model.addConstr(powerSubtr[n,d,t] == 0)
                    model.addConstr(powerInjPV[n, d, t] == 0)
                    model.addConstr(powerNetDisBat[n, d, t] == 0)
                    model.addConstr(powerNetChBat[n, d, t] == 0)
                    model.addConstr(powerNetLoad[n, d, t] == 0)
                    model.addConstr(powerUsePV[n, d, t] == 0)
                    model.addConstr(powerUseBat[n, d, t] == 0)
                    model.addConstr(powerPVChBat[n, d, t] == 0)

    
                elif n in nodes["load"]:
                    #energy balance node with net
                    model.addConstr(powerLine.sum(n,'*',d,t) - powerLine.sum('*',n,d,t) ==
                                    powerSubtr[n,d,t] - powerInj[n,d,t], name="node_balance net_"+str(n))

                    # energy balance node with pv-bat-house complex
                    model.addConstr(powerInj[n, d, t] - powerSubtr[n, d, t] ==
                                    powerInjPV[n, d, t] + powerNetDisBat[n, d, t] - powerNetLoad[n, d, t] -
                                    powerNetChBat[n, d, t], name="node_balance_complex_" + str(n))
                    #define powerInj and powerSubtr
                    model.addConstr(powerInj[n, d, t] ==
                                    powerInjPV[n, d, t] + powerNetDisBat[n, d, t], name="balance_powerInj_" + str(n))
                    model.addConstr(powerSubtr[n, d, t] ==
                                    powerNetLoad[n, d, t] + powerNetChBat[n, d, t], name="balance_powerSubtr_" + str(n))
                    # energy balance PV device
                    model.addConstr(-powerUsePV[n, d, t] ==
                                    -powerGenReal[n, d, t] + powerInjPV[n, d, t] + powerPVChBat[n, d, t],
                                    name="node_balance_pv_" + str(n) + str(d) + str(t))
                    # energy balance load
                    model.addConstr(-powerUsePV[n, d, t] ==
                                    -powerLoad[n, d, t] + powerUseBat[n, d, t] + powerNetLoad[n, d, t],
                                    name="node_balance_load"+str(n)+str(d)+str(t))

                else:

                    # energy balance node with net
                    model.addConstr(powerLine.sum(n, '*', d, t) - powerLine.sum('*', n, d, t) == 0,
                                    name="node_balance net_" + str(n))

                    model.addConstr(powerInj[n, d, t] == 0)
                    model.addConstr(powerSubtr[n, d, t] == 0)
                    model.addConstr(powerInjPV[n, d, t] == 0)
                    model.addConstr(powerNetDisBat[n, d, t] == 0)
                    model.addConstr(powerNetChBat[n, d, t] == 0)
                    model.addConstr(powerNetLoad[n, d, t] == 0)
                    model.addConstr(powerUsePV[n, d, t] == 0)
                    model.addConstr(powerUseBat[n, d, t] == 0)
                    model.addConstr(powerPVChBat[n, d, t] == 0)

    # set line limits
    # TODO: check if it's better to set line limits like this or to set lb/ub of variable to min/max values      
    for [n,m] in nodeLines:
            for d in days:
                for t in timesteps:
                    model.addConstr(powerLine[n,m,d,t] <= powerLine_max[n,m], name="line_power_max_"+str(n)+str(m)+str(t))
                    model.addConstr(powerLine[n,m,d,t] >= (-1)*powerLine_max[n,m], name="line_power_min_"+str(n)+str(m)+str(t))
                
                
    #%% battery constraints
    
    # binary variables x/y needed? don't think so right  now -> build LP
    
    # maximum power is defined by power/capacity ratio
    # energy balance for bat
    for n in gridnodes:
        for d in days:                   
            for t in timesteps:
                if n in nodes["bat"]:

                    # added constraint bat, which gets changed for every iteration (+1kW per iteration) and critical flag to indicate whether voltage violations were struck
                    # prevent batteries from simultaneously charging and discharging (yBat integrated)
                    model.addConstr(powerCh[n,d,t]  >= yBat[n,d,t] * critical_flag[n,d,t], name="min_power_"+str(n)+str(t))
                    model.addConstr(powerCh[n,d,t]  <= yBat[n,d,t] * batData["pc_ratio"], name="max_power_"+str(n)+str(t))
                    model.addConstr(powerDis[n,d,t] >= 0, name="min_power_"+str(n)+str(t))
                    model.addConstr(powerDis[n,d,t] <= (1-yBat[n,d,t]) * batData["pc_ratio"], name="max_power_"+str(n)+str(t))



                    model.addConstr(powerDis[n,d,t] - powerCh[n,d,t] ==
                                    powerNetChBat[n,d,t] + powerPVChBat[n,d,t] - powerNetDisBat[n,d,t] - powerUseBat[n,d,t], name="energy_balance_bat_"+str(n)+str(d)+str(t))
    
    # set limitations for battery capacity
    for n in gridnodes:
        if n in nodes["bat"]:
    
            model.addConstr(capacity[n] <= capBat_max[n], name="Battery_capacity_max")
            model.addConstr(capacity[n] >= capBat_min[n], name="Battery_capacity_min")
            
            model.addConstrs((capacity[n] >= SOC_init[n,d] for d in days), name="Battery_capacity_SOC_init")
            model.addConstrs((capacity[n] >= SOC[n,d,t] for t in timesteps for d in days), name="Battery_capacity_SOC")
            #model.addConstrs((SOC[n,d,t] >= 0.05 * capacity[n] for t in timesteps for d in days), name="Battery_SOC_min")
            
        else:
            
            model.addConstr(capacity[n] == 0, name="Battery_capacity_max")
                    
    # SOC repetitions: SOC at the end of typeday == SOC at the beginning of typeday
    for n in gridnodes:   
        for d in days:
            if n in nodes["bat"]:
            
                model.addConstr(SOC_init[n,d] == SOC[n,d,len(timesteps)-1],
                                                           name="repetitions_" +str(d))
    
    for n in gridnodes:   
        for d in days:         
            for t in timesteps:
                if n in nodes["bat"]:
                    if t == 0:
                        SOC_previous = SOC_init[n,d]
                    else:
                        SOC_previous = SOC[n,d,t-1]

                    model.addConstr(SOC[n,d,t] == SOC_previous
                                + (dt * (powerCh[n,d,t] * devs["bat"]["eta"] - powerDis[n,d,t]/devs["bat"]["eta"]))
                                - devs["bat"]["k_loss"]*dt*SOC_previous, name="storage balance_"+str(n)+str(t))

    #%% start optimization
    
    # set objective function
    
    #model.setObjective(sum((emission_nodes[n])for n in gridnodes), gp.GRB.MINIMIZE)

    """model.setObjective(sum(sum(sum((powerSubtr[n, d, t] - powerInj[n, d, t])for n in gridnodes) * clustered["co2_stat"][d,t]
                               for t in timesteps) for d in days), gp.GRB.MINIMIZE)"""
    model.setObjective(c_total_nodes, gp.GRB.MINIMIZE)

    # adgust gurobi settings
    #model.Params.TimeLimit = 60
    
    model.Params.MIPGap = 0.05
    model.Params.NumericFocus = 3
    model.Params.MIPFocus = 3
    model.Params.Aggregate = 1
    model.Params.DualReductions = 0

    model.optimize()
    print('Optimization ended with status %d' % model.status)

    # TODO: something with: gp.GurobiError()? So he allways gives out an error report
    if model.status == gp.GRB.Status.INFEASIBLE:
        model.computeIIS()
        f=open('errorfile.txt','w')
        f.write('\nThe following constraint(s) cannot be satisfied:\n')
        for c in model.getConstrs():
            if c.IISConstr:
                f.write('%s' % c.constrName)
                f.write('\n')
        f.close()


    #%% retrieve results
    #fpo: TODO: either delete result section completely or change it properly
    
    # grid results
    res_powerTrafoLoad = np.array([[powerTrafoLoad[d,t].X for t in timesteps]for d in days])
    res_powerTrafoInj = np.array([[powerTrafoInj[d,t].X for t in timesteps]for d in days])
        
    res_powerLine = {}
    for [n,m] in nodeLines:
        res_powerLine[n,m] = np.array([[powerLine[n,m,d,t].X for t in timesteps] for d in days])
    
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
    res_powerGen = {}
    res_powerInj = {}
    res_powerSubtr = {}

    res_powerInjPV = {}
    res_powerUsePV = {}
    res_powerPVChBat = {}
    res_powerNetLoad = {}
    res_powerUseBat = {}
    res_powerNetChBat = {}
    res_powerNetDisBat = {}

    
    for n in gridnodes:
        #res_powerLoad[n] = np.array([[powerLoad[n,d,t].X for t in timesteps] for d in days])
        #res_powerGen[n] = np.array([[powerGen[n,d,t].X for t in timesteps] for d in days])
        res_powerInj[n] = np.array([[powerInj[n,d,t].X for t in timesteps] for d in days])
        res_powerSubtr[n] = np.array([[powerSubtr[n,d,t].X for t in timesteps] for d in days])
        res_powerInjPV[n] = np.array([[powerInjPV[n,d,t].X for t in timesteps] for d in days])
        res_powerUsePV[n] = np.array([[powerUsePV[n,d,t].X for t in timesteps] for d in days])
        res_powerPVChBat[n] = np.array([[powerPVChBat[n,d,t].X for t in timesteps] for d in days])
        res_powerNetLoad[n] = np.array([[powerNetLoad[n,d,t].X for t in timesteps] for d in days])
        res_powerUseBat[n] = np.array([[powerUseBat[n,d,t].X for t in timesteps] for d in days])
        res_powerNetChBat[n] = np.array([[powerNetChBat[n,d,t].X for t in timesteps] for d in days])
        res_powerNetDisBat[n] = np.array([[powerNetDisBat[n,d,t].X for t in timesteps] for d in days])
    
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


# save results
    with open(options["filename_results"], "wb") as fout:
        pickle.dump(model.ObjVal, fout, pickle.HIGHEST_PROTOCOL)
        pickle.dump(model.Runtime, fout, pickle.HIGHEST_PROTOCOL)
        pickle.dump(model.MIPGap, fout, pickle.HIGHEST_PROTOCOL)

        pickle.dump(res_powerTrafoLoad, fout, pickle.HIGHEST_PROTOCOL)
        pickle.dump(res_powerTrafoInj, fout, pickle.HIGHEST_PROTOCOL)
        pickle.dump(res_powerLine, fout, pickle.HIGHEST_PROTOCOL)

        pickle.dump(res_capacity, fout, pickle.HIGHEST_PROTOCOL)
        pickle.dump(res_powerCh, fout, pickle.HIGHEST_PROTOCOL)
        pickle.dump(res_powerDis, fout, pickle.HIGHEST_PROTOCOL)
        pickle.dump(res_SOC, fout, pickle.HIGHEST_PROTOCOL)
        pickle.dump(res_SOC_init, fout, pickle.HIGHEST_PROTOCOL)

        pickle.dump(res_powerLoad, fout, pickle.HIGHEST_PROTOCOL)
        pickle.dump(res_powerGen, fout, pickle.HIGHEST_PROTOCOL)
        pickle.dump(res_powerInj, fout, pickle.HIGHEST_PROTOCOL)
        pickle.dump(res_powerSubtr, fout, pickle.HIGHEST_PROTOCOL)
        pickle.dump(res_powerInjPV, fout, pickle.HIGHEST_PROTOCOL)
        pickle.dump(res_powerUsePV, fout, pickle.HIGHEST_PROTOCOL)
        pickle.dump(res_powerPVChBat, fout, pickle.HIGHEST_PROTOCOL)
        pickle.dump(res_powerNetLoad, fout, pickle.HIGHEST_PROTOCOL)
        pickle.dump(res_powerUseBat, fout, pickle.HIGHEST_PROTOCOL)
        pickle.dump(res_powerNetChBat, fout, pickle.HIGHEST_PROTOCOL)
        pickle.dump(res_powerNetDisBat, fout, pickle.HIGHEST_PROTOCOL)

        pickle.dump(res_c_inv, fout, pickle.HIGHEST_PROTOCOL)
        pickle.dump(res_c_om, fout, pickle.HIGHEST_PROTOCOL)
        pickle.dump(res_c_dem, fout, pickle.HIGHEST_PROTOCOL)
        pickle.dump(res_c_fix, fout, pickle.HIGHEST_PROTOCOL)
        pickle.dump(res_rev, fout, pickle.HIGHEST_PROTOCOL)
        pickle.dump(res_c_dem_grid, fout, pickle.HIGHEST_PROTOCOL)
        pickle.dump(res_rev_grid, fout, pickle.HIGHEST_PROTOCOL)
        pickle.dump(res_c_node, fout, pickle.HIGHEST_PROTOCOL)
        pickle.dump(res_c_total_nodes, fout, pickle.HIGHEST_PROTOCOL)
        pickle.dump(res_c_total_grid, fout, pickle.HIGHEST_PROTOCOL)

        pickle.dump(res_emission_nodes, fout, pickle.HIGHEST_PROTOCOL)
        pickle.dump(res_emission_grid, fout, pickle.HIGHEST_PROTOCOL)

        pickle.dump(nodes, fout, pickle.HIGHEST_PROTOCOL)
        pickle.dump(res_actHP, fout, pickle.HIGHEST_PROTOCOL)
        pickle.dump(res_powerHP, fout, pickle.HIGHEST_PROTOCOL)
        pickle.dump(res_powerEH, fout, pickle.HIGHEST_PROTOCOL)
        pickle.dump(res_SOC_tes, fout, pickle.HIGHEST_PROTOCOL)
        pickle.dump(res_ch_tes, fout, pickle.HIGHEST_PROTOCOL)
        pickle.dump(res_dch_tes, fout, pickle.HIGHEST_PROTOCOL)
        pickle.dump(res_heatHP, fout, pickle.HIGHEST_PROTOCOL)
        pickle.dump(res_heatEH, fout, pickle.HIGHEST_PROTOCOL)


    

    # introduce retrieving variables to give to timeloop
    # divide by 1000 to convert from kW to MW
    powInjRet = np.array([[[((powerInj[n,d,t].X)/1000) for t in timesteps]for n in gridnodes]for d in days])
    powSubtrRet= np.array([[[((powerSubtr[n,d,t].X)/1000) for t in timesteps] for n in gridnodes] for d in days])



    print("optimization successfull")

    return (res_c_total_grid, res_emission_grid, timesteps, days, powInjRet, powSubtrRet, gridnodes)



