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

def compute(net, nodes, gridnodes, days, timesteps, eco, devs, clustered, params, options, constraint_apc, constraint_InjMin, constraint_SubtrMin,constraint_InjMax,constraint_SubtrMax, emissions_max, costs_max):
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

#%% set and calculate building energy system data, as well as load and injection profiles
    
    # PV efficiency
    eta_inverter = 0.97
    # compute generated power
    powerPV = options["P_pv"] *(devs["pv"]["area_mean"]/devs["pv"]["p_nom"]) * devs["pv"]["eta_el"] * eta_inverter * clustered["solar_irrad"]
    # define max powerPV for curtailment (clustered["solar_irrad"] is a value between 0 and 1)
    powerPVMax = options["P_pv"] * eta_inverter

    #extract modulation level of heatpump
    mod_lvl = devs["hp_air"]["mod_lvl"]
    # extract COP-table for given heat flow temperature
    cop = devs["hp_air"]["cop_w" + str(options["T_VL"])]

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

        res_powerHPNet = {}
        res_powerHPPV = {}
        res_powerHPBat = {}
        res_powerEHNet = {}
        res_powerEHPV = {}
        res_powerEHBat = {}

        if options ["dhw_electric"]:
            powerElec = clustered["electricity"] + clustered["dhw"] + res_powerHP + res_powerEH
        else:
            powerElec = clustered["electricity"] + res_powerHP + res_powerEH
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
        res_powerHPNet = {}
        res_powerHPPV = {}
        res_powerHPBat = {}
        res_powerEHNet = {}
        res_powerEHPV = {}
        res_powerEHBat = {}
        
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
    chBat_max = {}
    chBat_min = {}

    for n in gridnodes:
        if n in nodes["bat"]:
            capBat_max[n] = devs["bat"]["cap_max"]
            capBat_min[n] = devs["bat"]["cap_min"]
            chBat_max[n] = devs["bat"]["cap_max"]
            chBat_min[n] = devs["bat"]["cap_min"]

    # attach plug-in loads and PV generatrion to building nodes
    # assume that PV is the only generated electricity
    # TODO: do the same with EV loads!?
    powerLoad = {}
    powerGen = {}
    heatload = {}
    dhwload = {}
    for n in gridnodes:
        for t in timesteps:
            for d in days:
                if n in nodes["load"]:
                    heatload[n, d, t] = clustered["heat"][d, t]
                    dhwload[n, d, t] = clustered["dhw"][d, t]
                    powerLoad[n,d,t] = powerElec[d,t]
                    powerGen[n,d,t] = powerPV[d,t]
                else:
                    heatload[n, d, t] = np.zeros_like(clustered["heat"][d, t])
                    dhwload[n, d, t] = np.zeros_like(clustered["dhw"][d, t])
                    powerLoad[n,d,t] = np.zeros_like(powerElec[d,t])
                    powerGen[n,d,t] = np.zeros_like(powerPV[d,t])

    if(options["allow_apc_opti"] == False):
        powerGenReal = {}
        powerGenRealMax = {}
        powerGenCurt = {}
        for n in gridnodes:
            for t in timesteps:
                for d in days:
                    if n in nodes["load"]:
                        powerGenRealMax[n,d] = constraint_apc[n,d] * powerPVMax
                        if(powerGen[n,d,t] > powerGenRealMax[n,d]):
                            powerGenReal[n,d,t] = powerGenRealMax[n,d]
                            powerGenCurt[n,d,t] = powerGen[n,d,t] - powerGenReal[n,d,t]
                        else:
                            powerGenReal[n,d,t] = powerGen[n,d,t]
                            powerGenCurt[n,d,t] = 0
                    else:
                        powerGenRealMax[n,d] = 0
                        powerGenReal[n,d,t] = 0
                        powerGenCurt[n,d,t] = 0

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
    # x for existence of battery
    x_bat = model.addVars(gridnodes, vtype="B", name="bat_existance_" + str(n))
    # y to prevent simultaneous charge and discharge
    y_bat = model.addVars(gridnodes, days, timesteps, vtype="B", name="activation_charge_" + str(n) + str(t))
    capacity = model.addVars(gridnodes, vtype="C", name="Cap_"+str(n))
    SOC = model.addVars(gridnodes,days, timesteps, vtype="C", name="SOC_"+str(n)+str(d)+str(t))
    SOC_init = model.addVars(gridnodes, days, vtype="C", name="SOC_init_"+str(n)+str(d))
    powerCh = model.addVars(gridnodes,days,timesteps, vtype="C", name="powerCh_"+str(n)+str(t))
    powerDis = model.addVars(gridnodes,days,timesteps, vtype="C", name="powerDis_"+str(n)+str(t))


    if options["allow_apc_opti"]:
        #add Active Power Curailment variables to model
        apc_var = model.addVars(gridnodes, days, vtype="C", ub=1, name="apc_var"+str(n)+str(d))
        apc_total = model.addVars(gridnodes, days, vtype="C", ub=1, name="apc_total"+str(n)+str(d))
        powerGenRealMax = model.addVars(gridnodes, days, timesteps, vtype="C", name="powerGenRealMax"+str(n)+str(d))
        powerGenReal = model.addVars(gridnodes, days, timesteps, vtype="C", name="powerGenReal"+str(n)+str(d)+str(t))
        # introduce powerGenCurt, which tells, how much of powerGen got curtailed
        powerGenCurt = model.addVars(gridnodes, days, timesteps, vtype="C")

    # power flowing from net into complex of bat, house, pv and eh (hp seperated)
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


    if options["hp_mode"] == "grid_opt":
        # heatpump operation (important to realize modulation level)
        y_hp = model.addVars(gridnodes, days, timesteps, vtype="B", name="y_hp")
        # heatpump heat and power
        heat_hp = model.addVars(gridnodes, days, timesteps, vtype="C", name="Q_hp")
        power_hp = model.addVars(gridnodes, days, timesteps, vtype="C", name="P_hp")

        # electrical auxiliary heater
        heat_eh = model.addVars(gridnodes, days, timesteps, vtype="C", name="Q_eh")
        power_eh = model.addVars(gridnodes, days, timesteps, vtype="C", name="P_eh")

        # tes variables for charging, discharging, SoC and initial SoC per typeday
        ch_tes = model.addVars(gridnodes, days, timesteps, vtype="C", name="ch_tes")
        dch_tes = model.addVars(gridnodes, days, timesteps, vtype="C", name="dch_tes")
        soc_tes = model.addVars(gridnodes, days, timesteps, vtype="C", name="soc_tes")
        soc_init_tes = model.addVars(gridnodes, days, vtype="C", name="soc_init_tes")

        # heatpump auxilary variables for energy balances
        powerHPNet = model.addVars(gridnodes, days, timesteps, vtype="C", name="powerHPNet")
        powerHPPV = model.addVars(gridnodes, days, timesteps, vtype="C", name="powerHPPV")
        powerHPBat = model.addVars(gridnodes, days, timesteps, vtype="C", name="powerHPBat")

        powerEHNet = model.addVars(gridnodes, days, timesteps, vtype="C", name="powerEHNet")
        powerEHPV = model.addVars(gridnodes, days, timesteps, vtype="C", name="powerEHPV")
        powerEHBat = model.addVars(gridnodes, days, timesteps, vtype="C", name="powerEHBat")

    model.update()
    
    #%% define constraints
    
    #%% economical constraints
    #fpo: TODO: have to be worked through
    
    model.addConstr(c_total_nodes == c_inv.sum('*') + c_om.sum('*') + c_fix.sum('*')
                                     + c_dem.sum('*') - revenues.sum('*'), name="total_costs")
                                     
    
    model.addConstr(c_total_grid == c_inv.sum('*') + c_om.sum('*') + c_fix.sum('*')
                                    + c_dem_grid - revenues_grid, name="total_costs_grid")
                                     
         
    # compute annual investment costs per load node
    model.addConstrs((c_inv[n] == eco["crf"] * devs["bat"]["rval"] * (x_bat[n] * devs["bat"]["c_inv_fix"] + capacity[n] * devs["bat"]["c_inv_var"])
                        for n in gridnodes), name="investment_costs"+str(n))
    
    # compute annual operation and maintenance costs per load node
    model.addConstrs((c_om[n] == eco["b"]["infl"] * devs["bat"]["c_om_rel"] * c_inv[n]
                        for n in gridnodes), name="maintenance_costs"+str(n))

    if options["heatpump_seperated_costs"]:
        # compute annual fix costs for electricity per load node
        # approximated costs for "rundsteuerempfänger bei einspeisemanagement" at 700 Euros
        model.addConstrs((c_fix[n] ==900 + eco["el"]["el_hp"]["fix"][0] + eco["el"]["el_sta"]["fix"][0] for n in nodes["load"]), name="fix_costs"+str(n))
    else:
        model.addConstrs((c_fix[n] ==900 + eco["el"]["el_sta"]["fix"][0] for n in nodes["load"]), name="fix_costs" + str(n))


    # compute annual demand related costs load node
    Load_total_node = {}
    for n in gridnodes:
        Load_total_node[n] = (sum(clustered["weights"][d] * sum(powerSubtr[n,d,t]
                            for t in timesteps) for d in days) * dt)

    if options["hp_mode"] == "grid_opt":
        Load_HP_total_node = {}
        for n in gridnodes:
            Load_HP_total_node[n] = (sum(clustered["weights"][d] * sum(powerHPNet[n, d, t]
                                                                       for t in timesteps) for d in days) * dt)

        if options["static_prices"]:
            model.addConstrs((c_dem[n] == sum(sum(((powerHPNet[n, d, t] + powerSubtr[n,d,t]) * clustered["elcost_stat"][d,t] for t in timesteps)
                              * clustered["weights"][d]) for d in days) * dt) for n in gridnodes)
        else:
            model.addConstrs(c_dem[n] == (sum((sum((powerHPNet[n,d,t] + powerSubtr[n,d,t]) * clustered["elcost_dyn"][d,t] for t in timesteps) * clustered["weights"][d]) for d in days)
                             * dt ) for n in gridnodes)

        """if options["heatpump_seperated_costs"]:

            model.addConstrs((c_dem[n] == eco["crf"] * eco["b"]["el"] * Load_total_node[n] * eco["el"]["el_sta"]["var"][0]
                                + eco["crf"] * eco["b"]["el"] * Load_HP_total_node[n] * eco["el"]["el_hp"]["var"][0]
                                for n in gridnodes), name="demand_costs"+str(n))
        else:

            model.addConstrs((c_dem[n] == eco["crf"] * eco["b"]["el"] * (Load_total_node[n] + Load_HP_total_node[n]) * eco["el"]["el_sta"]["var"][0]
                              for n in gridnodes), name="demand_costs" + str(n))"""
    else:
        if options["static_prices"]:
            model.addConstrs((c_dem[n] == sum(sum((powerHPNet[n, d, t] + powerSubtr[n, d, t])* clustered["elcost_stat"][d,t] for t in timesteps) * clustered["weights"][d]
                              for d in days) * dt) for n in gridnodes)
        else:
            model.addConstrs(c_dem[n] == ((sum(sum((powerHPNet[n, d, t] + powerSubtr[n, d, t]) * clustered["elcost_dyn"][d,t] for t in timesteps)
                                ) * clustered["weights"][d] for d in days)  * dt) for n in gridnodes)


    #fpo: TODO: for future work you might want to implement the different heatpump power costs for the whole grid as well!
    # compute annual demand related costs per grid
    Load_total_grid = (sum(clustered["weights"][d] * sum(powerTrafoLoad[d,t]
                            for t in timesteps) for d in days) * dt)
        
    model.addConstr(c_dem_grid == 
                     eco["crf"] * eco["b"]["el"] * Load_total_grid * eco["el"]["el_sta"]["var"][0], 
                     name="demand_costs_grid")
    
    # compute annual revenues for electricity feed-in per node
    # here: it's assumed that revenues are generated only for PV power and curtailed generation
    InjPV_total_node = {}
    for n in gridnodes:
        InjPV_total_node[n] = (sum(clustered["weights"][d] * sum(powerInjPV[n,d,t]
                            for t in timesteps) for d in days) * dt)
    
    Inj_total_node = {}
    for n in gridnodes:
        Inj_total_node[n] = (sum(clustered["weights"][d] * sum(powerInj[n,d,t]
                            for t in timesteps) for d in days) * dt)

    Curt_total_node = {}
    for n in gridnodes:
        Curt_total_node[n] = (sum(clustered["weights"][d] * sum(powerGenCurt[n,d,t]
                            for t in timesteps) for d in days) * dt)

    #fpo: TODO: this is missing the possibility of buying electricity and selling it at another time.
    if options["rev_price_manner"] == "eeg":
        model.addConstrs((revenues[n] == eco["crf"] * eco["b"]["infl"] * (InjPV_total_node[n] + Curt_total_node[n]) * eco["price_sell_eeg"]
                        for n in gridnodes), name="revenues"+str(n))
    elif options["rev_price_manner"] == "real":
        if options["static_prices"]:
            model.addConstrs((revenues[n] == (InjPV_total_node[n] + Curt_total_node[n]) *
                              clustered["elcost_stat"][d,t] for n in gridnodes), name="revenues" + str(n))
        else:
            model.addConstrs(revenues[n] == (sum(sum((powerInjPV[n,d,t] + powerGenCurt[n,d,t]) * clustered["elcost_dyn"][d,t] for t in timesteps) * clustered["weights"][d] for d in days) * dt) for n in gridnodes)
    else:
        print("ERROR! CHECK OPTIONS FOR POWER PRICE!")

    
    # compute annual revenues for electricity feed-in per node
    # here: it's assumed that revenues are generated for all injections to the higher level grid
    # plus the curtailed energy of every node needs to be considered aswell
    Inj_total_grid = (sum(clustered["weights"][d] * sum(powerTrafoInj[d,t]
                            for t in timesteps) for d in days) * dt)
    Curt_total_grid = (sum(Curt_total_node[n] for n in gridnodes) * dt)

    #fpo: TODO: price sell eeg has to be changed aswell, if possibility of charging bat out of net is open
    model.addConstr(revenues_grid == 
                     eco["crf"] * eco["b"]["infl"] * (Inj_total_grid + Curt_total_grid)* eco["price_sell_eeg"],
                     name="revenues_grid")
    
    #%% ecological constraints


    if options["static_emissions"]:
        # compute annual emissions and emission revenues
        # for single nodes
        emissions_Load_nodes = {}
        emissions_Inj_nodes = {}
        for n in gridnodes:
            if options["hp_mode"] == "grid_opt":
                emissions_Load_nodes[n] = (sum(clustered["weights"][d] * sum(((powerHPNet[n,d,t] +powerSubtr[n,d,t]) *clustered["co2_stat"][d,t])
                                            for t in timesteps) for d in days) * dt)
            else:
                emissions_Load_nodes[n] = (sum(clustered["weights"][d] * sum((powerSubtr[n, d, t] * clustered["co2_stat"][d, t])
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
            if options["hp_mode"] == "grid_opt":
                emissions_Load_nodes[n] = (sum(clustered["weights"][d] * sum(((powerHPNet[n,d,t] +powerSubtr[n,d,t]) *clustered["co2_dyn"][d,t])
                                            for t in timesteps) for d in days) * dt)
            else:
                emissions_Load_nodes[n] = (sum(clustered["weights"][d] * sum((powerSubtr[n, d, t] * clustered["co2_dyn"][d, t])
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
    model.addConstrs((powerTrafoLoad[d,t] <= (yTrafo[d,t] * trafo_max) for d in days for t in timesteps),
                     name="trafoLoad_activation")
    model.addConstrs((powerTrafoInj[d,t] <= ((1 - yTrafo[d,t]) * trafo_max) for d in days for t in timesteps),
                     name="trafoInj_activation")

    if options["allow_apc_opti"]:
        #apc constraints
        for n in gridnodes:
            for d in days:
                model.addConstr(apc_total[n, d] == apc_var[n, d] + constraint_apc[n, d],name="total_apc_constr" + str(n) + str(d))
                for t in timesteps:
                    model.addConstr(powerGenRealMax[n, d, t] == (1 - apc_total[n, d]) * powerPVMax)
                    model.addConstr(powerGenReal[n, d, t] <= powerGen[n, d, t])
                    model.addConstr(powerGenReal[n,d,t] <= powerGenRealMax[n,d,t])
                    model.addConstr(powerGenCurt[n,d,t] <= powerGen[n,d,t] - powerGenReal[n,d,t])

    
    # set energy balance for all nodes

    if options["hp_mode"] == "grid_opt":
        for n in gridnodes:
            for d in days:
                for t in timesteps:
                    if n in nodes["trafo"]:
                        # energy balance around trafo
                        model.addConstr(powerLine.sum(n, '*', d, t) - powerLine.sum('*', n, d, t) ==
                                        powerTrafoLoad[d, t] - powerTrafoInj[d, t], name="node_balance_trafo_" + str(n))

                        model.addConstr(powerInj[n, d, t] == 0)
                        model.addConstr(powerSubtr[n, d, t] == 0)
                        model.addConstr(powerHPNet[n,d,t] == 0)

                    elif n in nodes["load"]:
                        # energy balance node with net
                        model.addConstr(powerLine.sum(n, '*', d, t) - powerLine.sum('*', n, d, t) ==
                                        powerSubtr[n, d, t] - powerInj[n, d, t] + powerHPNet[n, d, t],
                                        name="node_balance net_" + str(n))
                    else:
                        # energy balance node with net
                        model.addConstr(powerLine.sum(n, '*', d, t) - powerLine.sum('*', n, d, t) == 0,
                                        name="node_balance net_" + str(n))
                        model.addConstr(powerInj[n, d, t] == 0)
                        model.addConstr(powerSubtr[n, d, t] == 0)
                        model.addConstr(powerHPNet[n, d, t] == 0)




    else:
        for n in gridnodes:
            for d in days:
                for t in timesteps:
                    if n in nodes["trafo"]:
                        #energy balance around trafo
                        model.addConstr(powerLine.sum(n,'*',d,t) - powerLine.sum('*',n,d,t) ==
                                        powerTrafoLoad[d,t] - powerTrafoInj[d,t], name="node_balance_trafo_"+str(n))

                        model.addConstr(powerInj[n,d,t] == 0)
                        model.addConstr(powerSubtr[n,d,t] == 0)


                    elif n in nodes["load"]:

                        # energy balance node with net
                        model.addConstr(powerLine.sum(n, '*', d, t) - powerLine.sum('*', n, d, t) ==
                                        powerSubtr[n, d, t] - powerInj[n, d, t],
                                        name="node_balance net_" + str(n))

                    else:

                        # energy balance node with net
                        model.addConstr(powerLine.sum(n, '*', d, t) - powerLine.sum('*', n, d, t) == 0,
                                        name="node_balance net_" + str(n))

                        model.addConstr(powerInj[n, d, t] == 0)
                        model.addConstr(powerSubtr[n, d, t] == 0)

    # set line limits
    # TODO: check if it's better to set line limits like this or to set lb/ub of variable to min/max values      
    for [n,m] in nodeLines:
            for d in days:
                for t in timesteps:
                    model.addConstr(powerLine[n,m,d,t] <= powerLine_max[n,m], name="line_power_max_"+str(n)+str(m)+str(t))
                    model.addConstr(powerLine[n,m,d,t] >= (-1)*powerLine_max[n,m], name="line_power_min_"+str(n)+str(m)+str(t))
                
                
    #%% battery constraints

    # Battery can be switched on only if it has been purchased
    for n in gridnodes:
        model.addConstr(params["time_steps"] * params["days"] * x_bat[n] >= sum(sum(y_bat[n, d, t] for t in timesteps) for d in days), name="Activation_bat"+str(n)+str(d)+str(t))

    # maximum power is defined by power/capacity ratio
    # energy balance for bat
    for n in gridnodes:
        for d in days:                   
            for t in timesteps:
                if n in nodes["bat"]:

                    # prevent batteries from simultaneously charging and discharging (yBat integrated)
                    model.addConstr(powerCh[n,d,t]  <= (devs["bat"]["P_ch_fix"] + capacity[n]*devs["bat"]["P_ch_var"]), name="max_power_"+str(n)+str(t))
                    model.addConstr(powerCh[n,d,t] <= y_bat[n,d,t] * capBat_max[n], name= "bigM_power_"+str(n) +str(d)+str(t))
                    model.addConstr(powerDis[n,d,t] <= (devs["bat"]["P_dch_fix"] + capacity[n]*devs["bat"]["P_dch_var"]), name="max_power_"+str(n)+str(t))
                    model.addConstr(powerDis[n, d, t] <= (1 - y_bat[n, d, t]) * capBat_max[n], name="bigM_power_" + str(n) + str(d) + str(t))
    
    # set limitations for battery capacity
    for n in gridnodes:
        if n in nodes["bat"]:
    
            model.addConstr(capacity[n] <= x_bat[n] * capBat_max[n], name="Battery_capacity_max")
            model.addConstr(capacity[n] >= x_bat[n] * capBat_min[n], name="Battery_capacity_min")
            
            model.addConstrs((capacity[n] >= SOC_init[n,d] for d in days), name="Battery_capacity_SOC_init")
            model.addConstrs((capacity[n] >= SOC[n,d,t] for t in timesteps for d in days), name="Battery_capacity_SOC")
            model.addConstrs((SOC[n,d,t] >= 0.05 * capacity[n] for t in timesteps for d in days), name="Battery_SOC_min")

                    
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

    # %% heat pump and tes constraints

    if options["hp_mode"] == "grid_opt":
        # heatpump power and heat
        model.addConstrs((power_hp[n, d, t] <= (y_hp[n, d, t] * capa_hp) for n in gridnodes for d in days for t in
             timesteps),name="Max_power_hp")

        model.addConstrs((power_hp[n, d, t] >= (y_hp[n, d, t] * capa_hp * mod_lvl) for n in gridnodes for d in days
                          for t in timesteps), name="Min_power_hp")

        model.addConstrs((power_hp[n, d, t] == heat_hp[n, d, t] / cop[d, t] for n in gridnodes for d in days
                          for t in timesteps),name="heat_power_coupling")

        # electric heater power and heat
        model.addConstrs((power_eh[n, d, t] == heat_eh[n, d, t] / devs["eh"]["eta"] for n in gridnodes for d in days
                          for t in timesteps), name="heat_power_eh")

        # tes state of charge
        # initial SOC per typeday
        model.addConstrs((soc_init_tes[n, d] <= capa_tes for n in gridnodes for d in days), name="SOC_init_tes")
        # SOC limit for every timestep
        model.addConstrs((soc_tes[n, d, t] <= capa_tes for n in gridnodes for d in days for t in timesteps),
                         name="SOC_tes")
        # SOC repetitions >> SOC at the end of the day has to be SOC at the beginning of this day
        if np.max(clustered["weights"]) > 1:
            model.addConstrs((soc_init_tes[n, d] == soc_tes[n, d, params["time_steps"] - 1] for n in gridnodes
                              for d in days), name="repetitions_tes")

        # k_loss = devs["tes"]["k_loss"]
        k_loss = options["k_loss"]

        for n in gridnodes:
            for d in days:
                for t in timesteps:

                    if t == 0:
                        soc_prev = soc_init_tes[n, d]
                    else:
                        soc_prev = soc_tes[n, d, t - 1]
                    model.addConstr((soc_tes[n, d, t] == (1 - k_loss) * soc_prev + dt *
                                     (ch_tes[n, d, t] * devs["tes"]["eta_ch"] - dch_tes[n, d, t] / devs["tes"]["eta_dch"])),name="Storage_balance_tes")



        model.addConstrs((ch_tes[n, d, t] == (heat_hp[n, d, t] + heat_eh[n, d, t]) for n in gridnodes for d in days
                          for t in timesteps), name="Thermal_max_charge_tes")

        # differentiation for dhw-heating: either electric or via heating system
        if options["dhw_electric"]:
            model.addConstrs((dch_tes[n, d, t] == heatload[n, d, t] for n in gridnodes for d in days for t in timesteps),
                name="Thermal_max_discharge_tes")
        else:
            model.addConstrs((dch_tes[n, d, t] == (heatload[n, d, t] + dhwload[n, d, t]) for n in gridnodes for d in days
                            for t in timesteps), name="Thermal_max_discharge_tes")




    #%% energy balances for every node

    if options["hp_mode"] == "grid_opt":

        # powerHPNet kürzt sich aus der nachfolgenden Gleichung raus
        """# energy balance node with pv-bat-house-hp-eh complex
        model.addConstrs((powerInj[n, d, t] - powerSubtr[n, d, t] ==
                        powerInjPV[n, d, t] + powerNetDisBat[n, d, t] - powerNetLoad[n, d, t] -
                        powerNetChBat[n, d, t] - powerEHNet[n, d, t] for n in gridnodes for d in days for t in timesteps) , name="node_balance_complex_" + str(n))
"""
        # define powerInj and powerSubtr
        model.addConstrs((powerInj[n, d, t] ==
                        powerInjPV[n, d, t] + powerNetDisBat[n, d, t] for n in gridnodes for d in days for t in timesteps),
                        name="balance_powerInj_" + str(n))
        # auch hier kürzt sich powerHPNet raus!
        model.addConstrs((powerSubtr[n, d, t] ==
                        powerNetLoad[n, d, t] + powerNetChBat[n, d, t] + powerEHNet[n, d, t] for n in gridnodes for d in days for t in timesteps),
                        name="balance_powerSubtr_" + str(n))

        # split power from heatpump
        model.addConstrs((power_hp[n, d, t] == powerHPNet[n, d, t] + powerHPPV[n, d, t] + powerHPBat[n, d, t]
                          for n in gridnodes for d in days for t in timesteps),
                         name="powerInj_UseBat" + str(n) + str(d) + str(t))

        # split power from electric heater
        model.addConstrs((power_eh[n, d, t] == powerEHNet[n, d, t] + powerEHPV[n, d, t] + powerEHBat[n, d, t]
                          for n in gridnodes for d in days for t in timesteps),
                         name="powerInj_UseBat" + str(n) + str(d) + str(t))

        # energy balance PV device
        model.addConstrs((-powerUsePV[n, d, t] - powerEHPV[n, d, t] - powerHPPV[n, d, t] ==
                        -powerGenReal[n, d, t] + powerInjPV[n, d, t] + powerPVChBat[n, d, t] for n in gridnodes for d in days for t in timesteps),
                        name="node_balance_pv_" + str(n) + str(d) + str(t))

        # energy balance load
        model.addConstrs((-powerUsePV[n, d, t] ==
                        -powerLoad[n, d, t] + powerUseBat[n, d, t] + powerNetLoad[n, d, t] for n in gridnodes for d in days for t in timesteps),
                        name="node_balance_load" + str(n) + str(d) + str(t))

        # energy balance bat
        model.addConstrs((powerDis[n, d, t] ==
                        powerNetDisBat[n, d, t] + powerUseBat[n, d, t] for n in gridnodes for d in days for t in timesteps),
                        name="energy_balance_bat_" + str(n) + str(d) + str(t))
        model.addConstrs((powerCh[n, d, t] ==
                        powerNetChBat[n, d, t] + powerPVChBat[n, d, t] for n in gridnodes for d in days for t in timesteps),
                        name="energy_balance_bat_" + str(n) + str(d) + str(t))


    else:

        """# energy balance node with pv-bat-house complex
        model.addConstrs((powerInj[n, d, t] - powerSubtr[n, d, t] ==
                        powerInjPV[n, d, t] + powerNetDisBat[n, d, t] - powerNetLoad[n, d, t] -
                        powerNetChBat[n, d, t] for n in gridnodes for d in days for t in timesteps), name="node_balance_complex_" + str(n))"""
        # define powerInj and powerSubtr
        model.addConstrs((powerInj[n, d, t] ==
                        powerInjPV[n, d, t] + powerNetDisBat[n, d, t] for n in gridnodes for d in days for t in timesteps), name="balance_powerInj_" + str(n))
        model.addConstrs((powerSubtr[n, d, t] ==
                        powerNetLoad[n, d, t] + powerNetChBat[n, d, t] for n in gridnodes for d in days for t in timesteps), name="balance_powerSubtr_" + str(n))
        # energy balance PV device
        model.addConstrs((-powerUsePV[n, d, t]==
                         -powerGenReal[n, d, t] + powerInjPV[n, d, t] + powerPVChBat[n, d, t] for n in gridnodes for d in days for t in timesteps),
                        name="node_balance_pv_" + str(n) + str(d) + str(t))
        # energy balance load
        model.addConstrs((-powerUsePV[n, d, t] ==
                         -powerLoad[n, d, t] + powerUseBat[n, d, t] + powerNetLoad[n, d, t] for n in gridnodes for d in days for t in timesteps),
                        name="node_balance_load" + str(n) + str(d) + str(t))

        # energy balance bat
        model.addConstrs((powerDis[n, d, t] ==
                        powerNetDisBat[n, d, t] + powerUseBat[n, d, t] for n in gridnodes for d in days for t in timesteps), name="energy_balance_bat_" + str(n) + str(d) + str(t))
        model.addConstrs((powerCh[n,d,t] ==
                        powerNetChBat[n,d,t] + powerPVChBat[n,d,t] for n in gridnodes for d in days for t in timesteps), name="energy_balance_bat_" + str(n) + str(d) + str(t))

    # cutting Injection and Subtraction with constraints
    # for first optimization, both constraints are set to 10000
    for n in gridnodes:
        for d in days:
            for t in timesteps:
                model.addConstr(powerInj[n,d,t] >= constraint_InjMin[n,d,t])
                model.addConstr(powerInj[n,d,t] <= constraint_InjMax[n,d,t])
                if options["hp_mode"] == "grid_opt":
                    model.addConstr(powerSubtr[n,d,t] + powerHPNet[n,d,t] <= constraint_SubtrMax[n,d,t])
                    model.addConstr(powerSubtr[n,d,t] + powerHPNet[n,d,t] >= constraint_SubtrMin[n,d,t])
                else:
                    model.addConstr(powerSubtr[n, d, t] <= constraint_SubtrMax[n, d, t])
                    model.addConstr(powerSubtr[n,d,t] >= constraint_SubtrMin[n,d,t])

    if options["run_which_opti"] == "pareto_oNB" or "pareto_mNB":
        # introduce constraints for pareto evaluation
        model.addConstr(c_total_nodes <= costs_max)
        model.addConstr(sum(emission_nodes[n] for n in gridnodes) <= emissions_max)

    #%% start optimization
    
    # set objective function
    #ATTENTION: PLEASE ONLY OPTIMIZE AFTER SUM OF NODES; NOT AFTER GRID
    if options["opt_costs"]:
        model.setObjective(c_total_nodes, gp.GRB.MINIMIZE)
        #model.setObjective(c_total_grid, gp.GRB.MINIMIZE)
    elif options["opt_emissions"]:
        model.setObjective(sum((emission_nodes[n]) for n in gridnodes), gp.GRB.MINIMIZE)
        #model.setObjective(emission_grid, gp.GRB.MINIMIZE)
    else:
        model.setObjective(
            sum(sum(sum((powerSubtr[n, d, t] - powerInj[n, d, t]) for n in gridnodes) * clustered["co2_stat"][d, t]
                    for t in timesteps) for d in days), gp.GRB.MINIMIZE)

    # adgust gurobi settings

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
    res_yTrafo = np.array([[yTrafo[d,t].X for t in timesteps]for d in days])
        
    """res_powerLine = {}
    for [n,m] in nodeLines:
        res_powerLine[n,m] = np.array([[powerLine[n,m,d,t].X for t in timesteps] for d in days])"""
    res_powerLine = np.array([[[powerLine[n, m, d, t].X for t in timesteps] for d in days] for [n,m] in nodeLines])
    
    # battery operation results
    res_capacity = {}
    res_powerCh = {}
    res_powerDis = {}
    res_SOC = {}
    res_SOC_init = {}

    """for n in gridnodes:
        res_capacity = np.array([capacity[n].X for n in gridnodes])
        res_powerCh[n] = np.array([[powerCh[n,d,t].X for t in timesteps] for d in days])
        res_powerDis[n] = np.array([[powerDis[n,d,t].X for t in timesteps] for d in days])
        res_SOC[n] = np.array([[SOC[n,d,t].X for t in timesteps] for d in days])
        res_SOC_init[n] = np.array([SOC_init[n,d].X for d in days])"""

    res_capacity = np.array([capacity[n].X for n in gridnodes])
    res_powerCh = np.array([[[powerCh[n, d, t].X for t in timesteps] for d in days] for n in gridnodes])
    res_powerDis = np.array([[[powerDis[n, d, t].X for t in timesteps] for d in days] for n in gridnodes])
    res_SOC = np.array([[[SOC[n, d, t].X for t in timesteps] for d in days] for n in gridnodes])
    res_SOC_init = np.array([[SOC_init[n, d].X for d in days] for n in gridnodes])
    res_constraint_InjMin = np.array([[[constraint_InjMin[n, d, t] for t in timesteps] for d in days] for n in gridnodes])
    res_constraint_SubtrMin = np.array([[[constraint_SubtrMin[n, d, t] for t in timesteps] for d in days] for n in gridnodes])
    res_constraint_InjMax = np.array([[[constraint_InjMax[n, d, t] for t in timesteps] for d in days] for n in gridnodes])
    res_constraint_SubtrMax = np.array([[[constraint_SubtrMax[n, d, t] for t in timesteps] for d in days] for n in gridnodes])

    # retrieve apc results
    if options["allow_apc_opti"]:
        res_apc_var = np.array([[apc_var[n, d].X for d in days] for n in gridnodes])
        res_apc_total = np.array([[apc_total[n, d].X for d in days]for n in gridnodes])
        res_constraint_apc = np.array([[constraint_apc[n, d] for d in days]for n in gridnodes])
        res_powerGenRealMax = np.array([[[powerGenRealMax[n, d, t].X for d in days] for t in timesteps]for n in gridnodes])
        res_powerGenReal = np.array([[[powerGenReal[n, d, t].X for t in timesteps] for d in days]for n in gridnodes])
        res_powerGenCurt = np.array([[[powerGenCurt[n, d, t].X for t in timesteps] for d in days]for n in gridnodes])


    else:
        res_constraint_apc = np.array([[constraint_apc[n, d] for d in days] for n in gridnodes])
        res_powerGenRealMax = np.array([[powerGenRealMax[n, d] for d in days]for n in gridnodes])
        res_powerGenReal = np.array([[[powerGenReal[n, d, t] for t in timesteps] for d in days]for n in gridnodes])
        res_powerGenCurt = np.array([[[powerGenCurt[n,d,t]for t in timesteps] for d in days]for n in gridnodes])


    # node energy management results

    res_powerInj = np.array([[[powerInj[n, d, t].X for t in timesteps] for d in days]for n in gridnodes])
    res_powerSubtr = np.array([[[powerSubtr[n, d, t].X for t in timesteps] for d in days]for n in gridnodes])
    if options["hp_mode"] == "grid_opt":
        res_powerSubtrTotal= np.array([[[(((powerSubtr[n,d,t].X) + (powerHPNet[n,d,t].X))) for t in timesteps] for d in days] for n in gridnodes])
    else:
        res_powerSubtrTotal = np.array([[[((powerSubtr[n, d, t].X)) for t in timesteps] for d in days] for n in gridnodes])
    res_powerInjPV = np.array([[[powerInjPV[n, d, t].X for t in timesteps] for d in days]for n in gridnodes])
    res_powerUsePV = np.array([[[powerUsePV[n, d, t].X for t in timesteps] for d in days]for n in gridnodes])
    res_powerPVChBat = np.array([[[powerPVChBat[n, d, t].X for t in timesteps] for d in days]for n in gridnodes])
    res_powerNetLoad = np.array([[[powerNetLoad[n, d, t].X for t in timesteps] for d in days]for n in gridnodes])
    res_powerUseBat = np.array([[[powerUseBat[n, d, t].X for t in timesteps] for d in days]for n in gridnodes])
    res_powerNetChBat = np.array([[[powerNetChBat[n, d, t].X for t in timesteps] for d in days]for n in gridnodes])
    res_powerNetDisBat = np.array([[[powerNetDisBat[n, d, t].X for t in timesteps] for d in days]for n in gridnodes])

    # retrieve results for energy hp and eh
    if options["hp_mode"] == "grid_opt":

        res_actHP = np.array([[[y_hp[n, d, t].X for t in timesteps] for d in days]for n in gridnodes])
        res_powerHP = np.array([[[power_hp[n, d, t].X for t in timesteps] for d in days]for n in gridnodes])
        res_powerEH = np.array([[[power_eh[n, d, t].X for t in timesteps] for d in days]for n in gridnodes])
        res_heatHP = np.array([[[heat_hp[n, d, t].X for t in timesteps] for d in days]for n in gridnodes])
        res_heatEH = np.array([[[heat_eh[n, d, t].X for t in timesteps] for d in days]for n in gridnodes])

        res_SOC_tes = np.array([[[soc_tes[n, d, t].X for t in timesteps] for d in days]for n in gridnodes])
        res_SOC_init_tes = np.array([[soc_init_tes[n, d].X for d in days]for n in gridnodes])
        res_ch_tes = np.array([[[ch_tes[n, d, t].X for t in timesteps] for d in days]for n in gridnodes])
        res_dch_tes = np.array([[[dch_tes[n, d, t].X for t in timesteps] for d in days]for n in gridnodes])

        res_powerHPNet = np.array([[[powerHPNet[n, d, t].X for t in timesteps] for d in days]for n in gridnodes])
        res_powerHPPV = np.array([[[powerHPPV[n, d, t].X for t in timesteps] for d in days]for n in gridnodes])
        res_powerHPBat = np.array([[[powerHPBat[n, d, t].X for t in timesteps] for d in days]for n in gridnodes])
        res_powerEHNet = np.array([[[powerEHNet[n, d, t].X for t in timesteps] for d in days]for n in gridnodes])
        res_powerEHPV = np.array([[[powerEHPV[n, d, t].X for t in timesteps] for d in days]for n in gridnodes])
        res_powerEHBat = np.array([[[powerEHBat[n, d, t].X for t in timesteps] for d in days]for n in gridnodes])

    else:
        pass

    res_exBat = {}
    res_actBat = {}


    res_exBat = np.array([x_bat[n].X for n in gridnodes])
    res_actBat = np.array([[[y_bat[n, d, t].X for t in timesteps] for d in days]for n in gridnodes])

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
        pickle.dump(res_yTrafo, fout, pickle.HIGHEST_PROTOCOL)
        pickle.dump(res_powerLine, fout, pickle.HIGHEST_PROTOCOL)

        pickle.dump(res_capacity, fout, pickle.HIGHEST_PROTOCOL)
        pickle.dump(res_powerCh, fout, pickle.HIGHEST_PROTOCOL)
        pickle.dump(res_powerDis, fout, pickle.HIGHEST_PROTOCOL)
        pickle.dump(res_SOC, fout, pickle.HIGHEST_PROTOCOL)
        pickle.dump(res_SOC_init, fout, pickle.HIGHEST_PROTOCOL)
        pickle.dump(res_constraint_InjMin, fout, pickle.HIGHEST_PROTOCOL)
        pickle.dump(res_constraint_SubtrMin, fout, pickle.HIGHEST_PROTOCOL)
        pickle.dump(res_constraint_InjMax, fout, pickle.HIGHEST_PROTOCOL)
        pickle.dump(res_constraint_SubtrMax, fout, pickle.HIGHEST_PROTOCOL)
        pickle.dump(res_exBat, fout, pickle.HIGHEST_PROTOCOL)
        pickle.dump(res_actBat, fout, pickle.HIGHEST_PROTOCOL)

        pickle.dump(res_powerInj, fout, pickle.HIGHEST_PROTOCOL)
        pickle.dump(res_powerSubtr, fout, pickle.HIGHEST_PROTOCOL)
        pickle.dump(res_powerSubtrTotal, fout, pickle.HIGHEST_PROTOCOL)
        pickle.dump(res_powerInjPV, fout, pickle.HIGHEST_PROTOCOL)
        pickle.dump(res_powerUsePV, fout, pickle.HIGHEST_PROTOCOL)
        pickle.dump(res_powerPVChBat, fout, pickle.HIGHEST_PROTOCOL)
        pickle.dump(res_powerNetLoad, fout, pickle.HIGHEST_PROTOCOL)
        pickle.dump(res_powerUseBat, fout, pickle.HIGHEST_PROTOCOL)
        pickle.dump(res_powerNetChBat, fout, pickle.HIGHEST_PROTOCOL)
        pickle.dump(res_powerNetDisBat, fout, pickle.HIGHEST_PROTOCOL)

        if options["allow_apc_opti"]:
            pickle.dump(res_apc_var, fout, pickle.HIGHEST_PROTOCOL)
            pickle.dump(res_apc_total, fout, pickle.HIGHEST_PROTOCOL)
        pickle.dump(res_constraint_apc, fout, pickle.HIGHEST_PROTOCOL)
        pickle.dump(res_powerGenRealMax, fout, pickle.HIGHEST_PROTOCOL)
        pickle.dump(res_powerGenReal, fout, pickle.HIGHEST_PROTOCOL)
        pickle.dump(res_powerGenCurt, fout, pickle.HIGHEST_PROTOCOL)

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
        pickle.dump(res_powerHPNet, fout, pickle.HIGHEST_PROTOCOL)
        pickle.dump(res_powerHPPV, fout, pickle.HIGHEST_PROTOCOL)
        pickle.dump(res_powerHPBat, fout, pickle.HIGHEST_PROTOCOL)
        pickle.dump(res_powerEHNet, fout, pickle.HIGHEST_PROTOCOL)
        pickle.dump(res_powerEHPV, fout, pickle.HIGHEST_PROTOCOL)
        pickle.dump(res_powerEHBat, fout, pickle.HIGHEST_PROTOCOL)


    # introduce retrieving variables to give to timeloop
    # divide by 1000 to convert from kW to MW
    if options["hp_mode"] == "grid_opt":
        powSubtrRet= np.array([[[(((powerSubtr[n,d,t].X) + (powerHPNet[n,d,t].X))/1000) for t in timesteps] for n in gridnodes] for d in days])
    else:
        powSubtrRet = np.array([[[((powerSubtr[n, d, t].X)/1000) for t in timesteps] for n in gridnodes] for d in days])
    powInjRet = np.array([[[((powerInj[n, d, t].X)/1000) for t in timesteps] for n in gridnodes] for d in days])


    emissions_added = sum(res_emission_nodes)
    costs_added = res_c_total_nodes


    print("optimization successfull")

    return (res_c_total_grid, res_emission_grid, timesteps, days, powInjRet, powSubtrRet, gridnodes, res_exBat, res_powerInj, res_powerSubtrTotal, res_emission_nodes, res_c_total_nodes, emissions_added)