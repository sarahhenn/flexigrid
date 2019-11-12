# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 08:36:33 2019

@author: she
"""

from __future__ import division

import gurobipy as gp
import numpy as np

def optimize(options, params, clustered, devs, capa_hp, capa_tes):
    """
    Parameters
    ----------
    options: dict
        - dhw electric: true if decentral electric dhw supply
        - T_VL: 35 or 55 degree
        
    params : dict
        - c_w : heat capacity of water
        - days : quantity of clustered days
        - dt : time step length (h)
        - rho_w : density of water
        - time_steps : time steps per day

    clustered : dict
        - heat: Heat load profile
        - dhw : Domestic hot water load profile
        - solar_roof : solar irradiation on rooftop
        - temperature : Ambient temperature
        - weights : Weight factors from the clustering algorithm
        
    devs : dict
        - various devices
        - time resolved COP-table !
        
    capa_hp: float
        - nominal electrical heat pump capacity in kW
        
    capa_tes: float
        - nominal thermal tes capacity in kWh

    """
    # extract parameters
    dt          = params["dt"]
    timesteps   = [i for i in range(params["time_steps"])]
    days        = [i for i in range(params["days"])]   
    
    # extract modulation level of heatpump
    mod_lvl = devs["hp_air"]["mod_lvl"]
    # extract COP-table for given heat flow temperature
    cop = devs["hp_air"]["cop_w"+str(options["T_VL"])]
    
    #%% optimization model

    print("start modelling heat pump operation")
    model = gp.Model("HP energy operation optimization")
    
    #%% technical variables
    
    # heatpump operation (important to realize modulation level)
    y_hp = model.addVars(days, timesteps, vtype="B",  name="y_hp")
    # heatpump heat and power
    heat_hp = model.addVars(days, timesteps, vtype="C",  name="Q_hp")
    power_hp = model.addVars(days, timesteps, vtype="C", name="P_hp")
    
    # electrical auxiliary heater
    heat_eh = model.addVars(days, timesteps, vtype="C",  name="Q_eh")
    power_eh = model.addVars(days, timesteps, vtype="C", name="P_eh")
    
    # tes variables for charging, discharging, SoC and initial SoC per typeday
    ch = model.addVars(days, timesteps, vtype="C", name="ch_tes")
    dch = model.addVars(days, timesteps,vtype="C", name="dch_tes")
    soc = model.addVars(days, timesteps,vtype="C", name="soc_tes")
    soc_init = model.addVars(days,vtype="C", name="soc_init_tes")

    model.update()
    
    #%% technical contraints               
    
    # heatpump power and heat 
    model.addConstrs((power_hp[d,t] <= (y_hp[d,t] * capa_hp) for d in days for t in timesteps), name="Max_power_hp")
                    
    model.addConstrs((power_hp[d,t] >= (y_hp[d,t] * capa_hp * mod_lvl) for d in days for t in timesteps), name="Min_power_hp")
    
    model.addConstrs((power_hp[d,t] == heat_hp[d,t] / cop[d,t] for d in days for t in timesteps), name="heat_power_coupling")
    
    # electric heater power and heat    
    model.addConstrs((power_eh[d,t] == heat_eh[d,t]/devs["eh"]["eta"] for d in days for t in timesteps), name="heat_power_eh")    
           
    # tes state of charge
    # initial SOC per typeday
    model.addConstrs((soc_init[d] <= capa_tes for d in days), name="SOC_init")
    # SOC limit for every timestep            
    model.addConstrs((soc[d,t] <= capa_tes for d in days for t in timesteps), name="SOC")
    # SOC repetitions >> SOC at the end of the day has to be SOC at the beginning of this day
    if np.max(clustered["weights"]) > 1:
        model.addConstrs((soc_init[d] == soc[d,params["time_steps"]-1] for d in days), name="repetitions")
                     
        
    k_loss = devs["tes"]["k_loss"]
    eta_ch = devs["tes"]["eta_ch"]
    eta_dch = devs["tes"]["eta_dch"]
        
    for d in days:
        for t in timesteps:
            if t == 0:
                if np.max(clustered["weights"]) == 1:
                    if d == 0:
                       soc_prev = soc_init[d]
                    else:
                       soc_prev = soc[d-1,params["time_steps"]-1]
                else:
                    soc_prev = soc_init[d]
            else:
                soc_prev = soc[d,t-1]
            
            charge = eta_ch * ch[d,t]
            discharge = 1 / eta_dch * dch[d,t]
            
            model.addConstr(soc[d,t] == (1 - k_loss) * soc_prev + dt * (charge - discharge), name="Storage_balance")
                
    #%% thermal balances 
 
    model.addConstrs((ch[d,t]  == (heat_hp[d,t] + heat_eh[d,t]) for d in days for t in timesteps), name="Thermal_max_charge")
    
    # differentiation for dhw-heating: either electric or via heating system
    if options["dhw_electric"]:
        model.addConstrs((dch[d,t] == clustered["heat"][d,t] for d in days for t in timesteps), name="Thermal_max_discharge")            
    else:     
       model.addConstrs((dch[d,t] == (clustered["heat"][d,t] + clustered["dhw"][d,t]) for d in days for t in timesteps), name="Thermal_max_discharge")
    
    

    #%% start optimization
    
    # set objective function
    model.setObjective(sum(sum((power_hp[d,t] + power_eh[d,t]) for t in timesteps) for d in days), gp.GRB.MINIMIZE)                
    
    # adgust gurobi settings
    model.Params.TimeLimit = 25
    model.Params.MIPGap = 0.01
    model.Params.NumericFocus = 3
    model.Params.MIPFocus = 3
    model.Params.Aggregate = 1
    
    model.optimize()
    
    if model.status==gp.GRB.Status.INFEASIBLE:
        model.computeIIS()
        f=open('errorfile_hp.txt','w')
        f.write('\nThe following constraint(s) cannot be satisfied:\n')
        for c in model.getConstrs():
            if c.IISConstr:
                f.write('%s' % c.constrName)
                f.write('\n')
        f.close()
    
    
    #%% retrieve results
    
    res_actHP = np.array([[y_hp[d,t].X for t in timesteps]for d in days])
    res_powerHP = np.array([[power_hp[d,t].X for t in timesteps]for d in days])
    res_powerEH = np.array([[power_eh[d,t].X for t in timesteps]for d in days])
    res_heatHP = np.array([[heat_hp[d,t].X for t in timesteps]for d in days])
    res_heatEH = np.array([[heat_eh[d,t].X for t in timesteps]for d in days])
    res_SOC = np.array([[soc[d,t].X for t in timesteps]for d in days])
    res_SOC_init = np.array([soc[d].X for d in days])
    res_ch = np.array([[ch[d,t].X for t in timesteps]for d in days])
    res_dch = np.array([[dch[d,t].X for t in timesteps]for d in days])

    return (res_actHP, res_powerHP, res_powerEH, res_SOC, res_SOC_init, res_ch, res_dch, res_heatHP, res_heatEH)
