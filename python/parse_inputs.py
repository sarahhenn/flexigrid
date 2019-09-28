#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 18:42:33 2015
@author: Thomas
"""
from __future__ import division
import xlrd
from xml.dom import minidom
import numpy as np
#import statsmodels.api as sm
import scipy.stats as stats
from scipy.interpolate import griddata



def _parse_tariffs(fixed, variable):
    """
    """
    def _parse_tariff(tariff):
        """
        """
        lb  = []
        ub  = []
        tar = []
        
        tariff = tariff.replace(" ", "")
        tariff = tariff.replace("[", "")
        tariff = tariff.replace("]", "")
        split  = tariff.split(";")
        
        for val in split:
            (value_range, value) = val.split(":")
            (temp_lb, temp_ub)   = value_range.split("-")
            lb.append(float(temp_lb.replace(",",".")))
            ub.append(float(temp_ub.replace(",",".")))
            tar.append(float(value.replace(",",".")))
        
        return (lb, ub, tar)
    
    (lb, ub, fix) = _parse_tariff(fixed)
    (lb, ub, var) = _parse_tariff(variable)
    
    return (lb, ub, fix, var)
    
    
def read_economics(devices, filename="raw_inputs/economics.xlsx"):
    """
    Read in economic parameters and update residual values of devices.
    
    Parameters
    ----------
    devices : dictionary
        All device specific characteristics.
    filename : string, optional
        Excel-file with the economic and other parameters.
    
    Returns
    -------
    eco : dictionary
        Information on economic parameters.
    par : dictionary
        All non-economic and non-technical parameters.
    devices : dictionary
        All device specific characteristics.
    """
    book = xlrd.open_workbook(filename)
    
    sheet_eco  = book.sheet_by_name("gen_economics")    
    sheet_gas  = book.sheet_by_name("gas_economics")
    sheet_el   = book.sheet_by_name("el_economics")
    sheet_pel  = book.sheet_by_name("pel_economics")
    sheet_dev  = book.sheet_by_name("dev_economics")
    sheet_comp = book.sheet_by_name("comp_economics")
    sheet_par  = book.sheet_by_name("further_parameters")
    sheet_ep   = book.sheet_by_name("ep_table")
            
    eco = {}
    par = {}
          
    # Economics
    eco["t_calc"] = sheet_eco.cell_value(1,1)
    eco["tax"]    = sheet_eco.cell_value(2,1)
    eco["rate"]   = sheet_eco.cell_value(3,1)
    
    eco["q"]      = 1 + eco["rate"]
    eco["crf"]    = ((eco["q"] ** eco["t_calc"] * eco["rate"]) / 
                     (eco["q"] ** eco["t_calc"] - 1))                     
    
    eco["prChange"] = {}
    eco["prChange"]["el"]     = sheet_eco.cell_value(4,1)
    eco["prChange"]["gas"]    = sheet_eco.cell_value(5,1)
    eco["prChange"]["pel"] = sheet_eco.cell_value(6,1)
    eco["prChange"]["eex"]    = sheet_eco.cell_value(7,1)
    eco["prChange"]["infl"]   = sheet_eco.cell_value(8,1)
    
    eco["price_sell_el"] = sheet_eco.cell_value(9,1)
    eco["energy_tax"]    = sheet_eco.cell_value(10,1)  # in €/kWh
        
    pC = eco["prChange"]
    eco["b"] = {key: ((1 - (pC[key] / eco["q"]) ** eco["t_calc"]) / 
                      (eco["q"] - pC[key]))
                       for key in pC.keys()}
    
    # Prices and tariff amount gradations
    eco["el"]  = {}
    eco["gas"] = {}
    eco["pel"] = {}
    
    # Read gas prices
    for i in range(1, sheet_gas.nrows):
        (lb, ub, fix, var) = _parse_tariffs(sheet_gas.cell_value(i,1),
                                            sheet_gas.cell_value(i,2))
        eco["gas"][sheet_gas.cell_value(i,0)] = {"lb": lb, "ub": ub, 
                                                 "fix": fix, "var":var}
        eco["gas"][sheet_gas.cell_value(i,0)]["emi"] = float(sheet_gas.cell_value(i,3))
    
    # Read el prices
    for i in range(1, sheet_el.nrows):
        (lb, ub, fix, var) = _parse_tariffs(sheet_el.cell_value(i,1),
                                            sheet_el.cell_value(i,2))
        eco["el"][sheet_el.cell_value(i,0)] = {"lb": lb, "ub": ub, 
                                               "fix": fix, "var":var}
        eco["el"][sheet_el.cell_value(i,0)]["emi"] = float(sheet_el.cell_value(i,3))
        eco["el"][sheet_el.cell_value(i,0)]["hp"] = int(sheet_el.cell_value(i,4))       
    
    # Read pellet prices
    for i in range(1, sheet_pel.nrows):
        (lb, ub, fix, var) = _parse_tariffs(sheet_pel.cell_value(i,1),
                                            sheet_pel.cell_value(i,2))
        eco["pel"][sheet_pel.cell_value(i,0)] = {"lb": lb, "ub": ub, 
                                               "fix": fix, "var":var}
        eco["pel"][sheet_pel.cell_value(i,0)]["emi"] = float(sheet_pel.cell_value(i,3))
    
    # Determine residual values
    for dev in devices.keys():              
        
        T_n = devices[dev]["T_op"]
        T   = eco["t_calc"]        
        n   = int(T/T_n)
        r   = eco["prChange"]["infl"]
        q   = eco["q"] 
        
        rval = (sum((r/q)**(n*T_n) for n in range(0,n+1)) - ((r**(n*T_n) * ((n+1)*T_n - T)) / (T_n * q**T)))
        
        devices[dev]["rval"] = rval
       
    # Economic aspects of the building-shell: 
    shell_eco = {}
    for component in ("Window", "Rooftop", "OuterWall", "GroundFloor"): 
        shell_eco[component] = {}
        if component == "OuterWall": 
            row = 1
        if component == "Window": 
            row = 2
        if component == "Rooftop": 
            row = 3
        if component == "GroundFloor": 
            row = 4
            
        shell_eco[component]["c_var"]   = sheet_comp.cell_value(row,1)
        
        shell_eco[component]["c_const"] = sheet_comp.cell_value(row,2)
        
        T_n = sheet_comp.cell_value(row,3)
        T   = eco["t_calc"]        
        n   = int(T/T_n)
        r   = eco["prChange"]["infl"]
        q   = eco["q"] 

        rval = (sum((r/q)**(n*T_n) for n in range(0,n+1)) - ((r**(n*T_n) * ((n+1)*T_n - T)) / (T_n * q**T)))
        
        shell_eco[component]["rval"] = rval
        
        shell_eco[component]["c_o&m"] = sheet_comp.cell_value(row,4)
           
    # Installation costs 
    eco["inst_costs"] = {}
    
    eco["inst_costs"]["EFH"] = {}    
    eco["inst_costs"]["EFH"]["boiler"] = sheet_dev.cell_value(1,1)
    eco["inst_costs"]["EFH"]["hp_air"] = sheet_dev.cell_value(2,1)
    eco["inst_costs"]["EFH"]["hp_geo"] = sheet_dev.cell_value(3,1)
    eco["inst_costs"]["EFH"]["pellet"] = sheet_dev.cell_value(4,1)
    eco["inst_costs"]["EFH"]["chp"]    = sheet_dev.cell_value(5,1)
    eco["inst_costs"]["EFH"]["stc"]    = sheet_dev.cell_value(6,1)
    eco["inst_costs"]["EFH"]["pv"]     = sheet_dev.cell_value(7,1)
    eco["inst_costs"]["EFH"]["tes"]    = sheet_dev.cell_value(8,1)
    eco["inst_costs"]["EFH"]["bat"]    = sheet_dev.cell_value(9,1)
    eco["inst_costs"]["EFH"]["eh"]    = sheet_dev.cell_value(10,1)
    
    eco["inst_costs"]["MFH"] = {}
    eco["inst_costs"]["MFH"]["boiler"] = sheet_dev.cell_value(1,2)
    eco["inst_costs"]["MFH"]["hp_air"] = sheet_dev.cell_value(2,2)
    eco["inst_costs"]["MFH"]["hp_geo"] = sheet_dev.cell_value(3,2)
    eco["inst_costs"]["MFH"]["pellet"] = sheet_dev.cell_value(4,2)
    eco["inst_costs"]["MFH"]["chp"]    = sheet_dev.cell_value(5,2)
    eco["inst_costs"]["MFH"]["stc"]    = sheet_dev.cell_value(6,2)
    eco["inst_costs"]["MFH"]["pv"]     = sheet_dev.cell_value(7,2)
    eco["inst_costs"]["MFH"]["tes"]    = sheet_dev.cell_value(8,2)
    eco["inst_costs"]["MFH"]["bat"]    = sheet_dev.cell_value(9,2)
    eco["inst_costs"]["MFH"]["eh"]     = sheet_dev.cell_value(10,2)
                
    # Further parameters    
    par["mip_gap"]    = sheet_par.cell_value(1,1)
    par["time_limit"] = sheet_par.cell_value(2,1)
    par["rho_w"]      = sheet_par.cell_value(3,1)
    par["c_w"]        = sheet_par.cell_value(4,1)

    # The ep-table gives Information about the expenditure figures of different
    # combinations of heating technologies    
    ep_table = {}
    ep_table["ep"] = {}
    ep_table["boiler"] = {}
    ep_table["hp_air"] = {}
    ep_table["hp_geo"] = {}
    ep_table["eh"] = {}
    ep_table["chp"] = {}
    ep_table["pellet"] = {}
    ep_table["stc"] = {}
    ep_table["TVL35"] = {}
    
    for n in range(1, sheet_ep.nrows):
        ep_table["ep"][n] = sheet_ep.cell_value(n,2)
        ep_table["boiler"][n] = sheet_ep.cell_value(n,3)
        ep_table["hp_air"][n] = sheet_ep.cell_value(n,4)
        ep_table["hp_geo"][n] = sheet_ep.cell_value(n,5)
        ep_table["eh"][n] = sheet_ep.cell_value(n,6)
        ep_table["chp"][n] = sheet_ep.cell_value(n,7)
        ep_table["pellet"][n] = sheet_ep.cell_value(n,8)
        ep_table["stc"][n] = sheet_ep.cell_value(n,9)
        ep_table["TVL35"][n] = sheet_ep.cell_value(n,10)
             
    return (eco, par, devices)
            
            
def compute_parameters(par, number_clusters, len_day):
    """
    Add number of days, time steps per day and temporal discretization to par.
    
    Parameters
    ----------
    par : dictionary
        Dictionary which holds non-device-characteristic and non-economic 
        parameters.
    number_clusters : integer
        Number of allowed clusters.
    len_day : integer
        Time steps per day
    """
    par["days"] = number_clusters
    par["time_steps"] = len_day
    par["dt"] = 24 / len_day
    
    return par
    
def read_subsidies(economics, filename="raw_inputs/subsidies.xlsx"):
    """
    Read in subsdiy parameters.
    
    Parameters
    ----------
    filename : string, optional
        Excel-file with the subsidy parameters.
    
    Returns
    -------
    sub : dictionary
        Information on subsidy parameters.

    """
    book           = xlrd.open_workbook(filename)
    sheet_hp       = book.sheet_by_name("hp")
    sheet_stc      = book.sheet_by_name("stc")
    sheet_pellet   = book.sheet_by_name("pellet")
    sheet_bat      = book.sheet_by_name("bat")
    sheet_building = book.sheet_by_name("building")
    sheet_eeg      = book.sheet_by_name("eeg")
    sheet_kwkg     = book.sheet_by_name("kwkg")
    sheet_chp      = book.sheet_by_name("chp")
        
    sub_par = {}
    
    # EEG conditions for pv-systems
    sub_par["eeg"] = {}
    sub_par["eeg"]["10"]    = sheet_eeg.cell_value(1,1)
    sub_par["eeg"]["40"]    = sheet_eeg.cell_value(2,1)
    sub_par["eeg"]["750"]   = sheet_eeg.cell_value(3,1)
    sub_par["eeg"]["10000"] = sheet_eeg.cell_value(4,1)
    
    # Es wird davon ausgegangen, dass die Einnahmen durch die Einspeisevergütung
    # in jedem Jahr gleich hoch sind und maximal 20 Jahre gezahlt werden. Über
    # den Faktor sub_par["eeg_temp"] wird der Zinseffekt für die zu unterschiedlichen
    # Zeitpunkten anfallenden Zahlungen einbezogen. Sofern der Betrachtungszeitraum 
    # weniger als 20 Jahre beträgt, wird diese hier ebenfalls berücksichtigt.
    
    if economics["t_calc"] >= 20:        
        sub_par["eeg_temp"] = sum(1/economics["q"]**n for n in range(1,20))
    else:         
        sub_par["eeg_temp"] = sum(1/economics["q"]**n for n in range(1,int(economics["t_calc"]+1)))
    
    # KWKG conditions for chps    
    
    sub_par["kwkg"] = {}
#    sub_par["kwkg"]["lump"]       = sheet_kwkg.cell_value(0,1)
    sub_par["kwkg"]["t_50"]       = sheet_kwkg.cell_value(1,1)
#    sub_par["kwkg"]["t_100"]      = sheet_kwkg.cell_value(2,1)    
    
    sub_par["kwkg"]["self_50"]    = sheet_kwkg.cell_value(3,1)
#    sub_par["kwkg"]["self_100"]   = sheet_kwkg.cell_value(4,1)
#    sub_par["kwkg"]["self_250"]   = sheet_kwkg.cell_value(5,1)
#    sub_par["kwkg"]["self_2000"]  = sheet_kwkg.cell_value(6,1) 
#    sub_par["kwkg"]["self_10000"] = sheet_kwkg.cell_value(7,1)

    sub_par["kwkg"]["sell_50"]    = sheet_kwkg.cell_value(8,1)
#    sub_par["kwkg"]["sell_100"]   = sheet_kwkg.cell_value(9,1)
#    sub_par["kwkg"]["sell_250"]   = sheet_kwkg.cell_value(10,1)
#    sub_par["kwkg"]["sell_2000"]  = sheet_kwkg.cell_value(11,1)
#    sub_par["kwkg"]["sell_10000"] = sheet_kwkg.cell_value(12,1)
    
#    sub_par["kwkg"]["ave_50"]    = sheet_kwkg.cell_value(13,1)
#    sub_par["kwkg"]["ave_100"]   = sheet_kwkg.cell_value(14,1)
#    sub_par["kwkg"]["ave_250"]   = sheet_kwkg.cell_value(15,1)
#    sub_par["kwkg"]["ave_2000"]  = sheet_kwkg.cell_value(16,1)
#    sub_par["kwkg"]["ave_10000"] = sheet_kwkg.cell_value(17,1)
    
    sub_par["kwkg"]["vls"] = {}
    sub_par["kwkg"]["vls"]["2500"] = 2500
    sub_par["kwkg"]["vls"]["3000"] = 3000
    sub_par["kwkg"]["vls"]["3500"] = 3500
    sub_par["kwkg"]["vls"]["4000"] = 4000
    sub_par["kwkg"]["vls"]["4500"] = 4500
    sub_par["kwkg"]["vls"]["5000"] = 5000
    sub_par["kwkg"]["vls"]["5500"] = 5500
    sub_par["kwkg"]["vls"]["6000"] = 6000
    sub_par["kwkg"]["vls"]["6500"] = 6500
    
    sub_par["kwkg"]["i"] = {}
    for n in (2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500):
        t = round(sub_par["kwkg"]["t_50"] / n)          
        if economics["t_calc"] <= t:
            sub_par["kwkg"]["i"][str(n)] = sum(1/economics["q"]**n for n in range(1,int(economics["t_calc"])+1))
        else: 
            sub_par["kwkg"]["i"][str(n)] = sum(1/economics["q"]**n for n in range(1,t+1))
    
    # Bafa subsidy for Micro CHP
    
    sub_par["bafa_chp"] = {}
    sub_par["bafa_chp"]["sub_step_1"]  = sheet_chp.cell_value(0,1)
    sub_par["bafa_chp"]["sub_step_2"]  = sheet_chp.cell_value(1,1)
    sub_par["bafa_chp"]["sub_step_3"]  = sheet_chp.cell_value(2,1)
    sub_par["bafa_chp"]["sub_step_4"]  = sheet_chp.cell_value(3,1)
    sub_par["bafa_chp"]["sub_basic_max"]   = sheet_chp.cell_value(4,1)
    sub_par["bafa_chp"]["share_therm_eff"] = sheet_chp.cell_value(5,1)
    sub_par["bafa_chp"]["share_elec_eff"]  = sheet_chp.cell_value(6,1)
    
    #KfW program 275 for battery storages
    sub_par["bat"] = {}
    sub_par["bat"]["sub_bat_max"] = sheet_bat.cell_value(0,1)
    sub_par["bat"]["sub_bat"]     = sheet_bat.cell_value(1,1)
    sub_par["bat"]["share_max"]   = sheet_bat.cell_value(2,1)
    
    # Bafa Subsidy conditions for heating pumps
    sub_par["hp_air"] = {}  
    sub_par["hp_air"]["max_cap"]      = sheet_hp.cell_value(0,1)
    sub_par["hp_air"]["basic_fix"]    = sheet_hp.cell_value(2,1)
    sub_par["hp_air"]["basic_fix_pc"] = sheet_hp.cell_value(3,1)
    sub_par["hp_air"]["basic_var"]    = sheet_hp.cell_value(4,1)
    sub_par["hp_air"]["basic_scop"]   = sheet_hp.cell_value(5,1)
    sub_par["hp_air"]["inno_fix"]     = sheet_hp.cell_value(6,1)
    sub_par["hp_air"]["inno_fix_pc"]  = sheet_hp.cell_value(7,1)
    sub_par["hp_air"]["inno_var"]     = sheet_hp.cell_value(8,1)
    sub_par["hp_air"]["inno_scop"]    = sheet_hp.cell_value(9,1)
    sub_par["hp_air"]["smart_grid"]   = sheet_hp.cell_value(20,1)
    sub_par["hp_air"]["stor_restr"]   = sheet_hp.cell_value(21,1) 
    sub_par["hp_air"]["build_eff"]    = sheet_hp.cell_value(22,1) 
    
    sub_par["hp_geo"] = {}
    sub_par["hp_geo"]["max_cap"]      = sheet_hp.cell_value(0,1)
    sub_par["hp_geo"]["basic_fix"]    = sheet_hp.cell_value(11,1)
    sub_par["hp_geo"]["basic_fix_pc"] = sheet_hp.cell_value(12,1)
    sub_par["hp_geo"]["basic_var"]    = sheet_hp.cell_value(13,1)
    sub_par["hp_geo"]["basic_scop"]   = sheet_hp.cell_value(14,1)
    sub_par["hp_geo"]["inno_fix"]     = sheet_hp.cell_value(15,1)
    sub_par["hp_geo"]["inno_fix_pc"]  = sheet_hp.cell_value(16,1)
    sub_par["hp_geo"]["inno_var"]     = sheet_hp.cell_value(17,1)
    sub_par["hp_geo"]["inno_scop"]    = sheet_hp.cell_value(18,1)
    sub_par["hp_geo"]["smart_grid"]   = sheet_hp.cell_value(20,1)
    sub_par["hp_geo"]["stor_restr"]   = sheet_hp.cell_value(21,1) 
    sub_par["hp_geo"]["build_eff"]   = sheet_hp.cell_value(22,1) 
        
    #Bafa Subsidy conditions for solarthermal collectors
    sub_par["stc"] = {}
    sub_par["stc"]["basic_fix"]       = sheet_stc.cell_value(0,1)
    sub_par["stc"]["basic_var"]       = sheet_stc.cell_value(1,1)
    sub_par["stc"]["basic_area_min"]  = sheet_stc.cell_value(2,1)
    sub_par["stc"]["basic_area_max"]  = sheet_stc.cell_value(3,1)    
    sub_par["stc"]["inno_existing_b"] = sheet_stc.cell_value(4,1)
    sub_par["stc"]["inno_new_b"]      = sheet_stc.cell_value(5,1)
    sub_par["stc"]["inno_area_min"]   = sheet_stc.cell_value(6,1)
    sub_par["stc"]["inno_area_max"]   = sheet_stc.cell_value(7,1)    
    sub_par["stc"]["min_storage"]     = sheet_stc.cell_value(8,1)
    sub_par["stc"]["annual_gain"]     = sheet_stc.cell_value(9,1)
    sub_par["stc"]["stc_hp_combi"]    = sheet_stc.cell_value(11,1)
    sub_par["stc"]["build_eff"]       = sheet_stc.cell_value(12,1)
    
    #Bafa Subsidy conditions for a pellet heating
    sub_par["pellet"] = {}
    sub_par["pellet"]["min_cap"]           = sheet_pellet.cell_value(0,1)
    sub_par["pellet"]["max_cap"]           = sheet_pellet.cell_value(1,1)
    sub_par["pellet"]["basic_fix"]         = sheet_pellet.cell_value(2,1)
    sub_par["pellet"]["basic_storage"]     = sheet_pellet.cell_value(3,1)
    sub_par["pellet"]["basic_var"]         = sheet_pellet.cell_value(4,1)
    sub_par["pellet"]["inno_fix_new"]      = sheet_pellet.cell_value(5,1)
    sub_par["pellet"]["inno_fix_new_stor"] = sheet_pellet.cell_value(6,1)
    sub_par["pellet"]["inno_fix_old"]      = sheet_pellet.cell_value(7,1)
    sub_par["pellet"]["inno_fix_old_stor"] = sheet_pellet.cell_value(8,1)
    sub_par["pellet"]["stor_restr"]        = sheet_pellet.cell_value(9,1)
    sub_par["pellet"]["stc_pellet_combi"]  = sheet_pellet.cell_value(11,1)
    sub_par["pellet"]["build_eff"]         = sheet_pellet.cell_value(12,1)
        
    #KfW Subsidy conditions for the building envelope
    sub_par["building"] = {}
    sub_par["building"]["grant"] = {}
    sub_par["building"]["share_max"] = {}
    sub_par["building"]["grant"]["ind_mea"]         = sheet_building.cell_value(1,1)
    sub_par["building"]["share_max"]["ind_mea"]     = sheet_building.cell_value(1,2)
    sub_par["building"]["grant"]["kfw_eff_115"]     = sheet_building.cell_value(2,1)
    sub_par["building"]["share_max"]["kfw_eff_115"] = sheet_building.cell_value(2,2)
    sub_par["building"]["grant"]["kfw_eff_100"]     = sheet_building.cell_value(3,1)
    sub_par["building"]["share_max"]["kfw_eff_100"] = sheet_building.cell_value(3,2) 
    sub_par["building"]["grant"]["kfw_eff_85"]      = sheet_building.cell_value(4,1)
    sub_par["building"]["share_max"]["kfw_eff_85"]  = sheet_building.cell_value(4,2)
    sub_par["building"]["grant"]["kfw_eff_70"]      = sheet_building.cell_value(5,1)
    sub_par["building"]["share_max"]["kfw_eff_70"]  = sheet_building.cell_value(5,2)
    sub_par["building"]["grant"]["kfw_eff_55"]      = sheet_building.cell_value(6,1)
    sub_par["building"]["share_max"]["kfw_eff_55"]  = sheet_building.cell_value(6,2)        

    sub_par["building"]["eff_fact_Q"] = {}                                                                                                                      
    sub_par["building"]["eff_fact_Q"]["kfw_eff_115"] = sheet_building.cell_value(2,3)
    sub_par["building"]["eff_fact_Q"]["kfw_eff_100"] = sheet_building.cell_value(3,3)
    sub_par["building"]["eff_fact_Q"]["kfw_eff_85"]  = sheet_building.cell_value(4,3)
    sub_par["building"]["eff_fact_Q"]["kfw_eff_70"]  = sheet_building.cell_value(5,3)
    sub_par["building"]["eff_fact_Q"]["kfw_eff_55"]  = sheet_building.cell_value(6,3)
    
    sub_par["building"]["eff_fact_H"] = {}                                                                                                                      
    sub_par["building"]["eff_fact_H"]["kfw_eff_115"] = sheet_building.cell_value(2,4)
    sub_par["building"]["eff_fact_H"]["kfw_eff_100"] = sheet_building.cell_value(3,4)
    sub_par["building"]["eff_fact_H"]["kfw_eff_85"]  = sheet_building.cell_value(4,4)
    sub_par["building"]["eff_fact_H"]["kfw_eff_70"]  = sheet_building.cell_value(5,4)
    sub_par["building"]["eff_fact_H"]["kfw_eff_55"]  = sheet_building.cell_value(6,4)
    
    sub_par["building"]["u_value"] = {}
    sub_par["building"]["u_value"]["Window"]  = sheet_building.cell_value(9,1)
    sub_par["building"]["u_value"]["OuterWall"]    = sheet_building.cell_value(10,1)
    sub_par["building"]["u_value"]["Rooftop"] = sheet_building.cell_value(11,1)
    sub_par["building"]["u_value"]["GroundFloor"]  = sheet_building.cell_value(12,1)
    
    return sub_par
    
def read_devices(timesteps, days, 
                 temperature_ambient, solar_irradiation, 
                 days_per_cluster, filename="raw_inputs/devices.xlsx"):
    """
    Read all devices from a given file.
    
    Parameters
    ----------
    timesteps : integer
        Number of time steps per typical day
    days : integer
        Number of typical days
    temperature_ambient : array_like
        2-dimensional array [days, timesteps] with the ambient temperature in 
        degree Celsius
    solar_irradiation : array_like
        Solar irradiation in Watt per square meter on the tilted areas on 
        which STC or PV will be installed.
    filename : string, optional
        Path to the *.xlsx file containing all available devices
    
    Return
    ------
    results : dictionary
        Dictionary containing the information for each device specified in 
        the given input file.
    """
    # Initialize results
    results = {}
    
    # Open work book
    book = xlrd.open_workbook(filename)
    
    # Get all available sheets
    available_sheets = book.sheet_names()
    
    # Iterate over all sheets
    for dev in available_sheets:
        # Read each sheet
        current_sheet = _read_sheet(book.sheet_by_name(dev), dev, timesteps)
        
        results[dev] = _handle_sheet(current_sheet, dev, timesteps, days, 
                                     temperature_ambient,
                                     solar_irradiation, days_per_cluster)
    
    return results

def _handle_sheet(sheet, dev, timesteps, days,
                  temperature_ambient,
                  solar_irradiation, days_per_cluster):
    """
    Parameters
    ----------
    sheet : dictionary
        Read device characteristics
    dev : string
        - `"boiler"`    : Boiler
        - `"chp"`       : CHP unit
        - `"hp_air"`    : Air heat pump
        - `"hp_geo"` : Geothermal heat pump
        - `"eh"`        : Electrical heater
        - `"pv"`        : Photovoltaic modules
        - `"stc"`       : Solar thermal collectors
        - `"tes"`       : Thermal energy storage units
        - `"bat"`       : Battery units
        - `"inv"`       : Inverters    
    timesteps : integer
        Number of time steps per typical day
    days : integer
        Number of typical days
    temperature_ambient : array_like
        2-dimensional array [days, timesteps] with the ambient temperature in 
        degree Celsius
    filename : string, optional
        Path to the *.xlsx file containing all available devices
    solar_irradiation : array_like
        Solar irradiation in Watt per square meter on the tilted areas on 
        which STC or PV will be installed.
        
    Implemented characteristics
    ---------------------------
    - eta = Q/P
    - omega = (Q+P) / E
    """
    results = {}
    
    # Define infinity
    
    ones = np.ones((days, timesteps))
    
    keys = sheet.keys()
    
    if dev == "bat":
        capacity = np.array([sheet[i]["cap"] for i in keys])
        c_inv    = np.array([sheet[i]["c_inv"] for i in keys])
        c_om     = np.array([sheet[i]["c_om"] for i in keys])
        p_ch     = np.array([sheet[i]["P_ch_max"] for i in keys])
        p_dch    = np.array([sheet[i]["P_dch_max"] for i in keys])

        results["T_op"]   = np.mean([sheet[i]["T_op"] for i in keys])
        results["k_loss"] = np.mean([sheet[i]["k_loss"] for i in keys])
        results["eta"]    = np.mean([sheet[i]["eta"] for i in keys])

        results["cap_min"]  = np.min(capacity) # kWh
        results["cap_max"]  = np.max(capacity) # kWh
        results["c_om_rel"] = np.mean(c_om / c_inv)
        
        # Regression: p_ch = slope * capacity + intercept
        lin_reg = stats.linregress(x=capacity, y=p_ch)
        results["P_ch_fix"] = lin_reg[1]
        results["P_ch_var"] = lin_reg[0] # kW/kWh
        
        # Regression: p_dch = slope * capacity + intercept
        lin_reg = stats.linregress(x=capacity, y=p_dch)
        results["P_dch_fix"] = lin_reg[1]
        results["P_dch_var"] = lin_reg[0] # kW/kWh
        
        # Regression: c_inv = slope * capacity + intercept
        lin_reg = stats.linregress(x=capacity, y=c_inv)
        results["c_inv_fix"] = lin_reg[1]
        results["c_inv_var"] = lin_reg[0] # Euro/kWh
        
    elif dev == "pellet":
        
        c_inv       = np.array([sheet[i]["c_inv"] for i in keys])
        c_om        = np.array([sheet[i]["c_om"] for i in keys])
        heat_output = np.array([sheet[i]["Q_nom"] for i in keys])
                        
        results["T_op"]    = np.mean([sheet[i]["T_op"] for i in keys])
        results["mod_lvl"] = np.mean([sheet[i]["mod_lvl"] for i in keys])
        results["eta"]     = np.mean([sheet[i]["eta"] for i in keys])

        results["c_om_rel"]  = np.mean(c_om / c_inv)
        results["Q_nom_min"] = np.min(heat_output)
        results["Q_nom_max"] = np.max(heat_output)
        results["inno_ability"]  =  0
                
        # Regression: c_inv = slope * heat_output + intercept
        lin_reg = stats.linregress(x=heat_output, y=c_inv)
        results["c_inv_fix"] = lin_reg[1]
        results["c_inv_var"] = lin_reg[0]   # Euro/Watt
        
        
    elif dev == "boiler":
        
        c_inv       = np.array([sheet[i]["c_inv"] for i in keys])
        c_om        = np.array([sheet[i]["c_om"] for i in keys])
        heat_output = np.array([sheet[i]["Q_nom"] for i in keys])
                
        results["T_op"]      = np.mean([sheet[i]["T_op"] for i in keys])
        results["mod_lvl"]   = np.mean([sheet[i]["mod_lvl"] for i in keys])
        results["eta"]       = np.mean([sheet[i]["eta"] for i in keys])
        
        results["c_om_rel"]  = np.mean(c_om / c_inv)
        results["Q_nom_min"] = np.min(heat_output)
        results["Q_nom_max"] = np.max(heat_output)
        
        # Regression: c_inv = slope * heat_output + intercept
        lin_reg = stats.linregress(x=heat_output, y=c_inv)
        results["c_inv_fix"] = lin_reg[1]
        results["c_inv_var"] = lin_reg[0]   # Euro/Watt
        
        
    elif dev == "chp":
        
        c_inv       = np.array([sheet[i]["c_inv"] for i in keys])
        c_om        = np.array([sheet[i]["c_om"] for i in keys])
        heat_output = np.array([sheet[i]["Q_nom"] for i in keys])
        eta         = np.mean([sheet[i]["eta"] for i in keys])
        omega       = np.mean([sheet[i]["omega"] for i in keys])
        
        results["T_op"]    = np.mean([sheet[i]["T_op"] for i in keys])
        results["mod_lvl"] = np.mean([sheet[i]["mod_lvl"] for i in keys])
        
        results["c_om_rel"]  = np.mean(c_om / c_inv)
        results["Q_nom_min"] = np.min(heat_output)
        results["Q_nom_max"] = np.max(heat_output)
                
        results["omega"] = ones * omega
        results["eta"]   = ones * eta
        results["therm_eff_bonus"]  =  1
        results["power_eff_bonus"]  =  0
        results["sigma"] = 1/np.mean(results["eta"])
        results["omega"] = omega
                
        # Regression: c_inv = slope * heat_output + intercept
        lin_reg = stats.linregress(x=heat_output, y=c_inv)
        results["c_inv_fix"] = lin_reg[1]
        results["c_inv_var"] = lin_reg[0]   # Euro/Watt
        
        
    elif dev == "eh":
        
        c_inv       = np.array([sheet[i]["c_inv"] for i in keys])
        c_om        = np.array([sheet[i]["c_om"] for i in keys])
        heat_output = np.array([sheet[i]["Q_nom"] for i in keys])

        results["T_op"]    = np.mean([sheet[i]["T_op"] for i in keys])
        results["mod_lvl"] = np.mean([sheet[i]["mod_lvl"] for i in keys])
        results["eta"]     = np.mean([sheet[i]["eta"] for i in keys])
        
        results["c_om_rel"]  = np.mean(c_om / c_inv)
        results["Q_nom_min"] = np.min(heat_output)
        results["Q_nom_max"] = np.max(heat_output)
        
        # Regression: c_inv = slope * heat_output + intercept
        lin_reg = stats.linregress(x=heat_output, y=c_inv)
        results["c_inv_fix"] = lin_reg[1]
        results["c_inv_var"] = lin_reg[0]   # Euro/Watt
        
        
    elif dev == "hp_air":
        
        c_inv       = np.array([sheet[i]["c_inv"] for i in keys])
        c_om        = np.array([sheet[i]["c_om"] for i in keys])
        heat_output = np.array([sheet[i]["Q_nom"] for i in keys])
        
        # Regression: c_inv = slope * heat_output + intercept
        lin_reg = stats.linregress(x = heat_output, y = c_inv)
        results["c_inv_fix"] = lin_reg[1]
        results["c_inv_var"] = lin_reg[0]   # Euro/Watt
        
        results["c_om_rel"]  = np.mean(c_om / c_inv)
        results["Q_nom_min"] = np.min(heat_output)
        results["Q_nom_max"] = np.max(heat_output)
        
        results["pc"]  =  1
        results["Smart_Grid"]  =  1
        
        results["T_op"]    = np.mean([sheet[i]["T_op"] for i in keys])
        results["mod_lvl"] = np.mean([sheet[i]["mod_lvl"] for i in keys])
        results["dT_max"]  = np.mean([sheet[i]["dT_max"] for i in keys]) 
        
        results["cop_a-7w35"] = np.mean([sheet[i]["cop_a-7w35"] for i in keys])
        results["cop_a2w35"]  = np.mean([sheet[i]["cop_a2w35"] for i in keys])
        results["cop_a7w35"]  = np.mean([sheet[i]["cop_a7w35"] for i in keys])
        results["cop_a12w35"] = np.mean([sheet[i]["cop_a12w35"] for i in keys])
        results["cop_a-7w55"] = np.mean([sheet[i]["cop_a-7w55"] for i in keys])
        results["cop_a2w55"]  = np.mean([sheet[i]["cop_a2w55"] for i in keys])
        results["cop_a7w55"]  = np.mean([sheet[i]["cop_a7w55"] for i in keys])
        results["cop_a12w55"] = np.mean([sheet[i]["cop_a12w55"] for i in keys])

        results["cop_w35"] = np.ones_like(temperature_ambient)
        results["cop_w55"] = np.ones_like(temperature_ambient)
        
        cop_table = np.array([(35,-7,results["cop_a-7w35"]),(35,2,results["cop_a2w35"]),
                              (35,7,results["cop_a7w35"]),(35,12,results["cop_a12w35"]),             
                              (55,-7,results["cop_a-7w55"]),(55,2,results["cop_a2w55"]),
                              (55,7,results["cop_a7w55"]),(55,12,results["cop_a12w55"])])
                
        for TVL in (35,55):
            for d in range(0,days):
                for t in range(0,timesteps):
                    if temperature_ambient[d,t] < -7:
                        results["cop_w"+str(TVL)][d,t] = results["cop_a-7w"+str(TVL)]
                    elif temperature_ambient[d,t] > 12:
                        results["cop_w"+str(TVL)][d,t] = results["cop_a12w"+str(TVL)]
                    else:
                        results["cop_w"+str(TVL)][d,t] = griddata(cop_table[:,0:2], 
                        cop_table[:,2], [(TVL,temperature_ambient[d,t])], method='linear')
                        
    elif dev == "hp_geo":
        
        c_inv       = np.array([sheet[i]["c_inv"] for i in keys])
        c_om        = np.array([sheet[i]["c_om"] for i in keys])
        heat_output = np.array([sheet[i]["Q_nom"] for i in keys])
        
        # Regression: c_inv = slope * heat_output + intercept
        lin_reg = stats.linregress(x = heat_output, y = c_inv)
        results["c_inv_fix"] = lin_reg[1]
        results["c_inv_var"] = lin_reg[0]   # Euro/Watt
        
        results["c_om_rel"]  = np.mean(c_om / c_inv)
        results["Q_nom_min"] = np.min(heat_output)
        results["Q_nom_max"] = np.max(heat_output)
        
        results["pc"]  =  1
        results["Smart_Grid"]  =  1
        
        results["T_op"]    = np.mean([sheet[i]["T_op"] for i in keys])
        results["mod_lvl"] = np.mean([sheet[i]["mod_lvl"] for i in keys])
        results["dT_max"]  = np.mean([sheet[i]["dT_max"] for i in keys])
                      
        results["cop_a-7w35"] = np.mean([sheet[i]["cop_a-7w35"] for i in keys])
        results["cop_a2w35"]  = np.mean([sheet[i]["cop_a2w35"] for i in keys])
        results["cop_a7w35"]  = np.mean([sheet[i]["cop_a7w35"] for i in keys])
        results["cop_a12w35"] = np.mean([sheet[i]["cop_a12w35"] for i in keys])
        results["cop_a-7w55"] = np.mean([sheet[i]["cop_a-7w55"] for i in keys])
        results["cop_a2w55"]  = np.mean([sheet[i]["cop_a2w55"] for i in keys])
        results["cop_a7w55"]  = np.mean([sheet[i]["cop_a7w55"] for i in keys])
        results["cop_a12w55"] = np.mean([sheet[i]["cop_a12w55"] for i in keys])

        results["cop_w35"] = np.ones_like(temperature_ambient)
        results["cop_w55"] = np.ones_like(temperature_ambient)
        
        cop_table = np.array([(35,-7,results["cop_a-7w35"]),
                              (35,2,results["cop_a2w35"]),
                              (35,7,results["cop_a7w35"]),
                              (35,12,results["cop_a12w35"]),             
                              (55,-7,results["cop_a-7w55"]),
                              (55,2,results["cop_a2w55"]),
                              (55,7,results["cop_a7w55"]),
                              (55,12,results["cop_a12w55"])])

        for TVL in (35,55):
            for d in range(0,days):
                for t in range(0,timesteps):
                    if temperature_ambient[d,t] < -7:
                        results["cop_w"+str(TVL)][d,t] = results["cop_a-7w"+str(TVL)]
                    elif temperature_ambient[d,t] > 12:
                        results["cop_w"+str(TVL)][d,t] = results["cop_a12w"+str(TVL)]
                    else:
                        results["cop_w"+str(TVL)][d,t] = griddata(cop_table[:,0:2], 
                        cop_table[:,2], [(TVL,temperature_ambient[d,t])], method='linear')
        
    elif dev == "pv":
        c_inv = np.array([sheet[i]["c_inv"] for i in keys])
        c_om  = np.array([sheet[i]["c_om"] for i in keys])
        area  = np.array([sheet[i]["area"] for i in keys])
        
        results["c_om_rel"]  = np.mean(c_om / c_inv)
        results["area_mean"] = np.mean(area)
        results["area_min"]  = np.min(area)
        
        results["T_op"]   = np.mean([sheet[i]["T_op"] for i in keys])
        results["p_NOCT"] = np.mean([sheet[i]["p_NOCT"] for i in keys])
        results["t_NOCT"] = np.mean([sheet[i]["t_NOCT"] for i in keys])
        results["gamma"]  = np.mean([sheet[i]["gamma"] for i in keys])
        results["p_nom"]  = np.mean([sheet[i]["p_nom"] for i in keys])
        
        i_NOCT = 0.8 # kW / m2
        
        # Interpolate cell temperature.
        # Without solar irradiation, the cell temperature has to be equal
        # to the ambient temperature. At NOCT irradiation, the cell's 
        # temperature has to be equal to t_NOCT
        t_cell = (temperature_ambient + solar_irradiation / i_NOCT * 
                                  (results["t_NOCT"] - temperature_ambient))
        eta_NOCT = results["p_NOCT"] / (results["area_mean"] * i_NOCT)
        # Compute electrical efficiency of the cell
        eta_el   = eta_NOCT * (1 + results["gamma"] / 100 * 
                               (t_cell - results["t_NOCT"]))
        
        results["eta_el"] = eta_el
        
        results["c_inv_fix"] = 0
        results["c_inv_var"] = np.mean(c_inv) / results["area_mean"]  # Euro/m2
        
    elif dev == "stc":
        c_inv = np.array([sheet[i]["c_inv"] for i in keys])
        c_om  = np.array([sheet[i]["c_om"] for i in keys])
        area  = np.array([sheet[i]["area"] for i in keys])

        results["c_om_rel"]  = np.mean(c_om / c_inv)
        results["area_mean"] = np.mean(area)
        results["area_min"]  = np.min(area)
        
        results["T_op"] = np.mean([sheet[i]["T_op"] for i in keys])
        
        results["dT_max"] = 15#np.mean([sheet[i]["dT_max"] for i in keys])

        zero_loss    = np.mean([sheet[i]["zero_loss"] for i in keys])
        first_order  = np.mean([sheet[i]["first_order"] for i in keys])
        second_order = np.mean([sheet[i]["second_order"] for i in keys])

        temperature_flow = 55
        temp_diff = temperature_flow - temperature_ambient
        eta_th = (zero_loss - 
                  first_order / solar_irradiation * temp_diff - 
                  second_order / solar_irradiation * (temp_diff**2))
        eta_th = np.maximum(eta_th, 0)
        eta_th[solar_irradiation <= 0.00001] = 0
        results["eta_th"] = eta_th
        
        #Compute daily gain as subsidy restriction
        used_irradiation = np.multiply(eta_th,solar_irradiation)
        daily_gain = [None] * len(eta_th)
        for n in range(0,len(eta_th)):
            daily_gain[n] = sum(used_irradiation[n,m] for m in range(0,timesteps))
        gain_per_cluster = np.multiply(daily_gain,days_per_cluster)
        annual_gain = sum(gain_per_cluster[n] for n in range(0,len(eta_th)))
        results["annual_gain"] = annual_gain      

        results["c_inv_fix"] = 0
        results["c_inv_var"] = np.mean(c_inv) / results["area_mean"]  # Euro/m2
        
    elif dev == "tes":
        c_inv  = np.array([sheet[i]["c_inv"] for i in keys])
        c_om   = np.array([sheet[i]["c_om"] for i in keys])
        volume = np.array([sheet[i]["volume"] for i in keys])
        
        results["c_om_rel"]   = np.mean(c_om / c_inv)
        results["volume_min"] = np.min(volume)
        results["volume_max"] = np.max(volume)
        
        results["T_op"]    = np.mean([sheet[i]["T_op"] for i in keys])
        results["k_loss"]  = np.mean([sheet[i]["k_loss"] for i in keys])
        results["eta_ch"]  = np.mean([sheet[i]["eta_ch"] for i in keys])
        results["eta_dch"] = np.mean([sheet[i]["eta_dch"] for i in keys])                
        results["dT_max"] = np.mean([sheet[i]["dT_max"] for i in keys])
        
        # Regression: c_inv = slope * volume + intercept
        lin_reg = stats.linregress(x=volume, y=c_inv)
        results["c_inv_fix"] = lin_reg[1]
        results["c_inv_var"] = lin_reg[0]   # Euro/m3
                
    return results

def _read_sheet(sheet, device, timesteps):
    """
    sheet : sheet-object
        Sheet of the workbook containing all available devices
    device : string
        - `"boiler"`    : Boiler
        - `"chp"`       : CHP unit
        - `"hp"`        : Heat pump
        - `"eh"`        : Electrical heater
        - `"pv"`        : Photovoltaic modules
        - `"stc"`       : Solar thermal collectors
        - `"tes"`       : Thermal energy storage units
        - `"bat"`       : Battery units
        - `"inv"`       : Inverters
    timesteps : integer
        Number of time steps per typical day
    
    Implemented characteristics
    ---------------------------
    - eta = Q/P
    - omega = (Q+P) / E
    """
    
    # Initialize results
    results = {}
    
    # Read all rows but the headers:
    for row in range(1, sheet.nrows):
        # Create new dictionary for current entry. Add common inputs.
        current_results = {}
        
        # Handle each device separately
        if device == "bat":
            
            current_results["c_inv"]     = sheet.cell_value(row, 1)
            current_results["c_om"]      = sheet.cell_value(row, 2)
            current_results["T_op"]      = sheet.cell_value(row, 3)
            current_results["cap"]       = sheet.cell_value(row, 4)
            current_results["eta"]       = sheet.cell_value(row, 5)
            current_results["P_ch_max"]  = sheet.cell_value(row, 6)
            current_results["P_dch_max"] = sheet.cell_value(row, 7)
            current_results["k_loss"]    = 0

        elif device == "boiler":
            
            current_results["Q_nom"]   = sheet.cell_value(row, 1)
            current_results["mod_lvl"] = sheet.cell_value(row, 2)
            current_results["c_inv"]   = sheet.cell_value(row, 3)
            current_results["c_om"]    = sheet.cell_value(row, 4)
            current_results["T_op"]    = sheet.cell_value(row, 5)
            current_results["eta"]     = sheet.cell_value(row, 6)
            
        elif device == "pellet":
            
            current_results["Q_nom"]   = sheet.cell_value(row, 1)
            current_results["mod_lvl"] = sheet.cell_value(row, 2)
            current_results["c_inv"]   = sheet.cell_value(row, 3)
            current_results["c_om"]    = sheet.cell_value(row, 4)
            current_results["T_op"]    = sheet.cell_value(row, 5)
            current_results["eta"]     = sheet.cell_value(row, 6)

        elif device == "chp":
            
            current_results["Q_nom"]   = sheet.cell_value(row, 1)
            current_results["mod_lvl"] = sheet.cell_value(row, 2)
            current_results["c_inv"]   = sheet.cell_value(row, 3)
            current_results["c_om"]    = sheet.cell_value(row, 4)
            current_results["T_op"]    = sheet.cell_value(row, 5)
            current_results["eta"]     = 1 / sheet.cell_value(row, 6)
            current_results["omega"]   = sheet.cell_value(row, 7)
            
        elif device == "eh":
            
            current_results["Q_nom"]   = sheet.cell_value(row, 1)
            current_results["mod_lvl"] = sheet.cell_value(row, 2)
            current_results["c_inv"]   = sheet.cell_value(row, 3)
            current_results["c_om"]    = sheet.cell_value(row, 4)
            current_results["T_op"]    = sheet.cell_value(row, 5)
            current_results["eta"]     = sheet.cell_value(row, 6)
            
        elif device == "hp_air":
            
            current_results["Q_nom"]   = sheet.cell_value(row, 1)
            current_results["mod_lvl"] = sheet.cell_value(row, 2)
            current_results["c_inv"]   = sheet.cell_value(row, 3)
            current_results["c_om"]    = sheet.cell_value(row, 4)
            current_results["T_op"]    = sheet.cell_value(row, 5)
            current_results["dT_max"]  = sheet.cell_value(row, 6)            
            current_results["cop_a-7w35"] = sheet.cell_value(row, 7)
            current_results["cop_a2w35"]  = sheet.cell_value(row, 8)
            current_results["cop_a7w35"]  = sheet.cell_value(row, 9)
            current_results["cop_a12w35"] = sheet.cell_value(row, 10)
            current_results["cop_a-7w55"] = sheet.cell_value(row, 11)
            current_results["cop_a2w55"]  = sheet.cell_value(row, 12)
            current_results["cop_a7w55"]  = sheet.cell_value(row, 13)
            current_results["cop_a12w55"] = sheet.cell_value(row, 14)
            
        elif device == "hp_geo":
            
            current_results["Q_nom"]   = sheet.cell_value(row, 1)
            current_results["mod_lvl"] = sheet.cell_value(row, 2)
            current_results["c_inv"]   = sheet.cell_value(row, 3)
            current_results["c_om"]    = sheet.cell_value(row, 4)
            current_results["T_op"]    = sheet.cell_value(row, 5)
            current_results["dT_max"]  = sheet.cell_value(row, 6)            
            current_results["cop_a-7w35"] = sheet.cell_value(row, 7)
            current_results["cop_a2w35"]  = sheet.cell_value(row, 8)
            current_results["cop_a7w35"]  = sheet.cell_value(row, 9)
            current_results["cop_a12w35"] = sheet.cell_value(row, 10)
            current_results["cop_a-7w55"] = sheet.cell_value(row, 11)
            current_results["cop_a2w55"]  = sheet.cell_value(row, 12)
            current_results["cop_a7w55"]  = sheet.cell_value(row, 13)
            current_results["cop_a12w55"] = sheet.cell_value(row, 14)
            
        elif device == "pv":
            
            current_results["c_inv"] = sheet.cell_value(row, 2)
            current_results["c_om"]  = sheet.cell_value(row, 3)
            current_results["T_op"]  = sheet.cell_value(row, 4)
            current_results["area"]  = sheet.cell_value(row, 5)
            
            current_results["p_NOCT"]  = sheet.cell_value(row, 1)
            current_results["t_NOCT"]  = sheet.cell_value(row, 6)
            current_results["gamma"]   = sheet.cell_value(row, 7)
            current_results["p_nom"]   = sheet.cell_value(row, 8)

        elif device == "stc":
            
            current_results["c_inv"] = sheet.cell_value(row, 1)
            current_results["c_om"]  = sheet.cell_value(row, 2)
            current_results["T_op"]  = sheet.cell_value(row, 3)
            current_results["area"]  = sheet.cell_value(row, 4)

            current_results["zero_loss"]    = sheet.cell_value(row, 5)
            current_results["first_order"]  = sheet.cell_value(row, 6)
            current_results["second_order"] = sheet.cell_value(row, 7)
            
            current_results["dT_max"]  = sheet.cell_value(row, 8)         

        elif device == "tes":
            
            current_results["c_inv"]   = sheet.cell_value(row, 1)
            current_results["c_om"]    = sheet.cell_value(row, 2)
            current_results["T_op"]    = sheet.cell_value(row, 3)
            current_results["eta_ch"]  = sheet.cell_value(row, 6)
            current_results["eta_dch"] = sheet.cell_value(row, 7)
            current_results["dT_max"]  = sheet.cell_value(row, 8)
            
            standby_losses = sheet.cell_value(row, 4) # in kWh / 24h
            volume = sheet.cell_value(row, 5)         # in m3
            
            temp_diff_norm = 45 # K
            heat_cap       = 4180 # J/(kgK)
            density        = 1000 # kg/m3
            energy_content = volume * temp_diff_norm * heat_cap * density # J
            
            k_loss_day = 1 - standby_losses / energy_content * 3600*1000
            current_results["k_loss"] = 1 - (k_loss_day ** (1 / timesteps))
            current_results["volume"] = volume
            
        results[row] = current_results
        
    return results
    
    
def retrofit_scenarios():
    
    """
    sheet : sheet-object
        Sheet of the workbook containing all available devices
    device : string
        - `"boiler"`    : Boiler
        - `"chp"`       : CHP unit
        - `"hp"`        : Heat pump
        - `"eh"`        : Electrical heater
        - `"pv"`        : Photovoltaic modules
        - `"stc"`       : Solar thermal collectors
        - `"tes"`       : Thermal energy storage units
        - `"bat"`       : Battery units
        - `"inv"`       : Inverters
    timesteps : integer
        Number of time steps per typical day
    
    Implemented characteristics
    ---------------------------
    - eta = Q/P
    - omega = (Q+P) / E
    """
    
    # Read Material_File
    mat_file = minidom.parse("raw_inputs/materials.xml")
    materiallist = mat_file.getElementsByTagName("materials:Material")
    
    materials ={}
    values = ("thermal_conduc",
              "name")
    
    for i in range(len(materiallist)):
        material_id = materiallist[i].attributes["material_id"].value
        materials[material_id] = {}
        for j in values:
            try:
                materials[material_id][j] = materiallist[i].\
                getElementsByTagName("materials:"+j)[0].firstChild.nodeValue
            except:
                pass
    
    
    file = minidom.parse("raw_inputs/building_types.xml")
    building_elements = {}
    
    building_parts = ("OuterWall",
                      "Rooftop",
                      "GroundFloor",
                      "Window")
                
    building_types = ("SFH", "_TH", "MFH", "_AB")
    
    scenarios = ("standard", "retrofit", "adv_retr")
    
    timesteps = ("0 1859", "1860 1918", "1919 1948", "1949 1957",
                 "1958 1968", "1969 1978", "1979 1983", "1984 1994", 
                 "1995 2001", "2002 2009", "2010 2015", "2016 2100")
        
    values = ("building_age_group",
              "construction_type")
    
              
    insul_mat = ("glass_fibre_batt_40", "glass_fibre_batt_70", 
                 "EPS_perimeter_insulation_top_layer", "XPS_2_core_layer",
                 "XPS_55", "glass_fibre_glass_wool_80", "EPS_040_15",
                 "XPS_2_core_layer")
    
    for m in building_parts:
        building_elements[m] = {}
        elementlist = file.getElementsByTagName("elements:" + m)
    
        for i in range(len(elementlist)):
            building_elements[m][i] = {}
            for j in values:
                building_elements[m][i][j] = elementlist[i].getElementsByTagName("elements:"+j)[0].firstChild.nodeValue
            building_elements[m][i]["construction_type"] = building_elements[m][i]["construction_type"][7:]
                  
            layerlist = elementlist[i].getElementsByTagName("elements:Layers")[0].\
                                getElementsByTagName("elements:layer")
                
            for k in range(len(layerlist)):
                building_elements[m][i]["layer"+str(k)] = {}
                
                building_elements[m][i]["layer"+str(k)]["material_id"] = \
                elementlist[i].getElementsByTagName("elements:Layers")[0].\
                getElementsByTagName("elements:layer")[k].\
                getElementsByTagName("elements:material")[0].\
                attributes["material_id"].value
        
                building_elements[m][i]["layer"+str(k)]["thickness"] = \
                    elementlist[i].getElementsByTagName("elements:Layers")[0].\
                    getElementsByTagName("elements:layer")[k].\
                    getElementsByTagName("elements:thickness")[0].\
                    firstChild.nodeValue
    
    heat_trans_resis = {}
    heat_trans_resis["GroundFloor"] = 0.34 # (m2*K)/W
    heat_trans_resis["Rooftop"] = 0.21 # (m2*K)/W - Steildach
    #heat_trans_resis["Rooftop"] = 0.17 # (m2*K)/W - Flachdach    
    heat_trans_resis["OuterWall"] = 0.17 # (m2*K)/W
    heat_trans_resis["Window"] = 0.17 # (m2*K)/W
        
    tabula_scenarios = {}
    for a in building_types: 
        tabula_scenarios[a] = {}
        for b in timesteps:
            tabula_scenarios[a][b] = {}
            for c in scenarios: 
                tabula_scenarios[a][b][c] = {}
                for d in building_parts:
                    tabula_scenarios[a][b][c][d] = {}                
                    for e in building_elements[d].keys():
                        if a == building_elements[d][e]["construction_type"][-3:] and \
                           c == building_elements[d][e]["construction_type"][:8] and \
                           b == building_elements[d][e]["building_age_group"]:
                               for i in range (len(building_elements[d][e])-2):
                                   tabula_scenarios[a][b][c][d]["element"+str(i)] = {}
                                           
                                   tabula_scenarios[a][b][c][d]["element"+str(i)]["thickness"] = \
                                   building_elements[d][e]["layer"+str(i)]["thickness"]
                                   
                                   material_id = \
                                   building_elements[d][e]["layer"+str(i)]["material_id"]
                                                                 
                                   tabula_scenarios[a][b][c][d]["element"+str(i)]["material_name"] = \
                                   materials[material_id]["name"]
                                   
                                   tabula_scenarios[a][b][c][d]["element"+str(i)]["thermal_conduc"] = \
                                   materials[material_id]["thermal_conduc"]
                    
                               tabula_scenarios[a][b][c][d]["U-Value"] = \
                               round((1 / (heat_trans_resis[d] + 
                            sum(float(tabula_scenarios[a][b][c][d]["element"+str(i)]["thickness"]) / 
                                float(tabula_scenarios[a][b][c][d]["element"+str(i)]["thermal_conduc"])
                                for i in range (len(building_elements[d][e])-2)))),2)
    
    for a in tabula_scenarios.keys():
        for b in tabula_scenarios[a].keys():
            for c in tabula_scenarios[a][b].keys():
                for d in tabula_scenarios[a][b][c].keys():         
                    t = 0.0
                    for e in tabula_scenarios[a][b][c][d].keys():                    
                        if e != "U-Value" :
                            mat = tabula_scenarios[a][b][c][d][e]["material_name"]
                            if mat in insul_mat:
                                t = t + float(tabula_scenarios[a][b][c][d][e]["thickness"])
                    tabula_scenarios[a][b][c][d]["thick_insu"] = round(t,2)
     
    for a in tabula_scenarios.keys():
        for b in tabula_scenarios[a].keys():
            for c in tabula_scenarios[a][b].keys():
                for d in tabula_scenarios[a][b][c].keys():                    
                     tabula_scenarios[a][b][c][d]["thick_insu_add"] = \
                           round(tabula_scenarios[a][b][c][d]["thick_insu"] - \
                         tabula_scenarios[a][b]["standard"][d]["thick_insu"],2)
    
    tabula_scenarios["TH"] = tabula_scenarios.pop("_TH")
    tabula_scenarios["AB"] = tabula_scenarios.pop("_AB")

    for a in tabula_scenarios.keys():
        for b in tabula_scenarios[a].keys():
            for c in tabula_scenarios[a][b].keys():
                try:
                    if tabula_scenarios[a][b][c]["Window"]["U-Value"] >= 5.0:
                         tabula_scenarios[a][b][c]["Window"]["G-Value"] = 0.87
                    elif tabula_scenarios[a][b][c]["Window"]["U-Value"] < 5.0 and tabula_scenarios[a][b][c]["Window"]["U-Value"] > 1.9:
                         tabula_scenarios[a][b][c]["Window"]["G-Value"] = 0.75
                    elif tabula_scenarios[a][b][c]["Window"]["U-Value"] <= 1.9 and tabula_scenarios[a][b][c]["Window"]["U-Value"] > 1.2:
                         tabula_scenarios[a][b][c]["Window"]["G-Value"] = 0.6
                    elif tabula_scenarios[a][b][c]["Window"]["U-Value"] <= 1.2:
                         tabula_scenarios[a][b][c]["Window"]["G-Value"] = 0.5
                except:
                    None
                    
    return tabula_scenarios
        
def parse_building_parameters ():
    
    
    book  = xlrd.open_workbook("raw_inputs/buildings.xlsx")
    sheet = book.sheet_by_name("component_size")
    
    buildingtypes = {}
    for a in range(1,sheet.nrows):
        buildingtypes[sheet.cell_value(a,1)] ={}
        for b in range(1,sheet.nrows):
            buildingtypes[sheet.cell_value(a,1)][sheet.cell_value(b,2)] = {}
    
    for a in range(1,sheet.nrows):
        buildingtypes[sheet.cell_value(a,1)][sheet.cell_value(a,2)]\
                                               ["Area"]  = sheet.cell_value(a,3)
                                               
        buildingtypes[sheet.cell_value(a,1)][sheet.cell_value(a,2)]\
                                               ["Volume"]  = sheet.cell_value(a,4)
                                               
        buildingtypes[sheet.cell_value(a,1)][sheet.cell_value(a,2)]\
                                               ["Rooftop"]  = sheet.cell_value(a,5)
                                               
        buildingtypes[sheet.cell_value(a,1)][sheet.cell_value(a,2)]\
                                             ["OuterWall"]  = sheet.cell_value(a,6)
                                             
        buildingtypes[sheet.cell_value(a,1)][sheet.cell_value(a,2)]\
                                           ["GroundFloor"]  = sheet.cell_value(a,7)
                                           
        buildingtypes[sheet.cell_value(a,1)][sheet.cell_value(a,2)]\
                                                ["Window"]  = sheet.cell_value(a,8)
                                                
        buildingtypes[sheet.cell_value(a,1)][sheet.cell_value(a,2)]\
                                                ["Window_north"]  = sheet.cell_value(a,8) / 4
                                            
        buildingtypes[sheet.cell_value(a,1)][sheet.cell_value(a,2)]\
                                                ["Window_south"]  = sheet.cell_value(a,8) / 4
                                                
        buildingtypes[sheet.cell_value(a,1)][sheet.cell_value(a,2)]\
                                                ["Window_east"]  = sheet.cell_value(a,8) / 4
                                                
        buildingtypes[sheet.cell_value(a,1)][sheet.cell_value(a,2)]\
                                                ["Window_west"]  = sheet.cell_value(a,8) / 4
                                                                                                        
    return buildingtypes      

if __name__ == "__main__":
    timesteps = 24
    days = 5
    # Random temperatures between -10 and +20 degC:
    temperature_ambient = np.random.rand(days, timesteps) * 30 - 10
    
    temperature_design = -12 # Aachen
    
    solar_irradiation = np.random.rand(days, timesteps) * 800
    solar_irradiation
    
    devs = read_devices(timesteps, days, temperature_ambient,
                        solar_irradiation=solar_irradiation)
                        
    (eco, par, devs) = read_economics(devs)
