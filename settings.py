"""
created on Thursday 06/02/2020
@ author: fpo

file for setting the options for flexigrid ana eventually save TODO stuff

"""

import pandapower.networks as nw

# set parameters
building_type = "EFH"       # EFH, ZFH, MFH_6WE, MFH_10WE, MFH_15WE
building_age  = "2005"      # 1960, 1980, 2005
emission_year = "2030"      # 2017, 2030, 2050
number_clusters = 8

# set options
options = {"static_emissions": False,  # True: calculation with static emissions,
           # False: calculation with timevariant emissions
           "static_prices": False,  # True: static power prices
           # False: timevariant power prices
           "rev_emissions": False,  # True: emissions revenues for feed-in
           # False: no emissions revenues for feed-in
           "rev_price_manner": "real",  # "real" for revenues like power market price
           # "eeg" for revenues like eeg jurisdiction
           "dhw_electric": True,  # define if dhw is provided decentrally by electricity
           "P_pv": 10.00,  # installed peak PV power
           "with_hp": True,  # usage of heat pumps
           "hp_mode": "grid_opt",  # choose between "energy_opt" and "grid_opt"
           "T_VL": 35,  # choose between 35 and 55 "Vorlauftemperatur"
           "alpha_th": 0.8,  # relative size of heat pump (between 0 and 1)
           "beta_th": 0,  # relative size of thermal energy storage (between 0 and 1)
           "k_loss": 0,  # selbstberechneter Wert für k_loss
           "show_simple_plots": False,  # show gridplots before and after optimization
            "show_detailed_plot": False,
           "filename_results": "results/" + building_type + "_" + \
                               building_age + "TESEMISSIONS00.pkl",
           "filename_inputs": "results/inputs_" + building_type + "_" + \
                              building_age + "TESEMISSIONS00.pkl",
           "apc_while_voltage_violation": False,  # True: uses apc, when voltage violations occur
           # False: does not use apc, when voltage violations occur
           "cut_Inj/Subtr_while_voltage_violation": True,  # True: cuts Inj or Subtr, when voltage violations occur
           # depends automatically on the fact, whether voltage is too high or too low
           # note: power costs for heatpump can only be separated for a cost calculation for sum(nodes), not for grid in total!
           # select your cost function calculation through objective function in grid_optimization
           "heatpump_seperated_costs": True,  # True: Heatpumps power costs: 18.56 ct/kWh (apart from other power users)
           # False: Heatpump power costs: 27.8 ct/kWh (equal to other power users)
           "allow_apc_opti": False,  # True: Curtailment allowed to be set in optimization
           # False: Curtailment only through additional constraint
           "change_value_node_violation_abs": 1,
           # specify, for how much the absolute values of inj and subtr should change in case of voltage violations

           "rel1_or_abs0_violation_change": False,  # If false, use absolute power constraint generation
           # If True use relative power constraint generation
           "change_relative_node_violation_rel": 0.9,
           # specify what the new value should be relative to the previous value of inj or subtr in case of voltage violations
           "opt_costs": True,  # If true, minimize cost function (can be specified in opti script)
           "opt_emissions": False,  # If true, minimize emissions (can be specified in opti script)
           "run_which_opti": "normal"   # "normal" for usual opti script
                                        # "pareto_oNB" for running pareto opti script without any network-related constraints
                                        # "pareto_mNB" for running pareto opti script with network-related constraints
           }

#determine the optimization folder in which all input data and results are placed
operationFolder="C:\\users\\flori\\pycharmprojects\\flexigrid"
#the input data is always in this source folder
sourceFolder=operationFolder+"\\input"

# %% create network

# load example net (IEEE 9 buses)
'''
typical kerber grids:   landnetz_freileitung_1(), landnetz_freileitung_2(), landnetz_kabel_1(), landnetz_kabel_2(),
                        dorfnetz(), vorstadtnetz_kabel_1(), vorstadtnetz_kabel_2()
    -> create network with nw.create_kerber_name

extreme kerber grids:   landnetz_freileitung(), landnetz_kabel(), landnetz_freileitung_trafo(), landnetz_kabel_trafo(), 
                        dorfnetz(), dorfnetz_trafo(), 
                        vorstadtnetz_1(), vorstadtnetz_2(), vorstadtnetz_trafo_1(), vorstadtnetz_trafo_2()
    -> create network with nw.kb_extrem_name   

'''
net_name = "landnetz_kabel_trafo"
fkt_name = "kb_extrem_" + net_name
fkt = getattr(nw, fkt_name)
net = fkt()

# TODO: implement mixed shares of buildings
# TODO: adjust emission factors regarding to national weather conditions

# TODO: load data for useable roofarea per building type
# TODO: PV area as MILP-variable??
# useable_roofarea  = 0.30    #Default value: 0.25
