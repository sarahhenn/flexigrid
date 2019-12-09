# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 15:10:05 2019

@author: Chrissi
"""

import run_battery_opti_place_size_2 as run
import xlsxwriter

# resultname should be an explanation of the proposed evaluation
resultname = "Test1.xlsx"
resultFolder = "C:\\Users\\Chrissi\\Git\\Flexigrid\\Auswertung"

#%% Parameters

building = {}
building["case_1"] = {"building_type" : "EFH",       # EFH, ZFH, MFH_6WE, MFH_10WE, MFH_15WE
                      "building_type2" : "MFH_6WE",  # second building type, query "mfh"
                      "building_age"  : "2005",      # 1960, 1980, 2005 
                      "emission_year" : "2017",      # 2017, 2030, 2050 
          }
building["case_2"] = {"building_type" : "EFH",       # EFH, ZFH, MFH_6WE, MFH_10WE, MFH_15WE
                      "building_type2" : "MFH_6WE",  # second building type, query "mfh"
                      "building_age"  : "2005",      # 1960, 1980, 2005 
                      "emission_year" : "2017",      # 2017, 2030, 2050 
          }
building["case_3"] = {"building_type" : "EFH",       # EFH, ZFH, MFH_6WE, MFH_10WE, MFH_15WE
                      "building_type2" : "MFH_6WE",  # second building type, query "mfh"
                      "building_age"  : "2005",      # 1960, 1980, 2005 
                      "emission_year" : "2017",      # 2017, 2030, 2050 
          }
building["case_4"] = {"building_type" : "EFH",       # EFH, ZFH, MFH_6WE, MFH_10WE, MFH_15WE
                      "building_type2" : "MFH_6WE",  # second building type, query "mfh"
                      "building_age"  : "2005",      # 1960, 1980, 2005 
                      "emission_year" : "2017",      # 2017, 2030, 2050 
          }
building["case_5"] = {"building_type" : "EFH",       # EFH, ZFH, MFH_6WE, MFH_10WE, MFH_15WE
                      "building_type2" : "MFH_6WE",  # second building type, query "mfh"
                      "building_age"  : "2005",      # 1960, 1980, 2005 
                      "emission_year" : "2017",      # 2017, 2030, 2050 
          }

# District parameters
district_options = {}
district_options["case_1"] = {"mfh" : 0.33,                  # ratio of MFH to EFH in %
                              "pv" : 0.2,                    # ratio in %
                              "hp" : 0.2,                    # ratio in %
                              "ev" : 0.1,                    # ratio in %
                              "case" : "random"               # "best", "worst" and "random"
                              }
district_options["case_2"] = {"mfh" : 0.33,                  # ratio of MFH to EFH in %
                              "pv" : 0.2,                    # ratio in %
                              "hp" : 0.2,                    # ratio in %
                              "ev" : 0.1,                    # ratio in %
                              "case" : "random"               # "best", "worst" and "random"
                              }
district_options["case_3"] = {"mfh" : 0.33,                  # ratio of MFH to EFH in %
                              "pv" : 0.2,                    # ratio in %
                              "hp" : 0.2,                    # ratio in %
                              "ev" : 0.1,                    # ratio in %
                              "case" : "random"               # "best", "worst" and "random"
                              }
district_options["case_4"] = {"mfh" : 0.33,                  # ratio of MFH to EFH in %
                              "pv" : 0.2,                    # ratio in %
                              "hp" : 0.2,                    # ratio in %
                              "ev" : 0.1,                    # ratio in %
                              "case" : "random"               # "best", "worst" and "random"
                              }
district_options["case_5"] = {"mfh" : 0.33,                  # ratio of MFH to EFH in %
                              "pv" : 0.2,                    # ratio in %
                              "hp" : 0.2,                    # ratio in %
                              "ev" : 0.1,                    # ratio in %
                              "case" : "random"               # "best", "worst" and "random"
                              }

options = {}
options["case_1"] = {"static_emissions": True,   # True: calculation with static emissions, 
                                         # False: calculation with timevariant emissions
                     "rev_emissions": True,      # True: emissions revenues for feed-in
                                         # False: no emissions revenues for feed-in
                     "dhw_electric": True,       # define if dhw is provided decentrally by electricity
                     "P_pv": 20.0,               # installed peak PV power
                     "hp_mode": "energy_opt",    # choose between "off" (no hp) and "energy_opt" and "grid_opt"
                     "T_VL": 35,                 # choose between 35 and 55 "Vorlauftemperatur" 
                     "alpha_th": 0.8,            # relative size of heat pump (between 0 and 1)
                     "beta_th": 0.5,             # relative size of thermal energy storage (between 0 and 1)
                     "EV_mode": "off",         # choose between "off" (no EVs), "on_demand", "grid_reactive" and "bi_directional"
                     "show_grid_plots" : True,   # show gridplots before and after optimization
                     "phi" : 25.842              # 
                     }
options["case_2"] = {"static_emissions": True,   # True: calculation with static emissions, 
                                         # False: calculation with timevariant emissions
                     "rev_emissions": True,      # True: emissions revenues for feed-in
                                         # False: no emissions revenues for feed-in
                     "dhw_electric": True,       # define if dhw is provided decentrally by electricity
                     "P_pv": 20.0,               # installed peak PV power
                     "hp_mode": "energy_opt",    # choose between "off" (no hp) and "energy_opt" and "grid_opt"
                     "T_VL": 35,                 # choose between 35 and 55 "Vorlauftemperatur" 
                     "alpha_th": 0.8,            # relative size of heat pump (between 0 and 1)
                     "beta_th": 0.5,             # relative size of thermal energy storage (between 0 and 1)
                     "EV_mode": "off",         # choose between "off" (no EVs), "on_demand", "grid_reactive" and "bi_directional"
                     "show_grid_plots" : True,   # show gridplots before and after optimization
                     "phi" : 25.842              # 
                     }
options["case_3"] = {"static_emissions": True,   # True: calculation with static emissions, 
                                         # False: calculation with timevariant emissions
                     "rev_emissions": True,      # True: emissions revenues for feed-in
                                         # False: no emissions revenues for feed-in
                     "dhw_electric": True,       # define if dhw is provided decentrally by electricity
                     "P_pv": 20.0,               # installed peak PV power
                     "hp_mode": "energy_opt",    # choose between "off" (no hp) and "energy_opt" and "grid_opt"
                     "T_VL": 35,                 # choose between 35 and 55 "Vorlauftemperatur" 
                     "alpha_th": 0.8,            # relative size of heat pump (between 0 and 1)
                     "beta_th": 0.5,             # relative size of thermal energy storage (between 0 and 1)
                     "EV_mode": "off",         # choose between "off" (no EVs), "on_demand", "grid_reactive" and "bi_directional"
                     "show_grid_plots" : True,   # show gridplots before and after optimization
                     "phi" : 25.842              # 
                     }
options["case_4"] = {"static_emissions": True,   # True: calculation with static emissions, 
                                         # False: calculation with timevariant emissions
                     "rev_emissions": True,      # True: emissions revenues for feed-in
                                         # False: no emissions revenues for feed-in
                     "dhw_electric": True,       # define if dhw is provided decentrally by electricity
                     "P_pv": 20.0,               # installed peak PV power
                     "hp_mode": "energy_opt",    # choose between "off" (no hp) and "energy_opt" and "grid_opt"
                     "T_VL": 35,                 # choose between 35 and 55 "Vorlauftemperatur" 
                     "alpha_th": 0.8,            # relative size of heat pump (between 0 and 1)
                     "beta_th": 0.5,             # relative size of thermal energy storage (between 0 and 1)
                     "EV_mode": "on_demand",         # choose between "off" (no EVs), "on_demand", "grid_reactive" and "bi_directional"
                     "show_grid_plots" : True,   # show gridplots before and after optimization
                     "phi" : 25.842              # 
                     }
options["case_5"] = {"static_emissions": True,   # True: calculation with static emissions, 
                                         # False: calculation with timevariant emissions
                     "rev_emissions": True,      # True: emissions revenues for feed-in
                                         # False: no emissions revenues for feed-in
                     "dhw_electric": True,       # define if dhw is provided decentrally by electricity
                     "P_pv": 20.0,               # installed peak PV power
                     "hp_mode": "energy_opt",    # choose between "off" (no hp) and "energy_opt" and "grid_opt"
                     "T_VL": 35,                 # choose between 35 and 55 "Vorlauftemperatur" 
                     "alpha_th": 0.8,            # relative size of heat pump (between 0 and 1)
                     "beta_th": 0.5,             # relative size of thermal energy storage (between 0 and 1)
                     "EV_mode": "on_demand",         # choose between "off" (no EVs), "on_demand", "grid_reactive" and "bi_directional"
                     "show_grid_plots" : True,   # show gridplots before and after optimization
                     "phi" : 25.842              # 
                     }

#%% For loop
cases = [1]#,2]#,3,4,5]
results = {}
all_results = {}

result_book = xlsxwriter.Workbook(resultFolder + "\\" + resultname)
sheet_0 = result_book.add_worksheet("Info")    
sheet_0.write(0, 0, "Erläuterung der Auswertung")
sheet_0.write(1, 0, "bzw. Veränderungen zwischen den Cases")

for n in cases:
    print("Loop Nr. "+ str(n))
    bt1 = building["case_" + str(n)]["building_type"]
    bt2 = building["case_" + str(n)]["building_type"]
    bA = building["case_" + str(n)]["building_age"]
    emY = building["case_" + str(n)]["emission_year"]
    distOpt = district_options["case_" + str(n)]
    opt = options["case_" + str(n)]
    
    (gridnodes, load_with, outputs) = run.run(bt1, bt2, bA, emY, distOpt, opt)

#%% results     
    all_results["case_" + str(n)] = dict(outputs)
    
    # names of results to evaluate
    resultlist = {"nodeLines","res_capacity", "res_voltNode"}
    
    for key in resultlist:
        results["case_" + str(n) + "_" + key] = outputs[key].copy()
    for key in load_with:
        results["case_" + str(n) + "_" + key] = dict(load_with[key])
    
    results_each = {}
    #for key in resultlist:
    results_each["Capacity"] = outputs["res_capacity"]
    
    for key in load_with:
        #node = 0
        results_each[key] = list()
        for node in gridnodes:
            #results_each[key] = (load_with[key].copy())
            #results_each[key] = dict()
            
            if node in load_with[key]:
                #results_each[key][node] = node #load_with[key]
                results_each[key].append(node)
            else:
                #results_each[key][node] = 0
                results_each[key].append(0)

#    for key in gridnodes:
#        if key in load_with:    
#            results_each[key] = load_with[key]
#        else:
#            results_each[key] = 0
    
#    result_book = xlsxwriter.Workbook(resultFolder + "\\" + resultname)
#    sheet_0 = result_book.add_worksheet("Info")    
#    sheet_0.write(0, 0, "Erläuterung der Auswertung bzw. Veränderungen")
    
#    for sheets in cases:
    col = -1
    sheet = result_book.add_worksheet("case " + str(n))
    for key in results_each:
        row = 0
        col +=1
        sheet.write(row, col, key)
        for item in results_each[key]: 
            row += 1
            sheet.write(row, col, str(item)) # outputs[key][item]) 

#    for key in load_with:
#        row = 0
#        col +=1
#        sheet.write(row, col, key)
#        for item in results["case_" + str(n) + "_" + key]: 
#            row += 1
#            sheet.write(row, col, str(item))

result_book.close() 




