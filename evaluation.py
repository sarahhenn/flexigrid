# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 15:10:05 2019

@author: Chrissi
"""

import run_battery_opti_place_size_2 as run
import xlsxwriter
<<<<<<< HEAD
import numpy as np

# resultname should be an explanation of the proposed evaluation
resultname = "build_age_dorf_dist.xlsx"
voltname = "build_age_dorf_volt.xlsx"
=======

# resultname should be an explanation of the proposed evaluation
resultname = "Test1.xlsx"
>>>>>>> eea82d3593cb2cf14447edd603668a091d28651a
resultFolder = "C:\\Users\\Chrissi\\Git\\Flexigrid\\Auswertung"

#%% Parameters

building = {}
building["case_1"] = {"building_type" : "EFH",       # EFH, ZFH, MFH_6WE, MFH_10WE, MFH_15WE
                      "building_type2" : "MFH_6WE",  # second building type, query "mfh"
<<<<<<< HEAD
                      "building_age"  : "1960",      # 1960, 1980, 2005 
=======
                      "building_age"  : "2005",      # 1960, 1980, 2005 
>>>>>>> eea82d3593cb2cf14447edd603668a091d28651a
                      "emission_year" : "2017",      # 2017, 2030, 2050 
          }
building["case_2"] = {"building_type" : "EFH",       # EFH, ZFH, MFH_6WE, MFH_10WE, MFH_15WE
                      "building_type2" : "MFH_6WE",  # second building type, query "mfh"
<<<<<<< HEAD
                      "building_age"  : "1980",      # 1960, 1980, 2005 
=======
                      "building_age"  : "2005",      # 1960, 1980, 2005 
>>>>>>> eea82d3593cb2cf14447edd603668a091d28651a
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
<<<<<<< HEAD
district_options["case_1"] = {"mfh" : 0.35,                  # ratio of MFH to EFH in %
                              "pv" : 0.1,                    # ratio in %
                              "hp" : 0.1,                    # ratio in %
                              "ev" : 0.1,                    # ratio in %
                              "case" : "random"               # "best", "worst" and "random"
                              }
district_options["case_2"] = {"mfh" : 0.35,                  # ratio of MFH to EFH in %
                              "pv" : 0.1,                    # ratio in %
                              "hp" : 0.1,                    # ratio in %
                              "ev" : 0.1,                    # ratio in %
                              "case" : "random"               # "best", "worst" and "random"
                              }
district_options["case_3"] = {"mfh" : 0.35,                  # ratio of MFH to EFH in %
                              "pv" : 0.1,                    # ratio in %
                              "hp" : 0.1,                    # ratio in %
                              "ev" : 0.1,                    # ratio in %
                              "case" : "random"               # "best", "worst" and "random"
                              }
district_options["case_4"] = {"mfh" : 0.35,                  # ratio of MFH to EFH in %
                              "pv" : 0.1,                    # ratio in %
                              "hp" : 0.1,                    # ratio in %
                              "ev" : 0.1,                    # ratio in %
                              "case" : "random"               # "best", "worst" and "random"
                              }
district_options["case_5"] = {"mfh" : 0.35,                  # ratio of MFH to EFH in %
                              "pv" : 0.1,                    # ratio in %
                              "hp" : 0.1,                    # ratio in %
=======
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
>>>>>>> eea82d3593cb2cf14447edd603668a091d28651a
                              "ev" : 0.1,                    # ratio in %
                              "case" : "random"               # "best", "worst" and "random"
                              }

options = {}
<<<<<<< HEAD
options["case_1"] = {"static_emissions": False,   # True: calculation with static emissions, 
=======
options["case_1"] = {"static_emissions": True,   # True: calculation with static emissions, 
>>>>>>> eea82d3593cb2cf14447edd603668a091d28651a
                                         # False: calculation with timevariant emissions
                     "rev_emissions": True,      # True: emissions revenues for feed-in
                                         # False: no emissions revenues for feed-in
                     "dhw_electric": True,       # define if dhw is provided decentrally by electricity
<<<<<<< HEAD
                     "P_pv": 10.0,               # installed peak PV power
=======
                     "P_pv": 20.0,               # installed peak PV power
>>>>>>> eea82d3593cb2cf14447edd603668a091d28651a
                     "hp_mode": "energy_opt",    # choose between "off" (no hp) and "energy_opt" and "grid_opt"
                     "T_VL": 35,                 # choose between 35 and 55 "Vorlauftemperatur" 
                     "alpha_th": 0.8,            # relative size of heat pump (between 0 and 1)
                     "beta_th": 0.5,             # relative size of thermal energy storage (between 0 and 1)
<<<<<<< HEAD
                     "EV_mode": "on_demand",         # choose between "off" (no EVs), "on_demand", "grid_reactive" and "bi_directional"
                     "show_grid_plots" : True,   # show gridplots before and after optimization
                     "phi" : 25.842,              # 
                     "opt_costs": False
                     }
options["case_2"] = {"static_emissions": False,   # True: calculation with static emissions, 
=======
                     "EV_mode": "off",         # choose between "off" (no EVs), "on_demand", "grid_reactive" and "bi_directional"
                     "show_grid_plots" : True,   # show gridplots before and after optimization
                     "phi" : 25.842              # 
                     }
options["case_2"] = {"static_emissions": True,   # True: calculation with static emissions, 
>>>>>>> eea82d3593cb2cf14447edd603668a091d28651a
                                         # False: calculation with timevariant emissions
                     "rev_emissions": True,      # True: emissions revenues for feed-in
                                         # False: no emissions revenues for feed-in
                     "dhw_electric": True,       # define if dhw is provided decentrally by electricity
<<<<<<< HEAD
                     "P_pv": 10.0,               # installed peak PV power
=======
                     "P_pv": 20.0,               # installed peak PV power
>>>>>>> eea82d3593cb2cf14447edd603668a091d28651a
                     "hp_mode": "energy_opt",    # choose between "off" (no hp) and "energy_opt" and "grid_opt"
                     "T_VL": 35,                 # choose between 35 and 55 "Vorlauftemperatur" 
                     "alpha_th": 0.8,            # relative size of heat pump (between 0 and 1)
                     "beta_th": 0.5,             # relative size of thermal energy storage (between 0 and 1)
<<<<<<< HEAD
                     "EV_mode": "on_demand",         # choose between "off" (no EVs), "on_demand", "grid_reactive" and "bi_directional"
                     "show_grid_plots" : True,   # show gridplots before and after optimization
                     "phi" : 25.842,              # 
                     "opt_costs": False
                     }
options["case_3"] = {"static_emissions": False,   # True: calculation with static emissions, 
=======
                     "EV_mode": "off",         # choose between "off" (no EVs), "on_demand", "grid_reactive" and "bi_directional"
                     "show_grid_plots" : True,   # show gridplots before and after optimization
                     "phi" : 25.842              # 
                     }
options["case_3"] = {"static_emissions": True,   # True: calculation with static emissions, 
>>>>>>> eea82d3593cb2cf14447edd603668a091d28651a
                                         # False: calculation with timevariant emissions
                     "rev_emissions": True,      # True: emissions revenues for feed-in
                                         # False: no emissions revenues for feed-in
                     "dhw_electric": True,       # define if dhw is provided decentrally by electricity
<<<<<<< HEAD
                     "P_pv": 10.0,               # installed peak PV power
=======
                     "P_pv": 20.0,               # installed peak PV power
>>>>>>> eea82d3593cb2cf14447edd603668a091d28651a
                     "hp_mode": "energy_opt",    # choose between "off" (no hp) and "energy_opt" and "grid_opt"
                     "T_VL": 35,                 # choose between 35 and 55 "Vorlauftemperatur" 
                     "alpha_th": 0.8,            # relative size of heat pump (between 0 and 1)
                     "beta_th": 0.5,             # relative size of thermal energy storage (between 0 and 1)
<<<<<<< HEAD
                     "EV_mode": "on_demand",         # choose between "off" (no EVs), "on_demand", "grid_reactive" and "bi_directional"
                     "show_grid_plots" : True,   # show gridplots before and after optimization
                     "phi" : 25.842 ,              # 
                     "opt_costs": False
                     }
options["case_4"] = {"static_emissions": False,   # True: calculation with static emissions, 
=======
                     "EV_mode": "off",         # choose between "off" (no EVs), "on_demand", "grid_reactive" and "bi_directional"
                     "show_grid_plots" : True,   # show gridplots before and after optimization
                     "phi" : 25.842              # 
                     }
options["case_4"] = {"static_emissions": True,   # True: calculation with static emissions, 
>>>>>>> eea82d3593cb2cf14447edd603668a091d28651a
                                         # False: calculation with timevariant emissions
                     "rev_emissions": True,      # True: emissions revenues for feed-in
                                         # False: no emissions revenues for feed-in
                     "dhw_electric": True,       # define if dhw is provided decentrally by electricity
<<<<<<< HEAD
                     "P_pv": 10.0,               # installed peak PV power
=======
                     "P_pv": 20.0,               # installed peak PV power
>>>>>>> eea82d3593cb2cf14447edd603668a091d28651a
                     "hp_mode": "energy_opt",    # choose between "off" (no hp) and "energy_opt" and "grid_opt"
                     "T_VL": 35,                 # choose between 35 and 55 "Vorlauftemperatur" 
                     "alpha_th": 0.8,            # relative size of heat pump (between 0 and 1)
                     "beta_th": 0.5,             # relative size of thermal energy storage (between 0 and 1)
                     "EV_mode": "on_demand",         # choose between "off" (no EVs), "on_demand", "grid_reactive" and "bi_directional"
                     "show_grid_plots" : True,   # show gridplots before and after optimization
<<<<<<< HEAD
                     "phi" : 25.842 ,              # 
                     "opt_costs": False
                     }
options["case_5"] = {"static_emissions": False,   # True: calculation with static emissions, 
=======
                     "phi" : 25.842              # 
                     }
options["case_5"] = {"static_emissions": True,   # True: calculation with static emissions, 
>>>>>>> eea82d3593cb2cf14447edd603668a091d28651a
                                         # False: calculation with timevariant emissions
                     "rev_emissions": True,      # True: emissions revenues for feed-in
                                         # False: no emissions revenues for feed-in
                     "dhw_electric": True,       # define if dhw is provided decentrally by electricity
<<<<<<< HEAD
                     "P_pv": 10.0,               # installed peak PV power
=======
                     "P_pv": 20.0,               # installed peak PV power
>>>>>>> eea82d3593cb2cf14447edd603668a091d28651a
                     "hp_mode": "energy_opt",    # choose between "off" (no hp) and "energy_opt" and "grid_opt"
                     "T_VL": 35,                 # choose between 35 and 55 "Vorlauftemperatur" 
                     "alpha_th": 0.8,            # relative size of heat pump (between 0 and 1)
                     "beta_th": 0.5,             # relative size of thermal energy storage (between 0 and 1)
                     "EV_mode": "on_demand",         # choose between "off" (no EVs), "on_demand", "grid_reactive" and "bi_directional"
                     "show_grid_plots" : True,   # show gridplots before and after optimization
<<<<<<< HEAD
                     "phi" : 25.842,              # 
                     "opt_costs": False
                     }

#%% For loop
# Freischalten der Case-Anzahl
#cases = [1]
#cases = [1,2]
cases = [1,2,3]
#cases = [1,2,3,4]
#cases = [1,2,3,4,5]

results = {}
all_results = {}
results_each = {}

U_node_max_day = {}
U_node_min_day = {}
U_node_max = {}
U_node_min = {}
powerline_max_day = {}
powerline_min_day = {}
powerline_max = {}
powerline_min = {}
worstnode = {}

    
result_book = xlsxwriter.Workbook(resultFolder + "\\" + resultname)
rsheet_0 = result_book.add_worksheet("Info")    
rsheet_0.write(0, 0, "Erläuterung der Auswertung")
rsheet_0.write(1, 0, "bzw. Veränderungen zwischen den Cases")

voltage_book = xlsxwriter.Workbook(resultFolder + "\\" + voltname)
vSheet_0 = voltage_book.add_worksheet("Info Auswertung")  
vSheet_0.write(0, 0, "Vergleichsfall erster Test zur Auswertung")

for case in cases:
    print("Calculating case "+ str(case))
    bt1 = building["case_" + str(case)]["building_type"]
    bt2 = building["case_" + str(case)]["building_type"]
    bA = building["case_" + str(case)]["building_age"]
    emY = building["case_" + str(case)]["emission_year"]
    distOpt = district_options["case_" + str(case)]
    opt = options["case_" + str(case)]
    
    (gridnodes, load_with, outputs) = run.run(bt1, bt2, bA, emY, distOpt, opt)
    all_results["case_" + str(case)] = dict(outputs)
    nodelines = {}
    nodelines = outputs["nodeLines"]
    
    results_each["case_" + str(case)] = {}
    for key in load_with:
        results_each["case_" + str(case)][key] = dict()
        for node in gridnodes:
            if node in load_with[key]:
                results_each["case_" + str(case)][key][node] = node 
            else:
                results_each["case_" + str(case)][key][node] = 0    
        
    
#%% results     
    
#for case in cases:    
    
    for key in load_with:
        results["case_" + str(case) + "_" + key] = dict(load_with[key])
    
    U_node_max_day["case_" + str(case)] = {}
    U_node_min_day["case_" + str(case)] = {}
    U_node_max["case_" + str(case)] = {}
    U_node_min["case_" + str(case)] = {}
    days = range (0,12)  
    for node in gridnodes:
        U_node_max_day["case_" + str(case)][node] = {}
        U_node_min_day["case_" + str(case)][node] = {}
        for d in days:                             
            U_node_max_day["case_" + str(case)][node][d] = max(all_results["case_" + str(case)]["res_voltNode"][node][d])
            U_node_min_day["case_" + str(case)][node][d] = min(all_results["case_" + str(case)]["res_voltNode"][node][d])
        U_node_min["case_" + str(case)][node] = np.min(all_results["case_" + str(case)]["res_voltNode"][node])
        U_node_max["case_" + str(case)][node] = np.max(all_results["case_" + str(case)]["res_voltNode"][node])
        
    powerline_max_day["case_" + str(case)] = {}
    powerline_min_day["case_" + str(case)] = {}
    powerline_max["case_" + str(case)] = {}
    powerline_min["case_" + str(case)] = {}    
    days = range (0,12)  
    for n,m in nodelines:
        powerline_max_day["case_" + str(case)][n,m] = {}
        powerline_min_day["case_" + str(case)][n,m] = {}
        for d in days:                             
           powerline_max_day["case_" + str(case)][n,m][d] = max(all_results["case_" + str(case)]["res_powerLine"][n,m][d])
           powerline_min_day["case_" + str(case)][n,m][d] = min(all_results["case_" + str(case)]["res_powerLine"][n,m][d])
        powerline_max["case_" + str(case)][n,m] = np.max(all_results["case_" + str(case)]["res_powerLine"][n,m])
        powerline_min["case_" + str(case)][n,m] = np.min(all_results["case_" + str(case)]["res_powerLine"][n,m])      
        
        
#%% Print to excel    

    col = 0
    sheet = result_book.add_worksheet("case " + str(case))
    for key in results_each["case_" + str(case)]:
        row = 0
        col +=1
        sheet.write(row, col, key)
        for item in gridnodes:
            row += 1
            if item in results_each["case_" + str(case)][key]:
                sheet.write(row, col, results_each["case_" + str(case)][key][item]) 
            else:
                sheet.write(row, col, 0)
            sheet.write(node+1, 0, all_results["case_" + str(case)]["res_capacity"][item]) #into distsheet
        


    voltsheet_max = voltage_book.add_worksheet("case " + str(case) + "_max_volatge")
    voltsheet_min = voltage_book.add_worksheet("case " + str(case) + "_min_voltage")   
    
    voltsheet_max.write(0, 1, "Capacity")
    voltsheet_min.write(0, 1, "Capacity")

    voltsheet_max.write(0, 2, "max_overall")
    voltsheet_min.write(0, 2, "max_overall")
    
    for node in gridnodes:
        voltsheet_max.write(node+1, 0, "Node " +str (node))
        voltsheet_max.write(node+1, 1, all_results["case_" + str(case)]["res_capacity"][node]) #into distsheet
        voltsheet_max.write(node+1, 2, U_node_max["case_" + str(case)][node])
        
        voltsheet_min.write(node+1, 0, "Node " +str (node))
        voltsheet_min.write(node+1, 1, all_results["case_" + str(case)]["res_capacity"][node])
        voltsheet_min.write(node+1, 2, U_node_min["case_" + str(case)][node])
    for d in days:
        voltsheet_max.write(0, d+3, "Day " + str(d+1))
        voltsheet_min.write(0, d+3, "Day " + str(d+1))
    for node in gridnodes:
        for d in days:
            voltsheet_max.write(node+1, d+3, U_node_max_day["case_" + str(case)][node][d])
            voltsheet_min.write(node+1, d+3, U_node_min_day["case_" + str(case)][node][d])

    powersheet_max = voltage_book.add_worksheet("case " + str(case) + "_max_power")
    powersheet_min = voltage_book.add_worksheet("case " + str(case) + "_min_power")
    
    powersheet_max.write(0, 0, "Battery on target node")
    powersheet_min.write(0, 0, "Battery on target node")     
    
    powersheet_max.write(0, 1, "Capacity")
    powersheet_min.write(0, 1, "Capacity")

    powersheet_max.write(0, 2, "max_overall")
    powersheet_min.write(0, 2, "max_overall")

    for n,m in nodelines:
        powersheet_max.write(m-1, 0, "Line " + str(n) + "," + str(m))
        powersheet_max.write(m-1, 1, all_results["case_" + str(case)]["res_capacity"][m])
        powersheet_max.write(m-1, 2, powerline_max["case_" + str(case)][n,m])
        
        powersheet_min.write(m-1, 0, "Line " +str(n) + "," + str(m))
        powersheet_min.write(m-1, 1, all_results["case_" + str(case)]["res_capacity"][m])
        powersheet_min.write(m-1, 2, powerline_min["case_" + str(case)][n,m])
    for d in days:
        powersheet_max.write(0, d+3, "Day " + str(d+1))
        powersheet_min.write(0, d+3, "Day " + str(d+1))
    for n,m in nodelines:
        for d in days:
            powersheet_max.write(m-1, d+3, powerline_max_day["case_" + str(case)][n,m][d])
            powersheet_min.write(m-1, d+3, powerline_min_day["case_" + str(case)][n,m][d])     
      
        all_results["case_" + str(case)]["res_capacity"][node]

row = 3
col = 4
rsheet_0.write(row, col, "Loads on each branch")
#for branchloads in all_results["case_1"]["loads_per_branch"]:
#    row+=1 #falsch?
#    for branch in all_results["case_1"]["loads_per_branch"][branchloads]:
#        rsheet_0.write(row, col, all_results["case_1"]["loads_per_branch"][branch])
#        #for load in branch
#        col+=1
#
#

result_book.close() 
voltage_book.close()
=======
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

>>>>>>> eea82d3593cb2cf14447edd603668a091d28651a



