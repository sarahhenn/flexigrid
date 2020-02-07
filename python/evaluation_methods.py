"""
created on Thursday 06/02/2020
@ author: fpo

file for normal evaluation

"""

import pickle
# own functions
import python.grid_optimization as opti
import python.grid_optimization_oNB as opti_oNB
import python.timeloop_flexigrid as loop

def normal_evaluation(net, options, days, gridnodes, timesteps, nodes, eco, devs, params,clustered,fkt):

    # values for pareto opti, not relevant for other analysis
    emissions_max = 1000000000
    costs_max = 1000000000

    # solution_found as continuos variable for while loop
    solution_found = []
    for d in days:
        solution_found.append(False)
    boolean_loop = True

    # constraint_apc models APC, gets reduced from 1 to 0 in iteration steps with range 0.1
    constraint_apc = {}
    # constraint for Injection and Subtraction. Inj gets cut when voltage is too high, Subtr gets cut when voltage is too low
    constraint_InjMin = {}
    constraint_SubtrMin = {}
    constraint_InjMax = {}
    constraint_SubtrMax = {}

    # create array to flag whether values are critical for powerflow. If not critical: 0, if critical: 1
    critical_flag = {}
    iteration_counter = 0
    # introduce boolean to state infeasability
    infeasability = False

    change_relative = options["change_relative_node_violation_rel"]
    change_value = options["change_value_node_violation_abs"]

    for n in gridnodes:
        for d in days:
            constraint_apc[n, d] = 0
            for t in timesteps:
                critical_flag[n, d, t] = 0
                constraint_InjMin[n, d, t] = 0
                constraint_SubtrMin[n,d,t] = 0
                constraint_InjMax[n,d,t] = 10000
                constraint_SubtrMax[n, d, t] = 10000

    while boolean_loop:
        print("")
        print("!!! Iteration counter is currently at " +str(iteration_counter) + "!!!")
        print("")

        #run DC-optimization
        (costs_grid, emissions_grid, timesteps, days, powInjRet, powSubtrRet, gridnodes, res_exBat, powInjPrev, powSubtrPrev, emissions_nodes,
         costs_nodes, objective_function) = opti.compute(net, nodes, gridnodes, days, timesteps, eco, devs, clustered,params,
         options, constraint_apc, constraint_InjMin, constraint_SubtrMin, constraint_InjMax, constraint_SubtrMax,emissions_max, costs_max)

        # run AC-Powerflow-Solver
        (output_dir, critical_flag, solution_found,vm_pu_total) = loop.run_timeloop(fkt, timesteps, days, powInjRet, powSubtrRet,gridnodes, critical_flag, solution_found)
        print("zwischenstop")
        for d in days:
            if (solution_found[d] == False):

                print("Additional constrains have to be set for day" +str(d))
                if options["apc_while_voltage_violation"]:
                    if options["cut_Inj/Subtr_while_voltage_violation"]:
                        print("You selected both apc and Inj/Subtr cutting in case of voltage violations.")
                        for n in gridnodes:
                            if (any(critical_flag[n, d, t] == 1 for t in timesteps)):
                                if(any(vm_pu_total[n,d,t] < 0.96 for t in timesteps)):
                                    pass
                                elif(any(vm_pu_total[n,d,t] > 1.04 for t in timesteps)):
                                    constraint_apc[n,d] += 0.1
                                    if(constraint_apc[n,d] >= 1):
                                        print("You have reached the maximum amount of curtailment!")
                                        print("Will set curtailment to 100 Percent automatically.")
                                        constraint_apc[n,d] = 1
                                else:
                                    pass
                                for t in timesteps:
                                    if(critical_flag[n,d,t] == 1):
                                        if (vm_pu_total[n,d,t] < 0.96):
                                            if options["rel1_or_abs0_violation_change"]:
                                                # relative Lösung wirft Problem der Spannungsweiterleitung auf
                                                constraint_SubtrMax[n,d,t] = change_relative * powSubtrPrev[n,d,t]
                                            else:
                                                # absolute Regelung:
                                                if (powSubtrPrev[n, d, t] < change_value):
                                                    constraint_SubtrMax[n, d, t] = 0
                                                    print("Subtraction already set to 0 for node" +str(n) + " and timestep" +str(t))
                                                    print("Raising Injection now!")
                                                    constraint_InjMin[n,d,t] += change_value - powSubtrPrev[n,d,t]
                                                else:
                                                    constraint_SubtrMax[n,d,t] = powSubtrPrev[n,d,t] - change_value

                                        elif (vm_pu_total[n,d,t] > 1.04):
                                            if options["rel1_or_abs0_violation_change"]:
                                                constraint_InjMax[n, d, t] = change_relative * powInjPrev[n,d,t]
                                            else:
                                                #absolute änderung
                                                if (powInjPrev[n,d,t] < change_value):
                                                    constraint_InjMax[n,d,t] = 0
                                                    print("Injection already set to 0 for node" +str(n) + " and timestep" +str(t))
                                                    print("Raising Subtraction now!")
                                                    constraint_SubtrMin[n,d,t] += change_value -  powInjPrev[n,d,t]
                                                else:
                                                    constraint_InjMax[n,d,t] = powInjPrev[n,d,t] - change_value

                    else:
                        print("You selected only apc in case of voltage violations.")
                        for n in gridnodes:
                            if (any(critical_flag[n, d, t] == 1 for t in timesteps)):
                                if (any(vm_pu_total[n,d,t] < 0.96 for t in timesteps)):
                                    print("Only apc will not fix any voltage issues, because the load is too high on day" +str(d))
                                    infeasability = True
                                elif (any(vm_pu_total[n,d,t] > 1.04 for t in timesteps)):
                                    constraint_apc[n, d] += 0.1
                                    if (constraint_apc[n, d] >= 1):
                                        print("You have reached the maximal amount of curtailment!")
                                        print("Will set curtailment to 100 Percent automatically.")
                                        constraint_apc[n, d] = 1
                                        print("Since you only selected apc, it has reached 100 Percent and you haven't found a solution, the problem appears to be infeasable for these settings!")
                                        infeasability = True

                elif (options["cut_Inj/Subtr_while_voltage_violation"] == True and options["apc_while_voltage_violation"] == False):
                    print("You selected only Inj/Subtr cutting in case of voltage violations.")
                    for n in gridnodes:
                        for t in timesteps:
                            if (critical_flag[n, d, t] == 1):
                                if (vm_pu_total[n, d, t] < 0.96):
                                    if options["rel1_or_abs0_violation_change"]:
                                        constraint_SubtrMax[n, d, t] = change_relative * powSubtrPrev[n, d, t]
                                    else:
                                        # absolute Regelung:
                                        if (powSubtrPrev[n, d, t] < change_value):
                                            constraint_SubtrMax[n, d, t] = 0
                                            print("Subtraction already set to 0 for node" +str(n) + " and timestep" +str(t))
                                            print("Raising Injection now!")
                                            constraint_InjMin[n,d,t] += change_value - powSubtrPrev[n,d,t]
                                        else:
                                            constraint_SubtrMax[n,d,t] = powSubtrPrev[n,d,t] - change_value

                                elif (vm_pu_total[n, d, t] > 1.04):
                                    if options["rel1_or_abs0_violation_change"]:
                                        constraint_InjMax[n, d, t] = change_relative * powInjPrev[n,d,t]
                                    else:
                                        #absolute Regelung
                                        if (powInjPrev[n, d, t] < change_value):
                                            constraint_InjMax[n, d, t] = 0
                                            print("Injection already set to 0 for node" + str(n) + " and timestep" + str(t))
                                            print("Raising Subtraction now!")
                                            constraint_SubtrMin[n, d, t] += change_value - powInjPrev[n,d,t]
                                        else:
                                            constraint_InjMax[n, d, t] = powInjPrev[n, d, t] - change_value

                elif (options["cut_Inj/Subtr_while_voltage_violation"] == False and options["apc_while_voltage_violation"] == False):
                    print("Error: You did not select any measure in case of voltage violations!")
                    infeasability = True

            if(solution_found[d] == True):
                    print("Solution was successfully found for day" + str(d))

        if infeasability:
            print("Error: Model appears to be infeasable for the selected settings!")
            print("Reasons are stated above.")
            break

        iteration_counter += 1

        if all(solution_found[d] == True for d in days):
            print("Congratulations! Your optimization and loadflow calculation has been successfully finished after " + str(iteration_counter - 1) + " iteration steps!")
            break

    return(objective_function)

def pareto_evaluation_oNB(net, options, days, gridnodes, timesteps, nodes, eco, devs, params,clustered,fkt, building_type, building_age):

    # define starting values of emissions and cost max for pareto loop
    emissions_max_global = 1000000000
    costs_max_global = 1000000000

    # solution_found as continuos variable for while loop
    solution_found = []
    for d in days:
        solution_found.append(False)
    boolean_loop = True
    # constraint_apc models APC, gets reduced from 1 to 0 in iteration steps with range 0.1
    constraint_apc = {}
    # constraint for Injection and Subtraction. Inj gets cut when voltage is too high, Subtr gets cut when voltage is too low
    constraint_InjMin = {}
    constraint_SubtrMin = {}
    constraint_InjMax = {}
    constraint_SubtrMax = {}
    # create array to flag whether values are critical for powerflow. If not critical: 0, if critical: 1
    critical_flag = {}
    iteration_counter = 0
    # introduce boolean to state infeasability
    infeasability = False

    change_relative = options["change_relative_node_violation_rel"]
    change_value = options["change_value_node_violation_abs"]

    for n in gridnodes:
        for d in days:
            for t in timesteps:
                critical_flag[n, d, t] = 0
                constraint_apc[n, d] = 0
                constraint_InjMin[n, d, t] = 0
                constraint_SubtrMin[n, d, t] = 0
                constraint_InjMax[n, d, t] = 1000000000000000000000
                constraint_SubtrMax[n, d, t] = 1000000000000000000000

    print("")
    print("!!! Iteration counter is currently at " + str(iteration_counter) + "!!!")
    print("")

    # run DC-optimization
    (
        costs_grid, emissions_grid, timesteps, days, powInjRet, powSubtrRet, gridnodes, res_exBat, powInjPrev,
        powSubtrPrev,
        emissions_nodes, costs_min, emissions_max_global) = opti_oNB.compute(net, nodes, gridnodes, days, timesteps, eco,
                                                                         devs,
                                                                         clustered, params, options, constraint_apc,
                                                                         constraint_InjMin, constraint_SubtrMin,
                                                                         constraint_InjMax, constraint_SubtrMax,
                                                                         critical_flag, emissions_max_global,
                                                                         costs_max_global)

    print("this is the end")
    print("")
    print("")
    print("costs minimized are:")
    print(costs_min)
    print("emissions max are:")
    print(emissions_max_global)
    print("")
    print("")
    print("")

    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    print("hier ist das originalprogramm ")

    results_pareto = {}
    results_pareto["Minimale Kosten"] = costs_min
    results_pareto["Maximale Emissionen"] = emissions_max_global
    costlist = []
    emissionslist = []
    simulationslist = []

    # hier drunter müssen die anderen loops, neue printbefehle müssen individuell eingestellt werden und die results ordner müssen separat erstellt werden.

    options["opt_costs"] = False
    options["opt_emissions"] = True
    options["filename_results"] = "results/pareto_results/" + building_type + "_" + \
                                  building_age + "pareto_EmissionsMin.pkl"

    options["filename_inputs"] = "results/pareto_results/inputs_" + building_type + "_" + \
                                 building_age + "pareto_EmissionsMin.pkl"

    # %% Store clustered input parameters
    with open(options["filename_inputs"], "wb") as f_in:
        pickle.dump(clustered, f_in, pickle.HIGHEST_PROTOCOL)

    # !!!!!!!!!!! COPY FROM HERE FOR PARETO LOOOOOOOP !!!!!!!!!!!!!!!!!!!!!!!!

    # solution_found as continuos variable for while loop
    solution_found = []
    for d in days:
        solution_found.append(False)
    boolean_loop = True
    # constraint_apc models APC, gets reduced from 1 to 0 in iteration steps with range 0.1
    constraint_apc = {}
    # constraint for Injection and Subtraction. Inj gets cut when voltage is too high, Subtr gets cut when voltage is too low
    constraint_InjMin = {}
    constraint_SubtrMin = {}
    constraint_InjMax = {}
    constraint_SubtrMax = {}
    # create array to flag whether values are critical for powerflow. If not critical: 0, if critical: 1
    critical_flag = {}
    iteration_counter = 0
    # introduce boolean to state infeasability
    infeasability = False

    change_relative = options["change_relative_node_violation_rel"]
    change_value = options["change_value_node_violation_abs"]

    for n in gridnodes:
        for d in days:
            for t in timesteps:
                critical_flag[n, d, t] = 0
                constraint_apc[n, d] = 0
                constraint_InjMin[n, d, t] = 0
                constraint_SubtrMin[n, d, t] = 0
                constraint_InjMax[n, d, t] = 10000
                constraint_SubtrMax[n, d, t] = 10000

    print("")
    print("!!! Iteration counter is currently at " + str(iteration_counter) + "!!!")
    print("")

    # run DC-optimization
    (
        costs_grid, emissions_grid, timesteps, days, powInjRet, powSubtrRet, gridnodes, res_exBat, powInjPrev,
        powSubtrPrev,
        emissions_nodes, costs_max_global, emissions_min) = opti_oNB.compute(net, nodes, gridnodes, days, timesteps, eco,
                                                                         devs,
                                                                         clustered, params, options, constraint_apc,
                                                                         constraint_InjMin, constraint_SubtrMin,
                                                                         constraint_InjMax, constraint_SubtrMax,
                                                                         critical_flag, emissions_max_global,
                                                                         costs_max_global)

    print("this is the end")
    print("")
    print("")
    print("emissions minimized are:")
    print(emissions_min)
    print("costs max are:")
    print(costs_max_global)
    print("")
    print("")

    results_pareto["Maximale Kosten"] = costs_max_global
    results_pareto["Minimale Emissionen"] = emissions_min

    options["opt_costs"] = True
    options["opt_emissions"] = False

    number_simulations = 4
    for i in range(1, 1 + number_simulations):

        # !!!!!!!!!!! COPY FROM HERE FOR PARETO LOOOOOOOP !!!!!!!!!!!!!!!!!!!!!!!!

        options["filename_results"] = "results/pareto_results/" + building_type + "_" + \
                                      building_age + "pareto_CostsMin" + str(i / (number_simulations + 1)) + ".pkl"

        options["filename_inputs"] = "results/pareto_results/inputs_" + building_type + "_" + \
                                     building_age + "pareto_CostsMin" + str(i / (number_simulations + 1)) + ".pkl"

        emissions_max = emissions_min + (emissions_max_global - emissions_min) * i * (1 / (number_simulations + 1))

        # %% Store clustered input parameters

        with open(options["filename_inputs"], "wb") as f_in:
            pickle.dump(clustered, f_in, pickle.HIGHEST_PROTOCOL)

        # solution_found as continuos variable for while loop
        solution_found = []
        for d in days:
            solution_found.append(False)
        boolean_loop = True
        # constraint_apc models APC, gets reduced from 1 to 0 in iteration steps with range 0.1
        constraint_apc = {}
        # constraint for Injection and Subtraction. Inj gets cut when voltage is too high, Subtr gets cut when voltage is too low
        constraint_InjMin = {}
        constraint_SubtrMin = {}
        constraint_InjMax = {}
        constraint_SubtrMax = {}
        # create array to flag whether values are critical for powerflow. If not critical: 0, if critical: 1
        critical_flag = {}
        iteration_counter = 0
        # introduce boolean to state infeasability
        infeasability = False

        change_relative = options["change_relative_node_violation_rel"]
        change_value = options["change_value_node_violation_abs"]

        for n in gridnodes:
            for d in days:
                for t in timesteps:
                    critical_flag[n, d, t] = 0
                    constraint_apc[n, d] = 0
                    constraint_InjMin[n, d, t] = 0
                    constraint_SubtrMin[n, d, t] = 0
                    constraint_InjMax[n, d, t] = 1000000000000000000000
                    constraint_SubtrMax[n, d, t] = 1000000000000000000000

        print("")
        print("!!! Iteration counter is currently at " + str(iteration_counter) + "!!!")
        print("")

        # run DC-optimization
        (
            costs_grid, emissions_grid, timesteps, days, powInjRet, powSubtrRet, gridnodes, res_exBat, powInjPrev,
            powSubtrPrev,
            emissions_nodes, costs, emissions) = opti.compute(net, nodes, gridnodes, days, timesteps, eco, devs,
                                                              clustered, params, options, constraint_apc,
                                                              constraint_InjMin, constraint_SubtrMin,
                                                              constraint_InjMax, constraint_SubtrMax,
                                                              critical_flag, emissions_max, costs_max_global)

        print("this is the end")
        print("")
        print("")
        print("costs minimized are:")
        print(costs)
        print("for distance of emissions min from: " + str(i * 1 / (number_simulations + 1)))
        print("emissions are:")
        print(emissions)
        print("")
        print("")
        print("")

        costlist.append(costs)
        emissionslist.append(emissions)
        simulationslist.append(i * 1 / (number_simulations + 1))

    print("")
    print("")
    print("Extremwerte:")
    print(results_pareto)
    print("costlist:")
    print(costlist)
    print("emissionslist:")
    print(emissionslist)
    print("simulationspercentage:")
    print(simulationslist)

    print("strpo")

def pareto_evaluation_mNB(net, options, days, gridnodes, timesteps, nodes, eco, devs, params,clustered,fkt, building_type, building_age):

    # define starting values of emissions and cost max for pareto loop
    emissions_max_global = 1000000000
    costs_max_global = 1000000000

    # solution_found as continuos variable for while loop
    solution_found = []
    for d in days:
        solution_found.append(False)
    boolean_loop = True
    # constraint_apc models APC, gets reduced from 1 to 0 in iteration steps with range 0.1
    constraint_apc = {}
    # constraint for Injection and Subtraction. Inj gets cut when voltage is too high, Subtr gets cut when voltage is too low
    constraint_InjMin = {}
    constraint_SubtrMin = {}
    constraint_InjMax = {}
    constraint_SubtrMax = {}
    # create array to flag whether values are critical for powerflow. If not critical: 0, if critical: 1
    critical_flag = {}
    iteration_counter = 0
    # introduce boolean to state infeasability
    infeasability = False

    change_relative = options["change_relative_node_violation_rel"]
    change_value = options["change_value_node_violation_abs"]

    for n in gridnodes:
        for d in days:
            for t in timesteps:
                critical_flag[n, d, t] = 0
                constraint_apc[n, d] = 0
                constraint_InjMin[n, d, t] = 0
                constraint_SubtrMin[n, d, t] = 0
                constraint_InjMax[n, d, t] = 10000
                constraint_SubtrMax[n, d, t] = 10000

    while boolean_loop:
        print("")
        print("!!! Iteration counter is currently at " + str(iteration_counter) + "!!!")
        print("")

        # run DC-optimization
        (
            costs_grid, emissions_grid, timesteps, days, powInjRet, powSubtrRet, gridnodes, res_exBat, powInjPrev,
            powSubtrPrev,
            emissions_nodes, costs_min, emissions_max_global) = opti.compute(net, nodes, gridnodes, days, timesteps,
                                                                             eco, devs,
                                                                             clustered, params, options, constraint_apc,
                                                                             constraint_InjMin, constraint_SubtrMin,
                                                                             constraint_InjMax, constraint_SubtrMax,
                                                                             emissions_max_global, costs_max_global)

        # run AC-Powerflow-Solver
        (output_dir, critical_flag, solution_found, vm_pu_total) = loop.run_timeloop(fkt, timesteps, days, powInjRet,
                                                                                     powSubtrRet, gridnodes,
                                                                                     critical_flag,
                                                                                     solution_found)
        # vm_pu_total_array = np.array([[[vm_pu_total[n, d, t] for t in timesteps] for d in days] for n in gridnodes])
        print("zwischenstop")
        for d in days:
            if (solution_found[d] == False):

                print("Additional constrains have to be set for day" + str(d))
                if options["apc_while_voltage_violation"]:
                    if options["cut_Inj/Subtr_while_voltage_violation"]:
                        print("You selected both apc and Inj/Subtr cutting in case of voltage violations.")
                        for n in gridnodes:
                            if (any(critical_flag[n, d, t] == 1 for t in timesteps)):
                                if (any(vm_pu_total[n, d, t] < 0.96 for t in timesteps)):
                                    pass
                                elif (any(vm_pu_total[n, d, t] > 1.04 for t in timesteps)):
                                    constraint_apc[n, d] += 0.1
                                    if (constraint_apc[n, d] >= 1):
                                        print("You have reached the maximum amount of curtailment!")
                                        print("Will set curtailment to 100 Percent automatically.")
                                        constraint_apc[n, d] = 1
                                else:
                                    pass
                                for t in timesteps:
                                    if (critical_flag[n, d, t] == 1):
                                        if (vm_pu_total[n, d, t] < 0.96):
                                            if options["rel1_or_abs0_violation_change"]:
                                                # relative Lösung wirft Problem der Spannungsweiterleitung auf
                                                constraint_SubtrMax[n, d, t] = change_relative * powSubtrPrev[n, d, t]
                                            else:
                                                # absolute Regelung:
                                                if (powSubtrPrev[n, d, t] < change_value):
                                                    constraint_SubtrMax[n, d, t] = 0
                                                    print("Subtraction already set to 0 for node" + str(
                                                        n) + " and timestep" + str(t))
                                                    print("Raising Injection now!")
                                                    constraint_InjMin[n, d, t] += change_value - powSubtrPrev[n, d, t]
                                                else:
                                                    constraint_SubtrMax[n, d, t] = powSubtrPrev[n, d, t] - change_value

                                        elif (vm_pu_total[n, d, t] > 1.04):
                                            if options["rel1_or_abs0_violation_change"]:
                                                constraint_InjMax[n, d, t] = change_relative * powInjPrev[n, d, t]
                                            else:
                                                # absolute änderung
                                                if (powInjPrev[n, d, t] < change_value):
                                                    constraint_InjMax[n, d, t] = 0
                                                    print("Injection already set to 0 for node" + str(
                                                        n) + " and timestep" + str(t))
                                                    print("Raising Subtraction now!")
                                                    constraint_SubtrMin[n, d, t] += change_value - powInjPrev[n, d, t]
                                                else:
                                                    constraint_InjMax[n, d, t] = powInjPrev[n, d, t] - change_value

                    else:
                        print("You selected only apc in case of voltage violations.")
                        for n in gridnodes:
                            if (any(critical_flag[n, d, t] == 1 for t in timesteps)):
                                if (any(vm_pu_total[n, d, t] < 0.96 for t in timesteps)):
                                    print(
                                        "Only apc will not fix any voltage issues, because the load is too high on day" + str(
                                            d))
                                    infeasability = True
                                elif (any(vm_pu_total[n, d, t] > 1.04 for t in timesteps)):
                                    constraint_apc[n, d] += 0.1
                                    if (constraint_apc[n, d] >= 1):
                                        print("You have reached the maximal amount of curtailment!")
                                        print("Will set curtailment to 100 Percent automatically.")
                                        constraint_apc[n, d] = 1
                                        print(
                                            "Since you only selected apc, it has reached 100 Percent and you haven't found a solution, the problem appears to be infeasable for these settings!")
                                        infeasability = True

                elif (options["cut_Inj/Subtr_while_voltage_violation"] == True and options[
                    "apc_while_voltage_violation"] == False):
                    print("You selected only Inj/Subtr cutting in case of voltage violations.")
                    for n in gridnodes:
                        for t in timesteps:
                            if (critical_flag[n, d, t] == 1):
                                if (vm_pu_total[n, d, t] < 0.96):
                                    if options["rel1_or_abs0_violation_change"]:
                                        constraint_SubtrMax[n, d, t] = change_relative * powSubtrPrev[n, d, t]
                                    else:
                                        # absolute Regelung:
                                        if (powSubtrPrev[n, d, t] < change_value):
                                            constraint_SubtrMax[n, d, t] = 0
                                            print(
                                                "Subtraction already set to 0 for node" + str(
                                                    n) + " and timestep" + str(t))
                                            print("Raising Injection now!")
                                            constraint_InjMin[n, d, t] += change_value - powSubtrPrev[n, d, t]
                                        else:
                                            constraint_SubtrMax[n, d, t] = powSubtrPrev[n, d, t] - change_value

                                elif (vm_pu_total[n, d, t] > 1.04):
                                    if options["rel1_or_abs0_violation_change"]:
                                        constraint_InjMax[n, d, t] = change_relative * powInjPrev[n, d, t]
                                    else:
                                        # absolute Regelung
                                        if (powInjPrev[n, d, t] < change_value):
                                            constraint_InjMax[n, d, t] = 0
                                            print(
                                                "Injection already set to 0 for node" + str(n) + " and timestep" + str(
                                                    t))
                                            print("Raising Subtraction now!")
                                            constraint_SubtrMin[n, d, t] += change_value - powInjPrev[n, d, t]
                                        else:
                                            constraint_InjMax[n, d, t] = powInjPrev[n, d, t] - change_value

                elif (options["cut_Inj/Subtr_while_voltage_violation"] == False and options[
                    "apc_while_voltage_violation"] == False):
                    print("Error: You did not select any measure in case of voltage violations!")
                    infeasability = True

            if (solution_found[d] == True):
                print("Solution was successfully found for day" + str(d))

        if infeasability:
            print("Error: Model appears to be infeasable for the selected settings!")
            print("Reasons are stated above.")
            break

        iteration_counter += 1

        if all(solution_found[d] == True for d in days):
            print(
                "Congratulations! Your optimization and loadflow calculation has been successfully finished after " + str(
                    iteration_counter - 1) + " iteration steps!")
            break

    print("this is the end")
    print("")
    print("")
    print("costs minimized are:")
    print(costs_min)
    print("emissions max are:")
    print(emissions_max_global)
    print("")
    print("")
    print("")

    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    print("hier ist das originalprogramm ")

    results_pareto = {}
    results_pareto["Minimale Kosten"] = costs_min
    results_pareto["Maximale Emissionen"] = emissions_max_global
    costlist = []
    emissionslist = []
    simulationslist = []

    # hier drunter müssen die anderen loops, neue printbefehle müssen individuell eingestellt werden und die results ordner müssen separat erstellt werden.

    options["opt_costs"] = False
    options["opt_emissions"] = True
    options["filename_results"] = "results/pareto_results/" + building_type + "_" + \
                                  building_age + "pareto_EmissionsMin.pkl"

    options["filename_inputs"] = "results/pareto_results/inputs_" + building_type + "_" + \
                                 building_age + "pareto_EmissionsMin.pkl"

    # %% Store clustered input parameters
    with open(options["filename_inputs"], "wb") as f_in:
        pickle.dump(clustered, f_in, pickle.HIGHEST_PROTOCOL)

    # !!!!!!!!!!! COPY FROM HERE FOR PARETO LOOOOOOOP !!!!!!!!!!!!!!!!!!!!!!!!

    # solution_found as continuos variable for while loop
    solution_found = []
    for d in days:
        solution_found.append(False)
    boolean_loop = True
    # constraint_apc models APC, gets reduced from 1 to 0 in iteration steps with range 0.1
    constraint_apc = {}
    # constraint for Injection and Subtraction. Inj gets cut when voltage is too high, Subtr gets cut when voltage is too low
    constraint_InjMin = {}
    constraint_SubtrMin = {}
    constraint_InjMax = {}
    constraint_SubtrMax = {}
    # create array to flag whether values are critical for powerflow. If not critical: 0, if critical: 1
    critical_flag = {}
    iteration_counter = 0
    # introduce boolean to state infeasability
    infeasability = False

    change_relative = options["change_relative_node_violation_rel"]
    change_value = options["change_value_node_violation_abs"]

    for n in gridnodes:
        for d in days:
            for t in timesteps:
                critical_flag[n, d, t] = 0
                constraint_apc[n, d] = 0
                constraint_InjMin[n, d, t] = 0
                constraint_SubtrMin[n, d, t] = 0
                constraint_InjMax[n, d, t] = 10000
                constraint_SubtrMax[n, d, t] = 10000

    while boolean_loop:
        print("")
        print("!!! Iteration counter is currently at " + str(iteration_counter) + "!!!")
        print("")

        # run DC-optimization
        (
            costs_grid, emissions_grid, timesteps, days, powInjRet, powSubtrRet, gridnodes, res_exBat, powInjPrev,
            powSubtrPrev,
            emissions_nodes, costs_max_global, emissions_min) = opti.compute(net, nodes, gridnodes, days, timesteps,
                                                                             eco, devs,
                                                                             clustered, params, options, constraint_apc,
                                                                             constraint_InjMin, constraint_SubtrMin,
                                                                             constraint_InjMax, constraint_SubtrMax,
                                                                             critical_flag, emissions_max_global,
                                                                             costs_max_global)

        # run AC-Powerflow-Solver
        (output_dir, critical_flag, solution_found, vm_pu_total) = loop.run_timeloop(fkt, timesteps, days, powInjRet,
                                                                                     powSubtrRet, gridnodes,
                                                                                     critical_flag,
                                                                                     solution_found)
        # vm_pu_total_array = np.array([[[vm_pu_total[n, d, t] for t in timesteps] for d in days] for n in gridnodes])
        print("zwischenstop")
        for d in days:
            if (solution_found[d] == False):

                print("Additional constrains have to be set for day" + str(d))
                if options["apc_while_voltage_violation"]:
                    if options["cut_Inj/Subtr_while_voltage_violation"]:
                        print("You selected both apc and Inj/Subtr cutting in case of voltage violations.")
                        for n in gridnodes:
                            if (any(critical_flag[n, d, t] == 1 for t in timesteps)):
                                if (any(vm_pu_total[n, d, t] < 0.96 for t in timesteps)):
                                    pass
                                elif (any(vm_pu_total[n, d, t] > 1.04 for t in timesteps)):
                                    constraint_apc[n, d] += 0.1
                                    if (constraint_apc[n, d] >= 1):
                                        print("You have reached the maximum amount of curtailment!")
                                        print("Will set curtailment to 100 Percent automatically.")
                                        constraint_apc[n, d] = 1
                                else:
                                    pass
                                for t in timesteps:
                                    if (critical_flag[n, d, t] == 1):
                                        if (vm_pu_total[n, d, t] < 0.96):
                                            if options["rel1_or_abs0_violation_change"]:
                                                # relative Lösung wirft Problem der Spannungsweiterleitung auf
                                                constraint_SubtrMax[n, d, t] = change_relative * powSubtrPrev[n, d, t]
                                            else:
                                                # absolute Regelung:
                                                if (powSubtrPrev[n, d, t] < change_value):
                                                    constraint_SubtrMax[n, d, t] = 0
                                                    print("Subtraction already set to 0 for node" + str(
                                                        n) + " and timestep" + str(t))
                                                    print("Raising Injection now!")
                                                    constraint_InjMin[n, d, t] += change_value - powSubtrPrev[n, d, t]
                                                else:
                                                    constraint_SubtrMax[n, d, t] = powSubtrPrev[n, d, t] - change_value

                                        elif (vm_pu_total[n, d, t] > 1.04):
                                            if options["rel1_or_abs0_violation_change"]:
                                                constraint_InjMax[n, d, t] = change_relative * powInjPrev[n, d, t]
                                            else:
                                                # absolute änderung
                                                if (powInjPrev[n, d, t] < change_value):
                                                    constraint_InjMax[n, d, t] = 0
                                                    print("Injection already set to 0 for node" + str(
                                                        n) + " and timestep" + str(t))
                                                    print("Raising Subtraction now!")
                                                    constraint_SubtrMin[n, d, t] += change_value - powInjPrev[n, d, t]
                                                else:
                                                    constraint_InjMax[n, d, t] = powInjPrev[n, d, t] - change_value

                    else:
                        print("You selected only apc in case of voltage violations.")
                        for n in gridnodes:
                            if (any(critical_flag[n, d, t] == 1 for t in timesteps)):
                                if (any(vm_pu_total[n, d, t] < 0.96 for t in timesteps)):
                                    print(
                                        "Only apc will not fix any voltage issues, because the load is too high on day" + str(
                                            d))
                                    infeasability = True
                                elif (any(vm_pu_total[n, d, t] > 1.04 for t in timesteps)):
                                    constraint_apc[n, d] += 0.1
                                    if (constraint_apc[n, d] >= 1):
                                        print("You have reached the maximal amount of curtailment!")
                                        print("Will set curtailment to 100 Percent automatically.")
                                        constraint_apc[n, d] = 1
                                        print(
                                            "Since you only selected apc, it has reached 100 Percent and you haven't found a solution, the problem appears to be infeasable for these settings!")
                                        infeasability = True

                elif (options["cut_Inj/Subtr_while_voltage_violation"] == True and options[
                    "apc_while_voltage_violation"] == False):
                    print("You selected only Inj/Subtr cutting in case of voltage violations.")
                    for n in gridnodes:
                        for t in timesteps:
                            if (critical_flag[n, d, t] == 1):
                                if (vm_pu_total[n, d, t] < 0.96):
                                    if options["rel1_or_abs0_violation_change"]:
                                        constraint_SubtrMax[n, d, t] = change_relative * powSubtrPrev[n, d, t]
                                    else:
                                        # absolute Regelung:
                                        if (powSubtrPrev[n, d, t] < change_value):
                                            constraint_SubtrMax[n, d, t] = 0
                                            print(
                                                "Subtraction already set to 0 for node" + str(
                                                    n) + " and timestep" + str(t))
                                            print("Raising Injection now!")
                                            constraint_InjMin[n, d, t] += change_value - powSubtrPrev[n, d, t]
                                        else:
                                            constraint_SubtrMax[n, d, t] = powSubtrPrev[n, d, t] - change_value

                                elif (vm_pu_total[n, d, t] > 1.04):
                                    if options["rel1_or_abs0_violation_change"]:
                                        constraint_InjMax[n, d, t] = change_relative * powInjPrev[n, d, t]
                                    else:
                                        # absolute Regelung
                                        if (powInjPrev[n, d, t] < change_value):
                                            constraint_InjMax[n, d, t] = 0
                                            print(
                                                "Injection already set to 0 for node" + str(n) + " and timestep" + str(
                                                    t))
                                            print("Raising Subtraction now!")
                                            constraint_SubtrMin[n, d, t] += change_value - powInjPrev[n, d, t]
                                        else:
                                            constraint_InjMax[n, d, t] = powInjPrev[n, d, t] - change_value

                elif (options["cut_Inj/Subtr_while_voltage_violation"] == False and options[
                    "apc_while_voltage_violation"] == False):
                    print("Error: You did not select any measure in case of voltage violations!")
                    infeasability = True

            if (solution_found[d] == True):
                print("Solution was successfully found for day" + str(d))

        if infeasability:
            print("Error: Model appears to be infeasable for the selected settings!")
            print("Reasons are stated above.")
            break

        iteration_counter += 1

        if all(solution_found[d] == True for d in days):
            print(
                "Congratulations! Your optimization and loadflow calculation has been successfully finished after " + str(
                    iteration_counter - 1) + " iteration steps!")
            break

    print("this is the end")
    print("")
    print("")
    print("emissions minimized are:")
    print(emissions_min)
    print("costs max are:")
    print(costs_max_global)
    print("")
    print("")
    print("")

    results_pareto["Maximale Kosten"] = costs_max_global
    results_pareto["Minimale Emissionen"] = emissions_min

    options["opt_costs"] = True
    options["opt_emissions"] = False

    number_simulations = 4
    for i in range(1, 1 + number_simulations):

        # !!!!!!!!!!! COPY FROM HERE FOR PARETO LOOOOOOOP !!!!!!!!!!!!!!!!!!!!!!!!

        options["filename_results"] = "results/pareto_results/" + building_type + "_" + \
                                      building_age + "pareto_CostsMin" + str(i / (number_simulations + 1)) + ".pkl"

        options["filename_inputs"] = "results/pareto_results/inputs_" + building_type + "_" + \
                                     building_age + "pareto_CostsMin" + str(i / (number_simulations + 1)) + ".pkl"

        emissions_max = emissions_min + (emissions_max_global - emissions_min) * i * (1 / (number_simulations + 1))

        # %% Store clustered input parameters

        with open(options["filename_inputs"], "wb") as f_in:
            pickle.dump(clustered, f_in, pickle.HIGHEST_PROTOCOL)

        # solution_found as continuos variable for while loop
        solution_found = []
        for d in days:
            solution_found.append(False)
        boolean_loop = True
        # constraint_apc models APC, gets reduced from 1 to 0 in iteration steps with range 0.1
        constraint_apc = {}
        # constraint for Injection and Subtraction. Inj gets cut when voltage is too high, Subtr gets cut when voltage is too low
        constraint_InjMin = {}
        constraint_SubtrMin = {}
        constraint_InjMax = {}
        constraint_SubtrMax = {}
        # create array to flag whether values are critical for powerflow. If not critical: 0, if critical: 1
        critical_flag = {}
        iteration_counter = 0
        # introduce boolean to state infeasability
        infeasability = False

        change_relative = options["change_relative_node_violation_rel"]
        change_value = options["change_value_node_violation_abs"]

        for n in gridnodes:
            for d in days:
                for t in timesteps:
                    critical_flag[n, d, t] = 0
                    constraint_apc[n, d] = 0
                    constraint_InjMin[n, d, t] = 0
                    constraint_SubtrMin[n, d, t] = 0
                    constraint_InjMax[n, d, t] = 10000
                    constraint_SubtrMax[n, d, t] = 10000

        while boolean_loop:
            print("")
            print("!!! Iteration counter is currently at " + str(iteration_counter) + "!!!")
            print("")

            # run DC-optimization
            (
                costs_grid, emissions_grid, timesteps, days, powInjRet, powSubtrRet, gridnodes, res_exBat, powInjPrev,
                powSubtrPrev,
                emissions_nodes, costs, emissions) = opti.compute(net, nodes, gridnodes, days, timesteps, eco, devs,
                                                                  clustered, params, options, constraint_apc,
                                                                  constraint_InjMin, constraint_SubtrMin,
                                                                  constraint_InjMax, constraint_SubtrMax,
                                                                  critical_flag, emissions_max, costs_max_global)

            # run AC-Powerflow-Solver
            (output_dir, critical_flag, solution_found, vm_pu_total) = loop.run_timeloop(fkt, timesteps, days,
                                                                                         powInjRet,
                                                                                         powSubtrRet, gridnodes,
                                                                                         critical_flag,
                                                                                         solution_found)
            # vm_pu_total_array = np.array([[[vm_pu_total[n, d, t] for t in timesteps] for d in days] for n in gridnodes])
            print("zwischenstop")
            for d in days:
                if (solution_found[d] == False):

                    print("Additional constrains have to be set for day" + str(d))
                    if options["apc_while_voltage_violation"]:
                        if options["cut_Inj/Subtr_while_voltage_violation"]:
                            print("You selected both apc and Inj/Subtr cutting in case of voltage violations.")
                            for n in gridnodes:
                                if (any(critical_flag[n, d, t] == 1 for t in timesteps)):
                                    if (any(vm_pu_total[n, d, t] < 0.96 for t in timesteps)):
                                        pass
                                    elif (any(vm_pu_total[n, d, t] > 1.04 for t in timesteps)):
                                        constraint_apc[n, d] += 0.1
                                        if (constraint_apc[n, d] >= 1):
                                            print("You have reached the maximum amount of curtailment!")
                                            print("Will set curtailment to 100 Percent automatically.")
                                            constraint_apc[n, d] = 1
                                    else:
                                        pass
                                    for t in timesteps:
                                        if (critical_flag[n, d, t] == 1):
                                            if (vm_pu_total[n, d, t] < 0.96):
                                                if options["rel1_or_abs0_violation_change"]:
                                                    # relative Lösung wirft Problem der Spannungsweiterleitung auf
                                                    constraint_SubtrMax[n, d, t] = change_relative * powSubtrPrev[
                                                        n, d, t]
                                                else:
                                                    # absolute Regelung:
                                                    if (powSubtrPrev[n, d, t] < change_value):
                                                        constraint_SubtrMax[n, d, t] = 0
                                                        print("Subtraction already set to 0 for node" + str(
                                                            n) + " and timestep" + str(t))
                                                        print("Raising Injection now!")
                                                        constraint_InjMin[n, d, t] += change_value - powSubtrPrev[
                                                            n, d, t]
                                                    else:
                                                        constraint_SubtrMax[n, d, t] = powSubtrPrev[
                                                                                           n, d, t] - change_value

                                            elif (vm_pu_total[n, d, t] > 1.04):
                                                if options["rel1_or_abs0_violation_change"]:
                                                    constraint_InjMax[n, d, t] = change_relative * powInjPrev[n, d, t]
                                                else:
                                                    # absolute änderung
                                                    if (powInjPrev[n, d, t] < change_value):
                                                        constraint_InjMax[n, d, t] = 0
                                                        print("Injection already set to 0 for node" + str(
                                                            n) + " and timestep" + str(t))
                                                        print("Raising Subtraction now!")
                                                        constraint_SubtrMin[n, d, t] += change_value - powInjPrev[
                                                            n, d, t]
                                                    else:
                                                        constraint_InjMax[n, d, t] = powInjPrev[n, d, t] - change_value

                        else:
                            print("You selected only apc in case of voltage violations.")
                            for n in gridnodes:
                                if (any(critical_flag[n, d, t] == 1 for t in timesteps)):
                                    if (any(vm_pu_total[n, d, t] < 0.96 for t in timesteps)):
                                        print(
                                            "Only apc will not fix any voltage issues, because the load is too high on day" + str(
                                                d))
                                        infeasability = True
                                    elif (any(vm_pu_total[n, d, t] > 1.04 for t in timesteps)):
                                        constraint_apc[n, d] += 0.1
                                        if (constraint_apc[n, d] >= 1):
                                            print("You have reached the maximal amount of curtailment!")
                                            print("Will set curtailment to 100 Percent automatically.")
                                            constraint_apc[n, d] = 1
                                            print(
                                                "Since you only selected apc, it has reached 100 Percent and you haven't found a solution, the problem appears to be infeasable for these settings!")
                                            infeasability = True

                    elif (options["cut_Inj/Subtr_while_voltage_violation"] == True and options[
                        "apc_while_voltage_violation"] == False):
                        print("You selected only Inj/Subtr cutting in case of voltage violations.")
                        for n in gridnodes:
                            for t in timesteps:
                                if (critical_flag[n, d, t] == 1):
                                    if (vm_pu_total[n, d, t] < 0.96):
                                        if options["rel1_or_abs0_violation_change"]:
                                            constraint_SubtrMax[n, d, t] = change_relative * powSubtrPrev[n, d, t]
                                        else:
                                            # absolute Regelung:
                                            if (powSubtrPrev[n, d, t] < change_value):
                                                constraint_SubtrMax[n, d, t] = 0
                                                print(
                                                    "Subtraction already set to 0 for node" + str(
                                                        n) + " and timestep" + str(t))
                                                print("Raising Injection now!")
                                                constraint_InjMin[n, d, t] += change_value - powSubtrPrev[n, d, t]
                                            else:
                                                constraint_SubtrMax[n, d, t] = powSubtrPrev[n, d, t] - change_value

                                    elif (vm_pu_total[n, d, t] > 1.04):
                                        if options["rel1_or_abs0_violation_change"]:
                                            constraint_InjMax[n, d, t] = change_relative * powInjPrev[n, d, t]
                                        else:
                                            # absolute Regelung
                                            if (powInjPrev[n, d, t] < change_value):
                                                constraint_InjMax[n, d, t] = 0
                                                print("Injection already set to 0 for node" + str(
                                                    n) + " and timestep" + str(t))
                                                print("Raising Subtraction now!")
                                                constraint_SubtrMin[n, d, t] += change_value - powInjPrev[n, d, t]
                                            else:
                                                constraint_InjMax[n, d, t] = powInjPrev[n, d, t] - change_value

                    elif (options["cut_Inj/Subtr_while_voltage_violation"] == False and options[
                        "apc_while_voltage_violation"] == False):
                        print("Error: You did not select any measure in case of voltage violations!")
                        infeasability = True

                if (solution_found[d] == True):
                    print("Solution was successfully found for day" + str(d))

            if infeasability:
                print("Error: Model appears to be infeasable for the selected settings!")
                print("Reasons are stated above.")
                break

            iteration_counter += 1

            if all(solution_found[d] == True for d in days):
                print(
                    "Congratulations! Your optimization and loadflow calculation has been successfully finished after " + str(
                        iteration_counter - 1) + " iteration steps!")
                break

        print("this is the end")
        print("")
        print("")
        print("costs minimized are:")
        print(costs)
        print("for distance of emissions min from: " + str(i * 1 / (number_simulations + 1)))
        print("emissions are:")
        print(emissions)
        print("")
        print("")
        print("")

        costlist.append(costs)
        emissionslist.append(emissions)
        simulationslist.append(i * 1 / (number_simulations + 1))

    print("")
    print("")
    print("Extremwerte:")
    print(results_pareto)
    print("costlist:")
    print(costlist)
    print("emissionslist:")
    print(emissionslist)
    print("simulationspercentage:")
    print(simulationslist)

    print("strpo")
