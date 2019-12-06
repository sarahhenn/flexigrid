import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import pickle
import pandapower as pp
import tempfile
import tikzplotlib as tikz
import python.read_basic as reader



def plot_results(outputs, days, gridnodes, timesteps):

    plotting_options = {
                        "safe plots as tex": True,             # safe as latex tex file?
                        "plot_live"        : True,             # plot immediately on screen?
                        }


    for d in days:
        output_dir_pp = os.path.join(tempfile.gettempdir(), "time_series_example" + str(d))
        tex_folder = "C:\\users\\flori\\pycharmprojects\\flexigrid\\results\\LaTeX_results\\Beispiel"


        """# voltage results
        vm_pu_file = os.path.join(output_dir_pp, "res_bus", "vm_pu.json")
        vm_pu = pd.read_json(vm_pu_file)
        # sort dataframe to get timesteps(rows) in the correct order
        vm_pu = vm_pu.sort_index(axis=0)
        vm_pu.plot(label="vm_pu")
        plt.xlabel("time step")
        plt.ylabel("voltage mag. [p.u.]")
        plt.title("Voltage Magnitude day" + str(d))
        plt.grid()
        plt.show()

        # line loading results
        ll_file = os.path.join(output_dir_pp, "res_line", "loading_percent.json")
        line_loading = pd.read_json(ll_file)
        # sort dataframe to get timesteps(rows) in the correct order
        line_loading = line_loading.sort_index(axis=0)
        line_loading.plot(label="line_loading")
        plt.xlabel("time step")
        plt.ylabel("line loading [%]")
        plt.title("Line Loading")
        plt.grid()
        plt.show()"""

        """# load results
        load_file = os.path.join(output_dir_pp, "res_load", "p_mw.json")
        load = pd.read_json(load_file)
        # sort dataframe to get timesteps(rows) in the correct order
        load = load.sort_index(axis=0)
        load.plot(label="load")
        plt.xlabel("time step")
        plt.ylabel("P [MW]")
        plt.grid()
        plt.show()"""

        # state of charge battery results
        res_soc_bat = outputs["res_SOC"]
        res_soc_bat_2d = np.zeros((len(gridnodes),len(timesteps)))
        for n in gridnodes:
            for t in timesteps:
                res_soc_bat_2d[n,t] = res_soc_bat[n,d][t]
        res_soc_bat_df = pd.DataFrame.from_records(res_soc_bat_2d)
        res_soc_bat_df = res_soc_bat_df.transpose()
        res_soc_bat_df.plot(label ="SOC_Bat")
        plt.xlabel("Uhrzeit")
        plt.ylabel("SOC Battery [kWh]")
        plt.title("SOC Battery Day " +str(d))
        plt.grid()
        if plotting_options["plot_live"] == True:
            plt.show()
            fig = plt.figure()
            plt.savefig("soc_bat.png", dpi = fig.dpi )
        else:
            pass
        if plotting_options["safe plots as tex"] == True:
            tikz.save(figure = fig, filepath = tex_folder + "\\soc_bat.tex")
        else:
            pass


        """# power Inj results
        res_powerInj = outputs["res_powerInj"]
        res_powerInj_2d = np.zeros((len(gridnodes), len(timesteps)))
        for n in gridnodes:
            for t in timesteps:
                res_powerInj_2d[n, t] = res_powerInj[n, d][t]
        res_powerInj_df = pd.DataFrame.from_records(res_powerInj_2d)
        res_powerInj_df = res_powerInj_df.transpose()
        res_powerInj_df.plot(label="powerInj")
        plt.xlabel("Uhrzeit")
        plt.ylabel("Injected power [kWh]")
        plt.title("Injection from load-node into grid")
        plt.grid()
        if plotting_options["plot_live"] == True:
            plt.show()
            fig = plt.figure()
            plt.savefig("pow_Inj.png", dpi=fig.dpi)
        else:
            pass
        if plotting_options["safe plots as tex"] == True:
            tikz.save(figure = fig, filepath = tex_folder + "\\pow_Inj.tex")
        else:
            pass"""

        """# power Subtr results
        res_powerSubtr = outputs["res_powerSubtr"]
        res_powerSubtr_2d = np.zeros((len(gridnodes), len(timesteps)))
        for n in gridnodes:
            for t in timesteps:
                res_powerSubtr_2d[n, t] = res_powerSubtr[n, d][t]
        res_powerSubtr_df = pd.DataFrame.from_records(res_powerSubtr_2d)
        res_powerSubtr_df = res_powerSubtr_df.transpose()
        res_powerSubtr_df.plot(label="powerSubtr")
        plt.xlabel("Uhrzeit")
        plt.ylabel("Subtracted power [kWh]")
        plt.title("Subtraction from load-node into grid")
        if plotting_options["plot_live"] == True:
            plt.show()
            fig = plt.figure()
            plt.savefig("pow_Subtr.png", dpi=fig.dpi)
        else:
            pass
        if plotting_options["safe plots as tex"] == True:
            tikz.save(figure = fig, filepath = tex_folder + "\\pow_Subtr.tex")"""



        """# pow Subtr and Inj in one plot
        # Subtraction DataFrame
        res_powerSubtr = outputs["res_powerSubtr"]
        res_powerSubtr_2d = np.zeros((len(gridnodes), len(timesteps)))
        for n in gridnodes:
            for t in timesteps:
                res_powerSubtr_2d[n, t] = res_powerSubtr[n, d][t]
        res_powerSubtr_df = pd.DataFrame.from_records(res_powerSubtr_2d)
        res_powerSubtr_df = res_powerSubtr_df.transpose()"""

        """# Injection DataFrame
        res_powerInj = outputs["res_powerInj"]
        res_powerInj_2d = np.zeros((len(gridnodes), len(timesteps)))
        for n in gridnodes:
            for t in timesteps:
                res_powerInj_2d[n, t] = res_powerInj[n, d][t]
        res_powerInj_df = pd.DataFrame.from_records(res_powerInj_2d)
        res_powerInj_df = res_powerInj_df.transpose()"""

        #fig, ax1 = plt.subplots()

        """color = 'tab:red'
        ax1.set_xlabel('Zeit [Stunden]')
        ax1.set_ylabel('Power Subtracted [kW]', color=color)
        # hier schauen, wie im dataframe die name des columns ist!
        ax1.plot(efile["Taganzahl"], efile["Emissionswert"], color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('Power Injected [kW]', color=color)
        ax2.plot(, efile["Iterationsschritte"], color=color)
        ax2.tick_params(axis='y', labelcolor=color)

        fig.tight_layout()
        plt.show()
        tikz.save(figure=fig, filepath=tex_folder + "\\Typtageauswertung.tex")"""

    print("Successfully plotted the results!")
    print("You can look up yout plots in " + tex_folder)

