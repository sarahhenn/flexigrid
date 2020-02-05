import numpy as np
import pickle
import pandapower as pp
import pandapower.networks as nw
import pandas as pd
import tempfile
import os
import matplotlib.pyplot as plt
import tikzplotlib as tikz
import python.read_basic as reader

tex_folder = "D:\\studium_neu\\sciebo_synchro\\Bachelorarbeit\\Arbeit_LaTeX\\Figures\\Auswertung\\"

# set parameters
building_type = "EFH"       # EFH, ZFH, MFH_6WE, MFH_10WE, MFH_15WE
building_age  = "2005"      # 1960, 1980, 2005
emission_year = "2017"      # 2017, 2030, 2050

TES_Stufen = ["0","03","06","09","12","15"]
capacity_bat = {}
existence_bat = {}
apc_constraint_pre = {}
apc_constraint_added = {}

for TES_Stufe in TES_Stufen:

    plotting_options = {
                        "safe plots as tex": False,             # safe as latex tex file?
                        "plot_live"        : True,             # plot immediately on screen?
                        "filename_results": "results/" + building_type + "_" + \
                                                   building_age + "TES"+TES_Stufe+".pkl",
                        "filename_inputs": "results/inputs_" + building_type + "_" + \
                                                   building_age + "TES"+TES_Stufe+".pkl",
                        "allow_apc_opti": False
                        }

    results = reader.read_results(plotting_options)
    capacity_bat[TES_Stufe] = results["res_capacity"]
    existence_bat[TES_Stufe] = results["res_exBat"]
    apc_constraint_pre[TES_Stufe] = results["res_constraint_apc"]
    sum_list= []
    for n in range(56):
        sum_list.append(sum(apc_constraint_pre[TES_Stufe][n]))
    apc_constraint_added[TES_Stufe] = sum_list
    print("")

    # hier noch dataframes daraus machen
df_cap = pd.DataFrame(capacity_bat)
df_exBat = pd.DataFrame(existence_bat)
df_apc = pd.DataFrame(apc_constraint_added)

rows_to_drop = [0,1,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,55]
df_cap.drop(labels=rows_to_drop, axis=0, inplace=True)
df_exBat.drop(labels=rows_to_drop, axis=0, inplace=True)
df_apc.drop(labels=rows_to_drop, axis=0, inplace=True)

new_row_names = []
for i in range(1,27):
    new_row_names.append(i)
old_row_names = df_cap.index.tolist()

new_column_names = [0, 0.3, 0.6, 0.9, 1.2, 1.5]
old_column_names = df_cap.columns.tolist()

for i in range(len(df_cap)):
    df_cap.rename({old_row_names[i]: new_row_names[i]}, inplace=True)
    df_exBat.rename({old_row_names[i]: new_row_names[i]}, inplace=True)
    df_apc.rename({old_row_names[i]: new_row_names[i]}, inplace=True)

for i in range(len(df_cap.columns)):
    df_cap.rename({old_column_names[i]: new_column_names[i]}, inplace=True, axis=1)
    df_exBat.rename({old_column_names[i]: new_column_names[i]}, inplace=True, axis=1)
    df_apc.rename({old_column_names[i]: new_column_names[i]}, inplace=True, axis=1)


# plotting

df_cap.plot(label = 'Installed Battery capacity')
plt.xlabel("Lastknoten", size=18)
"""plt.tick_params(axis='y', size =18)
plt.tick_params(axis='x', size =18)"""
plt.ylabel("Installierte Batteriekap. [kWh]", size=18)
#plt.title("Installierte Batteriekapazität")
#plt.grid()
plt.show()
#tikz.save(filepath=tex_folder + "TES_Auswertung_2.tex")

print("")
