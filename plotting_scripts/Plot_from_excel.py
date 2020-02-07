import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib as tikz
import seaborn as sns

#tex_folder = "C:\\users\\flori\\pycharmprojects\\flexigrid\\results\\LaTeX_results"
tex_folder = "D:\\studium_neu\\sciebo_synchro\\Bachelorarbeit\\Arbeit_LaTeX\\Figures\\Auswertung\\"

def subplotting():

    efile = pd.read_excel("D:\\studium_neu\\sciebo_synchro\\Bachelorarbeit\\Graphiken_Modellentwicklung\\Typtageauswertung_Landnetz.xlsx",skiprows=0, usecols={0, 1, 2})

    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('Anzahl Typtage', size=18)
    ax1.set_ylabel('Zielfunktionswert', color = color, size=18)
    # hier schauen, wie im dataframe die name des columns ist!
    ax1.plot(efile["Taganzahl"], efile["Emissionswert"], color = color)
    #ax1.get_yaxis().set_visible(False)
    #ax1.ylabel = False
    ax1.tick_params(axis='y', labelcolor = color, size=14)
    #ax1.yaxis.grid(True)
    #ax1.axhline(linewidth = 2, color= 'black')

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Anzahl Iterationsschritte', color= color,size=18)
    ax2.plot(efile["Taganzahl"], efile["Iterationsschritte"], color = color)
    ax2.tick_params(axis='y', labelcolor = color, size =14)

    fig.tight_layout()
    plt.show()
    #tikz.save(figure = fig, filepath= tex_folder + "Typtageauswertung_Landnetz_ProzentualFarbe.tex")

#subplotting()

def otherplots():

    efile1= pd.read_excel("D:\\studium_neu\\sciebo_synchro\\Bachelorarbeit\\Graphiken_Modellentwicklung\\Pareto_1_Einzelgebaeude_mit_Beschraenkung.xlsx", skiprows=2, usecols={1,2})
    efile2= pd.read_excel("D:\\studium_neu\\sciebo_synchro\\Bachelorarbeit\\Graphiken_Modellentwicklung\\Pareto_2_Einzelgebaeude_ohne_Beschraenkung.xlsx", skiprows=2, usecols={1,2})

    color1 = 'tab:green'
    color2 = 'tab:red'

    plt.xlabel('Gesamte Emissionen [kgCO2]', size=16)
    plt.ylabel('Jährliche Gesamtkosten [€]', size= 16)

    #plt.Axes.ticklabel_format('both', style = 'plain')

    plt.plot(efile1["Emissionen"],efile1["Kosten"], marker = 'o', color = color1, label= "Mit Netzbeschränkungen", linewidth = 2, markersize = 8)
    plt.plot(efile2["Emissionen"],efile2["Kosten"], marker = '^', color = color2, label = "Ohne Netzbeschränkungen", linewidth = 2, markersize = 8)

    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.show()
    #tikz.save(filepath= tex_folder + "Pareto_Auswertung.tex")

otherplots()

def hpplotcosts():

    efile= pd.read_excel("D:\\studium_neu\\sciebo_synchro\\Bachelorarbeit\\Graphiken_Modellentwicklung\\BetriebWP.xlsx", skiprows=3, usecols={1,2})

    labels = ['Kostenminimiert', 'Emissionsminimiert']
    legend = ['J{\"a}hrliche Gesamtkosten [\euro{}]']
    # [ grid_opt val , energy_opt val ]
    costopti_values = [38760.40 , 108713.24]
    emissionopti_values = [52724.27, 135245.31]

    # label locations
    x = np.arange(len(labels))
    # width of the bars
    width = 0.25

    fig, ax = plt.subplots()
    ax.bar(x - width/2, costopti_values, width, label='Netzoptimierte HP')
    ax.bar(x + width/2, emissionopti_values, width, label='Energieoptimierte HP')

    ax.set_ylabel('J{\"a}hrliche Gesamtkosten [\euro{}]')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    fig.tight_layout()
    #plt.show()
    tikz.save(figure = fig, filepath=tex_folder + "BetriebWPKosten.tex")

#hpplotcosts()

def hpplotemissions():

    labels = ['Kostenminimiert', 'Emissionsminimiert']
    legend = ['Gesamte j{\"a}hrliche Emissionen [kg]']
    # [ grid_opt val , energy_opt val ]
    costopti_values = [32284.88, 18779.9]
    emissionopti_values = [64311.4, 37470.80]

    # label locations
    x = np.arange(len(labels))
    # width of the bars
    width = 0.25

    fig, ax = plt.subplots()
    ax.bar(x - width / 2, costopti_values, width, label='Netzoptimierte HP')
    ax.bar(x + width / 2, emissionopti_values, width, label='Energieoptimierte HP')

    ax.set_ylabel('Gesamte CO2-Emissionen [kg]')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    fig.tight_layout()
    #plt.show()
    tikz.save(figure = fig, filepath=tex_folder + "BetriebWPEmissionen.tex")

#hpplotemissions()

