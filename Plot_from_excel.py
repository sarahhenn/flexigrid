import pandas as pd
import matplotlib.pyplot as plt
import tikzplotlib as tikz

#tex_folder = "C:\\users\\flori\\pycharmprojects\\flexigrid\\results\\LaTeX_results"
tex_folder = "D:\\studium_neu\\sciebo_synchro\\Bachelorarbeit\\Arbeit_LaTeX\\Figures\\Auswertung"

efile = pd.read_excel("D:\\studium_neu\\sciebo_synchro\\Bachelorarbeit\\Graphiken_Modellentwicklung\\Typtageauswertung_Landnetz.xlsx",skiprows=2, usecols = {0,5,3})

fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Anzahl Typtage')
ax1.set_ylabel('Rel. Abweichung Funktionswert', color = color)
# hier schauen, wie im dataframe die name des columns ist!
ax1.plot(efile["Taganzahl"], efile["Abweichung Zielfunktionswert"], color = color)
#ax1.get_yaxis().set_visible(False)
#ax1.ylabel = False
ax1.tick_params(axis='y', labelcolor = color)
#ax1.yaxis.grid(True)
ax1.axhline(linewidth = 2, color= 'black')

"""ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Anzahl Iteratinsschritte', color= color)
ax2.plot(efile["Taganzahl"], efile["Iterationsschritte"], color = color)
ax2.tick_params(axis='y', labelcolor = color)"""

#fig.tight_layout()
#plt.show()
tikz.save(figure = fig, filepath= tex_folder + "\\Typtageauswertung_Landnetz_Prozentual.tex")