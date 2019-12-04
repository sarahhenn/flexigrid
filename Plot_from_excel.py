import pandas as pd
import matplotlib.pyplot as plt
import tikzplotlib as tikz

tex_folder = "C:\\users\\flori\\pycharmprojects\\flexigrid\\results\\LaTeX_results"

efile = pd.read_excel("D:\\studium_neu\\sciebo_synchro\\Bachelorarbeit\\Graphiken_Modellentwicklung\\Typtageauswertung_Landnetz.xlsx",skiprows=2, usecols = {0,1,3})

fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Anzahl Typtage')
ax1.set_ylabel('Zielfunktionswert Emissionen [kg]', color = color)
# hier schauen, wie im dataframe die name des columns ist!
ax1.plot(efile["Taganzahl"], efile["Emissionswert"], color = color)
ax1.tick_params(axis='y', labelcolor = color)

ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Anzahl Iteratinsschritte', color= color)
ax2.plot(efile["Taganzahl"], efile["Iterationsschritte"], color = color)
ax2.tick_params(axis='y', labelcolor = color)

fig.tight_layout()
plt.show()
tikz.save(figure = fig, filepath= tex_folder + "\\Typtageauswertung.tex")