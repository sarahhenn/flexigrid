import matplotlib as plt

# das ist alles Joels Code


font_dirs = ['Schriftarten']
font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
font_list = font_manager.createFontList(font_files)
font_manager.fontManager.ttflist.extend(font_list)
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Heuristica'
plt.rcParams['mathtext.it'] = 'Heuristica:bold'
plt.rcParams['mathtext.bf'] = 'Heuristica:bold'
plt.rcParams['font.family'] = 'Heuristica'
plt.rcParams['figure.figsize'] = [7.5, 5.6]
plt.rcParams.update({'figure.autolayout': True})
plt.rc('font', size=18)          # controls default text sizes
plt.rc('axes', titlesize=24)     # fontsize of the axes title
plt.rc('axes', labelsize=22)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=22)    # fontsize of the tick labels
plt.rc('ytick', labelsize=20)    # fontsize of the tick labels
plt.rc('legend', fontsize=18)    # legend fontsize
plt.rc('figure', titlesize=14)  # fontsize of the figure title