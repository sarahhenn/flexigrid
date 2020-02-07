import pandas as pd
import pandapower as pp
import pandapower.networks as nw
from pandapower.plotting import pf_res_plotly

net = nw.kb_extrem_landnetz_kabel()

# specify grid nodes for whole grid and trafo; choose and allocate load, injection and battery nodes
# draw parameters from pandapower network
nodes = {}

nodes["grid"] = net.bus.index.to_numpy()
nodes["trafo"] = net.trafo['lv_bus'].to_numpy()
nodes["load"] = net.load['bus'].to_numpy()
nodes["bat"] = net.load['bus'].to_numpy()

# define sgens for net in order to be able to include gen values in timeloop
nodesload = list(nodes["load"])
for n in nodesload:
    pp.create_sgen(net, n, p_mw=0.004)

for n in net.load.index:
    net.load.loc[n,"p_mw"] = 0.000

print(net.load)
print(net.sgen)

pf_res_plotly(net)