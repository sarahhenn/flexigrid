# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 13:03:15 2019

@author: she
tutorial from website: 
    https://github.com/e2nIEE/pandapower/blob/master/tutorials/create_simple.ipynb
"""

##

import numpy as np
import time
import pandapower as pp
import pandapower.networks as pn

#create empty network
net = pp.create_empty_network() #create an empty network

#create busses (nodes and busbars)
bus1 = pp.create_bus(net, name="HV Busbar", vn_kv=110, type="b")
bus2 = pp.create_bus(net, name="HV Busbar 2", vn_kv=110, type="b")
bus3 = pp.create_bus(net, name="HV Transformer Bus", vn_kv=110, type="n")
bus4 = pp.create_bus(net, name="MV Transformer Bus", vn_kv=20, type="n")
bus5 = pp.create_bus(net, name="MV Main Bus", vn_kv=20, type="b")
bus6 = pp.create_bus(net, name="MV Bus 1", vn_kv=20, type="b")
bus7 = pp.create_bus(net, name="MV Bus 2", vn_kv=20, type="b")

tab_bus = net.bus #bus table

#create external grid connection
pp.create_ext_grid(net, bus1, vm_pu=1.02, va_degree=50)

tab_ext_grid = net.ext_grid #external grid table

#create trafo
trafo1 = pp.create_transformer(net, bus3, bus4, name="110kV/20kV transformer", std_type="25 MVA 110/20 kV")

tab_trafo = net.trafo #transformer table

#create lines
line1 = pp.create_line(net, bus1, bus2, length_km=10, std_type="N2XS(FL)2Y 1x300 RM/35 64/110 kV",  name="Line 1")
line2 = pp.create_line(net, bus5, bus6, length_km=2.0, std_type="NA2XS2Y 1x240 RM/25 12/20 kV", name="Line 2")
line3 = pp.create_line(net, bus6, bus7, length_km=3.5, std_type="48-AL1/8-ST1A 20.0", name="Line 3")
line4 = pp.create_line(net, bus7, bus5, length_km=2.5, std_type="NA2XS2Y 1x240 RM/25 12/20 kV", name="Line 4")

tab_line = net.line #line table

#create switches
sw1 = pp.create_switch(net, bus2, bus3, et="b", type="CB", closed=True)
sw2 = pp.create_switch(net, bus4, bus5, et="b", type="CB", closed=True)

sw3 = pp.create_switch(net, bus5, line2, et="l", type="LBS", closed=True)
sw4 = pp.create_switch(net, bus6, line2, et="l", type="LBS", closed=True)
sw5 = pp.create_switch(net, bus6, line3, et="l", type="LBS", closed=True)
sw6 = pp.create_switch(net, bus7, line3, et="l", type="LBS", closed=False)
sw7 = pp.create_switch(net, bus7, line4, et="l", type="LBS", closed=True)
sw8 = pp.create_switch(net, bus5, line4, et="l", type="LBS", closed=True)

tab_switch = net.switch #switch table

#create load
pp.create_load(net, bus7, p_mw=2, q_mvar=4, scaling=1, name="load")
#pp.create_load(net, bus7, p_mw=2, q_mvar=4, const_z_percent=30, const_i_percent=20, name="zip_load")

tab_load = net.load #load table

#create static generator
pp.create_sgen(net, bus7, p_mw=2, q_mvar=-0.5, name="static generator")

tab_sgen = net.sgen #sgen table

#create voltage controlled generator
#pp.create_gen(net, bus6, p_mw=6, max_q_mvar=3, min_q_mvar=-3, vm_pu=1.03, name="generator") 

#tab_gen = net.gen #gen table

#create shunt
pp.create_shunt(net, bus3, q_mvar=-0.96, p_mw=0, name='Shunt')

tab_shunt = net.shunt #shunt table 

