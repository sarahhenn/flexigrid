import os
import tempfile

import pandapower as pp
import pandas as pd
import numpy as np

#from battery_opti_place_size import *
#Das ist zum importen unserer Variablen für den load und gen Verlauf aus der main-Datei. Statt dem * vielleicht was anderes nehmen, um Rechenzeit zu begrenzen? Oder geht das so in Ordnung?
#Projektstruktur mal grundlegend überarbeiten
#dazu die battery_opti_size datei in mehrere Methoden aufteilen und diese dann mit return usw bestücken
from pandapower.timeseries.data_sources.frame_data import DFData
from pandapower.timeseries.output_writer import OutputWriter
from pandapower.timeseries.run_time_series import run_timeseries
from pandapower.control.controller.const_control import ConstControl


# hier statt diesen testeingaben einmal die Daten unseres Modells einfügen

#output_dir = "C:\\users\\flori\\pycharmprojects\\Bachelorarbeit\\flexigrid\\output_zeitschleife"
#diese Zeile nur aktivieren, wenn unten der temp Befehl deaktiviert wird

def timeseries_example(output_dir):
    net = simple_test_net()
    #noch aus unserer Methode das abholen

    # 2. create data source
    n_timesteps = 24
    #stattdessen unsere Zeitabstände nach Clustering (12)
    profiles, ds = create_data_source(n_timesteps)

    # 3. create controllers (to control P values of the load and the sgen)
    create_controllers(net, ds)

    # time steps to be calculated. Could also be a list with non-consecutive time steps
    time_steps = range(0, n_timesteps)

    # 4. the output writer with the desired results to be stored to files
    ow = create_output_writer(net, time_steps, output_dir=output_dir)

    # 5. the main time series function
    run_timeseries(net, time_steps, output_writer=ow)


# hier wird nur ein Beispielnetz intiiert, das kann später raus

def simple_test_net():
    """
    simple net that looks like:

    ext_grid b0---b1 trafo(110/20) b2----b3 load
                                    |
                                    |
                                    b4 sgen
    """
    net = pp.create_empty_network()
    pp.set_user_pf_options(net, init_vm_pu="flat", init_va_degree="dc", calculate_voltage_angles=True)

    b0 = pp.create_bus(net, 110)
    b1 = pp.create_bus(net, 110)
    b2 = pp.create_bus(net, 20)
    b3 = pp.create_bus(net, 20)
    b4 = pp.create_bus(net, 20)

    pp.create_ext_grid(net, b0)
    pp.create_line(net, b0, b1, 10, "149-AL1/24-ST1A 110.0")
    pp.create_transformer(net, b1, b2, "25 MVA 110/20 kV", name='tr1')
    pp.create_line(net, b2, b3, 10, "184-AL1/30-ST1A 20.0")
    pp.create_line(net, b2, b4, 10, "184-AL1/30-ST1A 20.0")

    pp.create_load(net, b2, p_mw=20., q_mvar=10., name='load1')
    pp.create_sgen(net, b4, p_mw=20., q_mvar=0.15, name='sgen1')

    return net


def create_data_source(n_timesteps=24):
    profiles = pd.DataFrame()
    profiles['load1_p'] = np.random.random(n_timesteps) * 20.
    # da stattdessen dann unsere powerPlug Daten reinwerfen
    profiles['sgen1_p'] = np.random.random(n_timesteps) * 20.
    # Kann sein, dass nur eine Spalte benötigt wird. Also statt gen und load einfach nur unsere powerPlug Spalte

    ds = DFData(profiles)

    return profiles, ds


"""create the controllers by telling the function which element_index belongs to wich profile. In this case we map: 

    - first load in dataframe (element_index=[0]) to the profile_name "load1_p" 
    - first sgen in dataframe (element_index=[0]) to the profile_name "sgen1_p" 
"""


# Ist noch nicht ganz geklärt, ob ich den noch selbst schreiben muss, oder ob es den schon gibt.

def create_controllers(net, ds):
    ConstControl(net, element='load', variable='p_mw', element_index=[0],
                 data_source=ds, profile_name=["load1_p"])
    ConstControl(net, element='sgen', variable='p_mw', element_index=[0],
                 data_source=ds, profile_name=["sgen1_p"])


"""create the output writer. Instead of saving the whole net (which takes a lot of time), we extract only pre defined outputs. In this case we:

    save the results to "../timeseries/tests/outputs"
    write the results to ".xls" Excel files. (Possible are: .json, .p, .csv)
    log the variables "p_mw" from "res_load", "vm_pu" from "res_bus" and two res_line values.

"""


def create_output_writer(net, time_steps, output_dir):
    ow = OutputWriter(net, time_steps, output_path=output_dir, output_file_type=".xls")
    # these variables are saved to the harddisk after / during the time series loop
    # Hier eventuell einen anderen datentyp wählen, da das mit excel files langsamer ist
    ow.log_variable('res_load', 'p_mw')
    ow.log_variable('res_bus', 'vm_pu')
    ow.log_variable('res_line', 'loading_percent')
    ow.log_variable('res_line', 'i_ka')
    return ow


# execution follows

output_dir = os.path.join(tempfile.gettempdir(), "time_series_example")
# muss ich die output directory überhaupt definieren, oder wird mit dieser Zeile gesagt, dass sie eh nur temporär ist?
print("Results can be found in your local temp folder: {}".format(output_dir))
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
timeseries_example(output_dir)