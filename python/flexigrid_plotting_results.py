import matplotlib.pyplot as plt
import os
import pandapower as pd
import tempfile


def plot_results(output_dir):
    # voltage results
    vm_pu_file = os.path.join(output_dir, "res_bus", "vm_pu.xls")
    vm_pu = pd.read_excel(vm_pu_file)
    vm_pu.plot(label="vm_pu")
    plt.xlabel("time step")
    plt.ylabel("voltage mag. [p.u.]")
    plt.title("Voltage Magnitude")
    plt.grid()
    plt.show()

    # line loading results
    ll_file = os.path.join(output_dir, "res_line", "loading_percent.xls")
    line_loading = pd.read_excel(ll_file)
    line_loading.plot(label="line_loading")
    plt.xlabel("time step")
    plt.ylabel("line loading [%]")
    plt.title("Line Loading")
    plt.grid()
    plt.show()

    # load results
    load_file = os.path.join(output_dir, "res_load", "p_mw.xls")
    load = pd.read_excel(load_file)
    load.plot(label="load")
    plt.xlabel("time step")
    plt.ylabel("P [MW]")
    plt.grid()
    plt.show()