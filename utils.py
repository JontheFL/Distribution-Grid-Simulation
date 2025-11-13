import pandapower.networks as pn
import pandapower as pp
import pandapower.plotting as plot
import numpy as np
import pandas as pd

import pandapower.control as control
import pandapower.networks as nw
import pandapower.timeseries as timeseries
from pandapower.timeseries.data_sources.frame_data import DFData
import warnings

import matplotlib.pyplot as plt

warnings.simplefilter(action="ignore", category=FutureWarning)




def run_time_series(
    gen_data,
    load_data,
    net,
    index_order_gen,
    index_order_load,
    results_suffix="A",
):
    # preserve the original time index from the input gen_data (e.g., 0..23 hours)
    df_gen = pd.DataFrame(gen_data.values, index=gen_data.index, columns=net.sgen.index)
    ds_sgen = DFData(df_gen)

    # make a ConstControl object for the sgen
    const_sgen = control.ConstControl(
        net,
        element="sgen",
        element_index=index_order_gen,
        variable="p_mw",
        data_source=ds_sgen,
        profile_name=net.sgen.index,
    )

    # preserve the original time index from the input load_data
    df_load = pd.DataFrame(load_data.values, index=load_data.index, columns=net.load.index)
    ds_load = DFData(df_load)

    # make a ConstControl object for the load
    const_load = control.ConstControl(
        net,
        element="load",
        element_index=index_order_load,
        variable="p_mw",
        data_source=ds_load,
        profile_name=net.load.index,
    )

    # initialising the outputwriter to save data to excel files in the current folder. You can change this to .json, .csv, or .pickle as well
    ow = timeseries.OutputWriter(
        net, output_path=f"./results_{results_suffix}/", output_file_type=".csv"
    )
    # adding vm_pu of all buses and line_loading in percent of all lines as outputs to be stored
    ow.log_variable("res_bus", "vm_pu")
    ow.log_variable("res_line", "loading_percent")
    ow.log_variable("res_ext_grid", "p_mw")

    # run the time series using the explicit time steps from the dataframes' index
    # this avoids the library guessing the time steps and removes the informational message
    timeseries.run_timeseries(net, time_steps=list(df_gen.index))


    res_ext = pd.read_csv(
        f"./results_{results_suffix}/res_ext_grid/p_mw.csv", delimiter=";", index_col=0
    )
    res_lines = pd.read_csv(
        f"./results_{results_suffix}/res_line/loading_percent.csv", delimiter=";", index_col=0
    )

    return res_ext, res_lines