import pandapower as pp
import pandapower.shortcircuit as sc
import pandas as pd
import numpy as np

def create_grid_B():
    """
    Creates the pandapower network for Grid B.
    """
    # create empty net
    net = pp.create_empty_network()

    # create buses 0..17 (bus 0 is 20 kV, the rest are 0.4 kV)
    buses = [pp.create_bus(net, vn_kv=20.0 if i==0 else 0.4, name=f"Bus {i}") for i in range(18)]

    # create bus elements (use the first few buses as examples)
    pp.create_ext_grid(net, bus=buses[0], vm_pu=1.02, name="Grid Connection")
    for i in range(1, 18):
        pp.create_load(net, bus=buses[i], p_mw=0.1, q_mvar=0.005, name=f"Load {i}")# type: ignore
        pp.create_sgen(net, bus=buses[i], p_mw=0.1, q_mvar=0.005, name=f"Sgen {i}")# type: ignore


    # create branch elements
    tid = pp.create_transformer(
        net, hv_bus=buses[0], lv_bus=buses[1], std_type="0.4 MVA 20/0.4 kV", name="Trafo"
    )
    # create primary branch: buses 1 -> 2 -> ... -> 8
    for i in range(1, 8):
        pp.create_line(net, from_bus=buses[i], to_bus=buses[i+1], length_km=0.100, name=f"Line {i}-{i+1}", std_type="NAYY 4x50 SE")

    # create second branch: connect bus 1 to bus 9, then 9 -> 10 -> ... -> 17
    pp.create_line(net, from_bus=buses[1], to_bus=buses[9], length_km=0.100, name="Line 1-9", std_type="NAYY 4x50 SE")
    for i in range(9, 17):
        pp.create_line(net, from_bus=buses[i], to_bus=buses[i+1], length_km=0.100, name=f"Line {i}-{i+1}", std_type="NAYY 4x50 SE")
    
    return net

def rank_buses_by_impedance(net: pp.pandapowerNet, residential_bus_indices):
    """
    Ranks a list of buses from "weakest" to "strongest" using 
    the Short Circuit Impedance Method.
    Weakest bus = highest impedance to reference bus.
    Strongest bus = lowest impedance to reference bus.
    
    :param net: pandapower net object
    :param bus_indices_to_rank: list of bus indices to rank
    :return: list of bus indices ranked from weakest to strongest
    """
    print("Ranking buses by Short-Circuit Impedance (z_sc)...")

    # Define missing short-circuit parameters
    net.ext_grid["s_sc_min_mva"] = 100
    net.ext_grid["s_sc_max_mva"] = 100
    net.ext_grid["rx_min"] = 0.1
    net.ext_grid["rx_max"] = 0.1
    net.sgen["sn_mva"] = 0.1
    net.sgen["k"] = 0.1
    net.line["endtemp_degree"] = 80

    try:
        sc.calc_sc(net, case="max", fault="3ph", bus=residential_bus_indices)
    except Exception as e:
        #print("Error during short-circuit calculation:", e)
        return None
    
    #print(net.res_bus_sc)

    # Extract z_sc values for the specified buses
    r_sc_values = net.res_bus_sc["rk_ohm"]
    x_sc_values = net.res_bus_sc["xk_ohm"]
    z_sc_values = np.sqrt(r_sc_values**2 + x_sc_values**2)
    #print(z_sc_values)

    # Sort buses by z_sc values in descending order (highest impedance first)
    ranked_buses = z_sc_values.sort_values(ascending=False).index.tolist()

    # Subtract 1 from each index to convert from res_bus_sc indices to net bus indices
    ranked_buses = [bus_idx - 1 for bus_idx in ranked_buses]

    print("Ranked Buses (Weakest to Strongest):", ranked_buses)
    return ranked_buses


if __name__ == "__main__":
    net = create_grid_B()
    bus_indices = list(range(1, 18))  # Indices 1 to 17
    ranked_buses = rank_buses_by_impedance(net, bus_indices)
