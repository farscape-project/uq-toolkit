import argparse
from copy import copy
from sys import path
from warnings import warn
import re
import networkx as nx
try:
    import hjson as json
except:
    import json
import numpy as np


heat_transfer_coeff_x = '293.15 294.15 295.15 296.15 297.15 298.15 299.15 300.15 301.15 302.15 303.15 304.15 305.15 306.15 307.15 308.15 309.15 310.15 311.15 312.15 313.15 314.15 315.15 316.15 317.15 318.15 319.15 320.15 321.15 322.15 323.15 324.15 325.15 326.15 327.15 328.15 329.15 330.15 331.15 332.15 333.15 334.15 335.15 336.15 337.15 338.15 339.15 340.15 341.15 342.15 343.15 344.15 345.15 346.15 347.15 348.15 349.15 350.15 351.15 352.15 353.15 354.15 355.15 356.15 357.15 358.15 359.15 360.15 361.15 362.15 363.15 364.15 365.15 366.15 367.15 368.15 369.15 370.15 371.15 372.15 373.15 374.15 375.15 376.15 377.15 378.15 379.15 380.15 381.15 382.15 383.15 384.15 385.15 386.15 387.15 388.15 389.15 390.15 391.15 392.15 393.15 394.15 395.15 396.15 397.15 398.15 399.15 400.15 401.15 402.15 403.15 404.15 405.15 406.15 407.15 408.15 409.15 410.15 411.15 412.15 413.15 414.15 415.15 416.15 417.15 418.15 419.15 420.15 421.15 422.15 423.15 424.15 425.15 426.15 427.15 428.15 429.15 430.15 431.15 432.15 433.15 434.15 435.15 436.15 437.15 438.15 439.15 440.15 441.15 442.15 443.15 444.15 445.15 446.15 447.15 448.15 449.15 450.15 451.15 452.15 453.15 454.15 455.15 456.15 457.15 458.15 459.15 460.15 461.15 462.15 463.15 464.15 465.15 466.15 467.15 468.15 469.15 470.15 471.15 472.15 473.15 474.15 475.15 476.15 477.15 478.15 479.15 480.15 481.15 482.15 483.15 484.15 485.15 486.15 487.15 488.15 489.15 490.15 491.15 492.15 493.15 494.15 495.15 496.15 497.15 498.15 499.15 500.15 501.15 502.15 503.15 504.15 505.15 506.15 507.15 508.15 509.15 510.15 511.15 512.15' 
heat_transfer_coeff_y = '2.420000e+04 2.420000e+04 2.420000e+04 2.420000e+04 2.420000e+04 2.420000e+04 2.420000e+04 2.420000e+04 2.420000e+04 2.420000e+04 2.420000e+04 2.420000e+04 2.420000e+04 2.420000e+04 2.420000e+04 2.420000e+04 2.420000e+04 2.420000e+04 2.420000e+04 2.420000e+04 2.420000e+04 2.420000e+04 2.420000e+04 2.420000e+04 2.420000e+04 2.420000e+04 2.420000e+04 2.420000e+04 2.420000e+04 2.420000e+04 2.420000e+04 2.420000e+04 2.420000e+04 2.420000e+04 2.420000e+04 2.420000e+04 2.420000e+04 2.420000e+04 2.420000e+04 2.420000e+04 2.420000e+04 2.420000e+04 2.420000e+04 2.420000e+04 2.420000e+04 2.420000e+04 2.420000e+04 2.420000e+04 2.420000e+04 2.420000e+04 2.420000e+04 2.420000e+04 2.420000e+04 2.420000e+04 2.420000e+04 2.420000e+04 2.420000e+04 2.420000e+04 2.420000e+04 2.420000e+04 2.420000e+04 2.420000e+04 2.420000e+04 2.420000e+04 2.420000e+04 2.420000e+04 2.420000e+04 2.420000e+04 2.420000e+04 2.420000e+04 2.420000e+04 2.420000e+04 2.420000e+04 2.420000e+04 2.420000e+04 2.420000e+04 2.420000e+04 2.420000e+04 2.420000e+04 2.420000e+04 2.420000e+04 2.451700e+04 2.546500e+04 2.703400e+04 2.920600e+04 3.195700e+04 3.525800e+04 3.907200e+04 4.335700e+04 4.806700e+04 5.315000e+04 5.855000e+04 6.420800e+04 7.006200e+04 7.604800e+04 8.210000e+04 8.815200e+04 9.413800e+04 9.999200e+04 1.056500e+05 1.110500e+05 1.161300e+05 1.208400e+05 1.251300e+05 1.289400e+05 1.322400e+05 1.349900e+05 1.371700e+05 1.387300e+05 1.396800e+05 1.400000e+05 1.399700e+05 1.398600e+05 1.396900e+05 1.394500e+05 1.391500e+05 1.387800e+05 1.383400e+05 1.378300e+05 1.372600e+05 1.366200e+05 1.359200e+05 1.351600e+05 1.343400e+05 1.334500e+05 1.325100e+05 1.315100e+05 1.304500e+05 1.293400e+05 1.281700e+05 1.269500e+05 1.256800e+05 1.243600e+05 1.230000e+05 1.215900e+05 1.201300e+05 1.186400e+05 1.171000e+05 1.155300e+05 1.139300e+05 1.122900e+05 1.106200e+05 1.089200e+05 1.072000e+05 1.054500e+05 1.036800e+05 1.019000e+05 1.000900e+05 9.827800e+04 9.645000e+04 9.461300e+04 9.277100e+04 9.092400e+04 8.907600e+04 8.722900e+04 8.538700e+04 8.355000e+04 8.172200e+04 7.990600e+04 7.810300e+04 7.631700e+04 7.454900e+04 7.280300e+04 7.107900e+04 6.938200e+04 6.771300e+04 6.607400e+04 6.446800e+04 6.289700e+04 6.136300e+04 5.986800e+04 5.841400e+04 5.700400e+04 5.563800e+04 5.431900e+04 5.305000e+04 5.183000e+04 5.066300e+04 4.954900e+04 4.849100e+04 4.748900e+04 4.654600e+04 4.566100e+04 4.483800e+04 4.407600e+04 4.337600e+04 4.274100e+04 4.217000e+04 4.166400e+04 4.122400e+04 4.085100e+04 4.054500e+04 4.030700e+04 4.013700e+04 4.003400e+04 4.000000e+04 4.003400e+04 4.013700e+04 4.030700e+04 4.054500e+04 4.085100e+04 4.122400e+04 4.166400e+04 4.217000e+04 4.274100e+04 4.337600e+04 4.407600e+04 4.483800e+04 4.566100e+04 4.654600e+04 4.748900e+04 4.849100e+04 4.954900e+04 5.066300e+04 5.183000e+04 5.305000e+04 5.431900e+04 5.563800e+04 5.700400e+04 5.841400e+04'


def get_junction_nodes(graph, exclude_node_list=[]):
    degree = np.array(graph.degree)
    degree = degree[degree[:, 1].astype(int) >= 3]
    del_ind = []
    for i, node_i in enumerate(degree[:, 0]):
        if node_i in exclude_node_list:
            del_ind.append(i)
    return np.delete(degree[:, 0], del_ind)

def get_end_nodes(graph):
  degree = np.array(graph.degree).astype(int)
  return degree[:, 0][degree[:, 1]==1]

class MooseTHMProblem:
    def __init__(self):
        self.moose = MOOSEInput()

    def edge_to_pipe_name(self, graph, edge):
        if "name" in graph.edges[edge].keys():
            return graph.edges[edge]["name"]
        else:
            return f"pipe-{edge[0]}-{edge[1]}"

    def make_input(
        self,
        graph,
        network_params,
        config_key,
        restart_thm
    ):
        #### make some variables
        # NB in the TH all variables are automatically created by MOOSE, you
        # only need to store any that you create for auxkernels

        inlet_node, outlet_node = get_end_nodes(graph)
        inlet_node = str(inlet_node)
        outlet_node = str(outlet_node)

        closures = Closures()
        if network_params["closure"] == "simple":
            closure_i = Closures1PhaseSimple(name="simple_closure")
        elif network_params["closure"] == "thm":
            closure_i = Closures1PhaseTHM(name="thm_closure")
            if network_params["heat-transfer"]["source"] == "HeatTransferFromExternalAppHeatFlux1Phase":
                warn(
                    "The moose version used for WP4.6 of FARSCAPE3 does not support thm closure with HeatTransferFromExternalAppHeatFlux1Phase. "
                    "If the run fails, please switch to 'simple' for closures."
                )
        else:
            raise AssertionError("Unexpected closure. Options are ['simple', 'thm']")
        closures.closures[closure_i.name] = closure_i

        functions = Functions()
        function = PiecewiseLinear(
            name = "water-htc-function",
            axis = "x",
            x = heat_transfer_coeff_x,
            y = heat_transfer_coeff_y
        )
        functions.functions[function.name] = function

        #### make the fluid properties
        fluid_properties = FluidProperties()
        # TODO - maybe get material properties from lookup table
        if network_params["fluid"].lower() == "stiff-water":
            fluid = StiffenedGasFluidProperties(
                name="water",
                gamma=2.35,
                cv=4187,
                q=-0.1167e7,
                p_inf=1.0e9,
                q_prime=0,
                M=0.018,
                k=0.6,
                mu=1.0e-3,
                rho_c=1149.9,
                T_c=648.05,
            )
        else:
            if network_params["fluid"].lower() != "simple-water":
                warn(f"Unexpected entry for 'fluid' ({network_params['fluid']}) - defaulting to simple-water")
            fluid = SimpleFluidProperties(
                name="water",
                density0=1000,
                bulk_modulus=2.0e9,
                cp=4194,
                cv=4186,
                fp_type="single-phase-fp",
                molar_mass=0.018,
                porepressure_coefficient=1.0,
                specific_entropy=300,
                thermal_conductivity=0.6,
                thermal_expansion=0.000214,
                viscosity=0.001,
            )
        fluid_properties.fluidproperties[fluid.name] = fluid

        initial_param_dict = {}
        if restart_thm:
            pass
        else:
            initial_param_dict["initial_vel"] = 0.0
            initial_param_dict["initial_vel_x"] = 0.0
            initial_param_dict["initial_vel_y"] = 0.0
            initial_param_dict["initial_vel_z"] = 0.0
            initial_param_dict["initial_p"] = network_params["chimera-settings"][config_key]["p_init"]
            initial_param_dict["initial_T"] = network_params["chimera-settings"][config_key]["T_init"]

        global_params = GlobalParameters(
            # initial_vel=0.0, 
            # initial_vel_x=0.0, 
            # initial_vel_y=0.0,
            # initial_vel_z=0.0,  
            # initial_p=network_params["chimera-settings"][config_key]["p_init"], 
            # initial_T=network_params["chimera-settings"][config_key]["T_init"],
            closures=closure_i.name,
            fp="water",
            rdg_slope_reconstruction=network_params["rdg_slope_reconstruction"],
            **network_params["scaling-settings"],
            **initial_param_dict
        )
        self.moose.global_params = global_params

        self.moose.fluid_properties = fluid_properties

        #### write closures
        self.moose.closures = closures
        self.moose.functions = functions

        # add some components
        components = {}

        # convert system flowrate to SI units
        th_volumetric_flowrate = network_params["chimera-settings"][config_key]["flowrate_Lpermin"] / 60.0 / 1000.0

        # add info on diameters and lengths
        MIN_LENGTH = min(nx.get_edge_attributes(graph, "length").values())
        print(f"minimum length: {MIN_LENGTH}")
        MIN_DIAMETER = min(nx.get_edge_attributes(graph, "diameter").values())
        MAX_DIAMETER = max(nx.get_edge_attributes(graph, "diameter").values())
        print(f"minimum diameter: {MIN_DIAMETER}")
        print(f"maximum diameter: {MAX_DIAMETER}")

        ELEMENT_SIZE = MIN_LENGTH / network_params["element_length_ratio"]
        print(f"ELEMENT SIZE {ELEMENT_SIZE}")

        """
        As a first-order approximation, we will create FlowChannels for all segments,
        and nodes with degree 2 will be joined using 'JunctionOneToOne1Phase' which assumes
        no momentum loss.

        Later iterations could extend by accounting for momentum loss due to bends by using
        VolumeJunction with some specific loss factor, or ElbowPipe1Phase
        """
        for edge_i in graph.edges:
            # graph is digraph, so should be fixed ordering
            proximal_node, distal_node = edge_i

            position_start = graph.nodes[proximal_node]["position"]
            position_end = graph.nodes[distal_node]["position"]
            orientation = np.array(position_end) - np.array(position_start)
            orientation /= np.linalg.norm(orientation)
            pipe_name = self.edge_to_pipe_name(graph, [proximal_node, distal_node])
            graph.edges[edge_i]["name"] = pipe_name

            edge_length = graph.edges[edge_i]["length"]
            area = np.pi * (graph.edges[edge_i]["diameter"]/2)**2.0
            perimeter = np.pi * graph.edges[edge_i]["diameter"]
            
            kwargs_f = {}
            if graph.nodes[proximal_node]["bend_found"] and graph.nodes[distal_node]["bend_found"]:
                friction_factor = network_params["bent_pipe_friction_factor"]
            else:
                friction_factor = network_params["straight_pipe_friction_factor"]
            # only declare friction factor if simple closure is used
            if network_params["closure"] == "thm":
                pass
            else:
                kwargs_f["f"] = friction_factor

            num_elements = int(np.ceil(edge_length / ELEMENT_SIZE))

            component = FlowChannel1Phase(
                name=pipe_name,
                position=position_start,
                orientation=list(orientation),
                length=edge_length,
                n_elems=num_elements,
                A=area,
                D_h=graph.edges[edge_i]["diameter"],
                # f=friction_factor,
                **kwargs_f
            )
            del kwargs_f
            components[component.name] = component
            if network_params["heat-transfer"]["source"] == "HeatTransferFromExternalAppTemperature1Phase":
                initial_T_wall = {}
                if restart_thm:
                    pass
                else:
                    initial_T_wall["initial_T_wall"] = network_params["chimera-settings"][config_key]["T_init"]
                
                component = HeatTransferFromExternalAppTemperature1Phase(
                    name=f"hxcon-{pipe_name}",
                    flow_channel=pipe_name,
                    Hw="water-htc-function",
                    P_hf=perimeter,
                    **initial_T_wall
                )
                components[component.name] = component
            elif network_params["heat-transfer"]["source"] == "HeatTransferFromSpecifiedTemperature1Phase":
                component = HeatTransferFromSpecifiedTemperature1Phase(
                    name=f"hxcon-{pipe_name}",
                    flow_channel=pipe_name,
                    Hw="water-htc-function",
                    T_wall=network_params["heat-transfer"]["T_wall"],
                    P_hf=perimeter
                )
                components[component.name] = component
            elif network_params["heat-transfer"]["source"] == "HeatTransferFromExternalAppHeatFlux1Phase":
                component = HeatTransferFromExternalAppHeatFlux1Phase(
                    name=f"hxcon-{pipe_name}",
                    flow_channel=pipe_name,
                    Hw="water-htc-function",
                    P_hf=perimeter
                )
                components[component.name] = component
            elif network_params["heat-transfer"]["source"] == "none":
                pass
            else:
                raise ValueError(
                    f'Unexpected entry for network_params["heat-transfer"]["source"]'
                    f' = {network_params["heat-transfer"]["source"]}'
                )

        # initialise new attribute to check connectivity        
        for edge_i in graph.edges:
            graph.edges[edge_i]["connected_in"] = False
            graph.edges[edge_i]["connected_out"] = False
        for node_i in graph.nodes:
            graph.nodes[node_i]["connected"] = False

        degree_arr = np.array(graph.degree)
        for i, node_i in enumerate(graph.nodes):
            if int(degree_arr[i, 1]) == 2:
                name = f"junction-{node_i}"
                graph.nodes[node_i]["name"] = name
                upstream_edges = list(graph.in_edges(node_i))
                downstream_edges = list(graph.out_edges(node_i))
                
                if not (len(upstream_edges) > 0 and len(downstream_edges) > 0):
                    warn("size of both lists should be non-zero")

                upstream_connection_name = self.edge_to_pipe_name(graph, upstream_edges[0])
                downstream_connection_name = self.edge_to_pipe_name(graph, downstream_edges[0])

                connection_string = f"{upstream_connection_name}:out {downstream_connection_name}:in"
                component = JunctionOneToOne1Phase(
                    name=name,
                    connections=connection_string
                )
                components[component.name] = component            
                graph.edges[upstream_edges[0]]["connected_out"] = True
                graph.edges[downstream_edges[0]]["connected_in"] = True
                graph.nodes[node_i]["connected"] = True

        # # for junctions nodes with degree >= 3, we assign VolumeJunction1Phase
        junction_nodes = get_junction_nodes(graph)
        for i, node_i in enumerate(junction_nodes):
            position = graph.nodes[node_i]["position"]
            upstream_connections_list = list(graph.in_edges(node_i))
            downstream_connections_list = list(graph.out_edges(node_i))

            connections_to_write = []
            for upstream_edge_i in upstream_connections_list:
                # junction is connected to the OUTLET of upstream_edges -> :out
                connections_to_write.append(f"{graph.edges[upstream_edge_i]['name']}:out")
                graph.edges[upstream_edge_i]['connected_out'] = True
            for downstream_edge_i in downstream_connections_list:
                # junction is connected to the INLET of downstream_edges -> :in
                connections_to_write.append(f"{graph.edges[downstream_edge_i]['name']}:in")
                graph.edges[downstream_edge_i]['connected_in'] = True
            
            # assume junction is sphere with radius = diameter at node
            junction_avg_radius = graph.nodes[node_i]["diameter"] / 2.0
            junction_volume = 4.0/3.0 * np.pi * junction_avg_radius ** 3.0 

            component = VolumeJunction1Phase(
                name=f"junction-{node_i}",
                position=position,
                volume=junction_volume,
                connections=" ".join(connections_to_write),
                K=network_params["volume_junction_loss_factor"],
            )
            components[component.name] = component
            graph.nodes[node_i]["connected"] = True


        # Assign boundary conditions to coolant network, inlet with y-coord = min, outlet y-coord = max
        node_type = type(list(graph.nodes)[0])
        end_nodes = get_end_nodes(graph).astype(node_type)
        node_position = nx.get_node_attributes(graph, "position")
        ycoord_end_0 = node_position[end_nodes[0]][1]
        ycoord_end_1 = node_position[end_nodes[1]][1]
        inlet_node = end_nodes[np.argmin([ycoord_end_0, ycoord_end_1])]
        outlet_node = end_nodes[np.argmax([ycoord_end_0, ycoord_end_1])]
        path_inlet_to_outlet = nx.shortest_path(graph, inlet_node, outlet_node)

        print("adding inlet")
        inlet_pipe_id = self.edge_to_pipe_name(graph, [inlet_node, path_inlet_to_outlet[1]])
        graph.nodes[inlet_node]["connected"] = True
        inlet_area = np.pi * (graph.edges[[inlet_node, path_inlet_to_outlet[1]]]["diameter"]/2)**2.0
        inlet_velocity = th_volumetric_flowrate / inlet_area
        component = InletVelocityTemperature1Phase(
            name="inlet", input=components[inlet_pipe_id], vel=inlet_velocity, temperature=network_params["chimera-settings"][config_key]["T_inlet"]
        )
        components[component.name] = component

        estimated_propagation_time = ELEMENT_SIZE / inlet_velocity
        if "dt" in network_params["solve_options"]:
            estimated_courant_num = network_params["solve_options"]["dt"] / estimated_propagation_time
            estimated_courant_num_str = (
                f"Estimated courant number at inlet is {estimated_courant_num}. " 
                f"dt = {network_params['solve_options']['dt']}, characteristic time: {estimated_propagation_time}"
            )
            if estimated_courant_num > 1:
                warn(estimated_courant_num_str)
            else:
                print(estimated_courant_num_str)
        
        print("adding outlet")
        outlet_pipe_id = self.edge_to_pipe_name(graph, [path_inlet_to_outlet[-2], outlet_node])
        graph.nodes[outlet_node]["connected"] = True
        component = Outlet1Phase(name="outlet", input=components[outlet_pipe_id], pressure=network_params["chimera-settings"][config_key]["p_outlet"])
        components[component.name] = component

        # assert (
        #     outlet_found and inlet_found
        # ), f"not all BCs found inlet: {inlet_found}. outlet: {outlet_found}."

        components[component.name] = component
        comps = Components()
        comps.components = components
        self.moose.components = comps

        #### executioner
        exec = Executioner()
        solve_options = {}
        solve_options = network_params["solve_options"]
        exec.solve_objects = solve_options
        self.moose.executioner = exec

        num_nonconnected_out = np.array(list(nx.get_edge_attributes(graph, 'connected_out').values())) == False
        num_nonconnected_in = np.array(list(nx.get_edge_attributes(graph, 'connected_in').values())) == False
        # num_nonconnected_nodes = np.array(list(nx.get_node_attributes(graph, 'connected').values())) == False
        if (num_nonconnected_in.sum()-1) != 0 or (num_nonconnected_out.sum()-1) != 0:
            print(f"FAILED< CHECK VEDO {num_nonconnected_in.sum()} {num_nonconnected_out.sum()}")
            raise AssertionError("Unconnected edges found")

        postproc = PostProcessors()
        pressuredrop = PressureDrop(upstream_boundary="inlet", downstream_boundary="outlet", boundary="inlet outlet", pressure="p", name="pressuredrop")
        # postproc.post_processors = {}
        postproc.post_processors[pressuredrop.name] = pressuredrop
        self.moose.post_processors = postproc

        auxvar = AuxVariables()
        # family arg = 1 is Lagrange, 2 is Monomial
        auxvar.add_variable("vel_magnitude", 1, 2, "", "MooseVariableFVReal")
        self.moose.aux_variables = auxvar

        auxkernels = AuxKernels()
        auxkernels.kernels["vel_magnitude"] = VectorMagnitudeAux(name="vel_magnitude", variable="vel_magnitude", x="vel_x", y="vel_y", z="vel_z")
        self.moose.aux_kernels = auxkernels

        #### outputss
        output = Outputs(**network_params["outputs_kwargs"])
        self.moose.outputs = output

    def write_input(self, filename="test.i"):
        self.moose.write(filename)

    def add_extra_objs(self, file_name, extra_objects, restart_dict=None):
        with open(file_name, "r") as f:
            line_list = f.readlines()
        
        out_line_list = []
        if restart_dict is not None:
            out_line_list.append(restart_dict["restart-string"].replace("RESTARTDIR", restart_dict["restart-dir"])+"\n")
        
        for i, line_i in enumerate(line_list):
            out_line_list.append(line_i) # add current line to file
            # loop over all lines, remove whitespace and square brackets to find object e.g. [Executioner]
            line_clean = line_i.strip()
            # line_clean = re.sub(r"[\[\]]", "", line_clean)
            for key_i in extra_objects.keys():
                if line_clean == key_i:
                    print(f"appending {key_i}: \n{extra_objects[key_i]}")
                    # add string from json to list of strings to write. Add a newline to tidy up input file
                    out_line_list.append(extra_objects[key_i]+"\n")
        
        with open(file_name, "w") as f:
            [f.write(out_line_i) for out_line_i in out_line_list]
            
        

def get_inputs():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chimera_case", "-c",
        default="low-pressure", 
        type=str,
        choices=["low-pressure", "high-pressure"],
        help="case parameters to test",
    )
    parser.add_argument("--restart",
        default=False,
        action="store_true",
        help="If restart mode used, turn off all initial conditions from file",
    ) 
    return parser.parse_args()

if __name__ == "__main__":
    args = get_inputs()
    with open("chimera_params.jsonc", "r") as f:
        config_chimera = json.load(f)

    path.insert(0, config_chimera["moopy-path"])
    from moose.moose import MOOSEInput
    from moose.global_parameters import GlobalParameters
    from moose.fluidproperties import (
        FluidProperties,
        SimpleFluidProperties,
        StiffenedGasFluidProperties,
    )
    from moose.postprocessors import PostProcessors, PressureDrop
    from moose.functions import (
        Functions,
        PiecewiseLinear,
    )
    from moose.closures import Closures, Closures1PhaseSimple, Closures1PhaseTHM
    from moose.variables import AuxVariables
    from moose.kernels import AuxKernels, VectorMagnitudeAux
    from moose.components import (
        Components,
        FlowChannel1Phase,
        InletMassFlowRateTemperature1Phase,
        InletVelocityTemperature1Phase,
        Outlet1Phase,
        JunctionOneToOne1Phase,
        VolumeJunction1Phase,
        HeatTransferFromExternalAppTemperature1Phase,
        HeatTransferFromExternalAppHeatFlux1Phase,
        HeatTransferFromSpecifiedTemperature1Phase
    )
    from moose.executioner import Executioner
    from moose.outputs import Outputs

    filename_graph = "digraph.gml"
    graph = nx.read_gml(filename_graph)
    ORIG_NUM_ENDS = get_end_nodes(graph).size
    print(f"num ends {ORIG_NUM_ENDS}")
    print(f"num ends {get_end_nodes(graph).size}")
    print(
        f"Parsing graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges"
    )

    # class for thermal hydraulic stuff
    moose_th = MooseTHMProblem()
    moose_th.make_input(graph, config_chimera, args.chimera_case, args.restart)
    moose_th.write_input(config_chimera["config-name"]+".i")
    if args.restart:
        restart_dict = config_chimera["restart-dict"]
    else:
        restart_dict = None
    moose_th.add_extra_objs(config_chimera["config-name"]+".i", config_chimera["extra-objects"], restart_dict=restart_dict)

    # hacky way to add debug stuff
    # with open(config_chimera["config-name"]+".i", "r") as f:
    #     lines = f.readlines()
    
    # with open(config_chimera["config-name"]+".i", "a") as f:
    #     f.write("[Debug]\n show_var_residual_norms = true\n[]\n")
