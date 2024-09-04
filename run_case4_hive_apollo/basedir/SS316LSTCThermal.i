CoolantBulkTemperature = 300. # K

[Mesh]
  [meshed-coil-and-target]
    type = FileMeshGenerator
    file = ../Meshing/solid_meshed_stc.e
  []
  second_order = true
[]

[Variables]
  [temperature]
    family = LAGRANGE
    order = SECOND
    initial_condition = ${CoolantBulkTemperature}
  []
[]

[AuxVariables]
  [joule_heating_density]
    family = MONOMIAL
    order = CONSTANT
  []
[]

[Functions]
  [ss316l-sh-func]
    type = PiecewiseLinear
    data_file = MaterialData/steel_316L_specific_heat_capacity.csv
    format = columns
  []
  [ss316l-tc-func]
    type = PiecewiseLinear
    data_file = MaterialData/steel_316L_thermal_conductivity.csv
    format = columns
  []
  [ss316l-density-func]
    type = PiecewiseLinear
    data_file = MaterialData/steel_316L_density.csv
    format = columns
  []
  [water-htc-func]
    type = PiecewiseLinear
    data_file = MaterialData/water_htc.csv
    format = columns
  []  
[]

[Materials]
  [ss316l-density]
    type = ADCoupledValueFunctionMaterial
    v = temperature
    prop_name = density
    function = ss316l-density-func
  []
  [ss316l-thermal]
    type = ADHeatConductionMaterial
    temp = temperature
    specific_heat_temperature_function = ss316l-sh-func
    thermal_conductivity_temperature_function = ss316l-tc-func
  []
  [coolant_heat_transfer_coefficient]
    type = CoupledValueFunctionMaterial
    v = temperature
    prop_name = heat_transfer_coefficient
    function = water-htc-func
  []
[]

[Kernels]
  [heat]
    type = ADHeatConduction
    variable = temperature
  []
  [heat-time]
    type = ADHeatConductionTimeDerivative
    variable = temperature
  []
  [heat-source]
    type = CoupledForce
    variable = temperature
    v = joule_heating_density
  []
[]

[BCs]
  [heat_flux_out]
    type = ConvectiveHeatFluxBC
    variable = temperature
    boundary = 'inner_pipe'
    T_infinity = ${CoolantBulkTemperature}
    heat_transfer_coefficient = heat_transfer_coefficient
  []
[]

[Postprocessors]
  [max-T]
    type = NodalExtremeValue
    variable = temperature
  []
[]

[Executioner]
  automatic_scaling = true
  solve_type = 'NEWTON'
  type = Transient
  dt = 5.0
  start_time = 0.0
  end_time = 60.0
  line_search = none
  nl_abs_tol = 1e-8
  # nl_rel_tol = 1e-10
  l_tol = 1e-6
  l_max_its = 40
  nl_max_its = 40
  #petsc_options_iname = '-pc_type -pc_hypre_type -pc_hypre_boomeramg_strong_threshold -pc_hypre_boomeramg_coarsen_type -pc_hypre_boomeramg_interp_type '
  #petsc_options_value = '   hypre      boomeramg                         	 0.7                             HMIS                           ext+i '

[]

[Outputs]
  exodus = true
  csv = true
[]

[MultiApps]
  [sub_app]
    type = FullSolveMultiApp
    positions = '0 0 0'
    input_files = 'SS316LSTCComplexAFormEM.i'
    execute_on = timestep_begin
    max_procs_per_app = 1
  []
[]

[Transfers]
  [push_temperature]
    type = MultiAppGeneralFieldShapeEvaluationTransfer

    # Transfer to the sub-app from this app
    to_multi_app = sub_app

    # The name of the variable in this app
    source_variable = temperature

    # The name of the auxiliary variable in the sub-app
    variable = temperature
  []
  [pull_joule_heating]
    type = MultiAppGeneralFieldShapeEvaluationTransfer

    # Transfer from the sub-app to this app
    from_multi_app = sub_app

    # The name of the variable in the sub-app
    source_variable = joule_heating_density

    # The name of the auxiliary variable in this app
    variable = joule_heating_density
  []
[]
