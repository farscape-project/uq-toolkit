[Mesh]
  type = FileMesh
  file = 'cube_mesh.msh'
[]

[GlobalParams]
  displacements = 'disp_x disp_y disp_z'
  volumetric_locking_correction = true
[]

[Variables]
  [temp]
    initial_condition = 293.15
  []
  [disp_x]
  []
  [disp_y]
  []
  [disp_z]
  []
[]


[AuxVariables]
  [temp_in_C]
  []
  [radiation_flux]
  []
  [stress_xx_nodal]
    order = FIRST
    family = MONOMIAL
  []
  [strain_xx_nodal]
    order = FIRST
    family = MONOMIAL
  []
  [stress_yy_nodal]
    order = FIRST
    family = MONOMIAL
  []
  [strain_yy_nodal]
    order = FIRST
    family = MONOMIAL
  []
  [stress_zz_nodal]
    order = FIRST
    family = MONOMIAL
  []
  [strain_zz_nodal]
    order = FIRST
    family = MONOMIAL
  []
  [vonmises_nodal]
    order = FIRST
    family = MONOMIAL
  []
  [temp_received]
    # order = CONSTANT
    order = FIRST
    family = LAGRANGE
    initial_condition = 293.15
  []
  [aux_flux]
    order = CONSTANT
    family = MONOMIAL
  []
  [aux_flux_boundary]
    order = FIRST
    family = LAGRANGE
  []
[]

[Kernels]
  [isotropic-heat]
    type = ADHeatConduction
    variable = temp
    block = 'copper'
  []
  [heat-time]
    type = ADHeatConductionTimeDerivative
    variable = temp
    density_name = density
    block = 'copper'
  []
  [TensorMechanics] 
     displacements = 'disp_x disp_y disp_z'
     generate_output = 'strain_xx strain_yy strain_zz vonmises_stress'
     eigenstrain_names = 'eigenstrain'
     use_automatic_differentiation = true
     block = 'copper'
  []
  [gravity_z]
    type = ADGravity
    variable = disp_z
    value = -9.81
  []
[]

[AuxKernels]
 [aux_flux_kernel_proj]
    type = ProjectionAux
    variable = aux_flux_boundary
    v = aux_flux
  []
  [aux_flux_kernel]
    type = DiffusionFluxAux
    diffusion_variable = temp
    component = normal
    diffusivity = thermal_conductivity
    variable = aux_flux
    boundary = "innerpipe"
  []
  [K_to_C]
    type = ParsedAux
    variable = 'temp_in_C'
    function = 'temp-273.15'
    args = 'temp'
  []
  [stress_xx]
    type = ADRankTwoAux
    rank_two_tensor = stress
    variable = stress_xx_nodal
    index_i = 0
    index_j = 0
  []
  [strain_xx]
    type = ADRankTwoAux
    rank_two_tensor = total_strain
    index_i = 0
    index_j = 0
    variable = strain_xx_nodal
  []
  [stress_yy]
    type = ADRankTwoAux
    rank_two_tensor = stress
    variable = stress_yy_nodal
    index_i = 1
    index_j = 1
  []  
  [strain_yy]
    type = ADRankTwoAux
    rank_two_tensor = total_strain
    variable = strain_yy_nodal
    index_i = 1
    index_j = 1
  []
  [stress_zz]
    type = ADRankTwoAux
    rank_two_tensor = stress
    variable = stress_zz_nodal
    index_i = 2
    index_j = 2
  []
  [strain_zz]
    type = ADRankTwoAux
    rank_two_tensor = total_strain
    variable = strain_zz_nodal
    index_i = 2
    index_j = 2
  []
  [vonmises]
    type = ADRankTwoScalarAux
    rank_two_tensor = stress
    variable = vonmises_nodal
    scalar_type = VonMisesStress
  []
[]

[Functions]
  [water-htc-function]
    type = PiecewiseLinear
    scale_factor = 1.0
    data_file = "matprops/water_htc.csv"
  []
  [copper_tc]
    # Data from prebuilt material library
    # for material copper
    type = PiecewiseLinear
    data_file = "matprops/copper_tc.csv"
  []
  [copper_sh]
    # Data from prebuilt material library
    # for material copper
    type = PiecewiseLinear
    data_file = "matprops/copper_sh.csv"
  []
  [copper_thermal_expansion]
    # Data from prebuilt material library
    # for material
    type = PiecewiseLinear
    data_file = "matprops/copper_thermal_expansion.csv"
    scale_factor = '1e-6'
  []
[]

[BCs]
  [atmosphere-htc]
    type = ADConvectiveHeatFluxBC
    variable = temp
    boundary = 'top bottom front back right'
    T_infinity = 293.15
    heat_transfer_coefficient = 5 # W/m2K
  []
  [heat-load]
    type = ADNeumannBC
    variable = temp
    boundary = 'top'
    value = '500e3' # 100 kW
  []
  [zero-displacment-x]
    type = ADDirichletBC
    variable = disp_x
    boundary = 'left right'
    value = 0.0
  []
  [zero-displacment-y]
    type = ADDirichletBC
    variable = disp_y
    boundary = 'left right'
    value = 0.0
  []
  [zero-displacment-z]
    type = ADDirichletBC
    variable = disp_z
    boundary = 'left right'
    value = 0.0
  []
  [temp_innerpipe_from_THM]
    type = FunctorDirichletBC
    variable = temp
    boundary = 'innerpipe'
    functor = temp_received
  []
[]
			       
[Materials]
  [copper_density]
    type = ADPiecewiseLinearInterpolationMaterial
    x = '293.15 323.15 373.15 423.15 473.15 523.15 573.15 623.15 673.15 723.15 773.15 823.15 873.15 923.15 973.15 1023.15 1073.15 1123.15 1173.15'
    y = '8940.0 8926.0 8903.0 8879.0 8854.0 8829.0 8802.0 8774.0 8744.0 8713.0 8681.0 8647.0 8612.0 8575.0 8536.0 8495.0 8453.0 8409.0 8363.0'
    property = 'density'
    variable = temp
    block = 'copper'
  []
  [copper_youngs_modulus]
    # Data from prebuilt material library
    # for material
    type = ADPiecewiseLinearInterpolationMaterial
    x = '293.15 323.15 373.15 423.15 473.15 523.15 573.15 623.15 673.15'
    y = '117.0 116.0 114.0 112.0 110.0 108.0 105.0 102.0 98.0'
    variable = temp
    property = 'copper_youngs_modulus'
    scale_factor = 1e9    
    block = 'copper'
  []
  [copper_heat]
    type = ADHeatConductionMaterial
    block = 'copper'
    temp = temp
    specific_heat_temperature_function = 'copper_sh'
    thermal_conductivity_temperature_function = 'copper_tc'
  []
  [copper_elasticity]
    type = ADComputeVariableIsotropicElasticityTensor
    youngs_modulus = copper_youngs_modulus
    poissons_ratio = 0.33
    block = 'copper'
  []
  [copper_thermal_strain]
    type = ADComputeMeanThermalExpansionFunctionEigenstrain
    thermal_expansion_function = 'copper_thermal_expansion'
    thermal_expansion_function_reference_temperature = 293.15
    stress_free_temperature = 293.15
    temperature = temp
    eigenstrain_name = eigenstrain
    block = 'copper'
  []
  [copper_strain] #We use small deformation mechanics
    type = ADComputeSmallStrain
    displacements = 'disp_x disp_y disp_z'
    eigenstrain_names = 'eigenstrain'
    block = 'copper'
  []
  [copper_stress] #We use linear elasticity
    type = ADComputeLinearElasticStress
    block = 'copper'
  []
[]

[Postprocessors]
  [temp_xmin_wall]
    type = PointValue
    variable = temp
    point = '0 0 0.2'
  []
  [max_temp]
    type = NodalExtremeValue
    variable = temp
  []
  [max_disp_x]
    type = NodalExtremeValue
    variable = disp_x
  []
  [max_disp_y]
    type = NodalExtremeValue
    variable = disp_y
  []
  [max_disp_z]
    type = NodalExtremeValue
    variable = disp_z
  []
  [max_stress]
    type = ElementExtremeValue
    variable = vonmises_nodal
  []
[] 

[Executioner]
  automatic_scaling = true
  solve_type = 'NEWTON'
  type = Transient
  line_search = none
  nl_abs_tol = 1e-6
  nl_rel_tol = 1e-8
  l_tol = 1e-6

  l_max_its = 100
  nl_max_its = 10
  dt = 10
  num_steps = 10

  petsc_options_iname = '-pc_type -pc_hypre_type -pc_hypre_boomeramg_strong_threshold -pc_hypre_boomeramg_coarsen_type -pc_hypre_boomeramg_interp_type '
  petsc_options_value = '   hypre      boomeramg                                  0.7                             HMIS                           ext+i '
 

[]
  


[MultiApps]
  [flow_channel]
    type = TransientMultiApp
    app_type = ThermalHydraulicsApp
    input_files = coolant.i 
    execute_on = 'timestep_end'
    max_procs_per_app = 1
    sub_cycling = true
  []
[]

[Transfers]
  [T_from_child_to_parent]
    type = MultiAppGeneralFieldNearestLocationTransfer
    from_multi_app = flow_channel
    distance_weighted_average = true
    source_variable = 'T' 
    to_boundaries = "innerpipe"
    variable = temp_received # *to variable*
    num_nearest_points = 30
  []
  [heatflux_from_parent_to_child]
    type = MultiAppGeneralFieldNearestLocationTransfer
    to_multi_app = flow_channel
    distance_weighted_average = true
    source_variable = aux_flux_boundary # *from variable*
    from_boundaries = "innerpipe"
    variable = q_wall # *to variable*
    num_nearest_points = 200
  []
[]

[Outputs]
  exodus = false
  csv = true
[]

