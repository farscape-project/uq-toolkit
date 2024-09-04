[Mesh]
  type = CoupledMFEMMesh
  file = ../Meshing/vac_meshed_oval_coil_and_stc.e
  dim = 3
[]

[Problem]
  type = MFEMProblem
[]

[Formulation]
  type = ComplexAFormulation
  magnetic_vector_potential_name = magnetic_vector_potential
  magnetic_vector_potential_re_name = magnetic_vector_potential_re
  magnetic_vector_potential_im_name = magnetic_vector_potential_im
  frequency_name = frequency
  magnetic_reluctivity_name = magnetic_reluctivity
  magnetic_permeability_name = magnetic_permeability
  electric_conductivity_name = electrical_conductivity
  dielectric_permittivity_name = dielectric_permittivity

  electric_field_re_name = electric_field_re
  electric_field_im_name = electric_field_im
  current_density_re_name = current_density_re
  current_density_im_name = current_density_im
  magnetic_flux_density_re_name = magnetic_flux_density_re
  magnetic_flux_density_im_name = magnetic_flux_density_im
  joule_heating_density_name = joule_heating_density
[]

[FESpaces]
  [H1FESpace]
    type = MFEMFESpace
    fespace_type = H1
    order = FIRST
  []
  [HCurlFESpace]
    type = MFEMFESpace
    fespace_type = ND
    order = FIRST
  []
  [HDivFESpace]
    type = MFEMFESpace
    fespace_type = RT
    order = CONSTANT
  []
  [L2FESpace]
    type = MFEMFESpace
    fespace_type = L2
    order = CONSTANT
  []
[]

[AuxVariables]
  [magnetic_vector_potential_re]
    type = MFEMVariable
    fespace = HCurlFESpace
  []
  [magnetic_vector_potential_im]
    type = MFEMVariable
    fespace = HCurlFESpace
  []
  [electric_field_re]
    type = MFEMVariable
    fespace = HCurlFESpace
  []
  [electric_field_im]
    type = MFEMVariable
    fespace = HCurlFESpace
  []
  [current_density_re]
    type = MFEMVariable
    fespace = HDivFESpace
  []
  [current_density_im]
    type = MFEMVariable
    fespace = HDivFESpace
  []
  [magnetic_flux_density_re]
    type = MFEMVariable
    fespace = HDivFESpace
  []
  [magnetic_flux_density_im]
    type = MFEMVariable
    fespace = HDivFESpace
  []

  [source_current_density]
    type = MFEMVariable
    fespace = HDivFESpace
  []
  [source_electric_field]
    type = MFEMVariable
    fespace = HCurlFESpace
  []
  [source_electric_potential]
    type = MFEMVariable
    fespace = H1FESpace
  []
  [temperature]
    family = LAGRANGE
    order = FIRST
    initial_condition = 300.0 # K
  []
  [joule_heating_density]
    family = MONOMIAL
    order = CONSTANT
    initial_condition = 0.0
  []
[]

[Functions]
  [ss316l-rx]
    type = PiecewiseLinear
    data_file = MaterialData/steel_316L_electrical_resistivity.csv
    format = columns
  []
[]

[BCs]
  [tangential_E_bdr]
    # Sets tangential components of magnetic vector potential on surface.
    # Equivalent to setting the tangential components of E = 0 
    # at all times on this boundary. 
    type = MFEMComplexVectorDirichletBC
    variable = magnetic_vector_potential
    boundary = '1 2 3'
    real_vector_coefficient = ZeroVectorCoef
    imag_vector_coefficient = ZeroVectorCoef
  []
[]

[Materials]  
  [target]
    type = MFEMConductor
    electrical_conductivity_coeff = TargetEConductivity
    electric_permittivity_coeff = VacuumPermeability # should be VacuumPermittivity?
    magnetic_permeability_coeff = VacuumPermeability
    block = 2
  []
  [vacuum]
    type = MFEMConductor
    electrical_conductivity_coeff = VacuumEConductivity
    electric_permittivity_coeff = VacuumPermittivity
    magnetic_permeability_coeff = VacuumPermeability
    block = '1 3'
  []
[]

[VectorCoefficients]
  [ZeroVectorCoef]
    type = MFEMVectorConstantCoefficient
    value_x = 0.0
    value_y = 0.0
    value_z = 0.0
  []
[]

[Coefficients]
  [CoilEConductivity]
    type = MFEMConstantCoefficient
    value = 5.998e7 # S/m 
  []  
  [VacuumEConductivity]
    type = MFEMConstantCoefficient
    value = 1.0 # S/m
  []
  [VacuumPermeability]
    type = MFEMConstantCoefficient
    value = 1.25663706e-6 # T m/A
  []
  [VacuumPermittivity]
    type = MFEMConstantCoefficient
    value = 8.85418782e-12 # F/m
  []

  [TargetEConductivity]
    type = MFEMTemperatureDependentConductivityCoefficient
    temperature_variable = 'temperature'
    resistivity_function = ss316l-rx
  []

  # 1 kA RMS current source at 100 kHz
  [CurrentMagnitude]
    type = MFEMConstantCoefficient
    value = 1828.42712475 # A
  []
  [frequency]
    type = MFEMConstantCoefficient
    value = 1.0e5 # Hz
  []
[]

[Sources]
  [SourcePotential]
    type = MFEMOpenCoilSource
    total_current_coef = CurrentMagnitude
    # electrical_conductivity_coef = CoilEConductivity
    source_current_density_gridfunction = source_current_density
    source_electric_field_gridfunction = source_electric_field
    source_potential_gridfunction = source_electric_potential
    coil_in_boundary = 1
    coil_out_boundary = 2
    block = 1

    l_tol = 1e-16
    l_abs_tol = 1e-16
    l_max_its = 300
    print_level = 1
  []
[]

[Executioner]
  type = Steady
[]

