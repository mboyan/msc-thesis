module GerminationSignals

    include("analysis.jl")
    include("conversions.jl")
    include("datautils.jl")
    include("diffusion.jl")
    include("germstats.jl")
    include("plotting.jl")
    include("setup.jl")
    include("solver.jl")

    using .Analysis
    using .Conversions
    using .DataUtils
    using .Diffusion
    using .GermStats
    using .Plotting
    using .Setup
    using .Solver

    export get_concentration_evolution_from_file
    export get_coverage_and_exponent_from_files
    export get_density_and_exponent_from_files
    export summarise_fitted_parameters
    export cm_to_um
    export um_to_cm
    export nm_to_um
    export um_to_nm
    export cm2_to_um2
    export um2_to_cm2
    export mL_to_cubic_um
    export inverse_mL_to_cubic_um
    export cubic_um_to_mL
    export inverse_cubic_um_to_mL
    export inverse_um_to_mL
    export convert_D_to_Ps
    export convert_Ps_to_D
    export compute_stokes_radius
    export composite_Ps
    export compute_spore_area_and_volume_from_dia
    export compute_c_eq
    export compute_D_from_radius_and_viscosity
    export measure_coverage
    export measure_shielding_index
    export extract_mean_cw_concentration
    export compute_spore_concentration
    export generate_spore_positions
    export parse_ijadpanahsaravi_data
    export dantigny
    export generate_dantigny_dataset
    export fit_model_to_data
    export get_params_for_idx
    export fit_model_to_data_equilibrium
    export permeation_time_dependent_analytical
    export diffusion_time_dependent_analytical_src
    export concentration_at_spore_ambient_sources
    export slow_release_pt_src_grid
    export slow_release_pt_src_grid_at_src
    export slow_release_shell_src
    export slow_release_shell_src_at_src
    export compute_permeation_constant
    export diffusion_time_dependent_GPU!
    export diffusion_time_dependent_GPU_low_res
    export diffusion_time_dependent_GPU_hi_res!
    export diffusion_time_dependent_GPU_hi_res_implicit
    export compute_germination_response
    export germ_response_inhibitor_gh
    export germ_response_independent_factors_gh
    export germ_response_independent_factors_st_gh
    export germ_response_inducer_dep_inhibitor_thresh_st_gh
    export germ_response_inducer_dep_inhibitor_perm_st_gh
    export germ_response_inducer_dep_inhibitor_combined_st_gh
    export germ_response_inhibitor_dep_inducer_thresh_gh
    export germ_response_inhibitor_dep_inducer_thresh_st_gh
    export germ_response_inhibitor_dep_inducer_signal_gh
    export germ_response_inhibitor_dep_inducer_signal_st_gh
    export germ_response_inhibitor_dep_inducer_combined_gh
    export germ_response_inhibitor_dep_inducer_combined_st_gh
    export germ_response_inducer_dep_inhibitor_thresh_2_factor_st_gh
    export germ_response_inducer_dep_inhibitor_perm_2_factor_st_gh
    export germ_response_inducer_dep_inhibitor_combined_2_factor_st_gh
    export germ_response_inhibitor_dep_inducer_thresh_2_factor_st_gh
    export germ_response_inhibitor_dep_inducer_signal_2_factor_st_gh
    export germ_response_inhibitor_dep_inducer_combined_2_factor_st_gh
    export germ_response_inhibitor_var_perm_gh
    export germ_response_inducer_thresh_var_perm_st_gh
    export germ_response_inducer_combined_var_perm_st_gh
    export germ_response_independent_factors_var_perm_st_gh
    export germ_response_inducer_thresh_2_factors_var_perm_st_gh
    export germ_response_inducer_signal_2_factors_var_perm_st_gh
    export germ_response_inducer_combined_2_factors_var_perm_st_gh
    export germ_response_inducer_dep_inhibitor_combined_eq
    export germ_response_inducer_dep_inhibitor_combined_eq_c_ex
    export germ_response_inhibitor_dep_inducer_thresh_2_factors_eq
    export germ_response_inhibitor_dep_inducer_thresh_2_factors_eq_c_ex
    export germ_response_inhibitor_dep_inducer_signal_2_factors_eq
    export germ_response_inhibitor_dep_inducer_signal_2_factors_eq_c_ex
    export germ_response_inhibitor_dep_inducer_combined_2_factors_eq
    export germ_response_inhibitor_dep_inducer_combined_2_factors_eq_c_ex
    export germ_response_independent_eq
    export germ_response_independent_eq_c_ex
    export generate_ax_grid_pyplot
    export generate_grid_layout_glmakie
    export plot_spheres!
    export plot_spore_clusters
    export plot_concentration_lattice
    export compare_concentration_lattice
    export plot_concentration_evolution
    export compare_concentration_evolutions
    export compare_concentration_evolution_groups
    export plot_lattice_regions
    export plot_functional_relationship
    export compare_functional_relationships
    export compare_functional_relationships_groups
    export plot_spore_positions
    export plot_spore_arrangements
    export plot_dantigny_time_course
    export compare_time_course_to_dantigny
    export plot_germination_data_fit
    export plot_germination_data_fit_all
    export setup_spore_cluster
    export run_simulation
    export run_simulations
    export setup_model_comparison
    export invoke_smart_kernel_3D
    export max_reduce_kernel
    export update_GPU!
    export update_GPU_spore_cluster!
    export update_GPU_low_res!
    export update_GPU_low_res_spore_cluster!
    export update_GPU_hi_res!
    export update_GPU_hi_res_coeffs!
    export initialise_lattice_and_operator_GPU!
    export initialise_lattice_and_operator_GPU_abs_bndry!

end