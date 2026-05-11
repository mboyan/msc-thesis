module GermStats
__precompile__(false)
    """
    Contains tools for generating germination statistics
    """

    using QuadGK
    using Cubature
    using FastGaussQuadrature
    using LinearAlgebra
    using MeshGrid
    using Distributions
    using SpecialFunctions
    using ArgCheck
    using DifferentialEquations
    using QuasiMonteCarlo
    using Parameters

    include("./conversions.jl")
    using .Conversions
    
    export load_model_collection
    export compute_gresp_xform_params
    export compute_germination_response
    export inducer_concentration

    export gresp_independent_factors_gh
    export gresp_inducer_dep_inhibitor_thresh_gh
    export gresp_inducer_dep_inhibitor_perm_gh
    export gresp_inducer_dep_inhibitor_gh
    export gresp_inhibitor_dep_inducer_thresh_gh
    export gresp_inhibitor_dep_inducer_signal_gh
    export gresp_inhibitor_dep_inducer_gh
    export gresp_inducer_dep_inhibitor_thresh_2_factor_gh
    export gresp_inhibitor_dep_inducer_thresh_2_factor_gh
    export gresp_inhibitor_dep_inducer_signal_2_factor_gh
    export gresp_inhibitor_dep_inducer_2_factor_gh
    export gresp_inducer_thresh_var_perm_gh
    export gresp_inducer_var_perm_gh
    export gresp_independent_factors_var_perm_gh
    export gresp_inducer_thresh_2_factors_var_perm_gh
    export gresp_inducer_signal_2_factors_var_perm_gh
    export gresp_inducer_2_factors_var_perm_gh
    export gresp_inh_dep_ind_signal_ind_dep_inh_thresh_gh
    export gresp_inh_dep_ind_signal_ind_dep_inh_thresh_2_factor_gh
    export gresp_inh_dep_ind_thresh_ind_dep_inh_thresh_2_factor_gh
    export gresp_inh_dep_ind_thresh_signal_ind_dep_inh_thresh_2_factor_gh

    export gresp_inducer_dep_inhibitor_eq
    export gresp_inducer_dep_inhibitor_eq_c_ex
    export gresp_inhibitor_dep_inducer_thresh_2_factors_eq
    export gresp_inhibitor_dep_inducer_thresh_2_factors_eq_c_ex
    export gresp_inhibitor_dep_inducer_signal_2_factors_eq
    export gresp_inhibitor_dep_inducer_signal_2_factors_eq_c_ex
    export gresp_inhibitor_dep_inducer_2_factors_eq
    export gresp_inhibitor_dep_inducer_2_factors_eq_c_ex
    export gresp_independent_eq
    export gresp_independent_eq_c_ex

    export gresp_feedback

    export clamp_inplace!

    export ode_inducer_dependent_perm!
    export ode_inhibitor_dependent_perm!
    export ode_inducer_dependent_perm_inhibitor_dependent_signal!
    export ode_inducer_and_inhibitor_dependent_perm!
    export ode_inducer_and_inhibitor_dependent_perm_inhibitor_dependent_signal!

    export thresh_criterion_inhibitor
    export thresh_criterion_inducer
    export thresh_criterion_combined
    export thresh_criterion_inhibitor_shift
    export thresh_criterion_combined_inhibitor_shift
    export thresh_criterion_inducer_shift
    export thresh_criterion_combined_inducer_shift
    export thresh_criterion_inducer_signal
    export thresh_criterion_combined_inducer_signal
    export thresh_criterion_combined_shift
    export thresh_criterion_inhibitor_signal_shift
    export thresh_criterion_combined_inhibitor_signal_shift
    export thresh_criterion_inducer_signal_shift
    export thresh_criterion_combined_inducer_signal_shift
    export thresh_criterion_combined_inhibitor_shift_inducer_signal_shift


    # const DEBUG_MODE = false
    global PARAMS_BUFFER = []
    const LAMBDA_MAX = 1e6
    
    # Macro for printing code line in debug mode
    macro print_line(expr)
        line_str = string(__source__.line)
        code_str = string(expr)
        return quote
            global LINE_BUFFER = $code_str  # save code line in a variable as string
            # println("Executing at line $line_str: ", code_line)
            result = $(esc(expr))  # execute the original expression
            result     # return result
        end
    end

    macro note_param(expr)
        # global PARAMS_BUFFER
        code_str = string(expr)
        
        return quote
            
            substrings = split($code_str, r", | \[|ns=|ks=\[")
            for substring in substrings
                if startswith(substring, "prms[:")
                    global PARAMS_BUFFER
                    end_idx = findfirst(==(']'), substring)
                    safe_end = prevind(substring, end_idx)
                    push!(PARAMS_BUFFER, Symbol(SubString(substring, 7, safe_end)))
                end
            end
            result = $(esc(expr))  # execute the original expression
            result     # return result
        end
    end


    """
    Loads the entire collection of germination models,
    including their aliases, combination IDs and textual descriptions.
    output:
        aliases - model identifier strings
        combination_IDs - a string containing the interaction identifiers (A, B, C, D and E)
        descriptions - textual descriptions of the models
        param_key_sets - arrays of keys of the parameters used in each model
    """
    function load_model_collection()
        
        models = Dict(
            "T" => ["test",                                                                     "Toy model for testing\n",
                        [:Pₛ, :neg_δ_ω, :μ_ω]],
            "0" => ["independent",                                                              "Independent inducer/inhibitor model\n",
                        [:K_cC, :Pₛ, :Pₛ_cs, :neg_δ_γ, :neg_δ_ω, :μ_γ, :μ_ω]],
            "Ai" => ["feedback_inhibitor_inducer_perm",                                         "Inhibitor-dependent germination with\ninducer-dependent permeability",
                        [:K_cC, :Pₛ, :Pₛ_cs, :s, :neg_δ_γ, :neg_δ_ψ, :μ_γ, :μ_ψ]],
            "A" => ["feedback_combined_inducer_perm",                                           "2-factor germination with\ninducer-dependent permeability",
                        [:K_cC, :Pₛ, :Pₛ_cs, :s, :neg_δ_γ, :neg_δ_ψ, :neg_δ_ω, :μ_γ, :μ_ψ, :μ_ω]],
            "Bi" => ["inhibitor_thresh",                                                        "Inhibitor-dependent induction threshold\n",
                        [:K_cC, :Pₛ, :Pₛ_cs, :k_C, :neg_δ_γ, :μ_γ]],
            "B" => ["combined_inhibitor_thresh",                                                "2-factor germination with inducer-dependent\ninhibitor threshold",
                        [:K_cC, :Pₛ, :Pₛ_cs, :k_C, :neg_δ_γ, :neg_δ_ω, :μ_γ, :μ_ω]],
            "Cc" => ["inducer_signal" ,                                                         "Inhibitor-dependent induction signal\n",
                        [:K_I, :K_cC, :Pₛ, :Pₛ_cs, :n, :neg_δ_ψ, :neg_δ_ω, :μ_ψ, :μ_ω]],
            "C" => ["combined_inducer_signal" ,                                                 "2-factor germination with inhibitor-dependent\ninduction signal",
                        [:K_I, :K_cC, :Pₛ, :Pₛ_cs, :n, :neg_δ_γ, :neg_δ_ψ, :neg_δ_ω, :μ_γ, :μ_ψ, :μ_ω]],
            "Dc" => ["feedback_inducer_inhibitor_perm",                                         "Inducer-dependent germination with\ninhibitor-dependent permeability",
                        [:K_cC, :K_cI, :Pₛ, :Pₛ_cs, :b, :neg_δ_ψ, :neg_δ_ω, :μ_ψ, :μ_ω]],
            "D" => ["feedback_combined_inhibitor_perm",                                         "2-factor germination with\ninhibitor-dependent permeability",
                        [:K_cC, :K_cI, :Pₛ, :Pₛ_cs, :b, :neg_δ_γ, :neg_δ_ψ, :neg_δ_ω, :μ_γ, :μ_ψ, :μ_ω]],
            "Ec" => ["inducer_thresh",                                                          "Inhibitor-dependent induction threshold\n",
                        [:K_cC, :K_cI, :Pₛ, :Pₛ_cs, :k_I, :neg_δ_ψ, :neg_δ_ω, :μ_ψ, :μ_ω]],
            "E" => ["combined_inducer_thresh",                                                  "2-factor germination with inhibitor-dependent\ninduction threshold",
                        [:K_cC, :K_cI, :Pₛ, :Pₛ_cs, :k_I, :neg_δ_γ, :neg_δ_ψ, :neg_δ_ω, :μ_γ, :μ_ψ, :μ_ω]],
            "ABi" => ["feedback_inhibitor_inducer_perm_thresh",                                 "Inhibitor-dependent germination with\ninducer-dependent permeability/threshold",
                        [:K_cC, :Pₛ, :Pₛ_cs, :k_C, :s, :neg_δ_γ, :neg_δ_ψ, :μ_γ, :μ_ψ]],
            "AB" => ["feedback_combined_inducer_perm_thresh",                                   "2-factor germination with\ninducer-dependent permeability/threshold",
                        [:K_cC, :Pₛ, :Pₛ_cs, :k_C, :s, :neg_δ_γ, :neg_δ_ψ, :neg_δ_ω, :μ_γ, :μ_ψ, :μ_ω]],
            "ACi" => ["feedback_inhibitor_inducer_perm_inhibitor_signal",                       "Inhibitor-dependent germination with\ninducer-dep. permeability / inhibitor-dep. signal",
                        [:K_I, :K_cC, :Pₛ, :Pₛ_cs, :n, :s, :neg_δ_γ, :neg_δ_ψ, :μ_γ, :μ_ψ]],
            "ACc" => ["feedback_inducer_inducer_perm_inhibitor_signal",                         "Inducer-dependent germination with\ninducer-dep. permeability / inhibitor-dep. signal",
                        [:K_I, :K_cC, :Pₛ, :Pₛ_cs, :n, :s, :neg_δ_ψ, :neg_δ_ω, :μ_ψ, :μ_ω]],
            "AC" => ["feedback_combined_inducer_perm_inhibitor_signal",                         "2-factor germination with\ninducer-dep. permeability / inhibitor-dep. signal",
                        [:K_I, :K_cC, :Pₛ, :Pₛ_cs, :n, :s, :neg_δ_γ, :neg_δ_ψ, :neg_δ_ω, :μ_γ, :μ_ψ, :μ_ω]],
            "ADi" => ["feedback_inhibitor_inhibitor_inducer_perm",                              "Inhibitor-dependent germination with\ninhibitor+inducer-dependent permeability",
                        [:K_cC, :K_cI, :Pₛ, :Pₛ_cs, :b, :s, :neg_δ_γ, :neg_δ_ψ, :μ_γ, :μ_ψ]],
            "ADc" => ["feedback_inducer_inhibitor_inducer_perm",                                "Inducer-dependent germination with\ninhibitor+inducer-dependent permeability",
                        [:K_cC, :K_cI, :Pₛ, :Pₛ_cs, :b, :s, :neg_δ_ψ, :neg_δ_ω, :μ_ψ, :μ_ω]],
            "AD" => ["feedback_combined_inhibitor_inducer_perm",                                "2-factor germination with\ninhibitor+inducer-dependent permeability",
                        [:K_cC, :K_cI, :Pₛ, :Pₛ_cs, :b, :s, :neg_δ_γ, :neg_δ_ψ, :neg_δ_ω, :μ_γ, :μ_ψ, :μ_ω]],
            "AEc" => ["feedback_inducer_inhibitor_thresh_inducer_perm",                         "Inducer-dependent germination with\ninhibitor-dep. thresh., inducer-dep. perm.",
                        [:K_cC, :K_cI, :Pₛ, :Pₛ_cs, :k_I, :s, :neg_δ_ψ, :neg_δ_ω, :μ_ψ, :μ_ω]],
            "AE" => ["feedback_combined_inhibitor_thresh_inducer_perm",                         "2-factor germination with\ninhibitor-dep. thresh., inducer-dep. perm.",
                        [:K_cC, :K_cI, :Pₛ, :Pₛ_cs, :k_I, :s, :neg_δ_γ, :neg_δ_ψ, :neg_δ_ω, :μ_γ, :μ_ψ, :μ_ω]],
            "BCi" => ["inhibitor_thresh_inducer_signal",                                        "Inhibitor-dependent germination with\ninhibitor-dep. signal, inducer-dep. thresh",
                        [:K_I, :K_cC, :Pₛ, :Pₛ_cs, :k_C, :n, :neg_δ_γ, :neg_δ_ψ, :μ_γ, :μ_ψ]],
            "BC" => ["combined_inhibitor_thresh_inducer_signal",                                "2-factor germination with\ninhibitor-dep. signal, inducer-dep. thresh",
                        [:K_I, :K_cC, :Pₛ, :Pₛ_cs, :k_C, :n, :neg_δ_γ, :neg_δ_ψ, :neg_δ_ω, :μ_γ, :μ_ψ, :μ_ω]],
            "BDi" => ["feedback_inhibitor_inducer_thresh_inhibitor_perm",                       "Inhibitor-dependent germination with\ninducer-dep. thresh, inhibitor-dep. perm.",
                        [:K_cC, :K_cI, :Pₛ, :Pₛ_cs, :b, :k_C, :neg_δ_γ, :neg_δ_ψ, :μ_γ, :μ_ψ]],
            "BD" => ["feedback_combined_inducer_thresh_inhibitor_perm",                         "2-factor germination with\ninducer-dep. thresh., inhibitor-dep. perm.",
                        [:K_cC, :K_cI, :Pₛ, :Pₛ_cs, :b, :k_C, :neg_δ_γ, :neg_δ_ψ, :neg_δ_ω, :μ_γ, :μ_ψ, :μ_ω]],
            "BE" => ["combined_inhibitor_thresh_inducer_thresh",                                "2-factor germination with\ninducer-dep. thresh., inhibitor-dep. thresh.",
                        [:K_cC, :K_cI, :Pₛ, :Pₛ_cs, :k_C, :k_I, :neg_δ_γ, :neg_δ_ψ, :neg_δ_ω, :μ_γ, :μ_ψ, :μ_ω]],
            "CDc" => ["feedback_inducer_inhibitor_perm_signal",                                 "Inducer-dependent germination with\ninhibitor-dependent perm. and induction signal",
                        [:K_I, :K_cC, :K_cI, :Pₛ, :Pₛ_cs, :b, :neg_δ_ψ, :neg_δ_ω, :μ_ψ, :μ_ω, :n]],
            "CD" => ["feedback_combined_inhibitor_perm_signal",                                 "2-factor germination with\ninhibitor-dependent perm. and induction signal",
                        [:K_I, :K_cC, :K_cI, :Pₛ, :Pₛ_cs, :b, :neg_δ_γ, :neg_δ_ψ, :neg_δ_ω, :μ_γ, :μ_ψ, :μ_ω, :n]],
            "CEc" => ["inducer",                                                                "Inhibitor-dependent induction threshold and signal\n",
                        [:K_I, :K_cC, :K_cI, :Pₛ, :Pₛ_cs, :k_I, :n, :neg_δ_ψ, :neg_δ_ω, :μ_ψ, :μ_ω]],
            "CE" => ["combined_inducer",                                                        "Combined model with inhibitor-dependent\ninduction threshold and signal",
                        [:K_I, :K_cC, :K_cI, :Pₛ, :Pₛ_cs, :k_I, :n, :neg_δ_γ, :neg_δ_ψ, :neg_δ_ω, :μ_γ, :μ_ψ, :μ_ω]],
            "DEc" => ["feedback_inducer_inhibitor_perm_thresh",                                 "Inducer-dependent germination with\ninhibitor-dependent perm. and induction thresh.",
                        [:K_cC, :K_cI, :Pₛ, :Pₛ_cs, :b, :k_I, :neg_δ_ψ, :neg_δ_ω, :μ_ψ, :μ_ω]],
            "DE" => ["feedback_combined_inhibitor_perm_thresh",                                 "2-factor germination with\ninhibitor-dependent perm. and induction thresh.",
                        [:K_cC, :K_cI, :Pₛ, :Pₛ_cs, :b, :k_I, :neg_δ_γ, :neg_δ_ψ, :neg_δ_ω, :μ_γ, :μ_ψ, :μ_ω]],
            "ABCi" => ["feedback_inhibitor_inducer_perm_thresh_inhibitor_signal",               "Inhibitor-dependent germination with\ninducer-dep. perm./thresh., inhibitor-dep. signal",
                        [:K_I, :K_cC, :Pₛ, :Pₛ_cs, :k_C, :n, :s, :neg_δ_γ, :neg_δ_ψ, :μ_γ, :μ_ψ]],
            "ABC" => ["feedback_combined_inducer_perm_thresh_inhibitor_signal",                 "2-factor germination with\ninducer-dep. perm./thresh., inhibitor-dep. signal",
                        [:K_I, :K_cC, :Pₛ, :Pₛ_cs, :k_C, :n, :s, :neg_δ_γ, :neg_δ_ψ, :neg_δ_ω, :μ_γ, :μ_ψ, :μ_ω]],
            "ABDi" => ["feedback_inhibitor_inducer_perm_thresh_inhibitor_perm",                 "Inhibitor-dependent germination with\ninducer-dep. perm./thresh., inhibitor-dep. perm.",
                        [:K_cC, :K_cI, :Pₛ, :Pₛ_cs, :b, :k_C, :s, :neg_δ_γ, :neg_δ_ψ, :μ_γ, :μ_ψ]],
            "ABD" => ["feedback_combined_inducer_perm_thresh_inhibitor_perm",                   "2-factor germination with\ninducer-dep. perm./thresh., inhibitor-dep. perm.",
                        [:K_cC, :K_cI, :Pₛ, :Pₛ_cs, :b, :k_C, :s, :neg_δ_γ, :neg_δ_ψ, :neg_δ_ω, :μ_γ, :μ_ψ, :μ_ω]],
            "ABE" => ["feedback_combined_inducer_perm_thresh_inhibitor_thresh",                 "2-factor germination with\ninducer-dep. pern./thresh., inhibitor-dep. thresh.",
                        [:K_cC, :K_cI, :Pₛ, :Pₛ_cs, :k_C, :k_I, :s, :neg_δ_γ, :neg_δ_ψ, :neg_δ_ω, :μ_γ, :μ_ψ, :μ_ω]],
            "ACDi" => ["feedback_inhibitor_inhibitor_inducer_perm_inhibitor_signal",            "Inhibitor-dependent germination with\ninhibitor/inducer-dep. perm, inhibitor-dep. signal",
                        [:K_I, :K_cC, :K_cI, :Pₛ, :Pₛ_cs, :b, :n, :s, :neg_δ_γ, :neg_δ_ψ, :μ_γ, :μ_ψ]],
            "ACDc" => ["feedback_inducer_inhibitor_inducer_perm_inhibitor_signal",              "Inducer-dependent germination with\ninhibitor/inducer-dep. perm, inhibitor-dep. signal",
                        [:K_I, :K_cC, :K_cI, :Pₛ, :Pₛ_cs, :b, :n, :s, :neg_δ_ψ, :neg_δ_ω, :μ_ψ, :μ_ω]],
            "ACD" => ["feedback_combined_inhibitor_inducer_perm_inhibitor_signal",              "2-factor germination with\ninhibitor/inducer-dep. perm, inhibitor-dep. signal",
                        [:K_I, :K_cC, :K_cI, :Pₛ, :Pₛ_cs, :b, :n, :s, :neg_δ_γ, :neg_δ_ψ, :neg_δ_ω, :μ_γ, :μ_ψ, :μ_ω]],
            "ACEc" => ["feedback_inducer_inducer_perm_inhibitor_thresh_signal",                 "Inducer-dependent germination with\ninducer-dep. perm., inhbitor-dep. thresh./signal",
                        [:K_I, :K_cC, :K_cI, :Pₛ, :Pₛ_cs, :k_I, :n, :s, :neg_δ_ψ, :neg_δ_ω, :μ_ψ, :μ_ω]],
            "ACE" => ["feedback_combined_inducer_perm_inhibitor_thresh_signal",                 "2-factor germination with\ninducer-dep. perm., inhbitor-dep. thresh./signal",
                        [:K_I, :K_cC, :K_cI, :Pₛ, :Pₛ_cs, :k_I, :n, :s, :neg_δ_γ, :neg_δ_ψ, :neg_δ_ω, :μ_γ, :μ_ψ, :μ_ω]],
            "ADEc" => ["feedback_inducer_inhibitor_inducer_perm_inhibitor_thresh",              "Inhibitor-dependent germination with\ninhibitor/inducer-dep. perm, inhibitor-dep thresh.",
                        [:K_cC, :K_cI, :Pₛ, :Pₛ_cs, :b, :k_I, :s, :neg_δ_γ, :neg_δ_ψ, :μ_γ, :μ_ψ]],
            "ADE" => ["feedback_combined_inhibitor_inducer_perm_inhibitor_thresh",              "2-factor germination with\ninhibitor/inducer-dep. perm, inhibitor-dep thresh.",
                        [:K_cC, :K_cI, :Pₛ, :Pₛ_cs, :b, :k_I, :s, :neg_δ_γ, :neg_δ_ψ, :neg_δ_ω, :μ_γ, :μ_ψ, :μ_ω]],
            "BCDi" => ["feedback_inhibitor_inducer_thresh_inhibitor_perm_signal",               "Inhibitor-dependent germination with\ninducer-dep. thresh., inhibitor-dep. perm./signal",
                        [:K_I, :K_cC, :K_cI, :Pₛ, :Pₛ_cs, :b, :k_C, :n, :neg_δ_γ, :neg_δ_ψ, :μ_γ, :μ_ψ]],
            "BCD" => ["feedback_combined_inducer_thresh_inhibitor_perm_signal",                 "2-factor germination with\ninducer-dep. thresh., inhibitor-dep. perm./signal",
                        [:K_I, :K_cC, :K_cI, :Pₛ, :Pₛ_cs, :b, :k_C, :n, :neg_δ_γ, :neg_δ_ψ, :neg_δ_ω, :μ_γ, :μ_ψ, :μ_ω]],
            "BCE" => ["combined_inhibitor_thresh_signal_inducer_thresh",                        "2-factor germination with\ninhibitor-dep. thresh./signal, inducer-dep. thresh",
                        [:K_I, :K_cC, :K_cI, :Pₛ, :Pₛ_cs, :k_C, :k_I, :n, :neg_δ_γ, :neg_δ_ψ, :neg_δ_ω, :μ_γ, :μ_ψ, :μ_ω]],
            "BDE" => ["feedback_combined_inhibitor_perm_thresh_inducer_thresh",                 "2-factor germination with\ninhibitor-dep. perm./thresh., inducer-dep. thresh.",
                        [:K_cC, :K_cI, :Pₛ, :Pₛ_cs, :b, :k_C, :k_I, :neg_δ_γ, :neg_δ_ψ, :neg_δ_ω, :μ_γ, :μ_ψ, :μ_ω]],
            "CDEc" => ["feedback_inducer_inhibitor_perm_thresh_signal",                         "Inducer-dependent germination with\ninhibitor-dep. perm./thresh./signal",
                        [:K_I, :K_cC, :K_cI, :Pₛ, :Pₛ_cs, :b, :k_I, :n, :neg_δ_ψ, :neg_δ_ω, :μ_ψ, :μ_ω]],
            "CDE" => ["feedback_combined_inhibitor_perm_thresh_signal",                         "2-factor germination with\ninhibitor-dep. perm./thresh./signal",
                        [:K_I, :K_cC, :K_cI, :Pₛ, :Pₛ_cs, :b, :k_I, :n, :neg_δ_γ, :neg_δ_ψ, :neg_δ_ω, :μ_γ, :μ_ψ, :μ_ω]],
            "ABCDi" => ["feedback_inhibitor_inducer_perm_thresh_inhibitor_perm_signal",         "Inhibitor-dependent germination with\ninducer-dep. perm./thresh., inhibitor-dep. perm./signal",
                        [:K_I, :K_cC, :K_cI, :Pₛ, :Pₛ_cs, :b, :k_C, :n, :s, :neg_δ_γ, :neg_δ_ψ, :μ_γ, :μ_ψ]],
            "ABCD" => ["feedback_combined_inducer_perm_thresh_inhibitor_perm_signal",           "2-factor germination with\ninducer-dep. perm./thresh., inhibitor-dep. perm./signal",
                        [:K_I, :K_cC, :K_cI, :Pₛ, :Pₛ_cs, :b, :k_C, :n, :s, :neg_δ_γ, :neg_δ_ψ, :neg_δ_ω, :μ_γ, :μ_ψ, :μ_ω]],
            "ABCE" => ["feedback_combined_inhibitor_thresh_signal_inducer_perm_thresh",         "2-factor germination with inhibitor-dep.\nthresh./signal, inducer-dep. perm./thresh.",
                        [:K_I, :K_cC, :K_cI, :Pₛ, :Pₛ_cs, :k_C, :k_I, :n, :s, :neg_δ_γ, :neg_δ_ψ, :neg_δ_ω, :μ_γ, :μ_ψ, :μ_ω]],
            "ABDE" => ["feedback_combined_inhibitor_perm_thresh_inducer_perm_thresh",           "2-factor germination with inhibitor/inducer-dep. perm,\ninhibitor/inducer-dep. thresholds",
                        [:K_cC, :K_cI, :Pₛ, :Pₛ_cs, :b, :k_C, :k_I, :s, :neg_δ_γ, :neg_δ_ψ, :neg_δ_ω, :μ_γ, :μ_ψ, :μ_ω]],
            "ACDEc" => ["feedback_inducer_inhibitor_perm_thresh_signal_inducer_perm",           "Inducer-dependent germination with inhibitor/inducer-dep.\nperm., inhibitor-dep. thresh./signal",
                        [:K_I, :K_cC, :K_cI, :Pₛ, :Pₛ_cs, :b, :k_I, :n, :s, :neg_δ_ψ, :neg_δ_ω, :μ_ψ, :μ_ω]],
            "ACDE" => ["feedback_combined_inhibitor_perm_thresh_signal_inducer_perm",            "2-factor germination with inhibitor/inducer-dep. perm.,\ninhibitor-dep. thresh./signal",
                        [:K_I, :K_cC, :K_cI, :Pₛ, :Pₛ_cs, :b, :k_I, :n, :s, :neg_δ_γ, :neg_δ_ψ, :neg_δ_ω, :μ_γ, :μ_ψ, :μ_ω]],
            "BCDE" => ["feedback_combined_inhibitor_perm_thresh_signal_inducer_thresh",         "2-factor germination with inhibitor/inducer-dep. thresh.,\ninhibitor-dep. perm./signal",
                        [:K_I, :K_cC, :K_cI, :Pₛ, :Pₛ_cs, :b, :k_C, :k_I, :n, :neg_δ_γ, :neg_δ_ψ, :neg_δ_ω, :μ_γ, :μ_ψ, :μ_ω]],
            "ABCDE" => ["feedback_combined_inhibitor_perm_thresh_signal_inducer_perm_thresh",   "2-factor germination with inhibitor/inducer-dep.\nperm./thresh., inhibitor-dep. signal",
                        [:K_I, :K_cC, :K_cI, :Pₛ, :Pₛ_cs, :b, :k_C, :k_I, :n, :s, :neg_δ_γ, :neg_δ_ψ, :neg_δ_ω, :μ_γ, :μ_ψ, :μ_ω]] 
        )

        combination_IDs = sort(collect(keys(models)))
        aliases = [models[id][1] for id in combination_IDs]
        descriptions = [models[id][2] for id in combination_IDs]
        param_key_sets = [models[id][3] for id in combination_IDs]

        return aliases, combination_IDs, descriptions, param_key_sets
    end

    function clamp_inplace!(arr, eps=1e-12)
        @inbounds for i in eachindex(arr)
            if arr[i] < eps
                arr[i] = eps
            end
        end
        return arr
    end
    
    """
    Compute the concentration of carbon source in the cell wall.
    inputs:
        c_in (Float) - the initial concentration at the spore
        c_out (Float) - the initial external concentration
        t (Float) - time
        Pₛ (Float) - the hydrophobin layer permeation constant
        A (Float) - the surface area of the spore
        V_cw (Float) - the volume of the polysaccharide layer pores
    """
    function inducer_concentration(c_out, t, Pₛ, A, V_cw)
        τ = V_cw ./ (A * Pₛ)
        c = c_out .* (1 .- exp.(-t / τ))
        return c
    end

    """
    Wrapper function for the germination response
    that preprocesses parameters
    inputs:
        model_type (String): model type to fit
        times (Vector{Float64}): time points to compute the germination response (in seconds)
        ρₛ (Float) - spore density in spores/μm^3
        n_nodes (Int) - number of Gauss-Hermite nodes to use
        prms (Dict) - additional parameters for the germination response function
    output:
        p_out (Vector{Float64}) - germination fractions
    """
    function compute_gresp_xform_params(model_type, times, ρₛ, prms_raw, def_params)

        prms = Dict()
        for (key, val) in prms_raw
            if startswith(string(key), "neg_δ_")
                suffix = string(key)[end]
                prms[Symbol("σ_" * suffix)] = abs.(prms_raw[Symbol("μ_" * suffix)]) .* clamp.(exp.(-prms_raw[key]), 1e-12, 1e6)
            else
                prms[key] = prms_raw[key]
            end
        end
        # println("Input parameters: $prms")
        prms = merge(prms, def_params)
        return compute_germination_response(model_type, times, ρₛ, prms)

    end

    """
    Generic wrapper function for computing the germination response.
    inputs:
        model_type (String): model type to fit
        times (Vector{Float64}): time points to compute the germination response (in seconds)
        ρₛ (Float) - spore density in spores/μm^3
        n_nodes (Int) - number of Gauss-Hermite nodes to use
        prms (Dict) - additional parameters for the germination response function
        n_nodes (Int) - number of Gauss-Hermite nodes to use
        debug (Bool) - whether to print additional debugging messages
    output:
        p_out (Vector{Float64}) - germination fractions
    """
    function compute_germination_response(model_type, times, ρₛ, prms; n_nodes=nothing, debug=false)

        @argcheck model_type in load_model_collection()[1]

        # Empty global parameters buffer
        global PARAMS_BUFFER = []

        # Determine number of nodes depending on the integral dimension (if not specified)
        if isnothing(n_nodes)
            if model_type in ["test", "independent", "inhibitor", "inhibitor_thresh", "inhibitor_perm",
                                "combined_inhibitor", "combined_inhibitor_thresh", "combined_inhibitor_perm"]
                n_nodes = 36 # 2D integral
            elseif model_type in ["inducer", "inducer_thresh", "inducer_signal", 
                                "combined_inducer", "combined_inducer_thresh", "combined_inducer_signal", "special_independent",
                                "inhibitor_thresh_inducer_signal", "combined_inhibitor_thresh_inducer_signal", "combined_inhibitor_thresh_inducer_thresh", "combined_inhibitor_thresh_signal_inducer_thresh"]
                n_nodes = 10 # 3D integral
            elseif model_type in ["special_inducer", "special_combined", "special_thresh", "special_signal"]
                n_nodes = 6 # 4D integral
            elseif startswith(model_type, "feedback")
                n_nodes = 1024
            end
        end

        gh_integral = false
        if split(model_type, "_")[1] == "feedback"
            if (haskey(prms, :μ_γ) && haskey(prms, :μ_ω))
                sample_dim = 5
            else
                sample_dim = 4
            end
            sobol_pts = QuasiMonteCarlo.sample(n_nodes, sample_dim, SobolSample())
            # sobol_pts[1,:] for ξ
            # sobol_pts[2,:] for κ
            # sobol_pts[3,:] for ψ
            # sobol_pts[4,:] for γ
            # sobol_pts[5,:] for ω
        else
            gh_integral = true

            # Gauss-Hermite nodes and weights
            ghnodes, ghweights = gausshermite(n_nodes)
            u = √2 .* ghnodes
            hw = ghweights ./ √π
        end

        # Unpack means and stds and weight samples
        μ_ξ = prms[:μ_ξ]
        σ_ξ = prms[:σ_ξ]
        μ_ξ_log = log(μ_ξ^2 / sqrt(σ_ξ^2 + μ_ξ^2))
        σ_ξ_log = sqrt(log(σ_ξ^2 / μ_ξ^2 + 1))
        if gh_integral ξ = exp.(μ_ξ_log .+ σ_ξ_log .* u) end

        if haskey(prms, :μ_κ)
            μ_κ = prms[:μ_κ]
            σ_κ = prms[:σ_κ]
            μ_κ_log = log(μ_κ^2 / sqrt(σ_κ^2 + μ_κ^2))
            σ_κ_log = sqrt(log(σ_κ^2 / μ_κ^2 + 1))
            if gh_integral
                κ = exp.(μ_κ_log .+ σ_κ_log .* u)

                ξ2, κ2 = meshgrid(ξ, κ)
            end
        end

        # Weight tensors
        if model_type in ["independent", "inhibitor", "inhibitor_thresh", "inhibitor_perm",
                            "combined_inhibitor", "combined_inhibitor_thresh", "combined_inhibitor_perm"]
            W = hw * hw'
        elseif model_type in ["inducer", "inducer_thresh", "inducer_signal",
                            "combined_inducer", "combined_inducer_thresh", "combined_inducer_signal", "special_independent",
                            "inhibitor_thresh_inducer_signal", "combined_inhibitor_thresh_inducer_signal", "combined_inhibitor_thresh_inducer_thresh", "combined_inhibitor_thresh_signal_inducer_thresh"]
            W3 = reshape(hw, n_nodes,1,1) .* reshape(hw, 1,n_nodes,1) .* reshape(hw, 1,1,n_nodes)
        elseif model_type in ["special_inducer", "special_combined", "special_thresh", "special_signal"]
            W4 = reshape(hw, n_nodes,1,1,1) .* reshape(hw, 1,n_nodes,1,1) .* reshape(hw, 1,1,n_nodes,1) .* reshape(hw, 1,1,1,n_nodes)
        elseif model_type == "test"
            W = hw
        end

        # Construct distributions and geometric samples
        if !gh_integral
            dist_ξ = LogNormal(μ_ξ_log, σ_ξ_log)
            dist_κ = LogNormal(μ_κ_log, σ_κ_log)

            samples_ξ = clamp_inplace!(quantile(dist_ξ, sobol_pts[1,:]))
            samples_κ = clamp_inplace!(quantile(dist_κ, sobol_pts[2,:]))

            samples_AV = compute_spore_area_and_volume_from_dia.(2 .* samples_ξ)
            samples_A, samples_Vₛ = (getindex.(samples_AV, 1), getindex.(samples_AV, 2)) # (map(x -> x[1], samples_AV), map(x -> x[2], samples_AV))
            samples_V_out = 1.0/ρₛ .- samples_Vₛ
            samples_V_ps = compute_ps_layer_volume.(samples_ξ, prms[:d_hp], samples_κ)

            geom_samples = [samples_A, samples_Vₛ, samples_V_out, samples_V_ps]
        end

        # Compute the germination response
        if model_type == "independent" # 0
            @note_param germ_response = [gresp_independent_factors_gh(u, W, t, ρₛ, prms[:c₀_cs], prms[:d_hp], ξ2, κ2, prms[:Pₛ], prms[:Pₛ_cs], prms[:K_cC], prms[:μ_γ], prms[:σ_γ], prms[:μ_ω], prms[:σ_ω]) for t in times]
            
        elseif model_type == "inhibitor_thresh" # B
            @note_param germ_response = [gresp_inducer_dep_inhibitor_thresh_gh(u, W, t, ρₛ, prms[:c₀_cs], prms[:d_hp], ξ2, κ2, prms[:Pₛ], prms[:Pₛ_cs], prms[:k_C], prms[:K_cC], prms[:μ_γ], prms[:σ_γ]) for t in times]
            
        elseif model_type == "inducer" # CE
            @note_param germ_response = [gresp_inhibitor_dep_inducer_gh(u, W3, t, ρₛ, prms[:c₀_cs], prms[:d_hp], ξ2, κ2, prms[:Pₛ], prms[:Pₛ_cs], prms[:k_I], prms[:K_cI], prms[:K_cC], prms[:K_I], prms[:n], prms[:μ_ω], prms[:σ_ω], prms[:μ_ψ], prms[:σ_ψ]) for t in times]

        elseif model_type == "inducer_thresh" # E
            @note_param germ_response = [gresp_inhibitor_dep_inducer_thresh_gh(u, W3, t, ρₛ, prms[:c₀_cs], prms[:d_hp], ξ2, κ2, prms[:Pₛ], prms[:Pₛ_cs], prms[:K_cI], prms[:K_cC], prms[:k_I], prms[:μ_ω], prms[:σ_ω], prms[:μ_ψ], prms[:σ_ψ]) for t in times]
            
        elseif model_type == "inducer_signal" # C
            @note_param germ_response = [gresp_inhibitor_dep_inducer_signal_gh(u, W3, t, ρₛ, prms[:c₀_cs], prms[:d_hp], ξ2, κ2, prms[:Pₛ], prms[:Pₛ_cs], prms[:K_cC], prms[:K_I], prms[:n], prms[:μ_ω], prms[:σ_ω], prms[:μ_ψ], prms[:σ_ψ]) for t in times]
            
        elseif model_type == "combined_inhibitor_thresh" # B
            @note_param germ_response = [gresp_inducer_dep_inhibitor_thresh_2_factor_gh(u, W, t, ρₛ, prms[:c₀_cs], prms[:d_hp], ξ2, κ2, prms[:Pₛ], prms[:Pₛ_cs], prms[:k_C], prms[:K_cC], prms[:μ_γ], prms[:σ_γ], prms[:μ_ω], prms[:σ_ω]) for t in times]
            
        elseif model_type == "combined_inducer" # CE
            @note_param germ_response = [gresp_inhibitor_dep_inducer_2_factor_gh(u, W3, t, ρₛ, prms[:c₀_cs], prms[:d_hp], ξ2, κ2, prms[:Pₛ], prms[:Pₛ_cs], prms[:K_cI], prms[:K_cC], prms[:K_I], prms[:n], prms[:k_I], prms[:μ_γ], prms[:σ_γ], prms[:μ_ω], prms[:σ_ω], prms[:μ_ψ], prms[:σ_ψ]) for t in times]
            
        elseif model_type == "combined_inducer_thresh" # E
            @note_param germ_response = [gresp_inhibitor_dep_inducer_thresh_2_factor_gh(u, W3, t, ρₛ, prms[:c₀_cs], prms[:d_hp], ξ2, κ2, prms[:Pₛ], prms[:Pₛ_cs], prms[:K_cI], prms[:K_cC], prms[:k_I], prms[:μ_γ], prms[:σ_γ], prms[:μ_ω], prms[:σ_ω], prms[:μ_ψ], prms[:σ_ψ]) for t in times]
            
        elseif model_type == "combined_inducer_signal" # C
            @note_param germ_response = [gresp_inhibitor_dep_inducer_signal_2_factor_gh(u, W3, t, ρₛ, prms[:c₀_cs], prms[:d_hp], ξ2, κ2, prms[:Pₛ], prms[:Pₛ_cs], prms[:K_cC], prms[:K_I], prms[:n], prms[:μ_γ], prms[:σ_γ], prms[:μ_ω], prms[:σ_ω], prms[:μ_ψ], prms[:σ_ψ]) for t in times]
            
        elseif model_type == "feedback_inhibitor_inducer_perm" # A
            ode_func = ode_inducer_dependent_perm!
            thresh_crit = thresh_criterion_inhibitor
            @note_param f_maxs = [prms[:s]]
            @note_param K_fs = [nothing, prms[:K_cC]]
            @note_param thresh_means = [prms[:μ_γ]]
            @note_param thresh_sds = [prms[:σ_γ]]
            @note_param germ_response = gresp_feedback(ode_func, thresh_crit, sobol_pts, times, geom_samples, prms[:c₀_cs], f_maxs, prms[:Pₛ], prms[:Pₛ_cs], K_fs, prms[:μ_ψ], prms[:σ_ψ], thresh_means, thresh_sds)
            
        elseif model_type == "feedback_combined_inducer_perm" # A
            ode_func = ode_inducer_dependent_perm!
            thresh_crit = thresh_criterion_combined
            @note_param f_maxs = [prms[:s]]
            @note_param K_fs = [nothing, prms[:K_cC]]
            @note_param thresh_means = [prms[:μ_γ], prms[:μ_ω]]
            @note_param thresh_sds = [prms[:σ_γ], prms[:σ_ω]]
            @note_param germ_response = gresp_feedback(ode_func, thresh_crit, sobol_pts, times, geom_samples, prms[:c₀_cs], f_maxs, prms[:Pₛ], prms[:Pₛ_cs], K_fs, prms[:μ_ψ], prms[:σ_ψ], thresh_means, thresh_sds)
            
        elseif model_type == "feedback_inducer_inhibitor_perm" # D
            ode_func = ode_inhibitor_dependent_perm!
            thresh_crit = thresh_criterion_inducer
            @note_param f_maxs = [prms[:b]]
            @note_param K_fs = [prms[:K_cI], prms[:K_cC]]
            @note_param thresh_means = [prms[:μ_ω]]
            @note_param thresh_sds = [prms[:σ_ω]]
            @note_param germ_response = gresp_feedback(ode_func, thresh_crit, sobol_pts, times, geom_samples, prms[:c₀_cs], f_maxs, prms[:Pₛ], prms[:Pₛ_cs], K_fs, prms[:μ_ψ], prms[:σ_ψ], thresh_means, thresh_sds)
           
        elseif model_type == "feedback_combined_inhibitor_perm" # D
            ode_func = ode_inhibitor_dependent_perm!
            thresh_crit = thresh_criterion_combined
            @note_param f_maxs = [prms[:b]]
            @note_param K_fs = [prms[:K_cI], prms[:K_cC]]
            @note_param thresh_means = [prms[:μ_γ], prms[:μ_ω]]
            @note_param thresh_sds = [prms[:σ_γ], prms[:σ_ω]]
            @note_param germ_response = gresp_feedback(ode_func, thresh_crit, sobol_pts, times, geom_samples, prms[:c₀_cs], f_maxs, prms[:Pₛ], prms[:Pₛ_cs], K_fs, prms[:μ_ψ], prms[:σ_ψ], thresh_means, thresh_sds)
           
        elseif model_type == "feedback_inhibitor_inducer_perm_thresh" # AB
            ode_func = ode_inducer_dependent_perm!
            thresh_crit = thresh_criterion_inhibitor_shift
            @note_param f_maxs = [prms[:s]]
            @note_param K_fs = [nothing, prms[:K_cC]]
            @note_param thresh_means = [prms[:μ_γ]]
            @note_param thresh_sds = [prms[:σ_γ]]
            @note_param germ_response = gresp_feedback(ode_func, thresh_crit, sobol_pts, times, geom_samples, prms[:c₀_cs], f_maxs, prms[:Pₛ], prms[:Pₛ_cs], K_fs, prms[:μ_ψ], prms[:σ_ψ], thresh_means, thresh_sds; ks=[prms[:k_C]])
           
        elseif model_type == "feedback_combined_inducer_perm_thresh" # AB
            ode_func = ode_inducer_dependent_perm!
            thresh_crit = thresh_criterion_combined_inhibitor_shift
            @note_param f_maxs = [prms[:s]]
            @note_param K_fs = [nothing, prms[:K_cC]]
            @note_param thresh_means = [prms[:μ_γ], prms[:μ_ω]]
            @note_param thresh_sds = [prms[:σ_γ], prms[:σ_ω]]
            @note_param germ_response = gresp_feedback(ode_func, thresh_crit, sobol_pts, times, geom_samples, prms[:c₀_cs], f_maxs, prms[:Pₛ], prms[:Pₛ_cs], K_fs, prms[:μ_ψ], prms[:σ_ψ], thresh_means, thresh_sds; ks=[prms[:k_C]])
           
        elseif model_type == "feedback_inhibitor_inducer_perm_inhibitor_signal" # AC
            ode_func = ode_inducer_dependent_perm_inhibitor_dependent_signal!
            thresh_crit = thresh_criterion_inhibitor
            @note_param f_maxs = [prms[:s]]
            @note_param K_fs = [nothing, prms[:K_cC], prms[:K_I]]
            @note_param thresh_means = [prms[:μ_γ]]
            @note_param thresh_sds = [prms[:σ_γ]]
            @note_param germ_response = gresp_feedback(ode_func, thresh_crit, sobol_pts, times, geom_samples, prms[:c₀_cs], f_maxs, prms[:Pₛ], prms[:Pₛ_cs], K_fs, prms[:μ_ψ], prms[:σ_ψ], thresh_means, thresh_sds; n=prms[:n])

        elseif model_type == "feedback_inducer_inducer_perm_inhibitor_signal" # AC
            ode_func = ode_inducer_dependent_perm_inhibitor_dependent_signal!
            thresh_crit = thresh_criterion_inducer
            @note_param f_maxs = [prms[:s]]
            @note_param K_fs = [nothing, prms[:K_cC], prms[:K_I]]
            @note_param thresh_means = [prms[:μ_ω]]
            @note_param thresh_sds = [prms[:σ_ω]]
            @note_param germ_response = gresp_feedback(ode_func, thresh_crit, sobol_pts, times, geom_samples, prms[:c₀_cs], f_maxs, prms[:Pₛ], prms[:Pₛ_cs], K_fs, prms[:μ_ψ], prms[:σ_ψ], thresh_means, thresh_sds; n=prms[:n])
        
        elseif model_type == "feedback_combined_inducer_perm_inhibitor_signal" # AC
            ode_func = ode_inducer_dependent_perm_inhibitor_dependent_signal!
            thresh_crit = thresh_criterion_combined
            @note_param f_maxs = [prms[:s]]
            @note_param K_fs = [nothing, prms[:K_cC], prms[:K_I]]
            @note_param thresh_means = [prms[:μ_γ], prms[:μ_ω]]
            @note_param thresh_sds = [prms[:σ_γ], prms[:σ_ω]]
            @note_param germ_response = gresp_feedback(ode_func, thresh_crit, sobol_pts, times, geom_samples, prms[:c₀_cs], f_maxs, prms[:Pₛ], prms[:Pₛ_cs], K_fs, prms[:μ_ψ], prms[:σ_ψ], thresh_means, thresh_sds; n=prms[:n])
        
        elseif model_type == "feedback_inhibitor_inhibitor_inducer_perm" # AD
            ode_func = ode_inducer_and_inhibitor_dependent_perm!
            thresh_crit = thresh_criterion_inhibitor
            @note_param f_maxs = [prms[:b], prms[:s]]
            @note_param K_fs = [prms[:K_cI], prms[:K_cC]]
            @note_param thresh_means = [prms[:μ_γ]]
            @note_param thresh_sds = [prms[:σ_γ]]
            @note_param germ_response = gresp_feedback(ode_func, thresh_crit, sobol_pts, times, geom_samples, prms[:c₀_cs], f_maxs, prms[:Pₛ], prms[:Pₛ_cs], K_fs, prms[:μ_ψ], prms[:σ_ψ], thresh_means, thresh_sds)
            
        elseif model_type == "feedback_inducer_inhibitor_inducer_perm" # AD
            ode_func = ode_inducer_and_inhibitor_dependent_perm!
            thresh_crit = thresh_criterion_inducer
            @note_param f_maxs = [prms[:b], prms[:s]]
            @note_param K_fs = [prms[:K_cI], prms[:K_cC]]
            @note_param thresh_means = [prms[:μ_ω]]
            @note_param thresh_sds = [prms[:σ_ω]]
            @note_param germ_response = gresp_feedback(ode_func, thresh_crit, sobol_pts, times, geom_samples, prms[:c₀_cs], f_maxs, prms[:Pₛ], prms[:Pₛ_cs], K_fs, prms[:μ_ψ], prms[:σ_ψ], thresh_means, thresh_sds)
            
        elseif model_type == "feedback_combined_inhibitor_inducer_perm" # AD
            ode_func = ode_inducer_and_inhibitor_dependent_perm!
            thresh_crit = thresh_criterion_combined
            @note_param f_maxs = [prms[:b], prms[:s]]
            @note_param K_fs = [prms[:K_cI], prms[:K_cC]]
            @note_param thresh_means = [prms[:μ_γ], prms[:μ_ω]]
            @note_param thresh_sds = [prms[:σ_γ], prms[:σ_ω]]
            @note_param germ_response = gresp_feedback(ode_func, thresh_crit, sobol_pts, times, geom_samples, prms[:c₀_cs], f_maxs, prms[:Pₛ], prms[:Pₛ_cs], K_fs, prms[:μ_ψ], prms[:σ_ψ], thresh_means, thresh_sds)
            
        elseif model_type == "feedback_inducer_inhibitor_thresh_inducer_perm" # AE
            ode_func = ode_inducer_dependent_perm!
            thresh_crit = thresh_criterion_inducer_shift
            @note_param f_maxs = [prms[:s]]
            @note_param K_fs = [prms[:K_cI], prms[:K_cC]]
            @note_param thresh_means = [prms[:μ_ω]]
            @note_param thresh_sds = [prms[:σ_ω]]
            @note_param germ_response = gresp_feedback(ode_func, thresh_crit, sobol_pts, times, geom_samples, prms[:c₀_cs], f_maxs, prms[:Pₛ], prms[:Pₛ_cs], K_fs, prms[:μ_ψ], prms[:σ_ψ], thresh_means, thresh_sds, ks=[prms[:k_I]])
            
        elseif model_type == "feedback_combined_inhibitor_thresh_inducer_perm" # AE
            ode_func = ode_inducer_dependent_perm!
            thresh_crit = thresh_criterion_combined_inducer_shift
            @note_param f_maxs = [prms[:s]]
            @note_param K_fs = [prms[:K_cI], prms[:K_cC]]
            @note_param thresh_means = [prms[:μ_γ], prms[:μ_ω]]
            @note_param thresh_sds = [prms[:σ_γ], prms[:σ_ω]]
            @note_param germ_response = gresp_feedback(ode_func, thresh_crit, sobol_pts, times, geom_samples, prms[:c₀_cs], f_maxs, prms[:Pₛ], prms[:Pₛ_cs], K_fs, prms[:μ_ψ], prms[:σ_ψ], thresh_means, thresh_sds, ks=[prms[:k_I]])
            
        elseif model_type == "inhibitor_thresh_inducer_signal" # BC
            @note_param germ_response = [gresp_inh_dep_ind_signal_ind_dep_inh_thresh_gh(u, W3, t, ρₛ, prms[:c₀_cs], prms[:d_hp], ξ2, κ2, prms[:Pₛ], prms[:Pₛ_cs], prms[:k_C], prms[:K_cC], prms[:K_I], prms[:n], prms[:μ_γ], prms[:σ_γ], prms[:μ_ψ], prms[:σ_ψ]) for t in times]
            
        elseif model_type == "combined_inhibitor_thresh_inducer_signal" # BC
            @note_param germ_response = [gresp_inh_dep_ind_signal_ind_dep_inh_thresh_2_factor_gh(u, W3, t, ρₛ, prms[:c₀_cs], prms[:d_hp], ξ2, κ2, prms[:Pₛ], prms[:Pₛ_cs], prms[:k_C], prms[:K_cC], prms[:K_I], prms[:n], prms[:μ_γ], prms[:σ_γ], prms[:μ_ω], prms[:σ_ω], prms[:μ_ψ], prms[:σ_ψ]) for t in times]
            
        elseif model_type == "feedback_inhibitor_inducer_thresh_inhibitor_perm" # BD
            ode_func = ode_inhibitor_dependent_perm!
            thresh_crit = thresh_criterion_inhibitor_shift
            @note_param f_maxs = [prms[:b]]
            @note_param K_fs = [prms[:K_cI], prms[:K_cC]]
            @note_param thresh_means = [prms[:μ_γ]]
            @note_param thresh_sds = [prms[:σ_γ]]
            @note_param germ_response = gresp_feedback(ode_func, thresh_crit, sobol_pts, times, geom_samples, prms[:c₀_cs], f_maxs, prms[:Pₛ], prms[:Pₛ_cs], K_fs, prms[:μ_ψ], prms[:σ_ψ], thresh_means, thresh_sds, ks=[prms[:k_C]])
           
        elseif model_type == "feedback_combined_inducer_thresh_inhibitor_perm" # BD
            ode_func = ode_inhibitor_dependent_perm!
            thresh_crit = thresh_criterion_combined_inhibitor_shift
            @note_param f_maxs = [prms[:b]]
            @note_param K_fs = [prms[:K_cI], prms[:K_cC]]
            @note_param thresh_means = [prms[:μ_γ], prms[:μ_ω]]
            @note_param thresh_sds = [prms[:σ_γ], prms[:σ_ω]]
            @note_param germ_response = gresp_feedback(ode_func, thresh_crit, sobol_pts, times, geom_samples, prms[:c₀_cs], f_maxs, prms[:Pₛ], prms[:Pₛ_cs], K_fs, prms[:μ_ψ], prms[:σ_ψ], thresh_means, thresh_sds, ks=[prms[:k_C]])
           
        elseif model_type == "combined_inhibitor_thresh_inducer_thresh" # BE
            @note_param germ_response = [gresp_inh_dep_ind_thresh_ind_dep_inh_thresh_2_factor_gh(u, W3, t, ρₛ, prms[:c₀_cs], prms[:d_hp], ξ2, κ2, prms[:Pₛ], prms[:Pₛ_cs], prms[:K_cI], prms[:K_cC], prms[:k_I], prms[:k_C], prms[:μ_γ], prms[:σ_γ], prms[:μ_ω], prms[:σ_ω], prms[:μ_ψ], prms[:σ_ψ]) for t in times]
            
        elseif model_type == "feedback_inducer_inhibitor_perm_signal" # CD
            ode_func = ode_inhibitor_dependent_perm!
            thresh_crit = thresh_criterion_inducer_signal
            @note_param f_maxs = [prms[:b]]
            @note_param K_fs = [prms[:K_cI], prms[:K_cC], prms[:K_I]]
            @note_param thresh_means = [prms[:μ_ω]]
            @note_param thresh_sds = [prms[:σ_ω]]
            @note_param germ_response = gresp_feedback(ode_func, thresh_crit, sobol_pts, times, geom_samples, prms[:c₀_cs], f_maxs, prms[:Pₛ], prms[:Pₛ_cs], K_fs, prms[:μ_ψ], prms[:σ_ψ], thresh_means, thresh_sds, n=prms[:n])
           
        elseif model_type == "feedback_combined_inhibitor_perm_signal" # CD
            ode_func = ode_inhibitor_dependent_perm!
            thresh_crit = thresh_criterion_combined_inducer_signal
            @note_param f_maxs = [prms[:b]]
            @note_param K_fs = [prms[:K_cI], prms[:K_cC], prms[:K_I]]
            @note_param thresh_means = [prms[:μ_γ], prms[:μ_ω]]
            @note_param thresh_sds = [prms[:σ_γ], prms[:σ_ω]]
            @note_param germ_response = gresp_feedback(ode_func, thresh_crit, sobol_pts, times, geom_samples, prms[:c₀_cs], f_maxs, prms[:Pₛ], prms[:Pₛ_cs], K_fs, prms[:μ_ψ], prms[:σ_ψ], thresh_means, thresh_sds, n=prms[:n])
           
        elseif model_type == "feedback_inducer_inhibitor_perm_thresh" # DE
            ode_func = ode_inhibitor_dependent_perm!
            thresh_crit = thresh_criterion_inducer_shift
            @note_param f_maxs = [prms[:b]]
            @note_param K_fs = [prms[:K_cI], prms[:K_cC]]
            @note_param thresh_means = [prms[:μ_ω]]
            @note_param thresh_sds = [prms[:σ_ω]]
            @note_param germ_response = gresp_feedback(ode_func, thresh_crit, sobol_pts, times, geom_samples, prms[:c₀_cs], f_maxs, prms[:Pₛ], prms[:Pₛ_cs], K_fs, prms[:μ_ψ], prms[:σ_ψ], thresh_means, thresh_sds, ks=[prms[:k_I]])
           
        elseif model_type == "feedback_combined_inhibitor_perm_thresh" # DE
            ode_func = ode_inhibitor_dependent_perm!
            thresh_crit = thresh_criterion_combined_inducer_shift
            @note_param f_maxs = [prms[:b]]
            @note_param K_fs = [prms[:K_cI], prms[:K_cC]]
            @note_param thresh_means = [prms[:μ_γ], prms[:μ_ω]]
            @note_param thresh_sds = [prms[:σ_γ], prms[:σ_ω]]
            @note_param germ_response = gresp_feedback(ode_func, thresh_crit, sobol_pts, times, geom_samples, prms[:c₀_cs], f_maxs, prms[:Pₛ], prms[:Pₛ_cs], K_fs, prms[:μ_ψ], prms[:σ_ψ], thresh_means, thresh_sds, ks=[prms[:k_I]])
           
        elseif model_type == "feedback_inhibitor_inducer_perm_thresh_inhibitor_signal" # ABC
            ode_func = ode_inducer_dependent_perm_inhibitor_dependent_signal!
            thresh_crit = thresh_criterion_inhibitor_shift
            @note_param f_maxs = [prms[:s]]
            @note_param K_fs = [nothing, prms[:K_cC], prms[:K_I]]
            @note_param thresh_means = [prms[:μ_γ]]
            @note_param thresh_sds = [prms[:σ_γ]]
            @note_param germ_response = gresp_feedback(ode_func, thresh_crit, sobol_pts, times, geom_samples, prms[:c₀_cs], f_maxs, prms[:Pₛ], prms[:Pₛ_cs], K_fs, prms[:μ_ψ], prms[:σ_ψ], thresh_means, thresh_sds; ks=[prms[:k_C]], n=prms[:n])
        
        elseif model_type == "feedback_combined_inducer_perm_thresh_inhibitor_signal" # ABC
            ode_func = ode_inducer_dependent_perm_inhibitor_dependent_signal!
            thresh_crit = thresh_criterion_combined_inhibitor_shift
            @note_param f_maxs = [prms[:s]]
            @note_param K_fs = [nothing, prms[:K_cC], prms[:K_I]]
            @note_param thresh_means = [prms[:μ_γ], prms[:μ_ω]]
            @note_param thresh_sds = [prms[:σ_γ], prms[:σ_ω]]
            @note_param germ_response = gresp_feedback(ode_func, thresh_crit, sobol_pts, times, geom_samples, prms[:c₀_cs], f_maxs, prms[:Pₛ], prms[:Pₛ_cs], K_fs, prms[:μ_ψ], prms[:σ_ψ], thresh_means, thresh_sds; ks=[prms[:k_C]], n=prms[:n])
        
        elseif model_type == "feedback_inhibitor_inducer_perm_thresh_inhibitor_perm" # ABD
            ode_func = ode_inducer_and_inhibitor_dependent_perm!
            thresh_crit = thresh_criterion_inhibitor_shift
            @note_param f_maxs = [prms[:b], prms[:s]]
            @note_param K_fs = [prms[:K_cI], prms[:K_cC]]
            @note_param thresh_means = [prms[:μ_γ]]
            @note_param thresh_sds = [prms[:σ_γ]]
            @note_param germ_response = gresp_feedback(ode_func, thresh_crit, sobol_pts, times, geom_samples, prms[:c₀_cs], f_maxs, prms[:Pₛ], prms[:Pₛ_cs], K_fs, prms[:μ_ψ], prms[:σ_ψ], thresh_means, thresh_sds; ks=[prms[:k_C]])
            
        elseif model_type == "feedback_combined_inducer_perm_thresh_inhibitor_perm" # ABD
            ode_func = ode_inducer_and_inhibitor_dependent_perm!
            thresh_crit = thresh_criterion_combined_inhibitor_shift
            @note_param f_maxs = [prms[:b], prms[:s]]
            @note_param K_fs = [prms[:K_cI], prms[:K_cC]]
            @note_param thresh_means = [prms[:μ_γ], prms[:μ_ω]]
            @note_param thresh_sds = [prms[:σ_γ], prms[:σ_ω]]
            @note_param germ_response = gresp_feedback(ode_func, thresh_crit, sobol_pts, times, geom_samples, prms[:c₀_cs], f_maxs, prms[:Pₛ], prms[:Pₛ_cs], K_fs, prms[:μ_ψ], prms[:σ_ψ], thresh_means, thresh_sds; ks=[prms[:k_C]])
            
        elseif model_type == "feedback_combined_inducer_perm_thresh_inhibitor_thresh" # ABE
            ode_func = ode_inducer_dependent_perm!
            thresh_crit = thresh_criterion_combined_shift
            @note_param f_maxs = [prms[:s]]
            @note_param K_fs = [prms[:K_cI], prms[:K_cC]]
            @note_param thresh_means = [prms[:μ_γ], prms[:μ_ω]]
            @note_param thresh_sds = [prms[:σ_γ], prms[:σ_ω]]
            @note_param germ_response = gresp_feedback(ode_func, thresh_crit, sobol_pts, times, geom_samples, prms[:c₀_cs], f_maxs, prms[:Pₛ], prms[:Pₛ_cs], K_fs, prms[:μ_ψ], prms[:σ_ψ], thresh_means, thresh_sds; ks=[prms[:k_I], prms[:k_C]])
            
        elseif model_type == "feedback_inhibitor_inhibitor_inducer_perm_inhibitor_signal" # ACD
            ode_func = ode_inducer_and_inhibitor_dependent_perm_inhibitor_dependent_signal!
            thresh_crit = thresh_criterion_inhibitor
            @note_param f_maxs = [prms[:b], prms[:s]]
            @note_param K_fs = [prms[:K_cI], prms[:K_cC], prms[:K_I]]
            @note_param thresh_means = [prms[:μ_γ]]
            @note_param thresh_sds = [prms[:σ_γ]]
            @note_param germ_response = gresp_feedback(ode_func, thresh_crit, sobol_pts, times, geom_samples, prms[:c₀_cs], f_maxs, prms[:Pₛ], prms[:Pₛ_cs], K_fs, prms[:μ_ψ], prms[:σ_ψ], thresh_means, thresh_sds; n=prms[:n])
            
        elseif model_type == "feedback_inducer_inhibitor_inducer_perm_inhibitor_signal" # ACD
            ode_func = ode_inducer_and_inhibitor_dependent_perm_inhibitor_dependent_signal!
            thresh_crit = thresh_criterion_inducer
            @note_param f_maxs = [prms[:b], prms[:s]]
            @note_param K_fs = [prms[:K_cI], prms[:K_cC], prms[:K_I]]
            @note_param thresh_means = [prms[:μ_ω]]
            @note_param thresh_sds = [prms[:σ_ω]]
            @note_param germ_response = gresp_feedback(ode_func, thresh_crit, sobol_pts, times, geom_samples, prms[:c₀_cs], f_maxs, prms[:Pₛ], prms[:Pₛ_cs], K_fs, prms[:μ_ψ], prms[:σ_ψ], thresh_means, thresh_sds; n=prms[:n])
            
        elseif model_type == "feedback_combined_inhibitor_inducer_perm_inhibitor_signal" # ACD
            ode_func = ode_inducer_and_inhibitor_dependent_perm_inhibitor_dependent_signal!
            thresh_crit = thresh_criterion_combined
            @note_param f_maxs = [prms[:b], prms[:s]]
            @note_param K_fs = [prms[:K_cI], prms[:K_cC], prms[:K_I]]
            @note_param thresh_means = [prms[:μ_γ], prms[:μ_ω]]
            @note_param thresh_sds = [prms[:σ_γ], prms[:σ_ω]]
            @note_param germ_response = gresp_feedback(ode_func, thresh_crit, sobol_pts, times, geom_samples, prms[:c₀_cs], f_maxs, prms[:Pₛ], prms[:Pₛ_cs], K_fs, prms[:μ_ψ], prms[:σ_ψ], thresh_means, thresh_sds; n=prms[:n])
            
        elseif model_type == "feedback_inducer_inducer_perm_inhibitor_thresh_signal" # ACE
            ode_func = ode_inducer_dependent_perm_inhibitor_dependent_signal!
            thresh_crit = thresh_criterion_inducer_signal_shift
            @note_param f_maxs = [prms[:s]]
            @note_param K_fs = [prms[:K_cI], prms[:K_cC], prms[:K_I]]
            @note_param thresh_means = [prms[:μ_ω]]
            @note_param thresh_sds = [prms[:σ_ω]]
            @note_param germ_response = gresp_feedback(ode_func, thresh_crit, sobol_pts, times, geom_samples, prms[:c₀_cs], f_maxs, prms[:Pₛ], prms[:Pₛ_cs], K_fs, prms[:μ_ψ], prms[:σ_ψ], thresh_means, thresh_sds; ks=[prms[:k_I]], n=prms[:n])
        
        elseif model_type == "feedback_combined_inducer_perm_inhibitor_thresh_signal" # ACE
            ode_func = ode_inducer_dependent_perm_inhibitor_dependent_signal!
            thresh_crit = thresh_criterion_combined_inducer_signal_shift
            @note_param f_maxs = [prms[:s]]
            @note_param K_fs = [prms[:K_cI], prms[:K_cC], prms[:K_I]]
            @note_param thresh_means = [prms[:μ_γ], prms[:μ_ω]]
            @note_param thresh_sds = [prms[:σ_γ], prms[:σ_ω]]
            @note_param germ_response = gresp_feedback(ode_func, thresh_crit, sobol_pts, times, geom_samples, prms[:c₀_cs], f_maxs, prms[:Pₛ], prms[:Pₛ_cs], K_fs, prms[:μ_ψ], prms[:σ_ψ], thresh_means, thresh_sds; ks=[prms[:k_I]], n=prms[:n])
        
        elseif model_type == "feedback_inducer_inhibitor_inducer_perm_inhibitor_thresh" # ADE
            ode_func = ode_inducer_and_inhibitor_dependent_perm!
            thresh_crit = thresh_criterion_inducer_shift
            @note_param f_maxs = [prms[:b], prms[:s]]
            @note_param K_fs = [prms[:K_cI], prms[:K_cC]]
            @note_param thresh_means = [prms[:μ_γ]]
            @note_param thresh_sds = [prms[:σ_γ]]
            @note_param germ_response = gresp_feedback(ode_func, thresh_crit, sobol_pts, times, geom_samples, prms[:c₀_cs], f_maxs, prms[:Pₛ], prms[:Pₛ_cs], K_fs, prms[:μ_ψ], prms[:σ_ψ], thresh_means, thresh_sds; ks=[prms[:k_I]])
            
        elseif model_type == "feedback_combined_inhibitor_inducer_perm_inhibitor_thresh" # ADE
            ode_func = ode_inducer_and_inhibitor_dependent_perm!
            thresh_crit = thresh_criterion_combined_inducer_shift
            @note_param f_maxs = [prms[:b], prms[:s]]
            @note_param K_fs = [prms[:K_cI], prms[:K_cC]]
            @note_param thresh_means = [prms[:μ_γ], prms[:μ_ω]]
            @note_param thresh_sds = [prms[:σ_γ], prms[:σ_ω]]
            @note_param germ_response = gresp_feedback(ode_func, thresh_crit, sobol_pts, times, geom_samples, prms[:c₀_cs], f_maxs, prms[:Pₛ], prms[:Pₛ_cs], K_fs, prms[:μ_ψ], prms[:σ_ψ], thresh_means, thresh_sds; ks=[prms[:k_I]])
            
         elseif model_type == "feedback_inhibitor_inducer_thresh_inhibitor_perm_signal" # BCD
            ode_func = ode_inhibitor_dependent_perm!
            thresh_crit = thresh_criterion_inhibitor_signal_shift
            @note_param f_maxs = [prms[:b]]
            @note_param K_fs = [prms[:K_cI], prms[:K_cC], prms[:K_I]]
            @note_param thresh_means = [prms[:μ_γ]]
            @note_param thresh_sds = [prms[:σ_γ]]
            @note_param germ_response = gresp_feedback(ode_func, thresh_crit, sobol_pts, times, geom_samples, prms[:c₀_cs], f_maxs, prms[:Pₛ], prms[:Pₛ_cs], K_fs, prms[:μ_ψ], prms[:σ_ψ], thresh_means, thresh_sds; ks=[prms[:k_C]], n=prms[:n])
           
        elseif model_type == "feedback_combined_inducer_thresh_inhibitor_perm_signal" # BCD
            ode_func = ode_inhibitor_dependent_perm!
            thresh_crit = thresh_criterion_combined_inhibitor_signal_shift
            @note_param f_maxs = [prms[:b]]
            @note_param K_fs = [prms[:K_cI], prms[:K_cC], prms[:K_I]]
            @note_param thresh_means = [prms[:μ_γ], prms[:μ_ω]]
            @note_param thresh_sds = [prms[:σ_γ], prms[:σ_ω]]
            @note_param germ_response = gresp_feedback(ode_func, thresh_crit, sobol_pts, times, geom_samples, prms[:c₀_cs], f_maxs, prms[:Pₛ], prms[:Pₛ_cs], K_fs, prms[:μ_ψ], prms[:σ_ψ], thresh_means, thresh_sds; ks=[prms[:k_C]], n=prms[:n])
           
        elseif model_type == "combined_inhibitor_thresh_signal_inducer_thresh" # BCE
            @note_param germ_response = [gresp_inh_dep_ind_thresh_signal_ind_dep_inh_thresh_2_factor_gh(u, W3, t, ρₛ, prms[:c₀_cs], prms[:d_hp], ξ2, κ2, prms[:Pₛ], prms[:Pₛ_cs], prms[:K_cI], prms[:K_cC], prms[:K_I], prms[:n], prms[:k_I], prms[:k_C], prms[:μ_γ], prms[:σ_γ], prms[:μ_ω], prms[:σ_ω], prms[:μ_ψ], prms[:σ_ψ]) for t in times]
            
        elseif model_type == "feedback_combined_inhibitor_perm_thresh_inducer_thresh" # BDE
            ode_func = ode_inhibitor_dependent_perm!
            thresh_crit = thresh_criterion_combined_shift
            @note_param f_maxs = [prms[:b]]
            @note_param K_fs = [prms[:K_cI], prms[:K_cC]]
            @note_param thresh_means = [prms[:μ_γ], prms[:μ_ω]]
            @note_param thresh_sds = [prms[:σ_γ], prms[:σ_ω]]
            @note_param germ_response = gresp_feedback(ode_func, thresh_crit, sobol_pts, times, geom_samples, prms[:c₀_cs], f_maxs, prms[:Pₛ], prms[:Pₛ_cs], K_fs, prms[:μ_ψ], prms[:σ_ψ], thresh_means, thresh_sds; ks=[prms[:k_I], prms[:k_C]])
           
        elseif model_type == "feedback_inducer_inhibitor_perm_thresh_signal" # CDE
            ode_func = ode_inhibitor_dependent_perm!
            thresh_crit = thresh_criterion_inducer_signal_shift
            @note_param f_maxs = [prms[:b]]
            @note_param K_fs = [prms[:K_cI], prms[:K_cC], prms[:K_I]]
            @note_param thresh_means = [prms[:μ_ω]]
            @note_param thresh_sds = [prms[:σ_ω]]
            @note_param germ_response = gresp_feedback(ode_func, thresh_crit, sobol_pts, times, geom_samples, prms[:c₀_cs], f_maxs, prms[:Pₛ], prms[:Pₛ_cs], K_fs, prms[:μ_ψ], prms[:σ_ψ], thresh_means, thresh_sds; ks=[prms[:k_I]], n=prms[:n])
           
        elseif model_type == "feedback_combined_inhibitor_perm_thresh_signal" # CDE
            ode_func = ode_inhibitor_dependent_perm!
            thresh_crit = thresh_criterion_combined_inducer_signal_shift
            @note_param f_maxs = [prms[:b]]
            @note_param K_fs = [prms[:K_cI], prms[:K_cC], prms[:K_I]]
            @note_param thresh_means = [prms[:μ_γ], prms[:μ_ω]]
            @note_param thresh_sds = [prms[:σ_γ], prms[:σ_ω]]
            @note_param germ_response = gresp_feedback(ode_func, thresh_crit, sobol_pts, times, geom_samples, prms[:c₀_cs], f_maxs, prms[:Pₛ], prms[:Pₛ_cs], K_fs, prms[:μ_ψ], prms[:σ_ψ], thresh_means, thresh_sds; ks=[prms[:k_I]], n=prms[:n])
           
        elseif model_type == "feedback_inhibitor_inducer_perm_thresh_inhibitor_perm_signal" # ABCD
            ode_func = ode_inducer_and_inhibitor_dependent_perm_inhibitor_dependent_signal!
            thresh_crit = thresh_criterion_inhibitor_shift
            @note_param f_maxs = [prms[:b], prms[:s]]
            @note_param K_fs = [prms[:K_cI], prms[:K_cC], prms[:K_I]]
            @note_param thresh_means = [prms[:μ_γ]]
            @note_param thresh_sds = [prms[:σ_γ]]
            @note_param germ_response = gresp_feedback(ode_func, thresh_crit, sobol_pts, times, geom_samples, prms[:c₀_cs], f_maxs, prms[:Pₛ], prms[:Pₛ_cs], K_fs, prms[:μ_ψ], prms[:σ_ψ], thresh_means, thresh_sds; ks=[prms[:k_C]], n=prms[:n])
            
        elseif model_type == "feedback_combined_inducer_perm_thresh_inhibitor_perm_signal" # ABCD
            ode_func = ode_inducer_and_inhibitor_dependent_perm_inhibitor_dependent_signal!
            thresh_crit = thresh_criterion_combined_inhibitor_shift
            @note_param f_maxs = [prms[:b], prms[:s]]
            @note_param K_fs = [prms[:K_cI], prms[:K_cC], prms[:K_I]]
            @note_param thresh_means = [prms[:μ_γ], prms[:μ_ω]]
            @note_param thresh_sds = [prms[:σ_γ], prms[:σ_ω]]
            @note_param germ_response = gresp_feedback(ode_func, thresh_crit, sobol_pts, times, geom_samples, prms[:c₀_cs], f_maxs, prms[:Pₛ], prms[:Pₛ_cs], K_fs, prms[:μ_ψ], prms[:σ_ψ], thresh_means, thresh_sds; ks=[prms[:k_C]], n=prms[:n])
            
        elseif model_type == "feedback_combined_inhibitor_thresh_signal_inducer_perm_thresh" # ABCE
            ode_func = ode_inducer_dependent_perm_inhibitor_dependent_signal!
            thresh_crit = thresh_criterion_combined_inhibitor_shift_inducer_signal_shift
            @note_param f_maxs = [prms[:s]]
            @note_param K_fs = [prms[:K_cI], prms[:K_cC], prms[:K_I]]
            @note_param thresh_means = [prms[:μ_γ], prms[:μ_ω]]
            @note_param thresh_sds = [prms[:σ_γ], prms[:σ_ω]]
            @note_param germ_response = gresp_feedback(ode_func, thresh_crit, sobol_pts, times, geom_samples, prms[:c₀_cs], f_maxs, prms[:Pₛ], prms[:Pₛ_cs], K_fs, prms[:μ_ψ], prms[:σ_ψ], thresh_means, thresh_sds; ks=[prms[:k_I], prms[:k_C]], n=prms[:n])
        
        elseif model_type == "feedback_combined_inhibitor_perm_thresh_inducer_perm_thresh" # ABDE
            ode_func = ode_inducer_and_inhibitor_dependent_perm!
            thresh_crit = thresh_criterion_combined_shift
            @note_param f_maxs = [prms[:b], prms[:s]]
            @note_param K_fs = [prms[:K_cI], prms[:K_cC]]
            @note_param thresh_means = [prms[:μ_γ], prms[:μ_ω]]
            @note_param thresh_sds = [prms[:σ_γ], prms[:σ_ω]]
            @note_param germ_response = gresp_feedback(ode_func, thresh_crit, sobol_pts, times, geom_samples, prms[:c₀_cs], f_maxs, prms[:Pₛ], prms[:Pₛ_cs], K_fs, prms[:μ_ψ], prms[:σ_ψ], thresh_means, thresh_sds; ks=[prms[:k_I], prms[:k_C]])
            
        elseif model_type == "feedback_inducer_inhibitor_perm_thresh_signal_inducer_perm" # ACDE
            ode_func = ode_inducer_and_inhibitor_dependent_perm_inhibitor_dependent_signal!
            thresh_crit = thresh_criterion_inducer_signal_shift
            @note_param f_maxs = [prms[:b], prms[:s]]
            @note_param K_fs = [prms[:K_cI], prms[:K_cC], prms[:K_I]]
            @note_param thresh_means = [prms[:μ_ω]]
            @note_param thresh_sds = [prms[:σ_ω]]
            @note_param germ_response = gresp_feedback(ode_func, thresh_crit, sobol_pts, times, geom_samples, prms[:c₀_cs], f_maxs, prms[:Pₛ], prms[:Pₛ_cs], K_fs, prms[:μ_ψ], prms[:σ_ψ], thresh_means, thresh_sds; ks=[prms[:k_I]], n=prms[:n])
            
        elseif model_type == "feedback_combined_inhibitor_perm_thresh_signal_inducer_perm" # ACDE
            ode_func = ode_inducer_and_inhibitor_dependent_perm_inhibitor_dependent_signal!
            thresh_crit = thresh_criterion_combined_inducer_signal_shift
            @note_param f_maxs = [prms[:b], prms[:s]]
            @note_param K_fs = [prms[:K_cI], prms[:K_cC], prms[:K_I]]
            @note_param thresh_means = [prms[:μ_γ], prms[:μ_ω]]
            @note_param thresh_sds = [prms[:σ_γ], prms[:σ_ω]]
            @note_param germ_response = gresp_feedback(ode_func, thresh_crit, sobol_pts, times, geom_samples, prms[:c₀_cs], f_maxs, prms[:Pₛ], prms[:Pₛ_cs], K_fs, prms[:μ_ψ], prms[:σ_ψ], thresh_means, thresh_sds; ks=[prms[:k_I]], n=prms[:n])
            
        elseif model_type == "feedback_combined_inhibitor_perm_thresh_signal_inducer_thresh" # BCDE
            ode_func = ode_inhibitor_dependent_perm!
            thresh_crit = thresh_criterion_combined_inhibitor_shift_inducer_signal_shift
            @note_param f_maxs = [prms[:b]]
            @note_param K_fs = [prms[:K_cI], prms[:K_cC], prms[:K_I]]
            @note_param thresh_means = [prms[:μ_γ], prms[:μ_ω]]
            @note_param thresh_sds = [prms[:σ_γ], prms[:σ_ω]]
            @note_param germ_response = gresp_feedback(ode_func, thresh_crit, sobol_pts, times, geom_samples, prms[:c₀_cs], f_maxs, prms[:Pₛ], prms[:Pₛ_cs], K_fs, prms[:μ_ψ], prms[:σ_ψ], thresh_means, thresh_sds; ks=[prms[:k_I], prms[:k_C]], n=prms[:n])
           
        elseif model_type == "feedback_combined_inhibitor_perm_thresh_signal_inducer_perm_thresh" # ABCDE
            ode_func = ode_inducer_and_inhibitor_dependent_perm_inhibitor_dependent_signal!
            thresh_crit = thresh_criterion_combined_inhibitor_shift_inducer_signal_shift
            @note_param f_maxs = [prms[:b], prms[:s]]
            @note_param K_fs = [prms[:K_cI], prms[:K_cC], prms[:K_I]]
            @note_param thresh_means = [prms[:μ_γ], prms[:μ_ω]]
            @note_param thresh_sds = [prms[:σ_γ], prms[:σ_ω]]
            @note_param germ_response = gresp_feedback(ode_func, thresh_crit, sobol_pts, times, geom_samples, prms[:c₀_cs], f_maxs, prms[:Pₛ], prms[:Pₛ_cs], K_fs, prms[:μ_ψ], prms[:σ_ψ], thresh_means, thresh_sds; ks=[prms[:k_I], prms[:k_C]], n=prms[:n])
            
        elseif model_type == "test"
            @note_param germ_response = [gresp_test(u, W, t, ρₛ, prms[:d_hp], ξ2, κ2, prms[:Pₛ], prms[:μ_ω], prms[:σ_ω]) for t in times] # DELETE LATER!!!
        end

        if debug
            println("Parameters: ", sort(PARAMS_BUFFER))
            if startswith(model_type, "feedback")
                println("ODE function: ", ode_func)
                println("Threshold criterion: ", thresh_crit)
            else
                println("Integral function")
            end
        end

        return germ_response
    end

    """
    Compute the relative decrease in inhibitor concentration.
    inputs:
        V - spore volume in μm^3
        A - spore surface area in μm^2
        Pₛ - permeation constant for the inhibitor in μm/s
        ρₛ - spore density in spores/μm^3
        t - time in seconds
    output:
        β - relative inhibitor concentration decrease
    """
    function calc_beta(V, A, Pₛ, ρₛ, t)
        τ = V ./ (Pₛ * A)
        ϕ = ρₛ .* V
        β = ϕ .+ (1 .- ϕ) .* exp.(-t ./ (τ .* (1 .- ϕ)))
        return β
    end

    """
    Compute the induction signal strength.
    inputs:
        c₀_cs - initial concentration of carbon source in M
        t - time in seconds
        Pₛ_cs - permeation constant for the carbon source in μm/s
        A - spore surface area in μm^2
        V_cw - vacant cell wall volume in μm^3
        K_cC - half-saturation constant for the carbon source
    output:
        s - inducing signal strength
    """
    function calc_s(c₀_cs, t, Pₛ_cs, A, V_cw, K_cC)
        c_cs = inducer_concentration.(c₀_cs, t, Pₛ_cs, A, V_cw)
        s = c_cs ./ (K_cC .+ c_cs)
        return s
    end

    """
    Compute the spore volume & surface area
    and the vacant cell wall volume.
    inputs:
        ξ - spore radius in μm
        d_hp - thickness of the hydrophobin layer in μm
        κ - cell wall thickness in μm
    outputs:
        V - spore volume in μm^3
        A - spore surface area in μm^2
        V_cw - vacant cell wall volume in μm^3
    """
    function calc_geom_variables(ξ, d_hp, κ)
        V = 4/3 * π .* ξ.^3
        A = 4π .* ξ.^2
        V_cw = compute_ps_layer_volume(ξ, d_hp, κ)

        return V, A, V_cw
    end

    """
    Construct normal distributions for
    a collection of means and standard deviations.
    inputs:
        μs - means
        σs - standard deviations
    outputs:
        dists - distributions
    """
    function normal_distributions(μs, σs)
        return [Normal(μs[i], σs[i]) for i in eachindex(μs)]
    end

    """
    Compute inhibitory and inductive signals
    via geometric variables.
    inputs:
        ξ - spore radius in μm
        d_hp - thickness of the hydrophobin layer in μm
        κ - cell wall thickness in μm
        c₀_cs - initial concentration of the carbon source
        ρₛ - spore density in spores/μm^3
        Pₛ - permeation constant for the inhibitor in μm/s
        Pₛ_cs - permeation constant for the carbon source in μm/s
        K_cC - half-saturation constant for the carbon source
        t - time in seconds
    outputs:
        β - relative inhibitor concentration decrease
        s - inducing signal strength
    """
    function calc_signals(ξ, d_hp, κ, c₀_cs, ρₛ, Pₛ, Pₛ_cs, K_cC, t)
        V, A, V_cw = calc_geom_variables(ξ, d_hp, κ)
        β = calc_beta(V, A, Pₛ, ρₛ, t)
        s = calc_s(c₀_cs, t, Pₛ_cs, A, V_cw, K_cC)
        return β, s
    end

    """
    Compute lognormally distributed samples
    of the initial inhibitor concentration.
    inputs:
        μ_ψ - normal mean
        σ_ψ - normal standard deviation
        u - transformed Gauss-Hermite nodes
    outputs:
        ψ - lognormal samples
    """
    function lognormal_samples(μ, σ, u)
        μ_log = log(μ^2 / sqrt(σ^2 + μ^2))
        σ_log = sqrt(log(σ^2 / μ^2 + 1))
        return exp.(μ_log .+ σ_log .* u)
    end

    """
    Test model with 3 free parameters.
    inputs:
        u - transformed Gauss-Hermite nodes
        W - transformed Gauss-Hermite weights (matrix)
        t - time in seconds
        ρₛ - spore density in spores/μm^3
        c₀_cs - initial concentration of carbon source in M
        d_hp - thickness of the hydrophobin layer in μm
        ξ - spore radius in μm
        κ - cell wall thickness in μm
        Pₛ - permeation constant for the inhibitor in μm/s
        μ_γ - mean inhibition threshold
        σ_γ - standard deviation of inhibition threshold
    output:
        the germination response for the given parameters (normalized)
    """
    function gresp_test(u, W, t, ρₛ, d_hp, ξ, κ, Pₛ, μ_γ, σ_γ)

        # Distributions
        dist_γ = Normal(μ_γ, σ_γ)

        # Signals
        V, A, V_cw = calc_geom_variables(ξ, d_hp, κ)
        β = calc_beta(V, A, Pₛ, ρₛ, t)
        
        tail = 1 .- cdf.(dist_γ, β)
        
        return sum(W .* tail)
    end
    
    """
    Compute the germination response for independent
    inhibition and induction for a given set of parameters.
    Uses Gauss-Hermite approximation.
    inputs:
        u - transformed Gauss-Hermite nodes
        W - transformed Gauss-Hermite weights (matrix)
        t - time in seconds
        ρₛ - spore density in spores/μm^3
        c₀_cs - initial concentration of carbon source in M
        d_hp - thickness of the hydrophobin layer in μm
        ξ - spore radius in μm
        κ - cell wall thickness in μm
        Pₛ - permeation constant for the inhibitor in μm/s
        Pₛ_cs - permeation constant for the carbon source in μm/s
        K_cC - half-saturation constant for the carbon source
        μ_γ - mean inhibition threshold
        σ_γ - standard deviation of inhibition threshold
        μ_ω - mean induction threshold
        σ_ω - standard deviation of induction threshold
    output:
        the germination response for the given parameters (normalized)
    """
    function gresp_independent_factors_gh(u, W, t, ρₛ, c₀_cs, d_hp, ξ, κ, Pₛ, Pₛ_cs, K_cC, μ_γ, σ_γ, μ_ω, σ_ω)

        # Distributions
        dist_γ, dist_ω = normal_distributions([μ_γ, μ_ω], [σ_γ, σ_ω])

        # Signals
        β, s = calc_signals(ξ, d_hp, κ, c₀_cs, ρₛ, Pₛ, Pₛ_cs, K_cC, t)
        
        tail = cdf.(dist_ω, s) .* (1 .- cdf.(dist_γ, β))
        
        return sum(W .* tail)
    end

    """
    Compute the germination response for an inducer-dependent
    inhibitor threshold for a given set of parameters,
    without considering an external initial concentration.
    Uses Gauss-Hermite approximation.
    inputs:
        u - transformed Gauss-Hermite nodes
        W - transformed Gauss-Hermite weights (matrix)
        t - time in seconds
        ρₛ - spore density in spores/um^3
        c₀_cs - initial concentration of carbon source in M
        d_hp - thickness of the hydrophobin layer in um
        ξ - spore radius in um
        κ - cell wall thickness in um
        Pₛ - permeation constant for the inhibitor in um/s
        Pₛ_cs - permeation constant for the carbon source in um/s
        k - induction strength over inhibitor threshold
        K_cC - half-saturation constant for the carbon source
        μ_γ - mean inhibition threshold
        σ_γ - standard deviation of inhibition threshold
    output:
        the germination response for the given parameters (normalized)
    """
    function gresp_inducer_dep_inhibitor_thresh_gh(u, W, t, ρₛ, c₀_cs, d_hp, ξ, κ, Pₛ, Pₛ_cs, k, K_cC, μ_γ, σ_γ)

        # Distributions
        dist_γ = Normal(μ_γ, σ_γ)

        # Signals
        β, s = calc_signals(ξ, d_hp, κ, c₀_cs, ρₛ, Pₛ, Pₛ_cs, K_cC, t)

        tail = 1 .- cdf.(dist_γ, β .- k .* s)

        return sum(W .* tail)
    end
    
    """
    Compute the germination response for an inhibitor-dependent
    induction threshold for a given set of parameters.
    Uses Gauss-Hermite approximation.
    inputs:
        u - transformed Gauss-Hermite nodes
        W3 - transformed Gauss-Hermite weights (matrix)
        t - time in seconds
        ρₛ - spore density in spores/um^3
        c₀_cs - initial concentration of carbon source in M
        d_hp - thickness of the hydrophobin layer in um
        ξ - spore radius in um
        κ - cell wall thickness in um
        Pₛ - permeation constant for the inhibitor in um/s
        Pₛ_cs - permeation constant for the carbon source in um/s
        K_cI - half-saturation constant for the inhibitor threshold shift
        K_cC - half-saturation constant for the carbon source threshold shift
        k - inhibition strength over induction threshold
        μ_ω - mean induction threshold
        σ_ω - standard deviation of induction threshold
        μ_ψ - mean initial concentration
        σ_ψ - standard deviation of initial concentration
    output:
        the germination response for the given parameters (normalized)
    """
    function gresp_inhibitor_dep_inducer_thresh_gh(u, W3, t, ρₛ, c₀_cs, d_hp, ξ, κ, Pₛ, Pₛ_cs, K_cI, K_cC, k, μ_ω, σ_ω, μ_ψ, σ_ψ)

        # Transform to log-normal
        ψ = lognormal_samples(μ_ψ, σ_ψ, u)

        # Distributions
        dist_ω = Normal(μ_ω, σ_ω)

        # Signals
        β, s = calc_signals(ξ, d_hp, κ, c₀_cs, ρₛ, Pₛ, Pₛ_cs, K_cC, t)

        # Reshape
        n_nodes = size(u, 1)
        β = repeat(β, 1, 1, n_nodes)
        s = repeat(s, 1, 1, n_nodes)
        ψ = repeat(ψ, 1, n_nodes, n_nodes)
        ψ = permutedims(ψ, (2, 3, 1))

        c_in = ψ .* β

        tail = cdf.(dist_ω, s .- k .* c_in ./ (K_cI .+ c_in))

        return sum(W3 .* tail)
    end

    """
    Compute the germination response for an inhibitor-dependent
    induction signal for a given set of parameters.
    Uses Gauss-Hermite approximation.
    inputs:
        u - transformed Gauss-Hermite nodes
        W3 - transformed Gauss-Hermite weights (tensor)
        t - time in seconds
        ρₛ - spore density in spores/um^3
        c₀_cs - initial concentration of carbon source in M
        d_hp - thickness of the hydrophobin layer in um
        ξ - spore radius in um
        κ - cell wall thickness in um
        Pₛ - permeation constant for the inhibitor in um/s
        Pₛ_cs - permeation constant for the carbon source in um/s
        K_cC - half-saturation constant for the carbon source
        K_I - half-saturation constant for the inhibitor
        n - Hill coefficient for the inhibitor
        μ_ω - mean induction threshold
        σ_ω - standard deviation of induction threshold
        μ_ψ - mean initial concentration
        σ_ψ - standard deviation of initial concentration
    output:
        the germination response for the given parameters (normalized)
    """
    function gresp_inhibitor_dep_inducer_signal_gh(u, W3, t, ρₛ, c₀_cs, d_hp, ξ, κ, Pₛ, Pₛ_cs, K_cC, K_I, n, μ_ω, σ_ω, μ_ψ, σ_ψ)

        # Transform to log-normal
        ψ = lognormal_samples(μ_ψ, σ_ψ, u)

        # Distributions
        dist_ω = Normal(μ_ω, σ_ω)

        # Signals
        β, s = calc_signals(ξ, d_hp, κ, c₀_cs, ρₛ, Pₛ, Pₛ_cs, K_cC, t)

        # Reshape
        n_nodes = size(u, 1)
        β = repeat(β, 1, 1, n_nodes)
        s = repeat(s, 1, 1, n_nodes)
        ψ = repeat(ψ, 1, n_nodes, n_nodes)
        ψ = permutedims(ψ, (2, 3, 1))

        c_in = ψ .* β
        s_mod = s ./ (1 .+ (c_in ./ K_I).^n)

        tail = cdf.(dist_ω, s_mod)

        return sum(W3 .* tail)
    end

    """
    Compute the germination response for an inhibitor-dependent
    induction threshold and signal for a given set of parameters.
    Uses Gauss-Hermite approximation.
    inputs:
        u - transformed Gauss-Hermite nodes
        W3 - transformed Gauss-Hermite weights (tensor)
        t - time in seconds
        ρₛ - spore density in spores/um^3
        c₀_cs - initial concentration of carbon source in M
        d_hp - thickness of the hydrophobin layer in um
        ξ - spore radius in um
        κ - cell wall thickness in um
        Pₛ - permeation constant for the inhibitor in um/s
        Pₛ_cs - permeation constant for the carbon source in um/s
        k - proportionality constant for threshold modulation vs signal modulation
        K_cI - half-saturation constant for the inhibitor threshold shift
        K_cC - half-saturation constant for the carbon source threshold shift
        K_I - half-saturation constant for the signal inhibition
        n - Hill coefficient for the inhibitor
        k - inhibition strength over induction threshold
        μ_ω - mean induction threshold
        σ_ω - standard deviation of induction threshold
        μ_ψ - mean initial concentration
        σ_ψ - standard deviation of initial concentration
    output:
        the germination response for the given parameters (normalized)
    """
    function gresp_inhibitor_dep_inducer_gh(u, W3, t, ρₛ, c₀_cs, d_hp, ξ, κ, Pₛ, Pₛ_cs, k, K_cI, K_cC, K_I, n, μ_ω, σ_ω, μ_ψ, σ_ψ)

        # Transform to log-normal
        ψ = lognormal_samples(μ_ψ, σ_ψ, u)

        # Distributions
        dist_ω = Normal(μ_ω, σ_ω)

        # Signals
        β, s = calc_signals(ξ, d_hp, κ, c₀_cs, ρₛ, Pₛ, Pₛ_cs, K_cC, t)

        # Reshape
        n_nodes = size(u, 1)
        β = repeat(β, 1, 1, n_nodes)
        s = repeat(s, 1, 1, n_nodes)
        ψ = repeat(ψ, 1, n_nodes, n_nodes)
        ψ = permutedims(ψ, (2, 3, 1))

        c_in = ψ .* β
        s_mod = s ./ (1 .+ (c_in ./ K_I).^n)

        tail = cdf.(dist_ω, s_mod .- k .* c_in ./ (K_cI .+ c_in))

        return sum(W3 .* tail)
    end

    """
    Compute the germination response for an inhibitor-dependent
    induction signal and an inducer-dependent inhibition threshold
    for a given set of parameters.
    Uses Gauss-Hermite approximation.
    inputs:
        u - transformed Gauss-Hermite nodes
        W3 - transformed Gauss-Hermite weights (tensor)
        t - time in seconds
        ρₛ - spore density in spores/um^3
        c₀_cs - initial concentration of carbon source in M
        d_hp - thickness of the hydrophobin layer in um
        ξ - spore radius in um
        κ - cell wall thickness in um
        Pₛ - permeation constant for the inhibitor in um/s
        Pₛ_cs - permeation constant for the carbon source in um/s
        K_cC - half-saturation constant for the carbon source
        K_I - half-saturation constant for the inhibitor
        n - Hill coefficient for the inhibitor
        k_C - inducer strength over inhibition threshold
        μ_γ - mean inhibition threshold
        σ_γ - standard deviation of inhibition threshold
        μ_ψ - mean initial concentration
        σ_ψ - standard deviation of initial concentration
    output:
        the germination response for the given parameters (normalized)
    """
    function gresp_inh_dep_ind_signal_ind_dep_inh_thresh_gh(u, W3, t, ρₛ, c₀_cs, d_hp, ξ, κ, Pₛ, Pₛ_cs, k_C, K_cC, K_I, n, μ_γ, σ_γ, μ_ψ, σ_ψ)

        # Transform to log-normal
        ψ = lognormal_samples(μ_ψ, σ_ψ, u)

        # Distributions
        dist_γ = Normal(μ_γ, σ_γ)

        # Signals
        β, s = calc_signals(ξ, d_hp, κ, c₀_cs, ρₛ, Pₛ, Pₛ_cs, K_cC, t)

        # Reshape
        n_nodes = size(u, 1)
        β = repeat(β, 1, 1, n_nodes)
        s = repeat(s, 1, 1, n_nodes)
        ψ = repeat(ψ, 1, n_nodes, n_nodes)
        ψ = permutedims(ψ, (2, 3, 1))

        c_in = ψ .* β
        s_mod = s ./ (1 .+ (c_in ./ K_I).^n)

        tail = 1. .- cdf.(dist_γ, β .- k_C .* s_mod)

        return sum(W3 .* tail)
    end

    """
    Compute the germination response for a 2-factor germination
    with an inhibitor-dependent induction signal and an
    inducer-dependent inhibition threshold for a given set of parameters.
    Uses Gauss-Hermite approximation.
    inputs:
        u - transformed Gauss-Hermite nodes
        W3 - transformed Gauss-Hermite weights (tensor)
        t - time in seconds
        ρₛ - spore density in spores/um^3
        c₀_cs - initial concentration of carbon source in M
        d_hp - thickness of the hydrophobin layer in um
        ξ - spore radius in um
        κ - cell wall thickness in um
        Pₛ - permeation constant for the inhibitor in um/s
        Pₛ_cs - permeation constant for the carbon source in um/s
        K_cC - half-saturation constant for the carbon source
        K_I - half-saturation constant for the inhibitor
        n - Hill coefficient for the inhibitor
        k_C - inducer strength over inhibition threshold
        μ_γ - mean inhibition threshold
        σ_γ - standard deviation of inhibition threshold
        μ_ω - mean induction threshold
        σ_ω - standard deviation of induction threshold
        μ_ψ - mean initial concentration
        σ_ψ - standard deviation of initial concentration
    output:
        the germination response for the given parameters (normalized)
    """
    function gresp_inh_dep_ind_signal_ind_dep_inh_thresh_2_factor_gh(u, W3, t, ρₛ, c₀_cs, d_hp, ξ, κ, Pₛ, Pₛ_cs, k_C, K_cC, K_I, n, μ_γ, σ_γ, μ_ω, σ_ω, μ_ψ, σ_ψ)

        # Transform to log-normal
        ψ = lognormal_samples(μ_ψ, σ_ψ, u)

        # Distributions
        dist_γ, dist_ω = normal_distributions([μ_γ, μ_ω], [σ_γ, σ_ω])

        # Signals
        β, s = calc_signals(ξ, d_hp, κ, c₀_cs, ρₛ, Pₛ, Pₛ_cs, K_cC, t)

        # Reshape
        n_nodes = size(u, 1)
        β = repeat(β, 1, 1, n_nodes)
        s = repeat(s, 1, 1, n_nodes)
        ψ = repeat(ψ, 1, n_nodes, n_nodes)
        ψ = permutedims(ψ, (2, 3, 1))

        c_in = ψ .* β
        s_mod = s ./ (1 .+ (c_in ./ K_I).^n)

        tail = (1. .- cdf.(dist_γ, β .- k_C .* s_mod)) .* cdf.(dist_ω, s_mod)

        return sum(W3 .* tail)
    end
    
    """
    Compute the germination response for an inducer-dependent
    inhibitor threshold for a given set of parameters,
    without considering an external initial concentration.
    Uses Gauss-Hermite approximation.
    inputs:
        u - transformed Gauss-Hermite nodes
        W - transformed Gauss-Hermite weights (matrix)
        t - time in seconds
        ρₛ - spore density in spores/um^3
        c₀_cs - initial concentration of carbon source in M
        d_hp - thickness of the hydrophobin layer in um
        ξ - spore radius in um
        κ - cell wall thickness in um
        Pₛ - permeation constant for the inhibitor in um/s
        Pₛ_cs - permeation constant for the carbon source in um/s
        k - induction strength over inhibitor threshold
        K_cC - half-saturation constant for the carbon source
        μ_γ - mean inhibition threshold
        σ_γ - standard deviation of inhibition threshold
        μ_ω - mean induction threshold
        σ_ω - standard deviation of induction threshold
    output:
        the germination response for the given parameters (normalized)
    """
    function gresp_inducer_dep_inhibitor_thresh_2_factor_gh(u, W, t, ρₛ, c₀_cs, d_hp, ξ, κ, Pₛ, Pₛ_cs, k, K_cC, μ_γ, σ_γ, μ_ω, σ_ω)

        # Distributions
        dist_γ, dist_ω = normal_distributions([μ_γ, μ_ω], [σ_γ, σ_ω])

        # Signals
        β, s = calc_signals(ξ, d_hp, κ, c₀_cs, ρₛ, Pₛ, Pₛ_cs, K_cC, t)

        tail = (1 .- cdf.(dist_γ, β .- k .* s)) .* cdf.(dist_ω, s)

        return sum(W .* tail)
    end

    """
    Compute the germination response for an inhibitor-dependent
    induction threshold and an additional inhibitor-dependent
    germination for a given set of parameters.
    Uses Gauss-Hermite approximation.
    inputs:
        u - transformed Gauss-Hermite nodes
        W3 - transformed Gauss-Hermite weights (matrix)
        t - time in seconds
        ρₛ - spore density in spores/um^3
        c₀_cs - initial concentration of carbon source in M
        d_hp - thickness of the hydrophobin layer in um
        ξ - spore radius in um
        κ - cell wall thickness in um
        Pₛ - permeation constant for the inhibitor in um/s
        Pₛ_cs - permeation constant for the carbon source in um/s
        K_cI - half-saturation constant for the inhibitor threshold shift
        K_cC - half-saturation constant for the carbon source threshold shift
        k - inhibition strength over induction threshold
        μ_γ - mean inhibition threshold
        σ_γ - standard deviation of inhibition threshold
        μ_ω - mean induction threshold
        σ_ω - standard deviation of induction threshold
        μ_ψ - mean initial concentration
        σ_ψ - standard deviation of initial concentration
    output:
        the germination response for the given parameters (normalized)
    """
    function gresp_inhibitor_dep_inducer_thresh_2_factor_gh(u, W3, t, ρₛ, c₀_cs, d_hp, ξ, κ, Pₛ, Pₛ_cs, K_cI, K_cC, k, μ_γ, σ_γ, μ_ω, σ_ω, μ_ψ, σ_ψ)

        # Transform to log-normal
        ψ = lognormal_samples(μ_ψ, σ_ψ, u)

        # Distributions
        dist_γ, dist_ω = normal_distributions([μ_γ, μ_ω], [σ_γ, σ_ω])

        # Signals
        β, s = calc_signals(ξ, d_hp, κ, c₀_cs, ρₛ, Pₛ, Pₛ_cs, K_cC, t)

        # Inhibition CDF
        tail_γ = 1 .- cdf.(dist_γ, β)

        # Reshape
        n_nodes = size(u, 1)
        β = repeat(β, 1, 1, n_nodes)
        s = repeat(s, 1, 1, n_nodes)
        tail_γ = repeat(tail_γ, 1, 1, n_nodes)
        ψ = repeat(ψ, 1, n_nodes, n_nodes)
        ψ = permutedims(ψ, (2, 3, 1))

        c_in = ψ .* β

        tail = cdf.(dist_ω, s .- k .* c_in ./ (K_cI .+ c_in)) .* tail_γ

        return sum(W3 .* tail)
    end

    """
    Compute the germination response for an inhibitor-dependent
    induction signal and an additional inhibitor-dependent
    germination for a given set of parameters.
    Uses Gauss-Hermite approximation.
    inputs:
        u - transformed Gauss-Hermite nodes
        W3 - transformed Gauss-Hermite weights (matrix)
        t - time in seconds
        ρₛ - spore density in spores/um^3
        c₀_cs - initial concentration of carbon source in M
        d_hp - thickness of the hydrophobin layer in um
        ξ - spore radius in um
        κ - cell wall thickness in um
        Pₛ - permeation constant for the inhibitor in um/s
        Pₛ_cs - permeation constant for the carbon source in um/s
        K_cC - half-saturation constant for the carbon source
        K_I - half-saturation constant for the inhibitor
        n - Hill coefficient for the inhibitor
        μ_γ - mean inhibition threshold
        σ_γ - standard deviation of inhibition threshold
        μ_ω - mean induction threshold
        σ_ω - standard deviation of induction threshold
        μ_ψ - mean initial concentration
        σ_ψ - standard deviation of initial concentration
    output:
        the germination response for the given parameters (normalized)
    """
    function gresp_inhibitor_dep_inducer_signal_2_factor_gh(u, W3, t, ρₛ, c₀_cs, d_hp, ξ, κ, Pₛ, Pₛ_cs, K_cC, K_I, n, μ_γ, σ_γ, μ_ω, σ_ω, μ_ψ, σ_ψ)

        # Transform to log-normal
        ψ = lognormal_samples(μ_ψ, σ_ψ, u)

        # Distributions
        dist_γ, dist_ω = normal_distributions([μ_γ, μ_ω], [σ_γ, σ_ω])

        # Signals
        β, s = calc_signals(ξ, d_hp, κ, c₀_cs, ρₛ, Pₛ, Pₛ_cs, K_cC, t)

        # Inhibition CDF
        tail_γ = 1 .- cdf.(dist_γ, β)

        # Reshape
        n_nodes = size(u, 1)
        β = repeat(β, 1, 1, n_nodes)
        s = repeat(s, 1, 1, n_nodes)
        tail_γ = repeat(tail_γ, 1, 1, n_nodes)
        ψ = repeat(ψ, 1, n_nodes, n_nodes)
        ψ = permutedims(ψ, (2, 3, 1))

        c_in = ψ .* β
        s_mod = s ./ (1 .+ (c_in ./ K_I).^n)

        tail = cdf.(dist_ω, s_mod) .* tail_γ

        return sum(W3 .* tail)
    end

    """
    Compute the germination response for an inhibitor-dependent
    induction threshold and signal and an additional
    inhibitor-dependent germination for a given set of parameters.
    Uses Gauss-Hermite approximation.
    inputs:
        u - transformed Gauss-Hermite nodes
        W3 - transformed Gauss-Hermite weights (matrix)
        t - time in seconds
        ρₛ - spore density in spores/um^3
        c₀_cs - initial concentration of carbon source in M
        d_hp - thickness of the hydrophobin layer in um
        ξ - spore radius in um
        κ - cell wall thickness in um
        Pₛ - permeation constant for the inhibitor in um/s
        Pₛ_cs - permeation constant for the carbon source in um/s
        K_cI - half-saturation constant for the inhibitor threshold shift
        K_cC - half-saturation constant for the carbon source threshold shift
        K_I - half-saturation constant for the inhibitor
        n - Hill coefficient for the inhibitor
        k - inhibition strength over induction threshold
        μ_γ - mean inhibition threshold
        σ_γ - standard deviation of inhibition threshold
        μ_ω - mean induction threshold
        σ_ω - standard deviation of induction threshold
        μ_ψ - mean initial concentration
        σ_ψ - standard deviation of initial concentration
    output:
        the germination response for the given parameters (normalized)
    """
    function gresp_inhibitor_dep_inducer_2_factor_gh(u, W3, t, ρₛ, c₀_cs, d_hp, ξ, κ, Pₛ, Pₛ_cs, K_cI, K_cC, K_I, n, k, μ_γ, σ_γ, μ_ω, σ_ω, μ_ψ, σ_ψ)

        # Transform to log-normal
        ψ = lognormal_samples(μ_ψ, σ_ψ, u)

        # Distributions
        dist_γ, dist_ω = normal_distributions([μ_γ, μ_ω], [σ_γ, σ_ω])

        # Signals
        β, s = calc_signals(ξ, d_hp, κ, c₀_cs, ρₛ, Pₛ, Pₛ_cs, K_cC, t)

        # Inhibition CDF
        tail_γ = 1 .- cdf.(dist_γ, β)

        # Reshape
        n_nodes = size(u, 1)
        β = repeat(β, 1, 1, n_nodes)
        s = repeat(s, 1, 1, n_nodes)
        tail_γ = repeat(tail_γ, 1, 1, n_nodes)
        ψ = repeat(ψ, 1, n_nodes, n_nodes)
        ψ = permutedims(ψ, (2, 3, 1))

        c_in = ψ .* β
        s_mod = s ./ (1 .+ (c_in ./ K_I).^n)

        tail = cdf.(dist_ω, s_mod .- k .* c_in ./ (K_cI .+ c_in)) .* tail_γ

        return sum(W3 .* tail)
    end

    """
    Compute the germination response for an inhibitor-dependent
    induction threshold and an inducer-dependent inhibitor
    threshold for a given set of parameters.
    Uses Gauss-Hermite approximation.
    inputs:
        u - transformed Gauss-Hermite nodes
        W3 - transformed Gauss-Hermite weights (matrix)
        t - time in seconds
        ρₛ - spore density in spores/um^3
        c₀_cs - initial concentration of carbon source in M
        d_hp - thickness of the hydrophobin layer in um
        ξ - spore radius in um
        κ - cell wall thickness in um
        Pₛ - permeation constant for the inhibitor in um/s
        Pₛ_cs - permeation constant for the carbon source in um/s
        K_cI - half-saturation constant for the inhibitor threshold shift
        K_cC - half-saturation constant for the carbon source threshold shift
        k_I - inhibition strength over induction threshold
        k_C - induction strength over inhibition threshold
        μ_γ - mean inhibition threshold
        σ_γ - standard deviation of inhibition threshold
        μ_ω - mean induction threshold
        σ_ω - standard deviation of induction threshold
        μ_ψ - mean initial concentration
        σ_ψ - standard deviation of initial concentration
    output:
        the germination response for the given parameters (normalized)
    """
    function gresp_inh_dep_ind_thresh_ind_dep_inh_thresh_2_factor_gh(u, W3, t, ρₛ, c₀_cs, d_hp, ξ, κ, Pₛ, Pₛ_cs, K_cI, K_cC, k_I, k_C, μ_γ, σ_γ, μ_ω, σ_ω, μ_ψ, σ_ψ)

        # Transform to log-normal
        ψ = lognormal_samples(μ_ψ, σ_ψ, u)

        # Distributions
        dist_γ, dist_ω = normal_distributions([μ_γ, μ_ω], [σ_γ, σ_ω])

        # Signals
        β, s = calc_signals(ξ, d_hp, κ, c₀_cs, ρₛ, Pₛ, Pₛ_cs, K_cC, t)

        # Reshape
        n_nodes = size(u, 1)
        β = repeat(β, 1, 1, n_nodes)
        s = repeat(s, 1, 1, n_nodes)
        ψ = repeat(ψ, 1, n_nodes, n_nodes)
        ψ = permutedims(ψ, (2, 3, 1))

        c_in = ψ .* β

        tail = (1 .- cdf.(dist_γ, β .- k_C .* s)) .* cdf.(dist_ω, s .- k_I .* c_in ./ (K_cI .+ c_in))

        return sum(W3 .* tail)
    end

    """
    Compute the germination response for an inhibitor-dependent
    induction threshold/signal and an inducer-dependent inhibitor
    threshold for a given set of parameters.
    Uses Gauss-Hermite approximation.
    inputs:
        u - transformed Gauss-Hermite nodes
        W3 - transformed Gauss-Hermite weights (matrix)
        t - time in seconds
        ρₛ - spore density in spores/um^3
        c₀_cs - initial concentration of carbon source in M
        d_hp - thickness of the hydrophobin layer in um
        ξ - spore radius in um
        κ - cell wall thickness in um
        Pₛ - permeation constant for the inhibitor in um/s
        Pₛ_cs - permeation constant for the carbon source in um/s
        K_cI - half-saturation constant for the inhibitor threshold shift
        K_cC - half-saturation constant for the carbon source threshold shift
        K_I - half-saturation constant for the signal inhibition
        n - Hill exponent of inhibition
        k_I - inhibition strength over induction threshold
        k_C - induction strength over inhibition threshold
        μ_γ - mean inhibition threshold
        σ_γ - standard deviation of inhibition threshold
        μ_ω - mean induction threshold
        σ_ω - standard deviation of induction threshold
        μ_ψ - mean initial concentration
        σ_ψ - standard deviation of initial concentration
    output:
        the germination response for the given parameters (normalized)
    """
    function gresp_inh_dep_ind_thresh_signal_ind_dep_inh_thresh_2_factor_gh(u, W3, t, ρₛ, c₀_cs, d_hp, ξ, κ, Pₛ, Pₛ_cs, K_cI, K_cC, K_I, n, k_I, k_C, μ_γ, σ_γ, μ_ω, σ_ω, μ_ψ, σ_ψ)

        # Transform to log-normal
        ψ = lognormal_samples(μ_ψ, σ_ψ, u)

        # Distributions
        dist_γ, dist_ω = normal_distributions([μ_γ, μ_ω], [σ_γ, σ_ω])

        # Signals
        β, s = calc_signals(ξ, d_hp, κ, c₀_cs, ρₛ, Pₛ, Pₛ_cs, K_cC, t)

        # Reshape
        n_nodes = size(u, 1)
        β = repeat(β, 1, 1, n_nodes)
        s = repeat(s, 1, 1, n_nodes)
        ψ = repeat(ψ, 1, n_nodes, n_nodes)
        ψ = permutedims(ψ, (2, 3, 1))

        c_in = ψ .* β
        
        # Inhibition of signal
        s_mod = s ./ (1 .+ (c_in ./ K_I).^n)

        tail = (1 .- cdf.(dist_γ, β .- k_C .* s_mod)) .* cdf.(dist_ω, s_mod .- k_I .* c_in ./ (K_cI .+ c_in))

        return sum(W3 .* tail)
    end

    """
    Compute the germination response for an inhibitor-dependent
    induction threshold for a given set of parameters,
    whereby the permeation constant is a random variable.
    Uses Gauss-Hermite approximation.
    inputs:
        u - transformed Gauss-Hermite nodes
        W4 - transformed Gauss-Hermite weights (matrix)
        t - time in seconds
        ρₛ - spore density in spores/um^3
        c₀_cs - initial concentration of carbon source in M
        d_hp - thickness of the hydrophobin layer in um
        ξ - spore radius in um
        κ - cell wall thickness in um
        Pₛ - permeation constant for the inhibitor in um/s
        Pₛ_cs - permeation constant for the carbon source in um/s
        K_cC - half-saturation constant for the carbon source
        k - inhibition strength over induction threshold
        μ_ω - mean induction threshold
        σ_ω - standard deviation of induction threshold
        μ_ψ - mean initial concentration
        σ_ψ - standard deviation of initial concentration
        μ_α - mean cell wall porosity
        σ_α - standard deviation of cell wall porosity
    output:
        the germination response for the given parameters (normalized)
    """
    function gresp_inducer_thresh_var_perm_gh(u, W4, t, ρₛ, c₀_cs, d_hp, ξ, κ, Pₛ, Pₛ_cs, K_cC, k, μ_ω, σ_ω, μ_ψ, σ_ψ, μ_α, σ_α)

        # Transform to log-normal
        ψ = lognormal_samples(μ_ψ, σ_ψ, u)
        α = lognormal_samples(μ_α, σ_α, u)

        # Distributions
        dist_ω = Normal(μ_ω, σ_ω)

        # Modulate permeation
        Pₛ = Pₛ .* α
        Pₛ_cs = Pₛ_cs .* α

        # Geometric variables
        V, A, V_cw = calc_geom_variables(ξ, d_hp, κ)

        # Reshape
        n_nodes = size(u, 1)
        V_cw = repeat(V_cw, 1, 1, n_nodes)
        V = repeat(V, 1, 1, n_nodes)
        A = repeat(A, 1, 1, n_nodes)
        Pₛ = repeat(Pₛ, 1, n_nodes, n_nodes)
        Pₛ = permutedims(Pₛ, (2, 3, 1))
        Pₛ_cs = repeat(Pₛ_cs, 1, n_nodes, n_nodes)
        Pₛ_cs = permutedims(Pₛ_cs, (2, 3, 1))
        
        # Inducer
        s = calc_s(c₀_cs, t, Pₛ_cs, A, V_cw, K_cC)

        # Inhibitor
        β = calc_beta(V, A, Pₛ, ρₛ, t)

        # Reshape
        β = repeat(β, 1, 1, 1, n_nodes)
        s = repeat(s, 1, 1, 1, n_nodes)
        ψ = repeat(ψ, 1, n_nodes, n_nodes, n_nodes)
        ψ = permutedims(ψ, (2, 3, 4, 1))

        c_in = ψ .* β

        tail = cdf.(dist_ω, s .- k .* c_in)

        return sum(W4 .* tail)
    end

    """
    Compute the germination response for an inhibitor-dependent
    induction threshold and signal for a given set of parameters,
    whereby the permeation constant is a random variable.
    Uses Gauss-Hermite approximation.
    inputs:
        u - transformed Gauss-Hermite nodes
        W4 - transformed Gauss-Hermite weights (matrix)
        t - time in seconds
        ρₛ - spore density in spores/um^3
        c₀_cs - initial concentration of carbon source in M
        d_hp - thickness of the hydrophobin layer in um
        ξ - spore radius in um
        κ - cell wall thickness in um
        Pₛ - permeation constant for the inhibitor in um/s
        Pₛ_cs - permeation constant for the carbon source in um/s
        K_cC - half-saturation constant for the carbon source
        K_I - half-saturation constant for the inhibitor
        k - inhibition strength over induction threshold
        n - Hill coefficient for the inhibitor
        μ_ω - mean induction threshold
        σ_ω - standard deviation of induction threshold
        μ_ψ - mean initial concentration
        σ_ψ - standard deviation of initial concentration
        μ_α - mean cell wall porosity
        σ_α - standard deviation of cell wall porosity
    output:
        the germination response for the given parameters (normalized)
    """
    function gresp_inducer_var_perm_gh(u, W4, t, ρₛ, c₀_cs, d_hp, ξ, κ, Pₛ, Pₛ_cs, K_cC, K_I, k, n, μ_ω, σ_ω, μ_ψ, σ_ψ, μ_α, σ_α)

        # Transform to log-normal
        ψ = lognormal_samples(μ_ψ, σ_ψ, u)
        α = lognormal_samples(μ_α, σ_α, u)

        # Distributions
        dist_ω = Normal(μ_ω, σ_ω)

        # Modulate permeation
        Pₛ = Pₛ .* α
        Pₛ_cs = Pₛ_cs .* α

        # Geometric variables
        V, A, V_cw = calc_geom_variables(ξ, d_hp, κ)

        # Reshape
        n_nodes = size(u, 1)
        V_cw = repeat(V_cw, 1, 1, n_nodes)
        V = repeat(V, 1, 1, n_nodes)
        A = repeat(A, 1, 1, n_nodes)
        Pₛ = repeat(Pₛ, 1, n_nodes, n_nodes)
        Pₛ = permutedims(Pₛ, (2, 3, 1))
        Pₛ_cs = repeat(Pₛ_cs, 1, n_nodes, n_nodes)
        Pₛ_cs = permutedims(Pₛ_cs, (2, 3, 1))
        
        # Inducer
        s = calc_s(c₀_cs, t, Pₛ_cs, A, V_cw, K_cC)

        # Inhibitor
        β = calc_beta(V, A, Pₛ, ρₛ, t)

        # Reshape
        β = repeat(β, 1, 1, 1, n_nodes)
        s = repeat(s, 1, 1, 1, n_nodes)
        ψ = repeat(ψ, 1, n_nodes, n_nodes, n_nodes)
        ψ = permutedims(ψ, (2, 3, 4, 1))

        c_in = ψ .* β
        s_mod = s ./ (1 .+ (c_in ./ K_I).^n)

        tail = cdf.(dist_ω, s_mod .- k .* c_in)

        return sum(W4 .* tail)
    end

    """
    Compute the germination response for independent
    inhibition and induction for a given set of parameters,
    whereby the permeation constant is a random variable.
    Uses Gauss-Hermite approximation.
    inputs:
        u - transformed Gauss-Hermite nodes
        W3 - transformed Gauss-Hermite weights (tensor)
        t - time in seconds
        ρₛ - spore density in spores/um^3
        c₀_cs - initial concentration of carbon source in M
        d_hp - thickness of the hydrophobin layer in um
        ξ - spore radius in um
        κ - cell wall thickness in um
        Pₛ - baseline permeation constant for the inhibitor in um/s
        Pₛ_cs - baseline permeation constant for the carbon source in um/s
        K_cC - half-saturation constant for the carbon source
        μ_γ - mean inhibition threshold
        σ_γ - standard deviation of inhibition threshold
        μ_ω - mean induction threshold
        σ_ω - standard deviation of induction threshold
        μ_α - mean cell wall porosity
        σ_α - standard deviation of cell wall porosity
    output:
        the germination response for the given parameters (normalized)
    """
    function gresp_independent_factors_var_perm_gh(u, W3, t, ρₛ, c₀_cs, d_hp, ξ, κ, Pₛ, Pₛ_cs, K_cC, μ_γ, σ_γ, μ_ω, σ_ω, μ_α, σ_α)

        # Transform to log-normal
        α = lognormal_samples(μ_α, σ_α, u)

        # Distributions
        dist_γ, dist_ω = normal_distributions([μ_γ, μ_ω], [σ_γ, σ_ω])

        # Geometric variables
        V, A, V_cw = calc_geom_variables(ξ, d_hp, κ)

        # Modulate permeation
        Pₛ = Pₛ .* α
        Pₛ_cs = Pₛ_cs .* α
        
        # Reshape
        n_nodes = size(u, 1)
        V = repeat(V, 1, 1, n_nodes)
        A = repeat(A, 1, 1, n_nodes)
        Pₛ = repeat(Pₛ, 1, n_nodes, n_nodes)
        Pₛ = permutedims(Pₛ, (2, 3, 1))
        Pₛ_cs = repeat(Pₛ_cs, 1, n_nodes, n_nodes)
        Pₛ_cs = permutedims(Pₛ_cs, (2, 3, 1))
        
        # Inhibitor
        β = calc_beta(V, A, Pₛ, ρₛ, t)

        # Inducer
        s = calc_s(c₀_cs, t, Pₛ_cs, A, V_cw, K_cC)
        
        tail = cdf.(dist_ω, s) .* (1 .- cdf.(dist_γ, β))
        
        return sum(W3 .* tail)
    end

    """
    Compute the germination response for an inhibitor-dependent
    induction threshold for a given set of parameters,
    whereby the permeation constant is a random variable.
    Uses Gauss-Hermite approximation.
    inputs:
        u - transformed Gauss-Hermite nodes
        W4 - transformed Gauss-Hermite weights (matrix)
        t - time in seconds
        ρₛ - spore density in spores/um^3
        c₀_cs - initial concentration of carbon source in M
        d_hp - thickness of the hydrophobin layer in um
        ξ - spore radius in um
        κ - cell wall thickness in um
        Pₛ - permeation constant for the inhibitor in um/s
        Pₛ_cs - permeation constant for the carbon source in um/s
        K_cC - half-saturation constant for the carbon source
        k - inhibition strength over induction threshold
        μ_γ - mean inhibition threshold
        σ_γ - standard deviation of inhibition threshold
        μ_ω - mean induction threshold
        σ_ω - standard deviation of induction threshold
        μ_ψ - mean initial concentration
        σ_ψ - standard deviation of initial concentration
        μ_α - mean cell wall porosity
        σ_α - standard deviation of cell wall porosity
    output:
        the germination response for the given parameters (normalized)
    """
    function gresp_inducer_thresh_2_factors_var_perm_gh(u, W4, t, ρₛ, c₀_cs, d_hp, ξ, κ, Pₛ, Pₛ_cs, K_cC, k, μ_γ, σ_γ, μ_ω, σ_ω, μ_ψ, σ_ψ, μ_α, σ_α)

        # Transform to log-normal
        ψ = lognormal_samples(μ_ψ, σ_ψ, u)
        α = lognormal_samples(μ_α, σ_α, u)

        # Distributions
        dist_γ, dist_ω = normal_distributions([μ_γ, μ_ω], [σ_γ, σ_ω])

        # Modulate permeation
        Pₛ = Pₛ .* α
        Pₛ_cs = Pₛ_cs .* α

        # Geometric variables
        V, A, V_cw = calc_geom_variables(ξ, d_hp, κ)

        # Reshape
        n_nodes = size(u, 1)
        V_cw = repeat(V_cw, 1, 1, n_nodes)
        V = repeat(V, 1, 1, n_nodes)
        A = repeat(A, 1, 1, n_nodes)
        Pₛ = repeat(Pₛ, 1, n_nodes, n_nodes)
        Pₛ = permutedims(Pₛ, (2, 3, 1))
        Pₛ_cs = repeat(Pₛ_cs, 1, n_nodes, n_nodes)
        Pₛ_cs = permutedims(Pₛ_cs, (2, 3, 1))
        
        # Inducer
        s = calc_s(c₀_cs, t, Pₛ_cs, A, V_cw, K_cC)

        # Inhibitor
        β = calc_beta(V, A, Pₛ, ρₛ, t)

        # Reshape
        β = repeat(β, 1, 1, 1, n_nodes)
        s = repeat(s, 1, 1, 1, n_nodes)
        ψ = repeat(ψ, 1, n_nodes, n_nodes, n_nodes)
        ψ = permutedims(ψ, (2, 3, 4, 1))

        c_in = ψ .* β

        tail = cdf.(dist_ω, s .- k .* c_in) .* (1 .- cdf.(dist_γ, β))

        return sum(W4 .* tail)
    end

    """
    Compute the germination response for an inhibitor-dependent
    induction signal for a given set of parameters,
    whereby the permeation constant is a random variable.
    Uses Gauss-Hermite approximation.
    inputs:
        u - transformed Gauss-Hermite nodes
        W4 - transformed Gauss-Hermite weights (matrix)
        t - time in seconds
        ρₛ - spore density in spores/um^3
        c₀_cs - initial concentration of carbon source in M
        d_hp - thickness of the hydrophobin layer in um
        ξ - spore radius in um
        κ - cell wall thickness in um
        Pₛ - permeation constant for the inhibitor in um/s
        Pₛ_cs - permeation constant for the carbon source in um/s
        K_cC - half-saturation constant for the carbon source
        K_I - half-saturation constant for the inhibitor
        n - Hill coefficient for the inhibitor
        μ_γ - mean inhibition threshold
        σ_γ - standard deviation of inhibition threshold
        μ_ω - mean induction threshold
        σ_ω - standard deviation of induction threshold
        μ_ψ - mean initial concentration
        σ_ψ - standard deviation of initial concentration
        μ_α - mean cell wall porosity
        σ_α - standard deviation of cell wall porosity
    output:
        the germination response for the given parameters (normalized)
    """
    function gresp_inducer_signal_2_factors_var_perm_gh(u, W4, t, ρₛ, c₀_cs, d_hp, ξ, κ, Pₛ, Pₛ_cs, K_cC, K_I, n, μ_γ, σ_γ, μ_ω, σ_ω, μ_ψ, σ_ψ, μ_α, σ_α)

        # Transform to log-normal
        ψ = lognormal_samples(μ_ψ, σ_ψ, u)
        α = lognormal_samples(μ_α, σ_α, u)

        # Distributions
        dist_γ, dist_ω = normal_distributions([μ_γ, μ_ω], [σ_γ, σ_ω])

        # Modulate permeation
        Pₛ = Pₛ .* α
        Pₛ_cs = Pₛ_cs .* α

        # Geometric variables
        V, A, V_cw = calc_geom_variables(ξ, d_hp, κ)

        # Reshape
        n_nodes = size(u, 1)
        V_cw = repeat(V_cw, 1, 1, n_nodes)
        V = repeat(V, 1, 1, n_nodes)
        A = repeat(A, 1, 1, n_nodes)
        Pₛ = repeat(Pₛ, 1, n_nodes, n_nodes)
        Pₛ = permutedims(Pₛ, (2, 3, 1))
        Pₛ_cs = repeat(Pₛ_cs, 1, n_nodes, n_nodes)
        Pₛ_cs = permutedims(Pₛ_cs, (2, 3, 1))
        
        # Inducer
        s = calc_s(c₀_cs, t, Pₛ_cs, A, V_cw, K_cC)

        # Inhibitor
        β = calc_beta(V, A, Pₛ, ρₛ, t)

        # Reshape
        β = repeat(β, 1, 1, 1, n_nodes)
        s = repeat(s, 1, 1, 1, n_nodes)
        ψ = repeat(ψ, 1, n_nodes, n_nodes, n_nodes)
        ψ = permutedims(ψ, (2, 3, 4, 1))

        c_in = ψ .* β
        s_mod = s ./ (1 .+ (c_in ./ K_I).^n)

        tail = cdf.(dist_ω, s_mod) .* (1 .- cdf.(dist_γ, β))

        return sum(W4 .* tail)
    end

    """
    Compute the germination response for an inhibitor-dependent
    induction threshold and signal for a given set of parameters,
    whereby the permeation constant is a random variable.
    Uses Gauss-Hermite approximation.
    inputs:
        u - transformed Gauss-Hermite nodes
        W4 - transformed Gauss-Hermite weights (matrix)
        t - time in seconds
        ρₛ - spore density in spores/um^3
        c₀_cs - initial concentration of carbon source in M
        d_hp - thickness of the hydrophobin layer in um
        ξ - spore radius in um
        κ - cell wall thickness in um
        Pₛ - permeation constant for the inhibitor in um/s
        Pₛ_cs - permeation constant for the carbon source in um/s
        K_cC - half-saturation constant for the carbon source
        K_I - half-saturation constant for the inhibitor
        k - inhibition strength over induction threshold
        n - Hill coefficient for the inhibitor
        μ_γ - mean inhibition threshold
        σ_γ - standard deviation of inhibition threshold
        μ_ω - mean induction threshold
        σ_ω - standard deviation of induction threshold
        μ_ψ - mean initial concentration
        σ_ψ - standard deviation of initial concentration
        μ_α - mean cell wall porosity
        σ_α - standard deviation of cell wall porosity
    output:
        the germination response for the given parameters (normalized)
    """
    function gresp_inducer_2_factors_var_perm_gh(u, W4, t, ρₛ, c₀_cs, d_hp, ξ, κ, Pₛ, Pₛ_cs, K_cC, K_I, k, n, μ_γ, σ_γ, μ_ω, σ_ω, μ_ψ, σ_ψ, μ_α, σ_α)

        # Transform to log-normal
        ψ = lognormal_samples(μ_ψ, σ_ψ, u)
        α = lognormal_samples(μ_α, σ_α, u)

        # Distributions
        dist_γ, dist_ω = normal_distributions([μ_γ, μ_ω], [σ_γ, σ_ω])

        # Modulate permeation
        Pₛ = Pₛ .* α
        Pₛ_cs = Pₛ_cs .* α

        # Geometric variables
        V, A, V_cw = calc_geom_variables(ξ, d_hp, κ)

        # Reshape
        n_nodes = size(u, 1)
        V_cw = repeat(V_cw, 1, 1, n_nodes)
        V = repeat(V, 1, 1, n_nodes)
        A = repeat(A, 1, 1, n_nodes)
        Pₛ = repeat(Pₛ, 1, n_nodes, n_nodes)
        Pₛ = permutedims(Pₛ, (2, 3, 1))
        Pₛ_cs = repeat(Pₛ_cs, 1, n_nodes, n_nodes)
        Pₛ_cs = permutedims(Pₛ_cs, (2, 3, 1))
        
        # Inducer
        s = calc_s(c₀_cs, t, Pₛ_cs, A, V_cw, K_cC)

        # Inhibitor
        β = calc_beta(V, A, Pₛ, ρₛ, t)

        # Reshape
        β = repeat(β, 1, 1, 1, n_nodes)
        s = repeat(s, 1, 1, 1, n_nodes)
        ψ = repeat(ψ, 1, n_nodes, n_nodes, n_nodes)
        ψ = permutedims(ψ, (2, 3, 4, 1))

        c_in = ψ .* β
        s_mod = s ./ (1 .+ (c_in ./ K_I).^n)

        tail = cdf.(dist_ω, s_mod .- k .* c_in) .* (1 .- cdf.(dist_γ, β))

        return sum(W4 .* tail)
    end

    """
    Compute the equilibrium germination response
    for an inducer-dependent inhibitor threshold and release.
    inputs:
        ρₛ - spore density in spores/um^3
        dist_ξ - distribution of spore radii (LogNormal)
        μ_γ - mean inhibition threshold
        σ_γ - standard deviation of inhibition threshold
        reltol - relative tolerance for the integration
    output:
        the equilibrium germination response for the given parameters (normalized)
    """
    function gresp_inducer_dep_inhibitor_eq(ρₛ, dist_ξ, μ_γ, σ_γ; reltol=1e-4)

        # Distributions
        dist_γ = Normal(μ_γ, σ_γ)

        function integrand(ξ)
            V = 4/3 * π .* ξ^3
            ϕ = ρₛ .* V
            tail = 1 .- cdf(dist_γ, ϕ)
            return tail * pdf(dist_ξ, ξ)
        end

        return quadgk(x -> integrand(x), 0.0, Inf, rtol=reltol)[1]
    end

    """
    Compute the equilibrium germination response
    for an inducer-dependent inhibitor threshold and release.
    inputs:
        ρₛ - spore density in spores/um^3
        dist_ξ - distribution of spore radii (LogNormal)
        c_ex - external concentration of the inducer in M
        μ_γ - mean inhibition threshold
        σ_γ - standard deviation of inhibition threshold
        μ_ψ - mean initial concentration in M
        σ_ψ - standard deviation of initial concentration in M
        reltol - relative tolerance for the integration
    output:
        the equilibrium germination response for the given parameters (normalized)
    """
    function gresp_inducer_dep_inhibitor_eq_c_ex(ρₛ, dist_ξ, c_ex, μ_γ, σ_γ, μ_ψ, σ_ψ; reltol=1e-4)

        # Distributions
        dist_γ = Normal(μ_γ, σ_γ)
        μ_ψ_log = log(μ_ψ^2 / sqrt(σ_ψ^2 + μ_ψ^2))
        σ_ψ_log = sqrt(log(σ_ψ^2 / μ_ψ^2 + 1))
        dist_ψ = LogNormal(μ_ψ_log, σ_ψ_log)

        function integrand(input)
            ξ, ψ = input
            V = 4/3 * π .* ξ^3
            ϕ = ρₛ .* V
            tail = 1 .- cdf(dist_γ, ϕ .+ (1 .- ϕ) .* c_ex ./ ψ)
            return tail * pdf(dist_ξ, ξ) * pdf(dist_ψ, ψ)
        end

        return hcubature(integrand, [0.0, 0.0], [quantile(dist_ξ, 1-1e-8), quantile(dist_ψ, 1-1e-8)], reltol=reltol, abstol=1e-6)[1]
    end

    """
    Compute the equilibrium germination response
    for an inhibitor-dependent inducer threshold and signal and
    an additional inhibitor-dependent germination.
    inputs:
        ρₛ - spore density in spores/um^3
        dist_ξ - distribution of spore radii (LogNormal)
        c₀_cs - initial concentration of carbon source in M
        K_cI - half-saturation constant for the inhibitor threshold shift
        K_cC - half-saturation constant for the carbon source threshold shift
        K_I - half-saturation constant for the inhibitor
        k - inhibition strength over induction threshold
        n - Hill coefficient for the inhibitor
        μ_γ - mean inhibition threshold
        σ_γ - standard deviation of inhibition threshold
        μ_ω - mean induction threshold
        σ_ω - standard deviation of induction threshold
        μ_ψ - mean initial concentration
        σ_ψ - standard deviation of initial concentration
        reltol - relative tolerance for the integration
        abstol - absolute tolerance for the integration
    output:
        the equilibrium germination response for the given parameters (normalized)
    """
    function gresp_inhibitor_dep_inducer_2_factors_eq(ρₛ, dist_ξ, c₀_cs, K_cI, K_cC, K_I, k, n, μ_γ, σ_γ, μ_ω, σ_ω, μ_ψ, σ_ψ; reltol=1e-4, abstol=1e-6)
        
        # Distributions
        dist_γ = Normal(μ_γ, σ_γ)
        dist_ω = Normal(μ_ω, σ_ω)
        μ_ψ_log = log(μ_ψ^2 / sqrt(σ_ψ^2 + μ_ψ^2))
        σ_ψ_log = sqrt(log(σ_ψ^2 / μ_ψ^2 + 1))
        dist_ψ = LogNormal(μ_ψ_log, σ_ψ_log)

        # Signal
        s_eq = c₀_cs ./ (K_cC .+ c₀_cs)

        function integrand(input)
            ξ, ψ = input
            V = 4/3 * π .* ξ^3
            ϕ = ρₛ .* V
            c_eq = ψ .* ϕ
            s_mod = s_eq ./ (1 .+ (c_eq ./ K_I).^n)
            tail1 = 1 .- cdf(dist_γ, ϕ)
            tail2 = cdf(dist_ω, s_mod .- k .* c_eq ./ (K_cI .+ c_eq))
            return tail1 * tail2 * pdf(dist_ξ, ξ) * pdf(dist_ψ, ψ)
        end

        return hcubature(integrand, [0.0, 0.0], [quantile(dist_ξ, 1-1e-8), quantile(dist_ψ, 1-1e-9)], reltol=reltol, abstol=abstol)[1]
    end

    """
    Compute the equilibrium germination response
    for an inhibitor-dependent inducer threshold and signal and
    an additional inhibitor-dependent germination.
    inputs:
        ρₛ - spore density in spores/um^3
        dist_ξ - distribution of spore radii (LogNormal)
        c_ex - external concentration of the inducer in M
        c₀_cs - initial concentration of carbon source in M
        K_cI - half-saturation constant for the inhibitor threshold shift
        K_cC - half-saturation constant for the carbon source threshold shift
        K_I - half-saturation constant for the inhibitor
        k - inhibition strength over induction threshold
        n - Hill coefficient for the inhibitor
        μ_γ - mean inhibition threshold
        σ_γ - standard deviation of inhibition threshold
        μ_ω - mean induction threshold
        σ_ω - standard deviation of induction threshold
        μ_ψ - mean initial concentration
        σ_ψ - standard deviation of initial concentration
        reltol - relative tolerance for the integration
        abstol - absolute tolerance for the integration
    output:
        the equilibrium germination response for the given parameters (normalized)
    """
    function gresp_inhibitor_dep_inducer_2_factors_eq_c_ex(ρₛ, dist_ξ, c_ex, c₀_cs, K_cI, K_cC, K_I, k, n, μ_γ, σ_γ, μ_ω, σ_ω, μ_ψ, σ_ψ; reltol=1e-4, abstol=1e-6)
        
        # Distributions
        dist_γ = Normal(μ_γ, σ_γ)
        dist_ω = Normal(μ_ω, σ_ω)
        μ_ψ_log = log(μ_ψ^2 / sqrt(σ_ψ^2 + μ_ψ^2))
        σ_ψ_log = sqrt(log(σ_ψ^2 / μ_ψ^2 + 1))
        dist_ψ = LogNormal(μ_ψ_log, σ_ψ_log)

        # Signal
        s_eq = c₀_cs ./ (K_cC .+ c₀_cs)

         function integrand(input)
            ξ, ψ = input
            V = 4/3 * π .* ξ^3
            ϕ = ρₛ .* V
            c_eq = ϕ .* ψ .+ (1 .- ϕ) .* c_ex
            s_mod = s_eq ./ (1 .+ (c_eq ./ K_I).^n)
            tail1 = 1 .- cdf(dist_γ, ϕ .+ (1 .- ϕ) .* c_ex ./ ψ)
            tail2 = cdf(dist_ω, s_mod .- k .* c_eq ./ (K_cI .+ c_eq))
            return tail1 * tail2 * pdf(dist_ξ, ξ) * pdf(dist_ψ, ψ)
        end

        return hcubature(integrand, [0.0, 0.0], [quantile(dist_ξ, 1-1e-8), quantile(dist_ψ, 1-1e-8)], reltol=reltol, abstol=abstol)[1]
    end

    """
    Compute the equilibrium germination response
    for an inhibitor-dependent inducer threshold and
    an additional inhibitor-dependent germination.
    inputs:
        ρₛ - spore density in spores/um^3
        dist_ξ - distribution of spore radii (LogNormal)
        c₀_cs - initial concentration of carbon source in M
        K_cI - half-saturation constant for the inhibitor threshold shift
        K_cC - half-saturation constant for the carbon source threshold shift
        k - inhibition strength over induction threshold
        μ_γ - mean inhibition threshold
        σ_γ - standard deviation of inhibition threshold
        μ_ω - mean induction threshold
        σ_ω - standard deviation of induction threshold
        μ_ψ - mean initial concentration
        σ_ψ - standard deviation of initial concentration
        reltol - relative tolerance for the integration
        abstol - absolute tolerance for the integration
    output:
        the equilibrium germination response for the given parameters (normalized)
    """
    function gresp_inhibitor_dep_inducer_thresh_2_factors_eq(ρₛ, dist_ξ, c₀_cs, K_cI, K_cC, k, μ_γ, σ_γ, μ_ω, σ_ω, μ_ψ, σ_ψ; reltol=1e-4, abstol=1e-6)
        
        # Distributions
        dist_γ = Normal(μ_γ, σ_γ)
        dist_ω = Normal(μ_ω, σ_ω)
        μ_ψ_log = log(μ_ψ^2 / sqrt(σ_ψ^2 + μ_ψ^2))
        σ_ψ_log = sqrt(log(σ_ψ^2 / μ_ψ^2 + 1))
        dist_ψ = LogNormal(μ_ψ_log, σ_ψ_log)

        # Signal
        s_eq = c₀_cs ./ (K_cC .+ c₀_cs)

        function integrand(input)
            ξ, ψ = input
            V = 4/3 * π .* ξ^3
            ϕ = ρₛ .* V
            c_eq = ψ .* ϕ
            tail1 = 1 .- cdf(dist_γ, ϕ)
            tail2 = cdf(dist_ω, s_eq .- k .* c_eq ./ (K_cI + c_eq))
            return tail1 * tail2 * pdf(dist_ξ, ξ) * pdf(dist_ψ, ψ)
        end

        return hcubature(integrand, [0.0, 0.0], [quantile(dist_ξ, 1-1e-8), quantile(dist_ψ, 1-1e-9)], reltol=reltol, abstol=abstol)[1]
    end

    """
    Compute the equilibrium germination response
    for an inhibitor-dependent inducer threshold and
    an additional inhibitor-dependent germination.
    inputs:
        ρₛ - spore density in spores/um^3
        dist_ξ - distribution of spore radii (LogNormal)
        c_ex - external concentration of the inducer in M
        c₀_cs - initial concentration of carbon source in M
        K_cI - half-saturation constant for the inhibitor threshold shift
        K_cC - half-saturation constant for the carbon source threshold shift
        k - inhibition strength over induction threshold
        μ_γ - mean inhibition threshold
        σ_γ - standard deviation of inhibition threshold
        μ_ω - mean induction threshold
        σ_ω - standard deviation of induction threshold
        μ_ψ - mean initial concentration
        σ_ψ - standard deviation of initial concentration
        reltol - relative tolerance for the integration
    output:
        the equilibrium germination response for the given parameters (normalized)
    """
    function gresp_inhibitor_dep_inducer_thresh_2_factors_eq_c_ex(ρₛ, dist_ξ, c_ex, c₀_cs, K_cI, K_cC, k, μ_γ, σ_γ, μ_ω, σ_ω, μ_ψ, σ_ψ; reltol=1e-4)
        
        # Distributions
        dist_γ = Normal(μ_γ, σ_γ)
        dist_ω = Normal(μ_ω, σ_ω)
        μ_ψ_log = log(μ_ψ^2 / sqrt(σ_ψ^2 + μ_ψ^2))
        σ_ψ_log = sqrt(log(σ_ψ^2 / μ_ψ^2 + 1))
        dist_ψ = LogNormal(μ_ψ_log, σ_ψ_log)

        # Signal
        s_eq = c₀_cs ./ (K_cC .+ c₀_cs)

         function integrand(input)
            ξ, ψ = input
            V = 4/3 * π .* ξ^3
            ϕ = ρₛ .* V
            c_eq = ϕ .* ψ .+ (1 .- ϕ) .* c_ex
            tail1 = 1 .- cdf(dist_γ, ϕ .+ (1 .- ϕ) .* c_ex ./ ψ)
            tail2 = cdf(dist_ω, s_eq .- k .* c_eq ./ (K_cI .+ c_eq))
            return tail1 * tail2 * pdf(dist_ξ, ξ) * pdf(dist_ψ, ψ)
        end

        return hcubature(integrand, [0.0, 0.0], [quantile(dist_ξ, 1-1e-8), quantile(dist_ψ, 1-1e-8)], reltol=reltol, abstol=1e-6)[1]
    end

    """
    Compute the equilibrium germination response
    for an inhibitor-dependent inducer signal and
    an additional inhibitor-dependent germination.
    inputs:
        ρₛ - spore density in spores/um^3
        dist_ξ - distribution of spore radii (LogNormal)
        c₀_cs - initial concentration of carbon source in M
        K_cC - half-saturation constant for the carbon source
        K_I - half-saturation constant for the inhibitor
        n - Hill coefficient for the inhibitor
        μ_γ - mean inhibition threshold
        σ_γ - standard deviation of inhibition threshold
        μ_ω - mean induction threshold
        σ_ω - standard deviation of induction threshold
        μ_ψ - mean initial concentration
        σ_ψ - standard deviation of initial concentration
        reltol - relative tolerance for the integration
        abstol - absolute tolerance for the integration
    output:
        the equilibrium germination response for the given parameters (normalized)
    """
    function gresp_inhibitor_dep_inducer_signal_2_factors_eq(t, ρₛ, dist_ξ, c₀_cs, K_cC, K_I, n, μ_γ, σ_γ, μ_ω, σ_ω, μ_ψ, σ_ψ; reltol=1e-4, abstol=1e-6)
        
        # Distributions
        dist_γ = Normal(μ_γ, σ_γ)
        dist_ω = Normal(μ_ω, σ_ω)
        μ_ψ_log = log(μ_ψ^2 / sqrt(σ_ψ^2 + μ_ψ^2))
        σ_ψ_log = sqrt(log(σ_ψ^2 / μ_ψ^2 + 1))
        dist_ψ = LogNormal(μ_ψ_log, σ_ψ_log)

        # Signal
        s_eq = c₀_cs ./ (K_cC .+ c₀_cs)

        function integrand(input)
            ξ, ψ = input
            V = 4/3 * π .* ξ^3
            ϕ = ρₛ .* V
            c_eq = ψ .* ϕ
            s_mod = s_eq ./ (1 .+ (c_eq ./ K_I).^n)
            tail1 = 1 .- cdf(dist_γ, ϕ)
            tail2 = cdf(dist_ω, s_mod)
            return tail1 * tail2 * pdf(dist_ξ, ξ) * pdf(dist_ψ, ψ)
        end

        return hcubature(integrand, [0.0, 0.0], [quantile(dist_ξ, 1-1e-8), quantile(dist_ψ, 1-1e-9)], reltol=reltol, abstol=abstol)[1]
    end

    """
    Compute the equilibrium germination response
    for an inhibitor-dependent inducer signal and
    an additional inhibitor-dependent germination.
    inputs:
        ρₛ - spore density in spores/um^3
        dist_ξ - distribution of spore radii (LogNormal)
        c_ex - external concentration of the inducer in M
        c₀_cs - initial concentration of carbon source in M
        K_cC - half-saturation constant for the carbon source
        K_I - half-saturation constant for the inhibitor
        n - Hill coefficient for the inhibitor
        μ_γ - mean inhibition threshold
        σ_γ - standard deviation of inhibition threshold
        μ_ω - mean induction threshold
        σ_ω - standard deviation of induction threshold
        μ_ψ - mean initial concentration
        σ_ψ - standard deviation of initial concentration
        reltol - relative tolerance for the integration
        abstol - absolute tolerance for the integration
    output:
        the equilibrium germination response for the given parameters (normalized)
    """
    function gresp_inhibitor_dep_inducer_signal_2_factors_eq_c_ex(ρₛ, dist_ξ, c_ex, c₀_cs, K_cC, K_I, n, μ_γ, σ_γ, μ_ω, σ_ω, μ_ψ, σ_ψ; reltol=1e-4, abstol=1e-6)
        
        # Distributions
        dist_γ = Normal(μ_γ, σ_γ)
        dist_ω = Normal(μ_ω, σ_ω)
        μ_ψ_log = log(μ_ψ^2 / sqrt(σ_ψ^2 + μ_ψ^2))
        σ_ψ_log = sqrt(log(σ_ψ^2 / μ_ψ^2 + 1))
        dist_ψ = LogNormal(μ_ψ_log, σ_ψ_log)

        # Signal
        s_eq = c₀_cs ./ (K_cC .+ c₀_cs)

         function integrand(input)
            ξ, ψ = input
            V = 4/3 * π .* ξ^3
            ϕ = ρₛ .* V
            c_eq = ϕ .* ψ .+ (1 .- ϕ) .* c_ex
            s_mod = s_eq ./ (1 .+ (c_eq ./ K_I).^n)
            tail1 = 1 .- cdf(dist_γ, ϕ .+ (1 .- ϕ) .* c_ex ./ ψ)
            tail2 = cdf(dist_ω, s_mod)
            return tail1 * tail2 * pdf(dist_ξ, ξ) * pdf(dist_ψ, ψ)
        end

        return hcubature(integrand, [0.0, 0.0], [quantile(dist_ξ, 1-1e-8), quantile(dist_ψ, 1-1e-8)], reltol=reltol, abstol=abstol)[1]
    end

    """
    Compute the equilibrium germination response
    for independent inhibition and induction.
    inputs:
        ρₛ - spore density in spores/um^3
        dist_ξ - distribution of spore radii (LogNormal)
        c₀_cs - initial concentration of carbon source in M
        K_cC - half-saturation constant for the carbon source
        μ_γ - mean inhibition threshold
        σ_γ - standard deviation of inhibition threshold
        μ_ω - mean induction threshold
        σ_ω - standard deviation of induction threshold
        reltol - relative tolerance for the integration
    output:
        the equilibrium germination response for the given parameters (normalized)
    """
    function gresp_independent_eq(ρₛ, dist_ξ, c₀_cs, K_cC, μ_γ, σ_γ, μ_ω, σ_ω; reltol=1e-4)

        # Distributions
        dist_γ = Normal(μ_γ, σ_γ)
        dist_ω = Normal(μ_ω, σ_ω)

        # Signal
        s_eq = c₀_cs ./ (K_cC .+ c₀_cs)
        tail2 = cdf(dist_ω, s_eq)

        function integrand(ξ)
            V = 4/3 * π .* ξ^3
            ϕ = ρₛ .* V
            tail1 = 1 .- cdf(dist_γ, ϕ)
            return tail1 * tail2 * pdf(dist_ξ, ξ)
        end

        return quadgk(x -> integrand(x), 0.0, Inf, rtol=reltol)[1]
    end

    """
    Compute the equilibrium germination response
    for independent inhibition and induction.
    inputs:
        ρₛ - spore density in spores/um^3
        dist_ξ - distribution of spore radii (LogNormal)
        c_ex - external concentration of the inducer in M
        c₀_cs - initial concentration of carbon source in M
        K_cC - half-saturation constant for the carbon source
        μ_γ - mean inhibition threshold
        σ_γ - standard deviation of inhibition threshold
        μ_ω - mean induction threshold
        σ_ω - standard deviation of induction threshold
        μ_ψ - mean initial concentration
        σ_ψ - standard deviation of initial concentration
        reltol - relative tolerance for the integration
    output:
        the equilibrium germination response for the given parameters (normalized)
    """
    function gresp_independent_eq_c_ex(ρₛ, dist_ξ, c_ex, c₀_cs, K_cC, μ_γ, σ_γ, μ_ω, σ_ω, μ_ψ, σ_ψ; reltol=1e-4)

        # Distributions
        dist_γ = Normal(μ_γ, σ_γ)
        dist_ω = Normal(μ_ω, σ_ω)
        μ_ψ_log = log(μ_ψ^2 / sqrt(σ_ψ^2 + μ_ψ^2))
        σ_ψ_log = sqrt(log(σ_ψ^2 / μ_ψ^2 + 1))
        dist_ψ = LogNormal(μ_ψ_log, σ_ψ_log)

        # Signal
        s_eq = c₀_cs ./ (K_cC .+ c₀_cs)
        tail2 = cdf(dist_ω, s_eq)

        function integrand(inputs)
            ξ, ψ = inputs
            V = 4/3 * π .* ξ^3
            ϕ = ρₛ .* V
            tail1 = 1 .- cdf(dist_γ, ϕ .+ (1 .- ϕ) .* c_ex ./ ψ)
            return tail1 * tail2 * pdf(dist_ξ, ξ) * pdf(dist_ψ, ψ)
        end

        return hcubature(integrand, [0.0, 0.0], [quantile(dist_ξ, 1-1e-8), quantile(dist_ψ, 1-1e-8)], reltol=reltol, abstol=1e-6)[1]
    end

    # ===== FEEDBACK MODELS ===== #

    # ---- Predeclare a typed parameter struct ----
    @with_kw struct PermParams
        A::Float64
        Vₛ::Float64
        V_out::Float64
        V_ps::Float64
        Pₛ_I::Float64
        Pₛ_C::Float64
        K_fs::Vector{Union{Float64, Nothing}}
        f_maxs::Vector{Float64}
        c₀_cs::Float64
        n::Union{Float64, Nothing}
    end

    """
    ODE function for inducer-dependent
    cell wall permeability.
    """
    function ode_inducer_dependent_perm!(du, u, p::PermParams, t)
        cinI, cinC, coutI = u

        Δg = p.f_maxs[1] * cinC / (p.K_fs[2] + cinC) # f_maxs[1] is s, K_fs[2] is K_cC
        exponent = -(1 + Δg) * 1e-3
        PmaxA = 1000 * p.A # limit permeability to Pmax = 1000 μm/s
        rateI = PmaxA * (1 - exp(exponent * p.Pₛ_I)) 
        rateC = PmaxA * (1 - exp(exponent * p.Pₛ_C))

        diffI = cinI - coutI
        diffC = cinC - p.c₀_cs

        du[1] = -(rateI / p.Vₛ) * diffI
        du[2] = -(rateC / p.V_ps) * diffC
        du[3] = (rateI / p.V_out) * diffI
    end

    """
    ODE function for inhibitor-dependent
    cell wall permeability.
    """
    function ode_inhibitor_dependent_perm!(du, u, p::PermParams, t)
        cinI, cinC, coutI = u

        cinI = max(cinI, 0.0)
        cinC = max(cinC, 0.0)
        
        exponent = p.f_maxs[1] * cinI / (p.K_fs[1] + cinI) # f_maxs[1] is b, K_fs[1] is K_cI
        g = exp(-exponent) * p.A
        rateI = g * p.Pₛ_I
        rateC = g * p.Pₛ_C

        diffI = cinI - coutI
        diffC = cinC - p.c₀_cs
        
        du[1] = -(rateI / p.Vₛ) * diffI
        du[2] = -(rateC / p.V_ps) * diffC
        du[3] = (rateI / p.V_out) * diffI
    end

    """
    ODE function for inducer-dependent
    cell wall permeability and inhibitor-dependent signal.
    """
    function ode_inducer_dependent_perm_inhibitor_dependent_signal!(du, u, p::PermParams, t)
        cinI, cinC, coutI = u

        cinI = max(cinI, 0.0)
        
        s = p.f_maxs[1] / (1 + (cinI / p.K_fs[3]) ^ p.n) # f_maxs[1] is s, K_fs[3] is K_I
        # g = (1 + s * cinC / (p.K_fs[2] + cinC)) * p.A # K_fs[2] is K_cC
        Δg = s * cinC / (p.K_fs[2] + cinC) # K_fs[2] is K_cC
        exponent = -(1 + Δg) * 1e-3
        PmaxA = 1000 * p.A # limit permeability to Pmax = 1000 μm/s
        rateI = PmaxA * (1 - exp(exponent * p.Pₛ_I)) 
        rateC = PmaxA * (1 - exp(exponent * p.Pₛ_C))
        # if rateI / p.Vₛ > 1e6 || rateC / p.V_ps > 1e6 ||  rateI / p.V_out > 1e6 println("eeeeek!") end

        diffI = cinI - coutI
        diffC = cinC - p.c₀_cs
        
        du[1] = -rateI / p.Vₛ * diffI
        du[2] = -rateC / p.V_ps * diffC
        du[3] = rateI / p.V_out * diffI
    end

    """
    ODE function for inducer- and inhibitor-dependent
    cell wall permeability.
    """
    function ode_inducer_and_inhibitor_dependent_perm!(du, u, p::PermParams, t)
        
        cinI, cinC, coutI = u
        
        Δg_C = p.f_maxs[2] * cinC / (p.K_fs[2] + cinC) # f_maxs[2] is s, K_fs[2] is K_cC
        Δg_I = p.f_maxs[1] * cinI / (p.K_fs[1] + cinI) # f_maxs[1] is b, K_fs[2] is K_cI
        perturb_C = (1 + Δg_C)
        exponent = -perturb_C * exp(-Δg_I / perturb_C) * 1e-3
        PmaxA = 1000 * p.A # limit permeability to Pmax = 1000 μm/s
        rateI = PmaxA * (1 - exp(exponent * p.Pₛ_I))
        rateC = PmaxA * (1 - exp(exponent * p.Pₛ_C))

        diffI = cinI - coutI
        diffC = cinC - p.c₀_cs

        du[1] = -(rateI / p.Vₛ) * diffI
        du[2] = -(rateC / p.V_ps) * diffC
        du[3] = (rateI / p.V_out) * diffI
    end

    """
    ODE function for inducer- and inhibitor-dependent
    cell wall permeability and an inihibitor-dependent
    induction signal.
    """
    function ode_inducer_and_inhibitor_dependent_perm_inhibitor_dependent_signal!(du, u, p::PermParams, t)
        
        cinI, cinC, coutI = u

        cinI = max(cinI, 0.0)
        cinC = max(cinC, 0.0)
        
        inh_factor = (1 + (cinI / p.K_fs[3]) ^ p.n) # K_fs[3] is K_I
        Δg_C = p.f_maxs[2] * cinC / ((p.K_fs[2] + cinC) * inh_factor) # f_maxs[2] is s, K_fs[2] is K_cC
        Δg_I = p.f_maxs[1] * cinI / (p.K_fs[1] + cinI) # f_maxs[1] is b, K_fs[2] is K_cI
        perturb_C = (1 + Δg_C)
        exponent = -perturb_C * exp(-Δg_I / perturb_C) * 1e-3
        PmaxA = 1000 * p.A # limit permeability to Pmax = 1000 μm/s
        rateI = PmaxA * (1 - exp(exponent * p.Pₛ_I))
        rateC = PmaxA * (1 - exp(exponent * p.Pₛ_C))

        diffI = cinI - coutI
        diffC = cinC - p.c₀_cs

        du[1] = -(rateI / p.Vₛ) * diffI
        du[2] = -(rateC / p.V_ps) * diffC
        du[3] = (rateI / p.V_out) * diffI
    end
    
    """
    Computes simple germination criterion
    with regard to the inhibition threshold
    inputs:
        cins - inhibitor/inducer concentrations (M)
        thresholds - inhibitor thresholds
        ks - scaling factors
        K_fs - half-saturation constants (M)
    """
    function thresh_criterion_inhibitor(cins, thresholds, ks=nothing, K_fs=nothing, n=nothing)
        return cins[1, :] .< thresholds[1, :]
    end

    """
    Computes simple germination criterion
    with regard to the induction threshold
    inputs:
        cins - inhibitor/inducer concentrations (M)
        thresholds - inducer thresholds
        ks - scaling factors
        K_fs - inducer half-saturation constant as 1-element vector (M)
    """
    function thresh_criterion_inducer(cins, thresholds, ks, K_fs, n=nothing)
        return cins[2, :] ./ (cins[2, :] .+ K_fs[2]) .> thresholds[1, :]
    end

    """
    Computes simple germination criterion
    with regard to the induction and inhibition thresholds
    inputs:
        cins - inhibitor/inducer concentrations (M)
        thresholds - inhibitor thresholds
        ks - scaling factors
        K_fs - inducer half-saturation constant as 1-element vector (M)
    """
    function thresh_criterion_combined(cins, thresholds, ks, K_fs, n=nothing)
        return (cins[1, :] .< thresholds[1, :]) .* (cins[2, :] ./ (cins[2, :] .+ K_fs[2]) .> thresholds[2, :])
    end

    """
    Computes inhibitor-dependent germination criterion with
    a threshold shifting signal defined by the inducer concentration.
    inputs:
        cins - inhibitor/inducer concentrations (M)
        thresholds - inhibitor thresholds
        ks - scaling factors
        K_fs - inducer half-saturation constant as 1-element vector (M)
    """
    function thresh_criterion_inhibitor_shift(cins, thresholds, ks, K_fs, n=nothing)
        thresh_bias_signal = ks[1] .* cins[2, :]  ./ (cins[2, :] .+ K_fs[2])
        return cins[1, :] .< thresholds[1, :] .+ thresh_bias_signal
    end

    """
    Computes 2-factor germination criterion with
    a threshold shifting signal defined by the inducer concentration.
    inputs:
        cins - inhibitor/inducer concentrations (M)
        thresholds - inhibitor thresholds
        ks - scaling factors
        K_fs - inducer half-saturation constant as 1-element vector (M)
    """
    function thresh_criterion_combined_inhibitor_shift(cins, thresholds, ks, K_fs, n=nothing)
        inducer_signal = cins[2, :]  ./ (cins[2, :] .+ K_fs[2])
        thresh_bias_signal = ks[1] .* inducer_signal
        return (cins[1, :] .< thresholds[1, :] .+ thresh_bias_signal) .* (inducer_signal .> thresholds[2, :])
    end

    """
    Computes inhibitor-dependent germination criterion with
    a threshold shifting signal defined by the inducer concentration.
    inputs:
        cins - inhibitor/inducer concentrations (M)
        thresholds - inhibitor thresholds
        ks - scaling factors
        K_fs - half-saturation constants (M)
    """
    function thresh_criterion_inducer_shift(cins, thresholds, ks, K_fs, n=nothing)
        thresh_bias_signal = ks[1] .* cins[1, :]  ./ (cins[1, :] .+ K_fs[1])
        return cins[2, :] ./ (cins[2, :] .+ K_fs[2]) .> thresholds[1, :] .+ thresh_bias_signal
    end

    """
    Computes inhibitor-dependent germination criterion with
    a threshold shifting signal defined by the inducer concentration.
    inputs:
        cins - inhibitor/inducer concentrations (M)
        thresholds - inhibitor thresholds
        ks - scaling factors
        K_fs - half-saturation constants (M)
    """
    function thresh_criterion_combined_inducer_shift(cins, thresholds, ks, K_fs, n=nothing)
        thresh_bias_signal = ks[1] .* cins[1, :]  ./ (cins[1, :] .+ K_fs[1])
        return (cins[1, :] .< thresholds[1, :]) .* (cins[2, :] ./ (cins[2, :] .+ K_fs[2]) .> thresholds[2, :] .+ thresh_bias_signal)
    end

    """
    Computes inhibitor-dependent germination criterion with
    an induction signal modulated by the inhibitor concentration.
    inputs:
        cins - inhibitor/inducer concentrations (M)
        thresholds - inhibitor thresholds
        ks - scaling factors
        K_fs - half-saturation constants (M)
        n - inhibition Hill exponent
    """
    function thresh_criterion_inducer_signal(cins, thresholds, ks, K_fs, n)
        signal_inhibition = 1 ./ (1 .+ (cins[1, :] ./ K_fs[3]) .^ n)  # K_fs[3] is K_I
        return signal_inhibition .* cins[2, :] ./ (cins[2, :] .+ K_fs[2]) .> thresholds[1, :]
    end

    """
    Computes inhibitor-dependent germination criterion with
    an induction signal modulated by the inducer concentration.
    inputs:
        cins - inhibitor/inducer concentrations (M)
        thresholds - inhibitor thresholds
        ks - scaling factors
        K_fs - half-saturation constants (M)
        n - inhibition Hill exponent
    """
    function thresh_criterion_combined_inducer_signal(cins, thresholds, ks, K_fs, n)
        signal_inhibition = 1 ./ (1 .+ (cins[1, :] ./ K_fs[3]) .^ n)  # K_fs[3] is K_I
        return (cins[1, :] .< thresholds[1, :] ) .* (signal_inhibition .* cins[2, :] ./ (cins[2, :] .+ K_fs[2]) .> thresholds[1, :])
    end

    """
    Computes simple germination criterion
    with regard to the induction and inhibition thresholds
    inputs:
        cins - inhibitor/inducer concentrations (M)
        thresholds - inhibitor thresholds
        ks - scaling factors
        K_fs - inducer half-saturation constant as 1-element vector (M)
    """
    function thresh_criterion_combined_shift(cins, thresholds, ks, K_fs, n=nothing)
        thresh_bias_signal_I = ks[1] .* cins[1, :]  ./ (cins[1, :] .+ K_fs[1])
        thresh_bias_signal_C = ks[2] .* cins[2, :]  ./ (cins[2, :] .+ K_fs[2])
        return (cins[1, :] .< thresholds[1, :] .+ thresh_bias_signal_C) .* (cins[2, :] ./ (cins[2, :] .+ K_fs[2]) .> thresholds[2, :] .+ thresh_bias_signal_I)
    end

    """
    Computes inhibitor-dependent germination criterion with
    an induction signal modulated by the inhibitor concentration
    and the inhibition threshold shifted by the inducer signal.
    inputs:
        cins - inhibitor/inducer concentrations (M)
        thresholds - inhibitor thresholds
        ks - scaling factors
        K_fs - half-saturation constants (M)
        n - inhibition Hill exponent
    """
    function thresh_criterion_inhibitor_signal_shift(cins, thresholds, ks, K_fs, n)
        cins .= max.(cins, 0.0)
        thresh_bias_signal = ks[1] .* cins[2, :] ./ (cins[2, :] .+ K_fs[2]) ./ (1 .+ (cins[1, :] ./ K_fs[3]) .^ n)  # K_fs[3] is K_I
        return cins[1, :] .< thresholds[1, :] .+ thresh_bias_signal
    end

    """
    Computes 2-factor germination criterion with
    an induction signal modulated by the inhibitor concentration
    and the inhibition threshold shifted by the inducer signal.
    inputs:
        cins - inhibitor/inducer concentrations (M)
        thresholds - inhibitor thresholds
        ks - scaling factors
        K_fs - half-saturation constants (M)
        n - inhibition Hill exponent
    """
    function thresh_criterion_combined_inhibitor_signal_shift(cins, thresholds, ks, K_fs, n)
        cins .= max.(cins, 0.0)
        signal = cins[2, :] ./ (cins[2, :] .+ K_fs[2]) ./ (1 .+ (cins[1, :] ./ K_fs[3]) .^ n)  # K_fs[3] is K_I
        return (cins[1, :] .< thresholds[1, :] .+ ks[1] .* signal) .* (signal .> thresholds[2, :])
    end

    """
    Computes inhibitor-dependent germination criterion with
    an induction signal modulated by the inhibitor concentration
    and the induction threshold shifted by the inhibitor concentration.
    inputs:
        cins - inhibitor/inducer concentrations (M)
        thresholds - inhibitor thresholds
        ks - scaling factors
        K_fs - half-saturation constants (M)
        n - inhibition Hill exponent
    """
    function thresh_criterion_inducer_signal_shift(cins, thresholds, ks, K_fs, n)
        cins .= max.(cins, 0.0)
        signal_inhibition = 1 ./ (1 .+ (cins[1, :] ./ K_fs[3]) .^ n) # K_fs[3] is K_I
        thresh_bias_signal = ks[1] .* cins[1, :]  ./ (cins[1, :] .+ K_fs[1])
        return signal_inhibition .* cins[2, :] ./ (cins[2, :] .+ K_fs[2]) .> thresholds[1, :] .+ thresh_bias_signal
    end

    """
    Computes 2-factor germination criterion with
    an induction signal modulated by the inhibitor concentration
    and the induction threshold shifted by the inhibitor concentration.
    inputs:
        cins - inhibitor/inducer concentrations (M)
        thresholds - inhibitor thresholds
        ks - scaling factors
        K_fs - half-saturation constants (M)
        n - inhibition Hill exponent
    """
    function thresh_criterion_combined_inducer_signal_shift(cins, thresholds, ks, K_fs, n)
        cins .= max.(cins, 0.0)
        signal_inhibition = 1 ./ (1 .+ (cins[1, :] ./ K_fs[3]) .^ n)  # K_fs[3] is K_I
        thresh_bias_signal = ks[1] .* cins[1, :]  ./ (cins[1, :] .+ K_fs[1])
        return (cins[1, :] .< thresholds[1, :]) .* (signal_inhibition .* cins[2, :] ./ (cins[2, :] .+ K_fs[2]) .> thresholds[2, :] .+ thresh_bias_signal)
    end

    """
    Computes 2-factor germination criterion with
    an induction signal modulated by the inhibitor concentration
    and both thresholds shifted by the respective signals.
    inputs:
        cins - inhibitor/inducer concentrations (M)
        thresholds - inhibitor thresholds
        ks - scaling factors
        K_fs - half-saturation constants (M)
        n - inhibition Hill exponent
    """
    function thresh_criterion_combined_inhibitor_shift_inducer_signal_shift(cins, thresholds, ks, K_fs, n)
        cins .= max.(cins, 0.0)
        signal_inhibition = 1 ./ (1 .+ (cins[1, :] ./ K_fs[3]) .^ n)  # K_fs[3] is K_I
        thresh_bias_signal_I = ks[1] .* cins[1, :]  ./ (cins[1, :] .+ K_fs[1])
        thresh_bias_signal_C = ks[2] .* cins[2, :]  ./ (cins[2, :] .+ K_fs[2])
        return (cins[1, :] .< thresholds[1, :] .+ thresh_bias_signal_C) .* (signal_inhibition .* cins[2, :] ./ (cins[2, :] .+ K_fs[2]) .> thresholds[2, :] .+ thresh_bias_signal_I)
    end

    """
    Generic function for computing the germination response
    for inducer-dependent cell wall permeability
    inputs:
        ode_func - ODE function to integrate
        thresh_func - threshold criterion function
        sobol_pts - normalized Sobol samples
        times - integration time frames in seconds
        geom_samples - geometric samples corresponding to sobol pts [samples_A, samples_Vₛ, samples_V_out, samples_V_ps]
        c₀_cs - initial concentration of carbon source in M
        f_maxs - maximum concentration-related signal strengths (inhibitory and/or inductive)
        Pₛ_I - permeation constant for the inhibitor in um/s
        Pₛ_C - permeation constant for the carbon source in um/s
        K_fs - half-saturation constants (inhibitory and/or inductive)
        μ_γ - mean inhibition threshold
        σ_γ - standard deviation of inhibition threshold
        μ_ψ - mean initial concentration
        σ_ψ - standard deviation of initial concentration
        μs_thresh - means of thresholds (γ and/or ω)
        σs_thresh - standard deviations of thresholds (γ and/or ω)
        ks - scaling factors for threshold shifts (optional)
        n - inhibition Hill exponent (optional)
    output:
        the germination response for the given parameters (normalized)
    """
    function gresp_feedback(ode_func, thresh_func, sobol_pts, times, geom_samples, c₀_cs, f_maxs, Pₛ_I, Pₛ_C, K_fs, μ_ψ, σ_ψ, μs_thresh, σs_thresh; ks=nothing, n=nothing)
        
        # Unpack geometric samples
        samples_A, samples_Vₛ, samples_V_out, samples_V_ps = geom_samples

        μ_ψ_log = log(μ_ψ^2 / sqrt(σ_ψ^2 + μ_ψ^2))
        σ_ψ_log = sqrt(log(σ_ψ^2 / μ_ψ^2 + 1))
        dist_ψ = LogNormal(μ_ψ_log, σ_ψ_log)
        samples_ψ = clamp_inplace!(quantile(dist_ψ, sobol_pts[3,:]))
        
        n_thresh = length(μs_thresh)
        n_samp = size(sobol_pts, 2)
        n_times = length(times)

        samples_thresh = Matrix{Float64}(undef, n_thresh, n_samp)
        @inbounds for i in 1:n_thresh
            dist_thresh = Normal(μs_thresh[i], σs_thresh[i])
            samples_thresh[i, :] .= quantile(dist_thresh, sobol_pts[3 + i,:])
        end

        # Template problem
        p0 = PermParams(
            samples_A[1], samples_Vₛ[1], samples_V_out[1], samples_V_ps[1],
            Pₛ_I, Pₛ_C, K_fs, f_maxs, c₀_cs, n
        )
        u0 = [μ_ψ, 0.0, 0.0]
        tspan = (0.0, maximum(times))
        prob = ODEProblem(ode_func, u0, tspan, p0)

        # Ensemble integration function
        function prob_func(prob, i, repeat)
            p_new = PermParams(
                samples_A[i], samples_Vₛ[i], samples_V_out[i], samples_V_ps[i],
                Pₛ_I, Pₛ_C, K_fs, f_maxs, c₀_cs, n
            )
            remake(prob; u0 = [samples_ψ[i], 0.0, 0.0], p = p_new)
        end

        # condition(u, t, integrator) = any(u .< 0) || any(isnan.(u))
        # affect!(integrator) = error("Invalid state at t=$(integrator.t): u=$(integrator.u)")

        # Select solver
        if ode_func in [ode_inducer_dependent_perm_inhibitor_dependent_signal!, ode_inducer_and_inhibitor_dependent_perm_inhibitor_dependent_signal!]
            solver = KenCarp5()#Rodas5()
        else
            solver = Rosenbrock23()
        end

        condition(u, t, integrator) = any(u .< 0)
        affect!(integrator) = (integrator.u .= max.(integrator.u, 0.0))

        cb = DiscreteCallback(condition, affect!)

        unstable_check(dt, u, p, t) = any(isnan, u) || any(isinf, u) || dt < 1e-16

        # Run ODE ensembles
        ep = EnsembleProblem(prob; prob_func=prob_func)
        # sols = solve(ep, AutoTsit5(Rosenbrock23()), EnsembleThreads(), trajectories=n_samp, saveat=times, abstol=1e-6, reltol=1e-6, callback=cb, unstable_check=unstable_check)
        sols = solve(ep, AutoTsit5(solver), EnsembleThreads(), trajectories=n_samp, saveat=times, abstol=1e-6, reltol=1e-6, maxiters=1e8, callback=cb, unstable_check=unstable_check)#, verbose=true)

        # Evaluate fraction germinated
        n_times = length(times)
        germinated = Vector{Float64}(undef, n_times)
        c_in = Array{Float64}(undef, 2, n_samp)

        germ_term = zeros(Bool, n_samp) # termination tags
        @inbounds for (ti, t) in enumerate(times)
            for i in 1:n_samp
                u = sols[i](t)
                c_in[1, i] = max(u[1], 0.0)
                c_in[2, i] = max(u[2], 0.0)
            end
            gmask = thresh_func(c_in, samples_thresh, ks, K_fs, n) .| germ_term # Make sure once germinated does not oscillate
            germ_term = gmask

            germinated[ti] = mean(gmask)
        end
        
        return germinated
    end
    
end