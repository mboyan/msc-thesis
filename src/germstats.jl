module GermStats
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

    export compute_germination_response

    export germ_response_independent_factors_gh
    export germ_response_inducer_dep_inhibitor_thresh_gh
    export germ_response_inducer_dep_inhibitor_perm_gh
    export germ_response_inducer_dep_inhibitor_gh
    export germ_response_inhibitor_dep_inducer_thresh_gh
    export germ_response_inhibitor_dep_inducer_signal_gh
    export germ_response_inhibitor_dep_inducer_gh
    export germ_response_inducer_dep_inhibitor_thresh_2_factor_gh
    export germ_response_inducer_dep_inhibitor_perm_2_factor_gh
    export germ_response_inducer_dep_inhibitor_2_factor_gh
    export germ_response_inhibitor_dep_inducer_thresh_2_factor_gh
    export germ_response_inhibitor_dep_inducer_signal_2_factor_gh
    export germ_response_inhibitor_dep_inducer_2_factor_gh
    export germ_response_inducer_thresh_var_perm_gh
    export germ_response_inducer_var_perm_gh
    export germ_response_independent_factors_var_perm_gh
    export germ_response_inducer_thresh_2_factors_var_perm_gh
    export germ_response_inducer_signal_2_factors_var_perm_gh
    export germ_response_inducer_2_factors_var_perm_gh
    export germ_response_inh_dep_ind_signal_ind_dep_inh_thresh_gh

    export germ_response_inducer_dep_inhibitor_eq
    export germ_response_inducer_dep_inhibitor_eq_c_ex
    export germ_response_inhibitor_dep_inducer_thresh_2_factors_eq
    export germ_response_inhibitor_dep_inducer_thresh_2_factors_eq_c_ex
    export germ_response_inhibitor_dep_inducer_signal_2_factors_eq
    export germ_response_inhibitor_dep_inducer_signal_2_factors_eq_c_ex
    export germ_response_inhibitor_dep_inducer_2_factors_eq
    export germ_response_inhibitor_dep_inducer_2_factors_eq_c_ex
    export germ_response_independent_eq
    export germ_response_independent_eq_c_ex

    export germ_response_feedback_perm

    export clamp_inplace!

    export ode_inducer_dependent_perm!
    export ode_inhibitor_dependent_perm!
    export ode_inducer_dependent_perm_inhibitor_dependent_signal!
    export ode_inducer_and_inhibitor_dependent_perm!

    export thresh_criterion_inhibitor
    export thresh_criterion_inducer
    export thresh_criterion_combined
    export thresh_criterion_inhibitor_shift
    export thresh_criterion_combined_inhibitor_shift
    export thresh_criterion_inducer_shift
    export thresh_criterion_combined_inducer_shift


    function clamp_inplace!(arr, eps=1e-12)
        @inbounds for i in eachindex(arr)
            if arr[i] < eps
                arr[i] = eps
            end
        end
        return arr
    end
    
    
    function inducer_concentration(c_out, t, Pₛ, A, V_cw)
        """
        Compute the concentration of carbon source in the cell wall.
        inputs:
            c_in (float) - the initial concentration at the spore
            c_out (float) - the initial external concentration
            t (float) - time
            Pₛ (float) - the hydrophobin layer permeation constant
            A (float) - the surface area of the spore
            V_cw (float) - the volume of the polysaccharide layer pores
        """
        τ = V_cw ./ (A * Pₛ)
        c = c_out .* (1 .- exp.(-t / τ))
        return c
    end

    # ===== GAUSS-HERMITE APPROXIMATIONS =====
    function compute_germination_response(model_type, times, ρₛ, prms; n_nodes=nothing)
        """
        Generic wrapper function to compute the germination response.
        inputs:
            model_type (String): model type to fit
            times (Vector{Float64}): time points to compute the germination response
            ρₛ (float) - spore density in spores/μm^3
            n_nodes (int) - number of Gauss-Hermite nodes to use
            prms (Dict) - additional parameters for the germination response function
            n_nodes (int) - number of Gauss-Hermite nodes to use
        """

        @argcheck model_type in ["independent",
                                "inhibitor", "inhibitor_thresh", "inhibitor_perm",
                                "inducer", "inducer_thresh", "inducer_signal",
                                "combined_inhibitor", "combined_inhibitor_thresh", "combined_inhibitor_perm",
                                "combined_inducer", "combined_inducer_thresh", "combined_inducer_signal",
                                "inhibitor_thresh_inducer_signal", "combined_inhibitor_thresh_inducer_signal", # BC
                                "special_inducer", "special_independent", "special_combined", "special_thresh", "special_signal",
                                "feedback_inhibitor_inducer_perm", "feedback_combined_inducer_perm",
                                "feedback_inducer_inhibitor_perm", "feedback_combined_inhibitor_perm",
                                "feedback_inhibitor_inducer_perm_thresh", "feedback_combined_inducer_perm_thresh",
                                "feedback_inhibitor_inducer_perm_inhibitor_signal", "feedback_inducer_inducer_perm_inhibitor_signal", "feedback_combined_inducer_perm_inhibitor_signal",
                                "feedback_inhibitor_inhibitor_inducer_perm", "feedback_inducer_inhibitor_inducer_perm", "feedback_combined_inhibitor_inducer_perm",
                                "feedback_inducer_inhibitor_thresh_inducer_perm", "feedback_combined_inhibitor_thresh_inducer_perm"]

        # Determine number of nodes depending on the integral dimension (if not specified)
        if isnothing(n_nodes)
            if model_type in ["independent", "inhibitor", "inhibitor_thresh", "inhibitor_perm",
                                "combined_inhibitor", "combined_inhibitor_thresh", "combined_inhibitor_perm"]
                n_nodes = 36 # 2D integral
            elseif model_type in ["inducer", "inducer_thresh", "inducer_signal", 
                                "combined_inducer", "combined_inducer_thresh", "combined_inducer_signal", "special_independent",
                                "inhibitor_thresh_inducer_signal", "combined_inhibitor_thresh_inducer_signal"]
                n_nodes = 10 # 3D integral
            elseif model_type in ["special_inducer", "special_combined", "special_thresh", "special_signal"]
                n_nodes = 6 # 4D integral
            elseif model_type in ["feedback_inhibitor_inducer_perm", "feedback_combined_inducer_perm", "feedback_inducer_inhibitor_perm", "feedback_combined_inhibitor_perm",
                                "feedback_inhibitor_inducer_perm_thresh", "feedback_combined_inducer_perm_thresh",
                                "feedback_inhibitor_inducer_perm_inhibitor_signal", "feedback_inducer_inducer_perm_inhibitor_signal", "feedback_combined_inducer_perm_inhibitor_signal",
                                "feedback_inhibitor_inhibitor_inducer_perm", "feedback_inducer_inhibitor_inducer_perm", "feedback_combined_inhibitor_inducer_perm",
                                "feedback_inducer_inhibitor_thresh_inducer_perm", "feedback_combined_inhibitor_thresh_inducer_perm"]
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
                            "inhibitor_thresh_inducer_signal", "combined_inhibitor_thresh_inducer_signal"]
            W3 = reshape(hw, n_nodes,1,1) .* reshape(hw, 1,n_nodes,1) .* reshape(hw, 1,1,n_nodes)
        elseif model_type in ["special_inducer", "special_combined", "special_thresh", "special_signal"]
            W4 = reshape(hw, n_nodes,1,1,1) .* reshape(hw, 1,n_nodes,1,1) .* reshape(hw, 1,1,n_nodes,1) .* reshape(hw, 1,1,1,n_nodes)
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
            germ_response = [germ_response_independent_factors_gh(u, W, t, ρₛ, prms[:c₀_cs], prms[:d_hp], ξ2, κ2, prms[:Pₛ], prms[:Pₛ_cs], prms[:K_cC], prms[:μ_γ], prms[:σ_γ], prms[:μ_ω], prms[:σ_ω]) for t in times]
            
        elseif model_type == "inhibitor"
            germ_response = [germ_response_inducer_dep_inhibitor_gh(u, W, t, ρₛ, prms[:c₀_cs], prms[:d_hp], ξ2, κ2, prms[:Pₛ], prms[:Pₛ_cs], prms[:K_cC], prms[:k_C], prms[:μ_γ], prms[:σ_γ]) for t in times]
            
        elseif model_type == "inhibitor_thresh" # B
            germ_response = [germ_response_inducer_dep_inhibitor_thresh_gh(u, W, t, ρₛ, prms[:c₀_cs], prms[:d_hp], ξ2, κ2, prms[:Pₛ], prms[:Pₛ_cs], prms[:k_C], prms[:K_cC], prms[:μ_γ], prms[:σ_γ]) for t in times]
            
        elseif model_type == "inhibitor_perm"
            germ_response = [germ_response_inducer_dep_inhibitor_perm_gh(u, W, t, ρₛ, prms[:c₀_cs], prms[:d_hp], ξ2, κ2, prms[:Pₛ], prms[:Pₛ_cs], prms[:K_cC], prms[:μ_γ], prms[:σ_γ]) for t in times]
            
        elseif model_type == "inducer" # CE
            germ_response = [germ_response_inhibitor_dep_inducer_gh(u, W3, t, ρₛ, prms[:c₀_cs], prms[:d_hp], ξ2, κ2, prms[:Pₛ], prms[:Pₛ_cs], prms[:k_I], prms[:K_cC], prms[:K_I], prms[:n], prms[:μ_ω], prms[:σ_ω], prms[:μ_ψ], prms[:σ_ψ]) for t in times]

        elseif model_type == "inducer_thresh" # E
            germ_response = [germ_response_inhibitor_dep_inducer_thresh_gh(u, W3, t, ρₛ, prms[:c₀_cs], prms[:d_hp], ξ2, κ2, prms[:Pₛ], prms[:Pₛ_cs], prms[:K_cC], prms[:k_I], prms[:μ_ω], prms[:σ_ω], prms[:μ_ψ], prms[:σ_ψ]) for t in times]
            
        elseif model_type == "inducer_signal" # C
            germ_response = [germ_response_inhibitor_dep_inducer_signal_gh(u, W3, t, ρₛ, prms[:c₀_cs], prms[:d_hp], ξ2, κ2, prms[:Pₛ], prms[:Pₛ_cs], prms[:K_cC], prms[:K_I], prms[:n], prms[:μ_ω], prms[:σ_ω], prms[:μ_ψ], prms[:σ_ψ]) for t in times]
            
        elseif model_type == "combined_inhibitor"
            germ_response = [germ_response_inducer_dep_inhibitor_2_factor_gh(u, W, t, ρₛ, prms[:c₀_cs], prms[:d_hp], ξ2, κ2, prms[:Pₛ], prms[:Pₛ_cs], prms[:K_cC], prms[:k_C], prms[:μ_γ], prms[:σ_γ], prms[:μ_ω], prms[:σ_ω]) for t in times]
            
        elseif model_type == "combined_inhibitor_thresh" # B
            germ_response = [germ_response_inducer_dep_inhibitor_thresh_2_factor_gh(u, W, t, ρₛ, prms[:c₀_cs], prms[:d_hp], ξ2, κ2, prms[:Pₛ], prms[:Pₛ_cs], prms[:k_C], prms[:K_cC], prms[:μ_γ], prms[:σ_γ], prms[:μ_ω], prms[:σ_ω]) for t in times]
            
        elseif model_type == "combined_inhibitor_perm"
            germ_response = [germ_response_inducer_dep_inhibitor_perm_2_factor_gh(u, W, t, ρₛ, prms[:c₀_cs], prms[:d_hp], ξ2, κ2, prms[:Pₛ], prms[:Pₛ_cs], prms[:K_cC], prms[:μ_γ], prms[:σ_γ], prms[:μ_ω], prms[:σ_ω]) for t in times]
            
        elseif model_type == "combined_inducer" # CE
            germ_response = [germ_response_inhibitor_dep_inducer_2_factor_gh(u, W3, t, ρₛ, prms[:c₀_cs], prms[:d_hp], ξ2, κ2, prms[:Pₛ], prms[:Pₛ_cs], prms[:K_cC], prms[:K_I], prms[:n], prms[:k_I], prms[:μ_γ], prms[:σ_γ], prms[:μ_ω], prms[:σ_ω], prms[:μ_ψ], prms[:σ_ψ]) for t in times]
            
        elseif model_type == "combined_inducer_thresh" # E
            germ_response = [germ_response_inhibitor_dep_inducer_thresh_2_factor_gh(u, W3, t, ρₛ, prms[:c₀_cs], prms[:d_hp], ξ2, κ2, prms[:Pₛ], prms[:Pₛ_cs], prms[:K_cC], prms[:k_I], prms[:μ_γ], prms[:σ_γ], prms[:μ_ω], prms[:σ_ω], prms[:μ_ψ], prms[:σ_ψ]) for t in times]
            
        elseif model_type == "combined_inducer_signal" # C
            germ_response = [germ_response_inhibitor_dep_inducer_signal_2_factor_gh(u, W3, t, ρₛ, prms[:c₀_cs], prms[:d_hp], ξ2, κ2, prms[:Pₛ], prms[:Pₛ_cs], prms[:K_cC], prms[:K_I], prms[:n], prms[:μ_γ], prms[:σ_γ], prms[:μ_ω], prms[:σ_ω], prms[:μ_ψ], prms[:σ_ψ]) for t in times]
            
        elseif model_type == "special_inducer"
            germ_response = [germ_response_inducer_var_perm_gh(u, W4, t, ρₛ, prms[:c₀_cs], prms[:d_hp], ξ2, κ2, prms[:Pₛ], prms[:Pₛ_cs], prms[:K_cC], prms[:K_I], prms[:k_I], prms[:n], prms[:μ_ω], prms[:σ_ω], prms[:μ_ψ], prms[:σ_ψ], prms[:μ_α], prms[:σ_α]) for t in times]
            
        elseif model_type == "special_independent"
            germ_response = [germ_response_independent_factors_var_perm_gh(u, W3, t, ρₛ, prms[:c₀_cs], prms[:d_hp], ξ2, κ2, prms[:Pₛ], prms[:Pₛ_cs], prms[:K_cC], prms[:μ_γ], prms[:σ_γ], prms[:μ_ω], prms[:σ_ω], prms[:μ_α], prms[:σ_α]) for t in times]
            
        elseif model_type == "special_combined"
            germ_response = [germ_response_inducer_2_factors_var_perm_gh(u, W4, t, ρₛ, prms[:c₀_cs], prms[:d_hp], ξ2, κ2, prms[:Pₛ], prms[:Pₛ_cs], prms[:K_cC], prms[:K_I], prms[:k_I], prms[:n], prms[:μ_γ], prms[:σ_γ], prms[:μ_ω], prms[:σ_ω], prms[:μ_ψ], prms[:σ_ψ], prms[:μ_α], prms[:σ_α]) for t in times]
            
        elseif model_type == "special_thresh"
            germ_response = [germ_response_inducer_thresh_2_factors_var_perm_gh(u, W4, t, ρₛ, prms[:c₀_cs], prms[:d_hp], ξ2, κ2, prms[:Pₛ], prms[:Pₛ_cs], prms[:K_cC], prms[:k_I], prms[:μ_γ], prms[:σ_γ], prms[:μ_ω], prms[:σ_ω], prms[:μ_ψ], prms[:σ_ψ], prms[:μ_α], prms[:σ_α]) for t in times]
            
        elseif model_type == "special_signal"
            germ_response = [germ_response_inducer_signal_2_factors_var_perm_gh(u, W4, t, ρₛ, prms[:c₀_cs], prms[:d_hp], ξ2, κ2, prms[:Pₛ], prms[:Pₛ_cs], prms[:K_cC], prms[:K_I], prms[:n], prms[:μ_γ], prms[:σ_γ], prms[:μ_ω], prms[:σ_ω], prms[:μ_ψ], prms[:σ_ψ], prms[:μ_α], prms[:σ_α]) for t in times]
        
        elseif model_type == "feedback_inhibitor_inducer_perm" # A
            f_maxs = [prms[:s_max]]
            K_fs = [nothing, prms[:K_cC]]
            thresh_means = [prms[:μ_γ]]
            thresh_sds = [prms[:σ_γ]]
            germ_response = germ_response_feedback_perm(ode_inducer_dependent_perm!, thresh_criterion_inhibitor, sobol_pts, times, geom_samples, prms[:c₀_cs], f_maxs, prms[:Pₛ], prms[:Pₛ_cs], K_fs, prms[:μ_ψ], prms[:σ_ψ], thresh_means, thresh_sds)
            
        elseif model_type == "feedback_combined_inducer_perm" # A
            f_maxs = [prms[:s_max]]
            K_fs = [nothing, prms[:K_cC]]
            thresh_means = [prms[:μ_γ], prms[:μ_ω]]
            thresh_sds = [prms[:σ_γ], prms[:σ_ω]]
            germ_response = germ_response_feedback_perm(ode_inducer_dependent_perm!, thresh_criterion_combined, sobol_pts, times, geom_samples, prms[:c₀_cs], f_maxs, prms[:Pₛ], prms[:Pₛ_cs], K_fs, prms[:μ_ψ], prms[:σ_ψ], thresh_means, thresh_sds)
            
        elseif model_type == "feedback_inducer_inhibitor_perm" # D
            f_maxs = [prms[:b_max]]
            K_fs = [prms[:K_cI], nothing]
            thresh_means = [prms[:μ_ω]]
            thresh_sds = [prms[:σ_ω]]
            germ_response = germ_response_feedback_perm(ode_inhibitor_dependent_perm!, thresh_criterion_inducer, sobol_pts, times, geom_samples, prms[:c₀_cs], f_maxs, prms[:Pₛ], prms[:Pₛ_cs], K_fs, prms[:μ_ψ], prms[:σ_ψ], thresh_means, thresh_sds)
           
        elseif model_type == "feedback_combined_inhibitor_perm" # D
            f_maxs = [prms[:b_max]]
            K_fs = [prms[:K_cI], nothing]
            thresh_means = [prms[:μ_γ], prms[:μ_ω]]
            thresh_sds = [prms[:σ_γ], prms[:σ_ω]]
            germ_response = germ_response_feedback_perm(ode_inhibitor_dependent_perm!, thresh_criterion_combined, sobol_pts, times, geom_samples, prms[:c₀_cs], f_maxs, prms[:Pₛ], prms[:Pₛ_cs], K_fs, prms[:μ_ψ], prms[:σ_ψ], thresh_means, thresh_sds)
           
        elseif model_type == "feedback_inhibitor_inducer_perm_thresh" # AB
            f_maxs = [prms[:s_max]]
            K_fs = [nothing, prms[:K_cC]]
            thresh_means = [prms[:μ_γ]]
            thresh_sds = [prms[:σ_γ]]
            germ_response = germ_response_feedback_perm(ode_inducer_dependent_perm!, thresh_criterion_inhibitor_shift, sobol_pts, times, geom_samples, prms[:c₀_cs], f_maxs, prms[:Pₛ], prms[:Pₛ_cs], K_fs, prms[:μ_ψ], prms[:σ_ψ], thresh_means, thresh_sds; ks=[prms[:k_C]])
           
        elseif model_type == "feedback_combined_inducer_perm_thresh" # AB
            f_maxs = [prms[:s_max]]
            K_fs = [nothing, prms[:K_cC]]
            thresh_means = [prms[:μ_γ], prms[:μ_ω]]
            thresh_sds = [prms[:σ_γ], prms[:σ_ω]]
            germ_response = germ_response_feedback_perm(ode_inducer_dependent_perm!, thresh_criterion_combined_inhibitor_shift, sobol_pts, times, geom_samples, prms[:c₀_cs], f_maxs, prms[:Pₛ], prms[:Pₛ_cs], K_fs, prms[:μ_ψ], prms[:σ_ψ], thresh_means, thesh_sds; ks=[prms[:k_C]])
           
        elseif model_type == "feedback_inhibitor_inducer_perm_inhibitor_signal" # AC
            f_maxs = [prms[:s_max]]
            K_fs = [prms[:K_cI], prms[:K_cC]]
            thresh_means = [prms[:μ_γ]]
            thresh_sds = [prms[:σ_γ]]
            germ_response = germ_response_feedback_perm(ode_inducer_dependent_perm_inhibitor_dependent_signal!, thresh_criterion_inhibitor, sobol_pts, times, geom_samples, prms[:c₀_cs], f_maxs, prms[:Pₛ], prms[:Pₛ_cs], K_fs, prms[:μ_ψ], prms[:σ_ψ], thresh_means, thresh_sds; n=prms[:n])

        elseif model_type == "feedback_inducer_inducer_perm_inhibitor_signal" # AC
            f_maxs = [prms[:s_max]]
            K_fs = [prms[:K_cI], prms[:K_cC]]
            thresh_means = [prms[:μ_ω]]
            thresh_sds = [prms[:σ_ω]]
            germ_response = germ_response_feedback_perm(ode_inducer_dependent_perm_inhibitor_dependent_signal!, thresh_criterion_inducer, sobol_pts, times, geom_samples, prms[:c₀_cs], f_maxs, prms[:Pₛ], prms[:Pₛ_cs], K_fs, prms[:μ_ψ], prms[:σ_ψ], thresh_means, thresh_sds; n=prms[:n])
        
        elseif model_type == "feedback_combined_inducer_perm_inhibitor_signal" # AC
            f_maxs = [prms[:s_max]]
            K_fs = [prms[:K_cI], prms[:K_cC]]
            thresh_means = [prms[:μ_γ], prms[:μ_ω]]
            thresh_sds = [prms[:σ_γ], prms[:σ_ω]]
            germ_response = germ_response_feedback_perm(ode_inducer_dependent_perm_inhibitor_dependent_signal!, thresh_criterion_combined, sobol_pts, times, geom_samples, prms[:c₀_cs], f_maxs, prms[:Pₛ], prms[:Pₛ_cs], K_fs, prms[:μ_ψ], prms[:σ_ψ], thresh_means, thresh_sds; n=prms[:n])
        
        elseif model_type == "feedback_inhibitor_inhibitor_inducer_perm" # AD
            f_maxs = [prms[:b_max], prms[:s_max]]
            K_fs = [prms[:K_cI], prms[:K_cC]]
            thresh_means = [prms[:μ_γ]]
            thresh_sds = [prms[:σ_γ]]
            germ_response = germ_response_feedback_perm(ode_inducer_and_inhibitor_dependent_perm!, thresh_criterion_inhibitor, sobol_pts, times, geom_samples, prms[:c₀_cs], f_maxs, prms[:Pₛ], prms[:Pₛ_cs], K_fs, prms[:μ_ψ], prms[:σ_ψ], thresh_means, thresh_sds)
            
        elseif model_type == "feedback_inducer_inhibitor_inducer_perm" # AD
            f_maxs = [prms[:b_max], prms[:s_max]]
            K_fs = [prms[:K_cI], prms[:K_cC]]
            thresh_means = [prms[:μ_ω]]
            thresh_sds = [prms[:σ_ω]]
            germ_response = germ_response_feedback_perm(ode_inducer_and_inhibitor_dependent_perm!, thresh_criterion_inducer, sobol_pts, times, geom_samples, prms[:c₀_cs], f_maxs, prms[:Pₛ], prms[:Pₛ_cs], K_fs, prms[:μ_ψ], prms[:σ_ψ], thresh_means, thresh_sds)
            
        elseif model_type == "feedback_combined_inhibitor_inducer_perm" # AD
            f_maxs = [prms[:b_max], prms[:s_max]]
            K_fs = [prms[:K_cI], prms[:K_cC]]
            thresh_means = [prms[:μ_γ], prms[:μ_ω]]
            thresh_sds = [prms[:σ_γ], prms[:σ_ω]]
            germ_response = germ_response_feedback_perm(ode_inducer_and_inhibitor_dependent_perm!, thresh_criterion_combined, sobol_pts, times, geom_samples, prms[:c₀_cs], f_maxs, prms[:Pₛ], prms[:Pₛ_cs], K_fs, prms[:μ_ψ], prms[:σ_ψ], thresh_means, thresh_sds)
            
        elseif model_type == "feedback_inducer_inhibitor_thresh_inducer_perm" # AE
            f_maxs = [prms[:s_max]]
            K_fs = [prms[:K_cI], prms[:K_cC]]
            thresh_means = [prms[:μ_ω]]
            thresh_sds = [prms[:σ_ω]]
            germ_response = germ_response_feedback_perm(ode_inducer_dependent_perm!, thresh_criterion_inducer_shift, sobol_pts, times, geom_samples, prms[:c₀_cs], f_maxs, prms[:Pₛ], prms[:Pₛ_cs], K_fs, prms[:μ_ψ], prms[:σ_ψ], thresh_means, thresh_sds, ks=[prms[:k_I]])
            
        elseif model_type == "feedback_combined_inhibitor_thresh_inducer_perm" # AE
            f_maxs = [prms[:s_max]]
            K_fs = [prms[:K_cI], prms[:K_cC]]
            thresh_means = [prms[:μ_γ], prms[:μ_ω]]
            thresh_sds = [prms[:σ_γ], prms[:σ_ω]]
            germ_response = germ_response_feedback_perm(ode_inducer_dependent_perm!, thresh_criterion_combined_inducer_shift, sobol_pts, times, geom_samples, prms[:c₀_cs], f_maxs, prms[:Pₛ], prms[:Pₛ_cs], K_fs, prms[:μ_ψ], prms[:σ_ψ], thresh_means, thresh_sds, ks=[prms[:k_I]])
            
        elseif model_type == "inhibitor_thresh_inducer_signal" # BC
            germ_response = [germ_response_inh_dep_ind_signal_ind_dep_inh_thresh_gh(u, W3, t, ρₛ, prms[:c₀_cs], prms[:d_hp], ξ2, κ2, prms[:Pₛ], prms[:Pₛ_cs], prms[:k_C], prms[:K_cC], prms[:K_I], prms[:n], prms[:μ_γ], prms[:σ_γ], prms[:μ_ψ], prms[:σ_ψ]) for t in times]
            
        end

        return germ_response
    end


    function calc_beta(V, A, Pₛ, ρₛ, t)
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

        τ = V ./ (Pₛ * A)
        ϕ = ρₛ .* V
        β = ϕ .+ (1 .- ϕ) .* exp.(-t ./ (τ .* (1 .- ϕ)))

        return β
    end


    function calc_s(c₀_cs, t, Pₛ_cs, A, V_cw, K_cC)
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
        c_cs = inducer_concentration.(c₀_cs, t, Pₛ_cs, A, V_cw)
        s = c_cs ./ (K_cC .+ c_cs)

        return s
    end


    function calc_geom_variables(ξ, d_hp, κ)
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

        V = 4/3 * π .* ξ.^3
        A = 4π .* ξ.^2
        V_cw = compute_ps_layer_volume(ξ, d_hp, κ)

        return V, A, V_cw
    end


    function normal_distributions(μs, σs)
        """
        Construct normal distributions for
        a collection of means and standard deviations.
        inputs:
            μs - means
            σs - standard deviations
        outputs:
            dists - distributions
        """
        
        return [Normal(μs[i], σs[i]) for i in eachindex(μs)]
    end


    function calc_signals(ξ, d_hp, κ, c₀_cs, ρₛ, Pₛ, Pₛ_cs, K_cC, t)
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

        V, A, V_cw = calc_geom_variables(ξ, d_hp, κ)
        β = calc_beta(V, A, Pₛ, ρₛ, t)
        s = calc_s(c₀_cs, t, Pₛ_cs, A, V_cw, K_cC)

        return β, s
    end


    function lognormal_samples(μ, σ, u)
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

        μ_log = log(μ^2 / sqrt(σ^2 + μ^2))
        σ_log = sqrt(log(σ^2 / μ^2 + 1))
        
        return exp.(μ_log .+ σ_log .* u)
    end


    function germ_response_independent_factors_gh(u, W, t, ρₛ, c₀_cs, d_hp, ξ, κ, Pₛ, Pₛ_cs, K_cC, μ_γ, σ_γ, μ_ω, σ_ω)
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

        # Distributions
        dist_γ, dist_ω = normal_distributions([μ_γ, μ_ω], [σ_γ, σ_ω])

        # Signals
        β, s = calc_signals(ξ, d_hp, κ, c₀_cs, ρₛ, Pₛ, Pₛ_cs, K_cC, t)
        
        tail = cdf.(dist_ω, s) .* (1 .- cdf.(dist_γ, β))
        
        return sum(W .* tail)
    end


    function germ_response_inducer_dep_inhibitor_thresh_gh(u, W, t, ρₛ, c₀_cs, d_hp, ξ, κ, Pₛ, Pₛ_cs, k, K_cC, μ_γ, σ_γ)
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

        # Distributions
        dist_γ = Normal(μ_γ, σ_γ)

        # Signals
        β, s = calc_signals(ξ, d_hp, κ, c₀_cs, ρₛ, Pₛ, Pₛ_cs, K_cC, t)

        tail = 1 .- cdf.(dist_γ, β .- k .* s)

        return sum(W .* tail)
    end


    # function germ_response_inducer_dep_inhibitor_perm_gh(u, W, t, ρₛ, c₀_cs, d_hp, ξ, κ, Pₛ, Pₛ_cs, K_cC, μ_γ, σ_γ)
    #     """
    #     Compute the germination response for an inducer-dependent
    #     inhibitor permeation for a given set of parameters,
    #     without considering an external initial concentration.
    #     Uses Gauss-Hermite approximation.
    #     inputs:
    #         u - transformed Gauss-Hermite nodes
    #         W - transformed Gauss-Hermite weights (matrix)
    #         t - time in seconds
    #         ρₛ - spore density in spores/um^3
    #         c₀_cs - initial concentration of carbon source in M
    #         d_hp - thickness of the hydrophobin layer in um
    #         ξ - spore radius in um
    #         κ - cell wall thickness in um
    #         Pₛ - permeation constant for inhibitor in um/s
    #         Pₛ_cs - permeation constant for the carbon source in um/s
    #         K_cC - half-saturation constant for the carbon source
    #         μ_γ - mean inhibition threshold
    #         σ_γ - standard deviation of inhibition threshold
    #     output:
    #         the germination response for the given parameters (normalized)
    #     """

    #     # Distributions
    #     dist_γ = Normal(μ_γ, σ_γ)

    #     # Inducer
    #     A = 4 * π .* ξ.^2
    #     V_cw = compute_ps_layer_volume(ξ, d_hp, κ)
    #     c_cs = inducer_concentration.(c₀_cs, t, Pₛ_cs, A, V_cw)
    #     s = c_cs ./ (K_cC .+ c_cs)

    #     # Inhibitor
    #     V = 4/3 * π .* ξ.^3
    #     τ = V ./ (s .* Pₛ .* A)
    #     ϕ = ρₛ .* V
    #     β = ϕ .+ (1 .- ϕ) .* exp.(-t ./ (τ .* (1 .- ϕ)))

    #     tail = 1 .- cdf.(dist_γ, β)

    #     return sum(W .* tail)
    # end


    # function germ_response_inducer_dep_inhibitor_gh(u, W, t, ρₛ, c₀_cs, d_hp, ξ, κ, Pₛ, Pₛ_cs, K_cC, k, μ_γ, σ_γ)
    #     """
    #     Compute the germination response for an inducer-dependent
    #     inhibitor threshold and permeation for a given set of parameters,
    #     without considering an external initial concentration.
    #     Uses Gauss-Hermite approximation.
    #     inputs:
    #         u - transformed Gauss-Hermite nodes
    #         W - transformed Gauss-Hermite weights (matrix)
    #         t - time in seconds
    #         ρₛ - spore density in spores/um^3
    #         c₀_cs - initial concentration of carbon source in M
    #         d_hp - thickness of the hydrophobin layer in um
    #         ξ - spore radius in um
    #         κ - cell wall thickness in um
    #         Pₛ - permeation constant for the inhibitor in um/s
    #         Pₛ_cs - permeation constant for the carbon source in um/s
    #         K_cC - half-saturation constant for the carbon source
    #         k - proportionality constant for threshold modulation vs permeability modulation
    #         μ_γ - mean inhibition threshold
    #         σ_γ - standard deviation of inhibition threshold
    #     output:
    #         the germination response for the given parameters (normalized)
    #     """

    #     # Distributions
    #     dist_γ = Normal(μ_γ, σ_γ)

    #     # Inducer
    #     A = 4 * π .* ξ.^2
    #     V_cw = compute_ps_layer_volume(ξ, d_hp, κ)
    #     c_cs = inducer_concentration.(c₀_cs, t, Pₛ_cs, A, V_cw)
    #     s = c_cs ./ (K_cC .+ c_cs)

    #     # Inhibitor
    #     V = 4/3 * π .* ξ.^3
    #     τ = V ./ (s .* Pₛ .* A)
    #     ϕ = ρₛ .* V
    #     β = ϕ .+ (1 .- ϕ) .* exp.(-t ./ (τ .* (1 .- ϕ)))

    #     tail = 1 .- cdf.(dist_γ, β .- k .* s)

    #     return sum(W .* tail)
    # end


    function germ_response_inhibitor_dep_inducer_thresh_gh(u, W3, t, ρₛ, c₀_cs, d_hp, ξ, κ, Pₛ, Pₛ_cs, K_cC, k, μ_ω, σ_ω, μ_ψ, σ_ψ)
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
            K_cC - half-saturation constant for the carbon source
            k - inhibition strength over induction threshold
            μ_ω - mean induction threshold
            σ_ω - standard deviation of induction threshold
            μ_ψ - mean initial concentration
            σ_ψ - standard deviation of initial concentration
        output:
            the germination response for the given parameters (normalized)
        """

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

        tail = cdf.(dist_ω, s .- k .* c_in)

        return sum(W3 .* tail)
    end


    function germ_response_inhibitor_dep_inducer_signal_gh(u, W3, t, ρₛ, c₀_cs, d_hp, ξ, κ, Pₛ, Pₛ_cs, K_cC, K_I, n, μ_ω, σ_ω, μ_ψ, σ_ψ)
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

    
    function germ_response_inhibitor_dep_inducer_gh(u, W3, t, ρₛ, c₀_cs, d_hp, ξ, κ, Pₛ, Pₛ_cs, k, K_cC, K_I, n, μ_ω, σ_ω, μ_ψ, σ_ψ)
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
            K_cC - half-saturation constant for the carbon source
            K_I - half-saturation constant for the inhibitor
            n - Hill coefficient for the inhibitor
            k - inhibition strength over induction threshold
            μ_ω - mean induction threshold
            σ_ω - standard deviation of induction threshold
            μ_ψ - mean initial concentration
            σ_ψ - standard deviation of initial concentration
        output:
            the germination response for the given parameters (normalized)
        """

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

        tail = cdf.(dist_ω, s_mod .- k .* c_in)

        return sum(W3 .* tail)
    end

    # !!!!!!!!!!!!!!!!!!!!!!WIP!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    function germ_response_inh_dep_ind_signal_ind_dep_inh_thresh_gh(u, W3, t, ρₛ, c₀_cs, d_hp, ξ, κ, Pₛ, Pₛ_cs, k_C, K_cC, K_I, n, μ_γ, σ_γ, μ_ψ, σ_ψ)
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


    function germ_response_inducer_dep_inhibitor_thresh_2_factor_gh(u, W, t, ρₛ, c₀_cs, d_hp, ξ, κ, Pₛ, Pₛ_cs, k, K_cC, μ_γ, σ_γ, μ_ω, σ_ω)
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

        # Distributions
        dist_γ, dist_ω = normal_distributions([μ_γ, μ_ω], [σ_γ, σ_ω])

        # Signals
        β, s = calc_signals(ξ, d_hp, κ, c₀_cs, ρₛ, Pₛ, Pₛ_cs, K_cC, t)

        tail = (1 .- cdf.(dist_γ, β .- k .* s)) .* cdf.(dist_ω, s)

        return sum(W .* tail)
    end


    function germ_response_inducer_dep_inhibitor_perm_2_factor_gh(u, W, t, ρₛ, c₀_cs, d_hp, ξ, κ, Pₛ, Pₛ_cs, K_cC, μ_γ, σ_γ, μ_ω, σ_ω)
        """
        Compute the germination response for an inducer-dependent
        inhibitor permeation for a given set of parameters,
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
            Pₛ - permeation constant for inhibitor in um/s
            Pₛ_cs - permeation constant for the carbon source in um/s
            K_cC - half-saturation constant for the carbon source
            μ_γ - mean inhibition threshold
            σ_γ - standard deviation of inhibition threshold
            μ_ω - mean induction threshold
            σ_ω - standard deviation of induction threshold
        output:
            the germination response for the given parameters (normalized)
        """

        # Distributions
        dist_γ, dist_ω = normal_distributions([μ_γ, μ_ω], [σ_γ, σ_ω])

        # Signals
        β, s = calc_signals(ξ, d_hp, κ, c₀_cs, ρₛ, Pₛ, Pₛ_cs, K_cC, t)

        tail = (1 .- cdf.(dist_γ, β)) .* cdf.(dist_ω, s)

        return sum(W .* tail)
    end


    function germ_response_inducer_dep_inhibitor_2_factor_gh(u, W, t, ρₛ, c₀_cs, d_hp, ξ, κ, Pₛ, Pₛ_cs, K_cC, k, μ_γ, σ_γ, μ_ω, σ_ω)
        """
        Compute the germination response for an inducer-dependent
        inhibitor threshold and permeation and an additional
        inducer-dependent germination for a given set of parameters.
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
            K_cC - half-saturation constant for the carbon source
            k - proportionality constant for threshold modulation vs permeability modulation
            μ_γ - mean inhibition threshold
            σ_γ - standard deviation of inhibition threshold
            μ_ω - mean induction threshold
            σ_ω - standard deviation of induction threshold
        output:
            the germination response for the given parameters (normalized)
        """

        # Distributions
        dist_γ, dist_ω = normal_distributions([μ_γ, μ_ω], [σ_γ, σ_ω])

        # Signals
        β, s = calc_signals(ξ, d_hp, κ, c₀_cs, ρₛ, Pₛ, Pₛ_cs, K_cC, t)

        tail = (1 .- cdf.(dist_γ, β .- k .* s)) .* cdf.(dist_ω, s)

        return sum(W .* tail)
    end


    function germ_response_inhibitor_dep_inducer_thresh_2_factor_gh(u, W3, t, ρₛ, c₀_cs, d_hp, ξ, κ, Pₛ, Pₛ_cs, K_cC, k, μ_γ, σ_γ, μ_ω, σ_ω, μ_ψ, σ_ψ)
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
            K_cC - half-saturation constant for the carbon source
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

        tail = cdf.(dist_ω, s .- k .* c_in) .* tail_γ

        return sum(W3 .* tail)
    end

    function germ_response_inhibitor_dep_inducer_signal_2_factor_gh(u, W3, t, ρₛ, c₀_cs, d_hp, ξ, κ, Pₛ, Pₛ_cs, K_cC, K_I, n, μ_γ, σ_γ, μ_ω, σ_ω, μ_ψ, σ_ψ)
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


    function germ_response_inhibitor_dep_inducer_2_factor_gh(u, W3, t, ρₛ, c₀_cs, d_hp, ξ, κ, Pₛ, Pₛ_cs, K_cC, K_I, n, k, μ_γ, σ_γ, μ_ω, σ_ω, μ_ψ, σ_ψ)
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
            K_cC - half-saturation constant for the carbon source
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

        tail = cdf.(dist_ω, s_mod .- k .* c_in) .* tail_γ

        return sum(W3 .* tail)
    end


    function germ_response_inducer_thresh_var_perm_gh(u, W4, t, ρₛ, c₀_cs, d_hp, ξ, κ, Pₛ, Pₛ_cs, K_cC, k, μ_ω, σ_ω, μ_ψ, σ_ψ, μ_α, σ_α)
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


    function germ_response_inducer_var_perm_gh(u, W4, t, ρₛ, c₀_cs, d_hp, ξ, κ, Pₛ, Pₛ_cs, K_cC, K_I, k, n, μ_ω, σ_ω, μ_ψ, σ_ψ, μ_α, σ_α)
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


    function germ_response_independent_factors_var_perm_gh(u, W3, t, ρₛ, c₀_cs, d_hp, ξ, κ, Pₛ, Pₛ_cs, K_cC, μ_γ, σ_γ, μ_ω, σ_ω, μ_α, σ_α)
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


    function germ_response_inducer_thresh_2_factors_var_perm_gh(u, W4, t, ρₛ, c₀_cs, d_hp, ξ, κ, Pₛ, Pₛ_cs, K_cC, k, μ_γ, σ_γ, μ_ω, σ_ω, μ_ψ, σ_ψ, μ_α, σ_α)
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


    function germ_response_inducer_signal_2_factors_var_perm_gh(u, W4, t, ρₛ, c₀_cs, d_hp, ξ, κ, Pₛ, Pₛ_cs, K_cC, K_I, n, μ_γ, σ_γ, μ_ω, σ_ω, μ_ψ, σ_ψ, μ_α, σ_α)
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


    function germ_response_inducer_2_factors_var_perm_gh(u, W4, t, ρₛ, c₀_cs, d_hp, ξ, κ, Pₛ, Pₛ_cs, K_cC, K_I, k, n, μ_γ, σ_γ, μ_ω, σ_ω, μ_ψ, σ_ψ, μ_α, σ_α)
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


    function germ_response_inducer_dep_inhibitor_eq(ρₛ, dist_ξ, μ_γ, σ_γ; reltol=1e-4)
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


    function germ_response_inducer_dep_inhibitor_eq_c_ex(ρₛ, dist_ξ, c_ex, μ_γ, σ_γ, μ_ψ, σ_ψ; reltol=1e-4)
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


    function germ_response_inhibitor_dep_inducer_2_factors_eq(ρₛ, dist_ξ, c₀_cs, K_cC, K_I, k, n, μ_γ, σ_γ, μ_ω, σ_ω, μ_ψ, σ_ψ; reltol=1e-4, abstol=1e-6)
        """
        Compute the equilibrium germination response
        for an inhibitor-dependent inducer threshold and signal and
        an additional inhibitor-dependent germination.
        inputs:
            ρₛ - spore density in spores/um^3
            dist_ξ - distribution of spore radii (LogNormal)
            c₀_cs - initial concentration of carbon source in M
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
            reltol - relative tolerance for the integration
            abstol - absolute tolerance for the integration
        output:
            the equilibrium germination response for the given parameters (normalized)
        """
        
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
            tail2 = cdf(dist_ω, s_mod .- k .* c_eq)
            return tail1 * tail2 * pdf(dist_ξ, ξ) * pdf(dist_ψ, ψ)
        end

        return hcubature(integrand, [0.0, 0.0], [quantile(dist_ξ, 1-1e-8), quantile(dist_ψ, 1-1e-9)], reltol=reltol, abstol=abstol)[1]
    end


    function germ_response_inhibitor_dep_inducer_2_factors_eq_c_ex(ρₛ, dist_ξ, c_ex, c₀_cs, K_cC, K_I, k, n, μ_γ, σ_γ, μ_ω, σ_ω, μ_ψ, σ_ψ; reltol=1e-4, abstol=1e-6)
        """
        Compute the equilibrium germination response
        for an inhibitor-dependent inducer threshold and signal and
        an additional inhibitor-dependent germination.
        inputs:
            ρₛ - spore density in spores/um^3
            dist_ξ - distribution of spore radii (LogNormal)
            c_ex - external concentration of the inducer in M
            c₀_cs - initial concentration of carbon source in M
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
            reltol - relative tolerance for the integration
            abstol - absolute tolerance for the integration
        output:
            the equilibrium germination response for the given parameters (normalized)
        """
        
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
            tail2 = cdf(dist_ω, s_mod .- k .* c_eq)
            return tail1 * tail2 * pdf(dist_ξ, ξ) * pdf(dist_ψ, ψ)
        end

        return hcubature(integrand, [0.0, 0.0], [quantile(dist_ξ, 1-1e-8), quantile(dist_ψ, 1-1e-8)], reltol=reltol, abstol=abstol)[1]
    end


    function germ_response_inhibitor_dep_inducer_thresh_2_factors_eq(ρₛ, dist_ξ, c₀_cs, K_cC, k, μ_γ, σ_γ, μ_ω, σ_ω, μ_ψ, σ_ψ; reltol=1e-4, abstol=1e-6)
        """
        Compute the equilibrium germination response
        for an inhibitor-dependent inducer threshold and
        an additional inhibitor-dependent germination.
        inputs:
            ρₛ - spore density in spores/um^3
            dist_ξ - distribution of spore radii (LogNormal)
            c₀_cs - initial concentration of carbon source in M
            K_cC - half-saturation constant for the carbon source
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
            tail1 = 1 .- cdf(dist_γ, ϕ)
            tail2 = cdf(dist_ω, s_eq .- k .* ϕ .* ψ)
            return tail1 * tail2 * pdf(dist_ξ, ξ) * pdf(dist_ψ, ψ)
        end

        return hcubature(integrand, [0.0, 0.0], [quantile(dist_ξ, 1-1e-8), quantile(dist_ψ, 1-1e-9)], reltol=reltol, abstol=abstol)[1]
    end


    function germ_response_inhibitor_dep_inducer_thresh_2_factors_eq_c_ex(ρₛ, dist_ξ, c_ex, c₀_cs, K_cC, k, μ_γ, σ_γ, μ_ω, σ_ω, μ_ψ, σ_ψ; reltol=1e-4)
        """
        Compute the equilibrium germination response
        for an inhibitor-dependent inducer threshold and
        an additional inhibitor-dependent germination.
        inputs:
            ρₛ - spore density in spores/um^3
            dist_ξ - distribution of spore radii (LogNormal)
            c_ex - external concentration of the inducer in M
            c₀_cs - initial concentration of carbon source in M
            K_cC - half-saturation constant for the carbon source
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
            tail1 = 1 .- cdf(dist_γ, ϕ .+ (1 .- ϕ) .* c_ex ./ ψ)
            tail2 = cdf(dist_ω, s_eq .- k .* (ϕ .* ψ .+ (1 .- ϕ) .* c_ex))
            return tail1 * tail2 * pdf(dist_ξ, ξ) * pdf(dist_ψ, ψ)
        end

        return hcubature(integrand, [0.0, 0.0], [quantile(dist_ξ, 1-1e-8), quantile(dist_ψ, 1-1e-8)], reltol=reltol, abstol=1e-6)[1]
    end


    function germ_response_inhibitor_dep_inducer_signal_2_factors_eq(t, ρₛ, dist_ξ, c₀_cs, K_cC, K_I, n, μ_γ, σ_γ, μ_ω, σ_ω, μ_ψ, σ_ψ; reltol=1e-4, abstol=1e-6)
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


    function germ_response_inhibitor_dep_inducer_signal_2_factors_eq_c_ex(ρₛ, dist_ξ, c_ex, c₀_cs, K_cC, K_I, n, μ_γ, σ_γ, μ_ω, σ_ω, μ_ψ, σ_ψ; reltol=1e-4, abstol=1e-6)
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


    function germ_response_independent_eq(ρₛ, dist_ξ, c₀_cs, K_cC, μ_γ, σ_γ, μ_ω, σ_ω; reltol=1e-4)
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


    function germ_response_independent_eq_c_ex(ρₛ, dist_ξ, c_ex, c₀_cs, K_cC, μ_γ, σ_γ, μ_ω, σ_ω, μ_ψ, σ_ψ; reltol=1e-4)
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

    function ode_inducer_dependent_perm!(du, u, p::PermParams, t)
        """
        ODE function for inducer-dependent
        cell wall permeability.
        """
        cinI, cinC, coutI = u
        
        g = (1 + p.f_maxs[1] * cinC / (p.K_fs[2] + cinC)) * p.A # f_maxs[1] is s_max, K_fs[2] is K_cC
        rateI = g * p.Pₛ_I
        rateC = g * p.Pₛ_C

        diffI = cinI - coutI
        diffC = cinC - p.c₀_cs

        du[1] = -(rateI / p.Vₛ) * diffI
        du[2] = -(rateC / p.V_ps) * diffC
        du[3] = (rateI / p.V_out) * diffI
    end


    function ode_inhibitor_dependent_perm!(du, u, p::PermParams, t)
        """
        ODE function for inhibitor-dependent
        cell wall permeability.
        """
        cinI, cinC, coutI = u
        
        g = (1 - p.f_maxs[1] * cinI / (p.K_fs[1] + cinI)) * p.A # f_maxs[1] is b_max, K_fs[1] is K_cI
        rateI = g * p.Pₛ_I
        rateC = g * p.Pₛ_C

        diffI = cinI - coutI
        diffC = cinC - p.c₀_cs
        
        du[1] = -(rateI / p.Vₛ) * diffI
        du[2] = -(rateC / p.V_ps) * diffC
        du[3] = (rateI / p.V_out) * diffI
    end


    function ode_inducer_dependent_perm_inhibitor_dependent_signal!(du, u, p::PermParams, t)
        """
        ODE function for inducer-dependent
        cell wall permeability.
        """
        cinI, cinC, coutI = u

        cinI = max(cinI, 0.0)
        
        s = p.f_maxs[1] / (1 + (cinI / p.K_fs[1]) ^ p.n) # f_maxs[1] is s_max, K_fs[1] is K_cI
        g = (1 + s * cinC / (p.K_fs[2] + cinC)) * p.A # K_fs[2] is K_cC
        rateI = g * p.Pₛ_I
        rateC = g * p.Pₛ_C

        diffI = cinI - coutI
        diffC = cinC - p.c₀_cs

        du[1] = -(rateI / p.Vₛ) * diffI
        du[2] = -(rateC / p.V_ps) * diffC
        du[3] = (rateI / p.V_out) * diffI
    end


    function ode_inducer_and_inhibitor_dependent_perm!(du, u, p::PermParams, t)
        """
        ODE function for inducer-dependent
        cell wall permeability.
        """
        cinI, cinC, coutI = u
        
        g = (1 - p.f_maxs[1] * cinI / (p.K_fs[1] + cinI) + p.f_maxs[2] * cinC / (p.K_fs[2] + cinC)) * p.A # f_maxs[1] is b_max, f_maxs[2] is s_max, K_fs[1] is K_cI, K_fs[2] is K_cC
        rateI = g * p.Pₛ_I
        rateC = g * p.Pₛ_C

        diffI = cinI - coutI
        diffC = cinC - p.c₀_cs

        du[1] = -(rateI / p.Vₛ) * diffI
        du[2] = -(rateC / p.V_ps) * diffC
        du[3] = (rateI / p.V_out) * diffI
    end
    
    
    function thresh_criterion_inhibitor(cins, thresholds, ks=nothing, K_fs=nothing)
        """
        Computes simple germination criterion
        with regard to the inhibition threshold
        inputs:
            cins - inhibitor/inducer concentrations (M)
            thresholds - inhibitor thresholds
            ks - scaling factors
            K_fs - half-saturation constants (M)
        """
        return cins[1, :] .< thresholds[1, :]
    end


    function thresh_criterion_inducer(cins, thresholds, ks, K_fs)
        """
        Computes simple germination criterion
        with regard to the induction threshold
        inputs:
            cins - inhibitor/inducer concentrations (M)
            thresholds - inducer thresholds
            ks - scaling factors
            K_fs - inducer half-saturation constant as 1-element vector (M)
        """
        return cins[2, :] ./ (cins[2, :] .+ K_fs[2]) .> thresholds[1, :]
    end


    function thresh_criterion_combined(cins, thresholds, ks, K_fs)
        """
        Computes simple germination criterion
        with regard to the induction and inhibition thresholds
        inputs:
            cins - inhibitor/inducer concentrations (M)
            thresholds - inhibitor thresholds
            ks - scaling factors
            K_fs - inducer half-saturation constant as 1-element vector (M)
        """
        return (cins[1, :] .< thresholds[1, :]) .* (cins[2, :] ./ (cins[2, :] .+ K_fs[2]) .> thresholds[2, :])
    end


    function thresh_criterion_inhibitor_shift(cins, thresholds, ks, K_fs)
        """
        Computes inhibitor-dependent germination criterion with
        a threshold shifting signal defined by the inducer concentration.
        inputs:
            cins - inhibitor/inducer concentrations (M)
            thresholds - inhibitor thresholds
            ks - scaling factors
            K_fs - inducer half-saturation constant as 1-element vector (M)
        """
        thresh_bias_signal = ks[1] .* cins[2, :]  ./ (cins[2, :] .+ K_fs[2])
        return cins[1, :] .< thresholds[1, :] .+ thresh_bias_signal
    end


    function thresh_criterion_combined_inhibitor_shift(cins, thresholds, ks, K_fs)
        """
        Computes 2-factor germination criterion with
        a threshold shifting signal defined by the inducer concentration.
        inputs:
            cins - inhibitor/inducer concentrations (M)
            thresholds - inhibitor thresholds
            ks - scaling factors
            K_fs - inducer half-saturation constant as 1-element vector (M)
        """
        thresh_bias_signal = ks[1] .* cins[2, :]  ./ (cins[2, :] .+ K_fs[2])
        return (cins[1, :] .< thresholds[1, :] .+ thresh_bias_signal) .* (cins[2, :] ./ (cins[2, :] .+ K_fs[2]) .> thresholds[2, :])
    end


    function thresh_criterion_inducer_shift(cins, thresholds, ks, K_fs)
        """
        Computes inhibitor-dependent germination criterion with
        a threshold shifting signal defined by the inducer concentration.
        inputs:
            cins - inhibitor/inducer concentrations (M)
            thresholds - inhibitor thresholds
            ks - scaling factors
            K_fs - inducer half-saturation constant as 1-element vector (M)
        """
        thresh_bias_signal = ks[1] .* cins[1, :]  ./ (cins[1, :] .+ K_fs[1])
        return cins[2, :] ./ (cins[2, :] .+ K_fs[2]) .> thresholds[1, :] .+ thresh_bias_signal
    end


    function thresh_criterion_combined_inducer_shift(cins, thresholds, ks, K_fs)
        """
        Computes inhibitor-dependent germination criterion with
        a threshold shifting signal defined by the inducer concentration.
        inputs:
            cins - inhibitor/inducer concentrations (M)
            thresholds - inhibitor thresholds
            ks - scaling factors
            K_fs - inducer half-saturation constant as 1-element vector (M)
        """
        thresh_bias_signal = ks[1] .* cins[1, :]  ./ (cins[1, :] .+ K_fs[1])
        return (cins[1, :] .< thresholds[1, :] .+ thresh_bias_signal) .* (cins[2, :] ./ (cins[2, :] .+ K_fs[2]) .> thresholds[2, :] .+ thresh_bias_signal)
    end


    function germ_response_feedback_perm(ode_func, thresh_func, sobol_pts, times, geom_samples, c₀_cs, f_maxs, Pₛ_I, Pₛ_C, K_fs, μ_ψ, σ_ψ, μs_thresh, σs_thresh; ks=nothing, n=nothing)
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

        # Run ODE ensembles
        ep = EnsembleProblem(prob; prob_func=prob_func)
        sols = solve(ep, AutoTsit5(Rosenbrock23()), EnsembleThreads(), trajectories=n_samp, saveat=times, abstol=1e-6, reltol=1e-6)

        # Evaluate fraction germinated
        n_times = length(times)
        germinated = Vector{Float64}(undef, n_times)
        c_in = Array{Float64}(undef, 2, n_samp)

        @inbounds for (ti, t) in enumerate(times)
            for i in 1:n_samp
                u = sols[i](t)
                c_in[1, i] = u[1]
                c_in[2, i] = u[2]
            end
            gmask = thresh_func(c_in, samples_thresh, ks, K_fs)
            germinated[ti] = mean(gmask)
        end
        
        return germinated
    end
    
end