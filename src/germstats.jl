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

    export germ_response_feedback_inhibitor_perm

    export clamp_inplace!


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
    function compute_germination_response(model_type, times, ρₛ, params; n_nodes=nothing)
        """
        Generic wrapper function to compute the germination response.
        inputs:
            model_type (String): model type to fit
            times (Vector{Float64}): time points to compute the germination response
            ρₛ (float) - spore density in spores/μm^3
            n_nodes (int) - number of Gauss-Hermite nodes to use
            params (Dict) - additional parameters for the germination response function
            n_nodes (int) - number of Gauss-Hermite nodes to use
        """

        @argcheck model_type in ["independent",
                                "inhibitor", "inhibitor_thresh", "inhibitor_perm",
                                "inducer", "inducer_thresh", "inducer_signal",
                                "combined_inhibitor", "combined_inhibitor_thresh", "combined_inhibitor_perm",
                                "combined_inducer", "combined_inducer_thresh", "combined_inducer_signal",
                                "special_inducer", "special_independent", "special_combined", "special_thresh", "special_signal",
                                "feedback_inhibitor_perm"]

        # Determine number of nodes depending on the integral dimension (if not specified)
        if isnothing(n_nodes)
            if model_type in ["independent", "inhibitor", "inhibitor_thresh", "inhibitor_perm",
                                "combined_inhibitor", "combined_inhibitor_thresh", "combined_inhibitor_perm"]
                n_nodes = 36 # 2D integral
            elseif model_type in ["inducer", "inducer_thresh", "inducer_signal", 
                                "combined_inducer", "combined_inducer_thresh", "combined_inducer_signal", "special_independent"]
                n_nodes = 10 # 3D integral
            elseif model_type in ["special_inducer", "special_combined", "special_thresh", "special_signal"]
                n_nodes = 6 # 4D integral
            elseif model_type in ["feedback_inhibitor_perm"]
                n_nodes = 1024
            end
        end

        gh_integral = false
        if split(model_type, "_")[1] == "feedback"
            sobol_pts = QuasiMonteCarlo.sample(n_nodes, 4, SobolSample())
        else
            gh_integral = true

            # Gauss-Hermite nodes and weights
            ghnodes, ghweights = gausshermite(n_nodes)
            u = √2 .* ghnodes
            hw = ghweights ./ √π
        end

        # Unpack means and stds and weight samples
        μ_ξ = params[:μ_ξ]
        σ_ξ = params[:σ_ξ]
        μ_ξ_log = log(μ_ξ^2 / sqrt(σ_ξ^2 + μ_ξ^2))
        σ_ξ_log = sqrt(log(σ_ξ^2 / μ_ξ^2 + 1))
        if gh_integral ξ = exp.(μ_ξ_log .+ σ_ξ_log .* u) end

        if haskey(params, :μ_κ)
            μ_κ = params[:μ_κ]
            σ_κ = params[:σ_κ]
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
                            "combined_inducer", "combined_inducer_thresh", "combined_inducer_signal", "special_independent"]
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
            samples_V_ps = compute_ps_layer_volume.(samples_ξ, params[:d_hp], samples_κ)
        end

        # Compute the germination response
        if model_type == "independent"
            germ_response = [germ_response_independent_factors_gh(u, W, t, ρₛ, params[:c₀_cs], params[:d_hp], ξ2, κ2, params[:Pₛ], params[:Pₛ_cs], params[:K_cs], params[:μ_γ], params[:σ_γ], params[:μ_ω], params[:σ_ω]) for t in times]
            
        elseif model_type == "inhibitor"
            germ_response = [germ_response_inducer_dep_inhibitor_gh(u, W, t, ρₛ, params[:c₀_cs], params[:d_hp], ξ2, κ2, params[:Pₛ], params[:Pₛ_cs], params[:K_cs], params[:k], params[:μ_γ], params[:σ_γ]) for t in times]
            
        elseif model_type == "inhibitor_thresh"
            germ_response = [germ_response_inducer_dep_inhibitor_thresh_gh(u, W, t, ρₛ, params[:c₀_cs], params[:d_hp], ξ2, κ2, params[:Pₛ], params[:Pₛ_cs], params[:k], params[:K_cs], params[:μ_γ], params[:σ_γ]) for t in times]
            
        elseif model_type == "inhibitor_perm"
            germ_response = [germ_response_inducer_dep_inhibitor_perm_gh(u, W, t, ρₛ, params[:c₀_cs], params[:d_hp], ξ2, κ2, params[:Pₛ], params[:Pₛ_cs], params[:K_cs], params[:μ_γ], params[:σ_γ]) for t in times]
            
        elseif model_type == "inducer"
            germ_response = [germ_response_inhibitor_dep_inducer_gh(u, W3, t, ρₛ, params[:c₀_cs], params[:d_hp], ξ2, κ2, params[:Pₛ], params[:Pₛ_cs], params[:k], params[:K_cs], params[:K_I], params[:n], params[:μ_ω], params[:σ_ω], params[:μ_ψ], params[:σ_ψ]) for t in times]

        elseif model_type == "inducer_thresh"
            germ_response = [germ_response_inhibitor_dep_inducer_thresh_gh(u, W3, t, ρₛ, params[:c₀_cs], params[:d_hp], ξ2, κ2, params[:Pₛ], params[:Pₛ_cs], params[:K_cs], params[:k], params[:μ_ω], params[:σ_ω], params[:μ_ψ], params[:σ_ψ]) for t in times]
            
        elseif model_type == "inducer_signal"
            germ_response = [germ_response_inhibitor_dep_inducer_signal_gh(u, W3, t, ρₛ, params[:c₀_cs], params[:d_hp], ξ2, κ2, params[:Pₛ], params[:Pₛ_cs], params[:K_cs], params[:K_I], params[:n], params[:μ_ω], params[:σ_ω], params[:μ_ψ], params[:σ_ψ]) for t in times]
            
        elseif model_type == "combined_inhibitor"
            germ_response = [germ_response_inducer_dep_inhibitor_2_factor_gh(u, W, t, ρₛ, params[:c₀_cs], params[:d_hp], ξ2, κ2, params[:Pₛ], params[:Pₛ_cs], params[:K_cs], params[:k], params[:μ_γ], params[:σ_γ], params[:μ_ω], params[:σ_ω]) for t in times]
            
        elseif model_type == "combined_inhibitor_thresh"
            germ_response = [germ_response_inducer_dep_inhibitor_thresh_2_factor_gh(u, W, t, ρₛ, params[:c₀_cs], params[:d_hp], ξ2, κ2, params[:Pₛ], params[:Pₛ_cs], params[:k], params[:K_cs], params[:μ_γ], params[:σ_γ], params[:μ_ω], params[:σ_ω]) for t in times]
            
        elseif model_type == "combined_inhibitor_perm"
            germ_response = [germ_response_inducer_dep_inhibitor_perm_2_factor_gh(u, W, t, ρₛ, params[:c₀_cs], params[:d_hp], ξ2, κ2, params[:Pₛ], params[:Pₛ_cs], params[:K_cs], params[:μ_γ], params[:σ_γ], params[:μ_ω], params[:σ_ω]) for t in times]
            
        elseif model_type == "combined_inducer"
            germ_response = [germ_response_inhibitor_dep_inducer_2_factor_gh(u, W3, t, ρₛ, params[:c₀_cs], params[:d_hp], ξ2, κ2, params[:Pₛ], params[:Pₛ_cs], params[:K_cs], params[:K_I], params[:n], params[:k], params[:μ_γ], params[:σ_γ], params[:μ_ω], params[:σ_ω], params[:μ_ψ], params[:σ_ψ]) for t in times]
            
        elseif model_type == "combined_inducer_thresh"
            germ_response = [germ_response_inhibitor_dep_inducer_thresh_2_factor_gh(u, W3, t, ρₛ, params[:c₀_cs], params[:d_hp], ξ2, κ2, params[:Pₛ], params[:Pₛ_cs], params[:K_cs], params[:k], params[:μ_γ], params[:σ_γ], params[:μ_ω], params[:σ_ω], params[:μ_ψ], params[:σ_ψ]) for t in times]
            
        elseif model_type == "combined_inducer_signal"
            germ_response = [germ_response_inhibitor_dep_inducer_signal_2_factor_gh(u, W3, t, ρₛ, params[:c₀_cs], params[:d_hp], ξ2, κ2, params[:Pₛ], params[:Pₛ_cs], params[:K_cs], params[:K_I], params[:n], params[:μ_γ], params[:σ_γ], params[:μ_ω], params[:σ_ω], params[:μ_ψ], params[:σ_ψ]) for t in times]
            
        elseif model_type == "special_inducer"
            germ_response = [germ_response_inducer_var_perm_gh(u, W4, t, ρₛ, params[:c₀_cs], params[:d_hp], ξ2, κ2, params[:Pₛ], params[:Pₛ_cs], params[:K_cs], params[:K_I], params[:k], params[:n], params[:μ_ω], params[:σ_ω], params[:μ_ψ], params[:σ_ψ], params[:μ_α], params[:σ_α]) for t in times]
            
        elseif model_type == "special_independent"
            germ_response = [germ_response_independent_factors_var_perm_gh(u, W3, t, ρₛ, params[:c₀_cs], params[:d_hp], ξ2, κ2, params[:Pₛ], params[:Pₛ_cs], params[:K_cs], params[:μ_γ], params[:σ_γ], params[:μ_ω], params[:σ_ω], params[:μ_α], params[:σ_α]) for t in times]
            
        elseif model_type == "special_combined"
            germ_response = [germ_response_inducer_2_factors_var_perm_gh(u, W4, t, ρₛ, params[:c₀_cs], params[:d_hp], ξ2, κ2, params[:Pₛ], params[:Pₛ_cs], params[:K_cs], params[:K_I], params[:k], params[:n], params[:μ_γ], params[:σ_γ], params[:μ_ω], params[:σ_ω], params[:μ_ψ], params[:σ_ψ], params[:μ_α], params[:σ_α]) for t in times]
            
        elseif model_type == "special_thresh"
            germ_response = [germ_response_inducer_thresh_2_factors_var_perm_gh(u, W4, t, ρₛ, params[:c₀_cs], params[:d_hp], ξ2, κ2, params[:Pₛ], params[:Pₛ_cs], params[:K_cs], params[:k], params[:μ_γ], params[:σ_γ], params[:μ_ω], params[:σ_ω], params[:μ_ψ], params[:σ_ψ], params[:μ_α], params[:σ_α]) for t in times]
            
        elseif model_type == "special_signal"
            germ_response = [germ_response_inducer_signal_2_factors_var_perm_gh(u, W4, t, ρₛ, params[:c₀_cs], params[:d_hp], ξ2, κ2, params[:Pₛ], params[:Pₛ_cs], params[:K_cs], params[:K_I], params[:n], params[:μ_γ], params[:σ_γ], params[:μ_ω], params[:σ_ω], params[:μ_ψ], params[:σ_ψ], params[:μ_α], params[:σ_α]) for t in times]
        
        elseif model_type == "feedback_inhibitor_perm"
            germ_response = germ_response_feedback_inhibitor_perm(sobol_pts, times, ρₛ, samples_A, samples_Vₛ, samples_V_out, samples_V_ps, params[:c₀_cs], params[:s_max], params[:Pₛ], params[:Pₛ_cs], params[:K_cs], params[:μ_γ], params[:σ_γ], params[:μ_ψ], params[:σ_ψ])
            
        end

        return germ_response
    end


    function germ_response_independent_factors_gh(u, W, t, ρₛ, c₀_cs, d_hp, ξ, κ, Pₛ, Pₛ_cs, K_cs, μ_γ, σ_γ, μ_ω, σ_ω)
        """
        Compute the germination response for independent
        inhibition and induction for a given set of parameters.
        The inducer signal is time-dependent.
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
            K_cs - half-saturation constant for the carbon source
            μ_γ - mean inhibition threshold
            σ_γ - standard deviation of inhibition threshold
            μ_ω - mean induction threshold
            σ_ω - standard deviation of induction threshold
        output:
            the germination response for the given parameters (normalized)
        """

        # Distributions
        dist_γ = Normal(μ_γ, σ_γ)
        dist_ω = Normal(μ_ω, σ_ω)

        # Inhibitor
        V = 4/3 * π .* ξ.^3
        A = 4 * π .* ξ.^2
        τ = V ./ (Pₛ * A)
        ϕ = ρₛ .* V
        β = ϕ .+ (1 .- ϕ) .* exp.(-t ./ (τ .* (1 .- ϕ)))

        # Inducer
        V_cw = compute_ps_layer_volume(ξ, d_hp, κ)
        c_cs = inducer_concentration.(c₀_cs, t, Pₛ_cs, A, V_cw)
        s = c_cs ./ (K_cs .+ c_cs)
        
        tail = cdf.(dist_ω, s) .* (1 .- cdf.(dist_γ, β))
        
        return sum(W .* tail)
    end


    function germ_response_inducer_dep_inhibitor_thresh_gh(u, W, t, ρₛ, c₀_cs, d_hp, ξ, κ, Pₛ, Pₛ_cs, k, K_cs, μ_γ, σ_γ)
        """
        Compute the germination response for an inducer-dependent
        inhibitor threshold for a given set of parameters,
        without considering an external initial concentration.
        The inducer signal is time-dependent.
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
            K_cs - half-saturation constant for the carbon source
            μ_γ - mean inhibition threshold
            σ_γ - standard deviation of inhibition threshold
        output:
            the germination response for the given parameters (normalized)
        """

        # Distributions
        dist_γ = Normal(μ_γ, σ_γ)

        # Inhibitor
        V = 4/3 * π .* ξ.^3
        A = 4 * π .* ξ.^2
        τ = V ./ (Pₛ * A)
        ϕ = ρₛ .* V
        β = ϕ .+ (1 .- ϕ) .* exp.(-t ./ (τ .* (1 .- ϕ)))

        # Inducer
        V_cw = compute_ps_layer_volume(ξ, d_hp, κ)
        c_cs = inducer_concentration.(c₀_cs, t, Pₛ_cs, A, V_cw)
        s = c_cs ./ (K_cs .+ c_cs)

        tail = 1 .- cdf.(dist_γ, β .- k .* s)

        return sum(W .* tail)
    end


    function germ_response_inducer_dep_inhibitor_perm_gh(u, W, t, ρₛ, c₀_cs, d_hp, ξ, κ, Pₛ, Pₛ_cs, K_cs, μ_γ, σ_γ)
        """
        Compute the germination response for an inducer-dependent
        inhibitor permeation for a given set of parameters,
        without considering an external initial concentration.
        The inducer signal is time-dependent.
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
            K_cs - half-saturation constant for the carbon source
            μ_γ - mean inhibition threshold
            σ_γ - standard deviation of inhibition threshold
        output:
            the germination response for the given parameters (normalized)
        """

        # Distributions
        dist_γ = Normal(μ_γ, σ_γ)

        # Inducer
        A = 4 * π .* ξ.^2
        V_cw = compute_ps_layer_volume(ξ, d_hp, κ)
        c_cs = inducer_concentration.(c₀_cs, t, Pₛ_cs, A, V_cw)
        s = c_cs ./ (K_cs .+ c_cs)

        # Inhibitor
        V = 4/3 * π .* ξ.^3
        τ = V ./ (s .* Pₛ .* A)
        ϕ = ρₛ .* V
        β = ϕ .+ (1 .- ϕ) .* exp.(-t ./ (τ .* (1 .- ϕ)))

        tail = 1 .- cdf.(dist_γ, β)

        return sum(W .* tail)
    end


    function germ_response_inducer_dep_inhibitor_gh(u, W, t, ρₛ, c₀_cs, d_hp, ξ, κ, Pₛ, Pₛ_cs, K_cs, k, μ_γ, σ_γ)
        """
        Compute the germination response for an inducer-dependent
        inhibitor threshold and permeation for a given set of parameters,
        without considering an external initial concentration.
        The inducer signal is time-dependent.
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
            K_cs - half-saturation constant for the carbon source
            k - proportionality constant for threshold modulation vs permeability modulation
            μ_γ - mean inhibition threshold
            σ_γ - standard deviation of inhibition threshold
        output:
            the germination response for the given parameters (normalized)
        """

        # Distributions
        dist_γ = Normal(μ_γ, σ_γ)

        # Inducer
        A = 4 * π .* ξ.^2
        V_cw = compute_ps_layer_volume(ξ, d_hp, κ)
        c_cs = inducer_concentration.(c₀_cs, t, Pₛ_cs, A, V_cw)
        s = c_cs ./ (K_cs .+ c_cs)

        # Inhibitor
        V = 4/3 * π .* ξ.^3
        τ = V ./ (s .* Pₛ .* A)
        ϕ = ρₛ .* V
        β = ϕ .+ (1 .- ϕ) .* exp.(-t ./ (τ .* (1 .- ϕ)))

        tail = 1 .- cdf.(dist_γ, β .- k .* s)

        return sum(W .* tail)
    end


    function germ_response_inhibitor_dep_inducer_thresh_gh(u, W3, t, ρₛ, c₀_cs, d_hp, ξ, κ, Pₛ, Pₛ_cs, K_cs, k, μ_ω, σ_ω, μ_ψ, σ_ψ)
        """
        Compute the germination response for an inhibitor-dependent
        induction threshold for a given set of parameters.
        The inducer signal is time-dependent.
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
            K_cs - half-saturation constant for the carbon source
            k - inhibition strength over induction threshold
            μ_ω - mean induction threshold
            σ_ω - standard deviation of induction threshold
            μ_ψ - mean initial concentration
            σ_ψ - standard deviation of initial concentration
        output:
            the germination response for the given parameters (normalized)
        """

        # Transform to log-normal
        μ_ψ_log = log(μ_ψ^2 / sqrt(σ_ψ^2 + μ_ψ^2))
        σ_ψ_log = sqrt(log(σ_ψ^2 / μ_ψ^2 + 1))
        ψ = exp.(μ_ψ_log .+ σ_ψ_log .* u)

        # Distributions
        dist_ω = Normal(μ_ω, σ_ω)

        # Inducer
        A = 4 * π .* ξ.^2
        V_cw = compute_ps_layer_volume(ξ, d_hp, κ)
        c_cs = inducer_concentration.(c₀_cs, t, Pₛ_cs, A, V_cw)
        s = c_cs ./ (K_cs .+ c_cs)

        # Inhibitor
        V = 4/3 * π .* ξ.^3
        τ = V ./ (Pₛ * A)
        ϕ = ρₛ .* V
        β = (ϕ .+ (1 .- ϕ) .* exp.(-t ./ (τ .* (1 .- ϕ))))

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


    function germ_response_inhibitor_dep_inducer_signal_gh(u, W3, t, ρₛ, c₀_cs, d_hp, ξ, κ, Pₛ, Pₛ_cs, K_cs, K_I, n, μ_ω, σ_ω, μ_ψ, σ_ψ)
        """
        Compute the germination response for an inhibitor-dependent
        induction signal for a given set of parameters.
        The inducer signal is time-dependent.
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
            K_cs - half-saturation constant for the carbon source
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
        μ_ψ_log = log(μ_ψ^2 / sqrt(σ_ψ^2 + μ_ψ^2))
        σ_ψ_log = sqrt(log(σ_ψ^2 / μ_ψ^2 + 1))
        ψ = exp.(μ_ψ_log .+ σ_ψ_log .* u)

        # Distributions
        dist_ω = Normal(μ_ω, σ_ω)

        # Inducer
        A = 4 * π .* ξ.^2
        V_cw = compute_ps_layer_volume(ξ, d_hp, κ)
        c_cs = inducer_concentration.(c₀_cs, t, Pₛ_cs, A, V_cw)
        s = c_cs ./ (K_cs .+ c_cs)

        # Inhibitor
        V = 4/3 * π .* ξ.^3
        τ = V ./ (Pₛ * A)
        ϕ = ρₛ .* V
        β = ϕ .+ (1 .- ϕ) .* exp.(-t ./ (τ .* (1 .- ϕ)))

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

    
    function germ_response_inhibitor_dep_inducer_gh(u, W3, t, ρₛ, c₀_cs, d_hp, ξ, κ, Pₛ, Pₛ_cs, k, K_cs, K_I, n, μ_ω, σ_ω, μ_ψ, σ_ψ)
        """
        Compute the germination response for an inhibitor-dependent
        induction threshold and signal for a given set of parameters.
        The inducer signal is time-dependent.
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
            K_cs - half-saturation constant for the carbon source
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
        μ_ψ_log = log(μ_ψ^2 / sqrt(σ_ψ^2 + μ_ψ^2))
        σ_ψ_log = sqrt(log(σ_ψ^2 / μ_ψ^2 + 1))
        ψ = exp.(μ_ψ_log .+ σ_ψ_log .* u)

        # Distributions
        dist_ω = Normal(μ_ω, σ_ω)

        # Inducer
        A = 4 * π .* ξ.^2
        V_cw = compute_ps_layer_volume(ξ, d_hp, κ)
        c_cs = inducer_concentration.(c₀_cs, t, Pₛ_cs, A, V_cw)
        s = c_cs ./ (K_cs .+ c_cs)

        # Inhibitor
        V = 4/3 * π .* ξ.^3
        τ = V ./ (Pₛ * A)
        ϕ = ρₛ .* V
        β = ϕ .+ (1 .- ϕ) .* exp.(-t ./ (τ .* (1 .- ϕ)))

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


    function germ_response_inducer_dep_inhibitor_thresh_2_factor_gh(u, W, t, ρₛ, c₀_cs, d_hp, ξ, κ, Pₛ, Pₛ_cs, k, K_cs, μ_γ, σ_γ, μ_ω, σ_ω)
        """
        Compute the germination response for an inducer-dependent
        inhibitor threshold for a given set of parameters,
        without considering an external initial concentration.
        The inducer signal is time-dependent.
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
            K_cs - half-saturation constant for the carbon source
            μ_γ - mean inhibition threshold
            σ_γ - standard deviation of inhibition threshold
            μ_ω - mean induction threshold
            σ_ω - standard deviation of induction threshold
        output:
            the germination response for the given parameters (normalized)
        """

        # Distributions
        dist_γ = Normal(μ_γ, σ_γ)
        dist_ω = Normal(μ_ω, σ_ω)

        # Inhibitor
        V = 4/3 * π .* ξ.^3
        A = 4 * π .* ξ.^2
        τ = V ./ (Pₛ * A)
        ϕ = ρₛ .* V
        β = ϕ .+ (1 .- ϕ) .* exp.(-t ./ (τ .* (1 .- ϕ)))

        # Inducer
        V_cw = compute_ps_layer_volume(ξ, d_hp, κ)
        c_cs = inducer_concentration.(c₀_cs, t, Pₛ_cs, A, V_cw)
        s = c_cs ./ (K_cs .+ c_cs)

        tail = (1 .- cdf.(dist_γ, β .- k .* s)) .* cdf.(dist_ω, s)

        return sum(W .* tail)
    end


    function germ_response_inducer_dep_inhibitor_perm_2_factor_gh(u, W, t, ρₛ, c₀_cs, d_hp, ξ, κ, Pₛ, Pₛ_cs, K_cs, μ_γ, σ_γ, μ_ω, σ_ω)
        """
        Compute the germination response for an inducer-dependent
        inhibitor permeation for a given set of parameters,
        without considering an external initial concentration.
        The inducer signal is time-dependent.
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
            K_cs - half-saturation constant for the carbon source
            μ_γ - mean inhibition threshold
            σ_γ - standard deviation of inhibition threshold
            μ_ω - mean induction threshold
            σ_ω - standard deviation of induction threshold
        output:
            the germination response for the given parameters (normalized)
        """

        # Distributions
        dist_γ = Normal(μ_γ, σ_γ)
        dist_ω = Normal(μ_ω, σ_ω)

        # Inducer
        A = 4 * π .* ξ.^2
        V_cw = compute_ps_layer_volume(ξ, d_hp, κ)
        c_cs = inducer_concentration.(c₀_cs, t, Pₛ_cs, A, V_cw)
        s = c_cs ./ (K_cs .+ c_cs)

        # Inhibitor
        V = 4/3 * π .* ξ.^3
        τ = V ./ (s .* Pₛ .* A)
        ϕ = ρₛ .* V
        β = ϕ .+ (1 .- ϕ) .* exp.(-t ./ (τ .* (1 .- ϕ)))

        tail = (1 .- cdf.(dist_γ, β)) .* cdf.(dist_ω, s)

        return sum(W .* tail)
    end


    function germ_response_inducer_dep_inhibitor_2_factor_gh(u, W, t, ρₛ, c₀_cs, d_hp, ξ, κ, Pₛ, Pₛ_cs, K_cs, k, μ_γ, σ_γ, μ_ω, σ_ω)
        """
        Compute the germination response for an inducer-dependent
        inhibitor threshold and permeation and an additional
        inducer-dependent germination for a given set of parameters.
        The inducer signal is time-dependent.
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
            K_cs - half-saturation constant for the carbon source
            k - proportionality constant for threshold modulation vs permeability modulation
            μ_γ - mean inhibition threshold
            σ_γ - standard deviation of inhibition threshold
            μ_ω - mean induction threshold
            σ_ω - standard deviation of induction threshold
        output:
            the germination response for the given parameters (normalized)
        """

        # Distributions
        dist_γ = Normal(μ_γ, σ_γ)
        dist_ω = Normal(μ_ω, σ_ω)

        # Inducer
        A = 4 * π .* ξ.^2
        V_cw = compute_ps_layer_volume(ξ, d_hp, κ)
        c_cs = inducer_concentration.(c₀_cs, t, Pₛ_cs, A, V_cw)
        s = c_cs ./ (K_cs .+ c_cs)

        # Inhibitor
        V = 4/3 * π .* ξ.^3
        τ = V ./ (s .* Pₛ .* A)
        ϕ = ρₛ .* V
        β = ϕ .+ (1 .- ϕ) .* exp.(-t ./ (τ .* (1 .- ϕ)))

        tail = (1 .- cdf.(dist_γ, β .- k .* s)) .* cdf.(dist_ω, s)

        return sum(W .* tail)
    end


    function germ_response_inhibitor_dep_inducer_thresh_2_factor_gh(u, W3, t, ρₛ, c₀_cs, d_hp, ξ, κ, Pₛ, Pₛ_cs, K_cs, k, μ_γ, σ_γ, μ_ω, σ_ω, μ_ψ, σ_ψ)
        """
        Compute the germination response for an inhibitor-dependent
        induction threshold and an additional inhibitor-dependent
        germination for a given set of parameters.
        The inducer signal is time-dependent.
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
            K_cs - half-saturation constant for the carbon source
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
        μ_ψ_log = log(μ_ψ^2 / sqrt(σ_ψ^2 + μ_ψ^2))
        σ_ψ_log = sqrt(log(σ_ψ^2 / μ_ψ^2 + 1))
        ψ = exp.(μ_ψ_log .+ σ_ψ_log .* u)

        # Distributions
        dist_γ = Normal(μ_γ, σ_γ)
        dist_ω = Normal(μ_ω, σ_ω)

        # Inducer
        A = 4 * π .* ξ.^2
        V_cw = compute_ps_layer_volume(ξ, d_hp, κ)
        c_cs = inducer_concentration.(c₀_cs, t, Pₛ_cs, A, V_cw)
        s = c_cs ./ (K_cs .+ c_cs)

        # Inhibitor
        V = 4/3 * π .* ξ.^3
        τ = V ./ (Pₛ * A)
        ϕ = ρₛ .* V
        β = (ϕ .+ (1 .- ϕ) .* exp.(-t ./ (τ .* (1 .- ϕ))))

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

    function germ_response_inhibitor_dep_inducer_signal_2_factor_gh(u, W3, t, ρₛ, c₀_cs, d_hp, ξ, κ, Pₛ, Pₛ_cs, K_cs, K_I, n, μ_γ, σ_γ, μ_ω, σ_ω, μ_ψ, σ_ψ)
        """
        Compute the germination response for an inhibitor-dependent
        induction signal and an additional inhibitor-dependent
        germination for a given set of parameters.
        The inducer signal is time-dependent.
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
            K_cs - half-saturation constant for the carbon source
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
        μ_ψ_log = log(μ_ψ^2 / sqrt(σ_ψ^2 + μ_ψ^2))
        σ_ψ_log = sqrt(log(σ_ψ^2 / μ_ψ^2 + 1))
        ψ = exp.(μ_ψ_log .+ σ_ψ_log .* u)

        # Distributions
        dist_γ = Normal(μ_γ, σ_γ)
        dist_ω = Normal(μ_ω, σ_ω)

        # Inducer
        A = 4 * π .* ξ.^2
        V_cw = compute_ps_layer_volume(ξ, d_hp, κ)
        c_cs = inducer_concentration.(c₀_cs, t, Pₛ_cs, A, V_cw)
        s = c_cs ./ (K_cs .+ c_cs)

        # Inhibitor
        V = 4/3 * π .* ξ.^3
        τ = V ./ (Pₛ * A)
        ϕ = ρₛ .* V
        β = (ϕ .+ (1 .- ϕ) .* exp.(-t ./ (τ .* (1 .- ϕ))))

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


    function germ_response_inhibitor_dep_inducer_2_factor_gh(u, W3, t, ρₛ, c₀_cs, d_hp, ξ, κ, Pₛ, Pₛ_cs, K_cs, K_I, n, k, μ_γ, σ_γ, μ_ω, σ_ω, μ_ψ, σ_ψ)
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
            K_cs - half-saturation constant for the carbon source
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
        μ_ψ_log = log(μ_ψ^2 / sqrt(σ_ψ^2 + μ_ψ^2))
        σ_ψ_log = sqrt(log(σ_ψ^2 / μ_ψ^2 + 1))
        ψ = exp.(μ_ψ_log .+ σ_ψ_log .* u)

        # Distributions
        dist_γ = Normal(μ_γ, σ_γ)
        dist_ω = Normal(μ_ω, σ_ω)

        # Inducer
        A = 4 * π .* ξ.^2
        V_cw = compute_ps_layer_volume(ξ, d_hp, κ)
        c_cs = inducer_concentration.(c₀_cs, t, Pₛ_cs, A, V_cw)
        s = c_cs ./ (K_cs .+ c_cs)

        # Inhibitor
        V = 4/3 * π .* ξ.^3
        τ = V ./ (Pₛ * A)
        ϕ = ρₛ .* V
        β = (ϕ .+ (1 .- ϕ) .* exp.(-t ./ (τ .* (1 .- ϕ))))

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


    function germ_response_inducer_thresh_var_perm_gh(u, W4, t, ρₛ, c₀_cs, d_hp, ξ, κ, Pₛ, Pₛ_cs, K_cs, k, μ_ω, σ_ω, μ_ψ, σ_ψ, μ_α, σ_α)
        """
        Compute the germination response for an inhibitor-dependent
        induction threshold for a given set of parameters,
        whereby the permeation constant is a random variable.
        The inducer signal is time-dependent.
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
            K_cs - half-saturation constant for the carbon source
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
        μ_ψ_log = log(μ_ψ^2 / sqrt(σ_ψ^2 + μ_ψ^2))
        σ_ψ_log = sqrt(log(σ_ψ^2 / μ_ψ^2 + 1))
        ψ = exp.(μ_ψ_log .+ σ_ψ_log .* u)
        μ_α_log = log(μ_α^2 / sqrt(σ_α^2 + μ_α^2))
        σ_α_log = sqrt(log(σ_α^2 / μ_α^2 + 1))
        α = exp.(μ_α_log .+ σ_α_log .* u)

        # Distributions
        dist_ω = Normal(μ_ω, σ_ω)

        # Modulate permeation
        Pₛ = Pₛ .* α
        Pₛ_cs = Pₛ_cs .* α

        # Cell wall and spore volumes
        A = 4 * π .* ξ.^2
        V_cw = compute_ps_layer_volume(ξ, d_hp, κ)
        V = 4/3 * π .* ξ.^3

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
        c_cs = inducer_concentration.(c₀_cs, t, Pₛ_cs, A, V_cw)
        s = c_cs ./ (K_cs .+ c_cs)

        # Inhibitor
        τ = V ./ (Pₛ .* A)
        ϕ = ρₛ .* V
        β = (ϕ .+ (1 .- ϕ) .* exp.(-t ./ (τ .* (1 .- ϕ))))

        # Reshape
        β = repeat(β, 1, 1, 1, n_nodes)
        s = repeat(s, 1, 1, 1, n_nodes)
        ψ = repeat(ψ, 1, n_nodes, n_nodes, n_nodes)
        ψ = permutedims(ψ, (2, 3, 4, 1))

        c_in = ψ .* β

        tail = cdf.(dist_ω, s .- k .* c_in)

        return sum(W4 .* tail)
    end


    function germ_response_inducer_var_perm_gh(u, W4, t, ρₛ, c₀_cs, d_hp, ξ, κ, Pₛ, Pₛ_cs, K_cs, K_I, k, n, μ_ω, σ_ω, μ_ψ, σ_ψ, μ_α, σ_α)
        """
        Compute the germination response for an inhibitor-dependent
        induction threshold and signal for a given set of parameters,
        whereby the permeation constant is a random variable.
        The inducer signal is time-dependent.
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
            K_cs - half-saturation constant for the carbon source
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
        μ_ψ_log = log(μ_ψ^2 / sqrt(σ_ψ^2 + μ_ψ^2))
        σ_ψ_log = sqrt(log(σ_ψ^2 / μ_ψ^2 + 1))
        ψ = exp.(μ_ψ_log .+ σ_ψ_log .* u)
        μ_α_log = log(μ_α^2 / sqrt(σ_α^2 + μ_α^2))
        σ_α_log = sqrt(log(σ_α^2 / μ_α^2 + 1))
        α = exp.(μ_α_log .+ σ_α_log .* u)

        # Distributions
        dist_ω = Normal(μ_ω, σ_ω)

        # Modulate permeation
        Pₛ = Pₛ .* α
        Pₛ_cs = Pₛ_cs .* α

        # Cell wall and spore volumes
        A = 4 * π .* ξ.^2
        V_cw = compute_ps_layer_volume(ξ, d_hp, κ)
        V = 4/3 * π .* ξ.^3

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
        c_cs = inducer_concentration.(c₀_cs, t, Pₛ_cs, A, V_cw)
        s = c_cs ./ (K_cs .+ c_cs)

        # Inhibitor
        τ = V ./ (Pₛ .* A)
        ϕ = ρₛ .* V
        β = (ϕ .+ (1 .- ϕ) .* exp.(-t ./ (τ .* (1 .- ϕ))))

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


    function germ_response_independent_factors_var_perm_gh(u, W3, t, ρₛ, c₀_cs, d_hp, ξ, κ, Pₛ, Pₛ_cs, K_cs, μ_γ, σ_γ, μ_ω, σ_ω, μ_α, σ_α)
        """
        Compute the germination response for independent
        inhibition and induction for a given set of parameters,
        whereby the permeation constant is a random variable.
        The inducer signal is time-dependent.
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
            K_cs - half-saturation constant for the carbon source
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
        μ_α_log = log(μ_α^2 / sqrt(σ_α^2 + μ_α^2))
        σ_α_log = sqrt(log(σ_α^2 / μ_α^2 + 1))
        α = exp.(μ_α_log .+ σ_α_log .* u)

        # Distributions
        dist_γ = Normal(μ_γ, σ_γ)
        dist_ω = Normal(μ_ω, σ_ω)

        # Spore volume and surface area
        V = 4/3 * π .* ξ.^3
        A = 4 * π .* ξ.^2

        # Cell wall volume
        V_cw = compute_ps_layer_volume(ξ, d_hp, κ)

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

        τ = V ./ (Pₛ .* A)
        ϕ = ρₛ .* V
        β = ϕ .+ (1 .- ϕ) .* exp.(-t ./ (τ .* (1 .- ϕ)))

        # Inducer
        c_cs = inducer_concentration.(c₀_cs, t, Pₛ_cs, A, V_cw)
        s = c_cs ./ (K_cs .+ c_cs)
        
        tail = cdf.(dist_ω, s) .* (1 .- cdf.(dist_γ, β))
        
        return sum(W3 .* tail)
    end


    function germ_response_inducer_thresh_2_factors_var_perm_gh(u, W4, t, ρₛ, c₀_cs, d_hp, ξ, κ, Pₛ, Pₛ_cs, K_cs, k, μ_γ, σ_γ, μ_ω, σ_ω, μ_ψ, σ_ψ, μ_α, σ_α)
        """
        Compute the germination response for an inhibitor-dependent
        induction threshold for a given set of parameters,
        whereby the permeation constant is a random variable.
        The inducer signal is time-dependent.
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
            K_cs - half-saturation constant for the carbon source
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
        μ_ψ_log = log(μ_ψ^2 / sqrt(σ_ψ^2 + μ_ψ^2))
        σ_ψ_log = sqrt(log(σ_ψ^2 / μ_ψ^2 + 1))
        ψ = exp.(μ_ψ_log .+ σ_ψ_log .* u)
        μ_α_log = log(μ_α^2 / sqrt(σ_α^2 + μ_α^2))
        σ_α_log = sqrt(log(σ_α^2 / μ_α^2 + 1))
        α = exp.(μ_α_log .+ σ_α_log .* u)

        # Distributions
        dist_γ = Normal(μ_γ, σ_γ)
        dist_ω = Normal(μ_ω, σ_ω)

        # Modulate permeation
        Pₛ = Pₛ .* α
        Pₛ_cs = Pₛ_cs .* α

        # Cell wall and spore volumes
        A = 4 * π .* ξ.^2
        V_cw = compute_ps_layer_volume(ξ, d_hp, κ)
        V = 4/3 * π .* ξ.^3

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
        c_cs = inducer_concentration.(c₀_cs, t, Pₛ_cs, A, V_cw)
        s = c_cs ./ (K_cs .+ c_cs)

        # Inhibitor
        τ = V ./ (Pₛ .* A)
        ϕ = ρₛ .* V
        β = (ϕ .+ (1 .- ϕ) .* exp.(-t ./ (τ .* (1 .- ϕ))))

        # Reshape
        β = repeat(β, 1, 1, 1, n_nodes)
        s = repeat(s, 1, 1, 1, n_nodes)
        ψ = repeat(ψ, 1, n_nodes, n_nodes, n_nodes)
        ψ = permutedims(ψ, (2, 3, 4, 1))

        c_in = ψ .* β

        tail = cdf.(dist_ω, s .- k .* c_in) .* (1 .- cdf.(dist_γ, β))

        return sum(W4 .* tail)
    end


    function germ_response_inducer_signal_2_factors_var_perm_gh(u, W4, t, ρₛ, c₀_cs, d_hp, ξ, κ, Pₛ, Pₛ_cs, K_cs, K_I, n, μ_γ, σ_γ, μ_ω, σ_ω, μ_ψ, σ_ψ, μ_α, σ_α)
        """
        Compute the germination response for an inhibitor-dependent
        induction signal for a given set of parameters,
        whereby the permeation constant is a random variable.
        The inducer signal is time-dependent.
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
            K_cs - half-saturation constant for the carbon source
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
        μ_ψ_log = log(μ_ψ^2 / sqrt(σ_ψ^2 + μ_ψ^2))
        σ_ψ_log = sqrt(log(σ_ψ^2 / μ_ψ^2 + 1))
        ψ = exp.(μ_ψ_log .+ σ_ψ_log .* u)
        μ_α_log = log(μ_α^2 / sqrt(σ_α^2 + μ_α^2))
        σ_α_log = sqrt(log(σ_α^2 / μ_α^2 + 1))
        α = exp.(μ_α_log .+ σ_α_log .* u)

        # Distributions
        dist_γ = Normal(μ_γ, σ_γ)
        dist_ω = Normal(μ_ω, σ_ω)

        # Modulate permeation
        Pₛ = Pₛ .* α
        Pₛ_cs = Pₛ_cs .* α

        # Cell wall and spore volumes
        A = 4 * π .* ξ.^2
        V_cw = compute_ps_layer_volume(ξ, d_hp, κ)
        V = 4/3 * π .* ξ.^3

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
        c_cs = inducer_concentration.(c₀_cs, t, Pₛ_cs, A, V_cw)
        s = c_cs ./ (K_cs .+ c_cs)

        # Inhibitor
        τ = V ./ (Pₛ .* A)
        ϕ = ρₛ .* V
        β = (ϕ .+ (1 .- ϕ) .* exp.(-t ./ (τ .* (1 .- ϕ))))

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


    function germ_response_inducer_2_factors_var_perm_gh(u, W4, t, ρₛ, c₀_cs, d_hp, ξ, κ, Pₛ, Pₛ_cs, K_cs, K_I, k, n, μ_γ, σ_γ, μ_ω, σ_ω, μ_ψ, σ_ψ, μ_α, σ_α)
        """
        Compute the germination response for an inhibitor-dependent
        induction threshold and signal for a given set of parameters,
        whereby the permeation constant is a random variable.
        The inducer signal is time-dependent.
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
            K_cs - half-saturation constant for the carbon source
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
        μ_ψ_log = log(μ_ψ^2 / sqrt(σ_ψ^2 + μ_ψ^2))
        σ_ψ_log = sqrt(log(σ_ψ^2 / μ_ψ^2 + 1))
        ψ = exp.(μ_ψ_log .+ σ_ψ_log .* u)
        μ_α_log = log(μ_α^2 / sqrt(σ_α^2 + μ_α^2))
        σ_α_log = sqrt(log(σ_α^2 / μ_α^2 + 1))
        α = exp.(μ_α_log .+ σ_α_log .* u)

        # Distributions
        dist_γ = Normal(μ_γ, σ_γ)
        dist_ω = Normal(μ_ω, σ_ω)

        # Modulate permeation
        Pₛ = Pₛ .* α
        Pₛ_cs = Pₛ_cs .* α

        # Cell wall and spore volumes
        A = 4 * π .* ξ.^2
        V_cw = compute_ps_layer_volume(ξ, d_hp, κ)
        V = 4/3 * π .* ξ.^3

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
        c_cs = inducer_concentration.(c₀_cs, t, Pₛ_cs, A, V_cw)
        s = c_cs ./ (K_cs .+ c_cs)

        # Inhibitor
        τ = V ./ (Pₛ .* A)
        ϕ = ρₛ .* V
        β = (ϕ .+ (1 .- ϕ) .* exp.(-t ./ (τ .* (1 .- ϕ))))

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


    function germ_response_inhibitor_dep_inducer_2_factors_eq(ρₛ, dist_ξ, c₀_cs, K_cs, K_I, k, n, μ_γ, σ_γ, μ_ω, σ_ω, μ_ψ, σ_ψ; reltol=1e-4, abstol=1e-6)
        """
        Compute the equilibrium germination response
        for an inhibitor-dependent inducer threshold and signal and
        an additional inhibitor-dependent germination.
        inputs:
            ρₛ - spore density in spores/um^3
            dist_ξ - distribution of spore radii (LogNormal)
            c₀_cs - initial concentration of carbon source in M
            K_cs - half-saturation constant for the carbon source
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
        s_eq = c₀_cs ./ (K_cs .+ c₀_cs)

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


    function germ_response_inhibitor_dep_inducer_2_factors_eq_c_ex(ρₛ, dist_ξ, c_ex, c₀_cs, K_cs, K_I, k, n, μ_γ, σ_γ, μ_ω, σ_ω, μ_ψ, σ_ψ; reltol=1e-4, abstol=1e-6)
        """
        Compute the equilibrium germination response
        for an inhibitor-dependent inducer threshold and signal and
        an additional inhibitor-dependent germination.
        inputs:
            ρₛ - spore density in spores/um^3
            dist_ξ - distribution of spore radii (LogNormal)
            c_ex - external concentration of the inducer in M
            c₀_cs - initial concentration of carbon source in M
            K_cs - half-saturation constant for the carbon source
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
        s_eq = c₀_cs ./ (K_cs .+ c₀_cs)

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


    function germ_response_inhibitor_dep_inducer_thresh_2_factors_eq(ρₛ, dist_ξ, c₀_cs, K_cs, k, μ_γ, σ_γ, μ_ω, σ_ω, μ_ψ, σ_ψ; reltol=1e-4, abstol=1e-6)
        """
        Compute the equilibrium germination response
        for an inhibitor-dependent inducer threshold and
        an additional inhibitor-dependent germination.
        inputs:
            ρₛ - spore density in spores/um^3
            dist_ξ - distribution of spore radii (LogNormal)
            c₀_cs - initial concentration of carbon source in M
            K_cs - half-saturation constant for the carbon source
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
        s_eq = c₀_cs ./ (K_cs .+ c₀_cs)

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


    function germ_response_inhibitor_dep_inducer_thresh_2_factors_eq_c_ex(ρₛ, dist_ξ, c_ex, c₀_cs, K_cs, k, μ_γ, σ_γ, μ_ω, σ_ω, μ_ψ, σ_ψ; reltol=1e-4)
        """
        Compute the equilibrium germination response
        for an inhibitor-dependent inducer threshold and
        an additional inhibitor-dependent germination.
        inputs:
            ρₛ - spore density in spores/um^3
            dist_ξ - distribution of spore radii (LogNormal)
            c_ex - external concentration of the inducer in M
            c₀_cs - initial concentration of carbon source in M
            K_cs - half-saturation constant for the carbon source
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
        s_eq = c₀_cs ./ (K_cs .+ c₀_cs)

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


    function germ_response_inhibitor_dep_inducer_signal_2_factors_eq(t, ρₛ, dist_ξ, c₀_cs, K_cs, K_I, n, μ_γ, σ_γ, μ_ω, σ_ω, μ_ψ, σ_ψ; reltol=1e-4, abstol=1e-6)
        """
        Compute the equilibrium germination response
        for an inhibitor-dependent inducer signal and
        an additional inhibitor-dependent germination.
        inputs:
            ρₛ - spore density in spores/um^3
            dist_ξ - distribution of spore radii (LogNormal)
            c₀_cs - initial concentration of carbon source in M
            K_cs - half-saturation constant for the carbon source
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
        s_eq = c₀_cs ./ (K_cs .+ c₀_cs)

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


    function germ_response_inhibitor_dep_inducer_signal_2_factors_eq_c_ex(ρₛ, dist_ξ, c_ex, c₀_cs, K_cs, K_I, n, μ_γ, σ_γ, μ_ω, σ_ω, μ_ψ, σ_ψ; reltol=1e-4, abstol=1e-6)
        """
        Compute the equilibrium germination response
        for an inhibitor-dependent inducer signal and
        an additional inhibitor-dependent germination.
        inputs:
            ρₛ - spore density in spores/um^3
            dist_ξ - distribution of spore radii (LogNormal)
            c_ex - external concentration of the inducer in M
            c₀_cs - initial concentration of carbon source in M
            K_cs - half-saturation constant for the carbon source
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
        s_eq = c₀_cs ./ (K_cs .+ c₀_cs)

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


    function germ_response_independent_eq(ρₛ, dist_ξ, c₀_cs, K_cs, μ_γ, σ_γ, μ_ω, σ_ω; reltol=1e-4)
        """
        Compute the equilibrium germination response
        for independent inhibition and induction.
        inputs:
            ρₛ - spore density in spores/um^3
            dist_ξ - distribution of spore radii (LogNormal)
            c₀_cs - initial concentration of carbon source in M
            K_cs - half-saturation constant for the carbon source
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
        s_eq = c₀_cs ./ (K_cs .+ c₀_cs)
        tail2 = cdf(dist_ω, s_eq)

        function integrand(ξ)
            V = 4/3 * π .* ξ^3
            ϕ = ρₛ .* V
            tail1 = 1 .- cdf(dist_γ, ϕ)
            return tail1 * tail2 * pdf(dist_ξ, ξ)
        end

        return quadgk(x -> integrand(x), 0.0, Inf, rtol=reltol)[1]
    end


    function germ_response_independent_eq_c_ex(ρₛ, dist_ξ, c_ex, c₀_cs, K_cs, μ_γ, σ_γ, μ_ω, σ_ω, μ_ψ, σ_ψ; reltol=1e-4)
        """
        Compute the equilibrium germination response
        for independent inhibition and induction.
        inputs:
            ρₛ - spore density in spores/um^3
            dist_ξ - distribution of spore radii (LogNormal)
            c_ex - external concentration of the inducer in M
            c₀_cs - initial concentration of carbon source in M
            K_cs - half-saturation constant for the carbon source
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
        s_eq = c₀_cs ./ (K_cs .+ c₀_cs)
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
    function germ_response_feedback_inhibitor_perm(sobol_pts, times, ρₛ, samples_A, samples_Vₛ, samples_V_out, samples_V_ps, c₀_cs, s_max, Pₛ_I, Pₛ_C, K_cs, μ_γ, σ_γ, μ_ψ, σ_ψ)
        """
        Compute the germination response for inducer-dependent
        cell wall permeability and an inhibitor-dependent germination.
        inputs:
            sobol_pts - normalized Sobol samples
            times - integration time frames in seconds
            ρₛ - spore density in spores/um^3
            samples_A - spore area samples (corresponding to sobol_pts)
            samples_Vₛ - spore volume samples (corresponding to sobol_pts)
            samples_V_out - outside volume samples (corresponding to sobol_pts)
            samples_V_ps - polysaccharide layer volume samples (corresponding to sobol_pts)
            c₀_cs - initial concentration of carbon source in M
            s_max - maximum inductive signal strength
            Pₛ_I - permeation constant for the inhibitor in um/s
            Pₛ_C - permeation constant for the carbon source in um/s
            K_cs - half-saturation constant for the carbon source
            μ_γ - mean inhibition threshold
            σ_γ - standard deviation of inhibition threshold
            μ_ψ - mean initial concentration
            σ_ψ - standard deviation of initial concentration
        output:
            the germination response for the given parameters (normalized)
        """

        μ_ψ_log = log(μ_ψ^2 / sqrt(σ_ψ^2 + μ_ψ^2))
        σ_ψ_log = sqrt(log(σ_ψ^2 / μ_ψ^2 + 1))
        dist_ψ = LogNormal(μ_ψ_log, σ_ψ_log)
        # samples_ψ = max.(quantile(dist_ψ, sobol_pts[2,:]), 1e-12)
        samples_ψ = clamp_inplace!(quantile(dist_ψ, sobol_pts[3,:]))

        μ_γ_log = log(μ_γ^2 / sqrt(σ_γ^2 + μ_γ^2))
        σ_γ_log = sqrt(log(σ_γ^2 / μ_γ^2 + 1))
        dist_γ = LogNormal(μ_γ_log, σ_γ_log)
        # samples_γ = max.(quantile(dist_γ, sobol_pts[3,:]), 1e-12)
        samples_γ = clamp_inplace!(quantile(dist_γ, sobol_pts[4,:]))

        # ODE function
        function ode!(du, u, p, t)
            cinI, coutI, cinC = u

            denom = p.K_cs + cinC
            g = (denom * (1 + p.s_max)) * p.A / denom
            rateI = g * p.Pₛ_I
            rateC = g * p.Pₛ_C

            du[1] = -(rateI / p.Vₛ) * (cinI - coutI)
            du[2] = (rateI / p.V_out) * (cinI - coutI)
            du[3] = -(rateC / p.V_ps) * (cinC - p.c₀_cs)
        end

        # Template problem
        p_template = (A=samples_A[1], Vₛ=samples_Vₛ[1], V_out=samples_V_out[1], V_ps=samples_V_ps[1],
                    Pₛ_I=Pₛ_I, Pₛ_C=Pₛ_C, K_cs=K_cs, s_max=s_max, c₀_cs=c₀_cs)
        u0_template = [μ_ψ, 0.0, 0.0]
        tspan = (0.0, maximum(times))
        prob = ODEProblem(ode!, u0_template, tspan, p_template)

        # Ensemble integration function
        function prob_func(prob, i, repeat)
            # r = max(samples_ξ[i], 1e-9)
            
            new_p = (A=samples_A[i], Vₛ=samples_Vₛ[i], V_out=samples_V_out[i], V_ps=samples_V_ps[i],
                    Pₛ_I=Pₛ_I, Pₛ_C=Pₛ_C, K_cs=K_cs, s_max=s_max, c₀_cs=c₀_cs)

            new_u0 = [samples_ψ[i], 0.0, 0.0]
            remake(prob; u0 = new_u0, p = new_p)
        end

        # Run ODE ensembles
        ep = EnsembleProblem(prob; prob_func=prob_func)
        # sols = solve(ep, Tsit5(), EnsembleThreads(), trajectories=n_samples, saveat=[t])
        # sols = solve(ep, Rodas5(), EnsembleThreads(), trajectories=n_samples, saveat=[t])
        sols = solve(ep, AutoTsit5(Rosenbrock23()), EnsembleThreads(), trajectories=size(sobol_pts, 2), saveat=times, abstol=1e-6, reltol=1e-6)
        # sols = solve(ep, Rosenbrock23(), EnsembleThreads(), trajectories=n_samples, saveat=[t])

        # Evaluate fraction germinated
        c_in_I_t = [[sol(t)[1] for sol in sols.u] for t in times]
        germinated = [mean(c_in_I_t[i] .< samples_γ) for i in 1:length(times)]
        
        return germinated
    end
    
end