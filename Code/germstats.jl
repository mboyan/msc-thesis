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
    using Revise

    include("./conversions.jl")
    Revise.includet("./conversions.jl")
    using .Conversions

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
    export germ_response_independent_factors_var_perm_st_gh
    export germ_response_inducer_dep_inhibitor_combined_eq
    export germ_response_inducer_dep_inhibitor_combined_eq_c_ex
    export germ_response_inhibitor_dep_inducer_thresh_2_factors_eq
    export germ_response_inhibitor_dep_inducer_thresh_2_factors_eq_c_ex
    export germ_response_independent_eq
    export germ_response_independent_eq_c_ex


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
    function compute_germination_response(model_type, st, times, ρₛ, params; n_nodes=nothing)
        """
        Generic wrapper function to compute the germination response.
        inputs:
            model_type (String): model type to fit
                ("independent", "inhibitor", "inhibitor_thresh", "inhibitor_perm", "inducer", "inducer_thresh", "inducer_signal")
            st (Bool): whether to use a time-dependent inducer
            times (Vector{Float64}): time points to compute the germination response
            ρₛ (float) - spore density in spores/um^3
            n_nodes (int) - number of Gauss-Hermite nodes to use
            params (Dict) - additional parameters for the germination response function
            n_nodes (int) - number of Gauss-Hermite nodes to use
        """

        @argcheck model_type in ["independent",
                                "inhibitor", "inhibitor_thresh", "inhibitor_perm",
                                "inducer", "inducer_thresh", "inducer_signal",
                                "combined_inhibitor", "combined_inhibitor_thresh", "combined_inhibitor_perm",
                                "combined_inducer", "combined_inducer_thresh", "combined_inducer_signal",
                                "special_inhibitor", "special_inducer", "special_independent"]

        # Determine number of nodes depending on the integral dimension (if not specified)
        if isnothing(n_nodes)
            if model_type in ["independent", "inhibitor", "inhibitor_thresh", "inhibitor_perm"] && !st
                n_nodes = 216 # 1D integral
            elseif (model_type in ["inducer", "inducer_thresh", "inducer_signal"] && !st) ||
                    (model_type in ["independent", "inhibitor", "inhibitor_thresh", "inhibitor_perm"] && st) ||
                    model_type in ["combined_inhibitor", "combined_inhibitor_thresh", "combined_inhibitor_perm"] ||
                    (model_type in ["special_inhibitor"])
                n_nodes = 36 # 2D integral
            elseif (model_type in ["inducer", "inducer_thresh", "inducer_signal"] && st) ||
                    model_type in ["combined_inducer", "combined_inducer_thresh", "combined_inducer_signal", "special_independent"]
                n_nodes = 10 # 3D integral
            elseif model_type == "special_inducer"
                n_nodes = 6 # 4D integral
            end
        end
        
        # Gauss-Hermite nodes and weights
        ghnodes, ghweights = gausshermite(n_nodes)
        u = √2 .* ghnodes
        hw = ghweights ./ √π

        # Unpack means and stds and weight samples
        μ_ξ = params[:μ_ξ]
        σ_ξ = params[:σ_ξ]
        μ_ξ_log = log(μ_ξ^2 / sqrt(σ_ξ^2 + μ_ξ^2))
        σ_ξ_log = sqrt(log(σ_ξ^2 / μ_ξ^2 + 1))
        ξ = exp.(μ_ξ_log .+ σ_ξ_log .* u)

        if haskey(params, :μ_κ)
            μ_κ = params[:μ_κ]
            σ_κ = params[:σ_κ]
            μ_κ_log = log(μ_κ^2 / sqrt(σ_κ^2 + μ_κ^2))
            σ_κ_log = sqrt(log(σ_κ^2 / μ_κ^2 + 1))
            κ = exp.(μ_κ_log .+ σ_κ_log .* u)

            ξ2, κ2 = meshgrid(ξ, κ)
        end

        # Weight tensors
        if (model_type in ["independent", "inhibitor", "inhibitor_thresh", "inhibitor_perm"] && st) ||
            (model_type in ["inducer", "inducer_thresh", "inducer_signal"] && !st) ||
            model_type in ["combined_inhibitor", "combined_inhibitor_thresh", "combined_inhibitor_perm"] ||
            model_type in ["special_inhibitor"]
            W = hw * hw'
        elseif (model_type in ["inducer", "inducer_thresh", "inducer_signal"] && st) ||
            model_type in ["combined_inducer", "combined_inducer_thresh", "combined_inducer_signal", "special_independent"]
            W3 = reshape(hw, n_nodes,1,1) .* reshape(hw, 1,n_nodes,1) .* reshape(hw, 1,1,n_nodes)
        elseif model_type == "special_inducer"
            W4 = reshape(hw, n_nodes,1,1,1) .* reshape(hw, 1,n_nodes,1,1) .* reshape(hw, 1,1,n_nodes,1) .* reshape(hw, 1,1,1,n_nodes)
        end

        # Compute the germination response
        if model_type == "independent"
            if !st
                germ_response = [germ_response_independent_factors_gh(u, hw, t, ρₛ, ξ, params[:Pₛ], params[:μ_γ], params[:σ_γ], params[:μ_ω], params[:σ_ω]) for t in times]
            else
                germ_response = [germ_response_independent_factors_st_gh(u, W, t, ρₛ, params[:c₀_cs], params[:d_hp], ξ2, κ2, params[:Pₛ], params[:Pₛ_cs], params[:K_cs], params[:μ_γ], params[:σ_γ], params[:μ_ω], params[:σ_ω]) for t in times]
            end
        elseif model_type == "inhibitor"
            if !st
                germ_response = [germ_response_inhibitor_gh(u, hw, t, ρₛ, ξ, params[:Pₛ], params[:μ_γ], params[:σ_γ]) for t in times]
            else
                germ_response = [germ_response_inducer_dep_inhibitor_combined_st_gh(u, W, t, ρₛ, params[:c₀_cs], params[:d_hp], ξ2, κ2, params[:Pₛ], params[:Pₛ_cs], params[:K_cs], params[:k], params[:μ_γ], params[:σ_γ]) for t in times]
            end
        elseif model_type == "inhibitor_thresh"
            if !st
                germ_response = [germ_response_inhibitor_gh(u, hw, t, ρₛ, ξ, params[:Pₛ], params[:μ_γ], params[:σ_γ]) for t in times]
            else
                germ_response = [germ_response_inducer_dep_inhibitor_thresh_st_gh(u, W, t, ρₛ, params[:c₀_cs], params[:d_hp], ξ2, κ2, params[:Pₛ], params[:Pₛ_cs], params[:k], params[:K_cs], params[:μ_γ], params[:σ_γ]) for t in times]
            end
        elseif model_type == "inhibitor_perm"
            if !st
                germ_response = [germ_response_inhibitor_gh(u, hw, t, ρₛ, ξ, params[:Pₛ], params[:μ_γ], params[:σ_γ]) for t in times]
            else
                germ_response = [germ_response_inducer_dep_inhibitor_perm_st_gh(u, W, t, ρₛ, params[:c₀_cs], params[:d_hp], ξ2, κ2, params[:Pₛ], params[:Pₛ_cs], params[:K_cs], params[:μ_γ], params[:σ_γ]) for t in times]
            end
        elseif model_type == "inducer"
            if !st
                germ_response = [germ_response_inhibitor_dep_inducer_combined_gh(u, W, t, ρₛ, ξ, params[:Pₛ], params[:k], params[:K_I], params[:n], params[:μ_ω], params[:σ_ω], params[:μ_ψ], params[:σ_ψ]) for t in times]
            else
                germ_response = [germ_response_inhibitor_dep_inducer_combined_st_gh(u, W3, t, ρₛ, params[:c₀_cs], params[:d_hp], ξ2, κ2, params[:Pₛ], params[:Pₛ_cs], params[:k], params[:K_cs], params[:K_I], params[:n], params[:μ_ω], params[:σ_ω], params[:μ_ψ], params[:σ_ψ]) for t in times]
            end
        elseif model_type == "inducer_thresh"
            if !st
                germ_response = [germ_response_inhibitor_dep_inducer_thresh_gh(u, W, t, ρₛ, ξ, params[:Pₛ], params[:k], params[:μ_ω], params[:σ_ω], params[:μ_ψ], params[:σ_ψ]) for t in times]
            else
                germ_response = [germ_response_inhibitor_dep_inducer_thresh_st_gh(u, W3, t, ρₛ, params[:c₀_cs], params[:d_hp], ξ2, κ2, params[:Pₛ], params[:Pₛ_cs], params[:K_cs], params[:k], params[:μ_ω], params[:σ_ω], params[:μ_ψ], params[:σ_ψ]) for t in times]
            end
        elseif model_type == "inducer_signal"
            if !st
                germ_response = [germ_response_inhibitor_dep_inducer_signal_gh(u, W, t, ρₛ, ξ, params[:Pₛ], params[:K_I], params[:n], params[:μ_ω], params[:σ_ω], params[:μ_ψ], params[:σ_ψ]) for t in times]
            else
                germ_response = [germ_response_inhibitor_dep_inducer_signal_st_gh(u, W3, t, ρₛ, params[:c₀_cs], params[:d_hp], ξ2, κ2, params[:Pₛ], params[:Pₛ_cs], params[:K_cs], params[:K_I], params[:n], params[:μ_ω], params[:σ_ω], params[:μ_ψ], params[:σ_ψ]) for t in times]
            end
        elseif model_type == "combined_inhibitor"
            germ_response = [germ_response_inducer_dep_inhibitor_combined_2_factor_st_gh(u, W, t, ρₛ, params[:c₀_cs], params[:d_hp], ξ2, κ2, params[:Pₛ], params[:Pₛ_cs], params[:K_cs], params[:k], params[:μ_γ], params[:σ_γ], params[:μ_ω], params[:σ_ω]) for t in times]
        elseif model_type == "combined_inhibitor_thresh"
            germ_response = [germ_response_inducer_dep_inhibitor_thresh_2_factor_st_gh(u, W, t, ρₛ, params[:c₀_cs], params[:d_hp], ξ2, κ2, params[:Pₛ], params[:Pₛ_cs], params[:k], params[:K_cs], params[:μ_γ], params[:σ_γ], params[:μ_ω], params[:σ_ω]) for t in times]
        elseif model_type == "combined_inhibitor_perm"
            germ_response = [germ_response_inducer_dep_inhibitor_perm_2_factor_st_gh(u, W, t, ρₛ, params[:c₀_cs], params[:d_hp], ξ2, κ2, params[:Pₛ], params[:Pₛ_cs], params[:K_cs], params[:μ_γ], params[:σ_γ], params[:μ_ω], params[:σ_ω]) for t in times]
        elseif model_type == "combined_inducer"
            germ_response = [germ_response_inhibitor_dep_inducer_combined_2_factor_st_gh(u, W3, t, ρₛ, params[:c₀_cs], params[:d_hp], ξ2, κ2, params[:Pₛ], params[:Pₛ_cs], params[:K_cs], params[:K_I], params[:n], params[:k], params[:μ_γ], params[:σ_γ], params[:μ_ω], params[:σ_ω], params[:μ_ψ], params[:σ_ψ]) for t in times]
        elseif model_type == "combined_inducer_thresh"
            germ_response = [germ_response_inhibitor_dep_inducer_thresh_2_factor_st_gh(u, W3, t, ρₛ, params[:c₀_cs], params[:d_hp], ξ2, κ2, params[:Pₛ], params[:Pₛ_cs], params[:K_cs], params[:k], params[:μ_γ], params[:σ_γ], params[:μ_ω], params[:σ_ω], params[:μ_ψ], params[:σ_ψ]) for t in times]
        elseif model_type == "combined_inducer_signal"
            germ_response = [germ_response_inhibitor_dep_inducer_signal_2_factor_st_gh(u, W3, t, ρₛ, params[:c₀_cs], params[:d_hp], ξ2, κ2, params[:Pₛ], params[:Pₛ_cs], params[:K_cs], params[:K_I], params[:n], params[:μ_γ], params[:σ_γ], params[:μ_ω], params[:σ_ω], params[:μ_ψ], params[:σ_ψ]) for t in times]
        elseif model_type == "special_inhibitor"
            germ_response = [germ_response_inhibitor_var_perm_gh(u, W, t, ρₛ, ξ, params[:μ_π], params[:σ_π], params[:μ_γ], params[:σ_γ]) for t in times]
        elseif model_type == "special_inducer"
            germ_response = [germ_response_inducer_thresh_var_perm_st_gh(u, W4, t, ρₛ, params[:c₀_cs], params[:d_hp], ξ2, κ2, params[:Pₛ], params[:Pₛ_cs], params[:K_cs], params[:k], params[:μ_ω], params[:σ_ω], params[:μ_ψ], params[:σ_ψ], params[:μ_α], params[:σ_α]) for t in times]
        elseif model_type == "special_independent"
            germ_response = [germ_response_independent_factors_var_perm_st_gh(u, W3, t, ρₛ, params[:c₀_cs], params[:d_hp], ξ2, κ2, params[:Pₛ], params[:Pₛ_cs], params[:K_cs], params[:μ_γ], params[:σ_γ], params[:μ_ω], params[:σ_ω], params[:μ_α], params[:σ_α]) for t in times]
        end

        return germ_response
    end


    function germ_response_inhibitor_gh(u, hw, t, ρₛ, ξ, Pₛ, μ_γ, σ_γ)
        """
        Compute the germination response for purely
        inhibition-dependent germination for a given set of parameters,
        without considering an external initial concentration.
        Uses Gauss-Hermite approximation.
        inputs:
            u - transformed Gauss-Hermite nodes
            hw - transformed Gauss-Hermite weights
            t - time in seconds
            ρₛ - spore density in spores/um^3
            ξ - spore radius in um
            Pₛ - permeation constant in um/s
            μ_γ - mean inhibition threshold
            σ_γ - standard deviation of inhibition threshold
        output:
            the germination response for the given parameters
        """

        # Distributions
        dist_γ = Normal(μ_γ, σ_γ)

        # Inhibitor
        V = 4/3 * π .* ξ.^3
        A = 4 * π .* ξ.^2
        τ = V ./ (Pₛ * A)
        ϕ = ρₛ .* V
        β = ϕ .+ (1 .- ϕ) .* exp.(-t ./ (τ .* (1 .- ϕ)))

        tail = 1 .- cdf.(dist_γ, β)

        return sum(hw .* tail)
    end


    function germ_response_independent_factors_gh(u, hw, t, ρₛ, ξ, Pₛ, μ_γ, σ_γ, μ_ω, σ_ω)
        """
        Compute the germination response for independent
        inhibition and induction for a given set of parameters.
        Uses Gauss-Hermite approximation.
        inputs:
            u - transformed Gauss-Hermite nodes
            hw - transformed Gauss-Hermite weights
            t - time in seconds
            ρₛ - spore density in spores/um^3
            ξ - spore radius in um
            Pₛ - permeation constant in um/s
            μ_γ - mean inhibition threshold
            σ_γ - standard deviation of inhibition threshold
            μ_ω - mean induction threshold
            σ_ω - standard deviation of induction threshold
        output:
            the germination response for the given parameters
        """

        # Distributions
        dist_ω = Normal(μ_ω, σ_ω)
        # μ_ω_log = log(μ_ω^2 / sqrt(σ_ω^2 + μ_ω^2))
        # σ_ω_log = sqrt(log(σ_ω^2 / μ_ω^2 + 1))
        # dist_ω = LogNormal(μ_ω_log, σ_ω_log)
        
        induction_factor = cdf.(dist_ω, 1)
        
        return induction_factor .* germ_response_inhibitor_gh(u, hw, t, ρₛ, ξ, Pₛ, μ_γ, σ_γ)
    end
    

    function germ_response_independent_factors_st_gh(u, W, t, ρₛ, c₀_cs, d_hp, ξ, κ, Pₛ, Pₛ_cs, K_cs, μ_γ, σ_γ, μ_ω, σ_ω)
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
            the germination response for the given parameters
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
        V_cw = 0.32 .* π .* ((ξ .- d_hp).^3 .- (ξ .- d_hp .- κ).^3)
        c_cs = inducer_concentration.(c₀_cs, t, Pₛ_cs, A, V_cw)
        s = c_cs ./ (K_cs .+ c_cs)
        
        tail = cdf.(dist_ω, s) .* (1 .- cdf.(dist_γ, β))
        
        return sum(W .* tail)
    end


    function germ_response_inducer_dep_inhibitor_thresh_st_gh(u, W, t, ρₛ, c₀_cs, d_hp, ξ, κ, Pₛ, Pₛ_cs, k, K_cs, μ_γ, σ_γ)
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
            the germination response for the given parameters
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
        V_cw = 0.32 .* π .* ((ξ .- d_hp).^3 .- (ξ .- d_hp .- κ).^3)
        c_cs = inducer_concentration.(c₀_cs, t, Pₛ_cs, A, V_cw)
        s = c_cs ./ (K_cs .+ c_cs)

        tail = 1 .- cdf.(dist_γ, β .- k .* s)

        return sum(W .* tail)
    end


    function germ_response_inducer_dep_inhibitor_perm_st_gh(u, W, t, ρₛ, c₀_cs, d_hp, ξ, κ, Pₛ, Pₛ_cs, K_cs, μ_γ, σ_γ)
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
            the germination response for the given parameters
        """

        # Distributions
        dist_γ = Normal(μ_γ, σ_γ)

        # Inducer
        A = 4 * π .* ξ.^2
        V_cw = 0.32 .* π .* ((ξ .- d_hp).^3 .- (ξ .- d_hp .- κ).^3)
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


    function germ_response_inducer_dep_inhibitor_combined_st_gh(u, W, t, ρₛ, c₀_cs, d_hp, ξ, κ, Pₛ, Pₛ_cs, K_cs, k, μ_γ, σ_γ)
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
            the germination response for the given parameters
        """

        # Distributions
        dist_γ = Normal(μ_γ, σ_γ)

        # Inducer
        A = 4 * π .* ξ.^2
        V_cw = 0.32 .* π .* ((ξ .- d_hp).^3 .- (ξ .- d_hp .- κ).^3)
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


    function germ_response_inhibitor_dep_inducer_thresh_gh(u, W, t, ρₛ, ξ, Pₛ, k, μ_ω, σ_ω, μ_ψ, σ_ψ)
        """
        Compute the germination response for an inhibitor-dependent
        induction threshold for a given set of parameters.
        Uses Gauss-Hermite approximation.
        inputs:
            u - transformed Gauss-Hermite nodes
            W - transformed Gauss-Hermite weights (matrix)
            t - time in seconds
            ρₛ - spore density in spores/um^3
            ξ - spore radius in um
            Pₛ - permeation constant for the inhibitor in um/s
            k - inhibition strength over induction threshold
            μ_ω - mean induction threshold
            σ_ω - standard deviation of induction threshold
            μ_ψ - mean initial concentration
            σ_ψ - standard deviation of initial concentration
        output:
            the germination response for the given parameters
        """

        # Transform to log-normal
        μ_ψ_log = log(μ_ψ^2 / sqrt(σ_ψ^2 + μ_ψ^2))
        σ_ψ_log = sqrt(log(σ_ψ^2 / μ_ψ^2 + 1))
        ψ = exp.(μ_ψ_log .+ σ_ψ_log .* u)

        # Distributions
        dist_ω = Normal(μ_ω, σ_ω)

        # Inhibitor
        V = 4/3 * π .* ξ.^3
        A = 4 * π .* ξ.^2
        τ = V ./ (Pₛ * A)
        ϕ = ρₛ .* V
        β = (ϕ .+ (1 .- ϕ) .* exp.(-t ./ (τ .* (1 .- ϕ))))

        # Reshape
        β, ψ = meshgrid(β, ψ)

        c_in = ψ .* β

        tail = cdf.(dist_ω, 1 .- k .* c_in)

        return sum(W .* tail)
    end


    function germ_response_inhibitor_dep_inducer_thresh_st_gh(u, W3, t, ρₛ, c₀_cs, d_hp, ξ, κ, Pₛ, Pₛ_cs, K_cs, k, μ_ω, σ_ω, μ_ψ, σ_ψ)
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
            the germination response for the given parameters
        """

        # Transform to log-normal
        μ_ψ_log = log(μ_ψ^2 / sqrt(σ_ψ^2 + μ_ψ^2))
        σ_ψ_log = sqrt(log(σ_ψ^2 / μ_ψ^2 + 1))
        ψ = exp.(μ_ψ_log .+ σ_ψ_log .* u)

        # Distributions
        dist_ω = Normal(μ_ω, σ_ω)

        # Inducer
        A = 4 * π .* ξ.^2
        V_cw = 0.32 .* π .* ((ξ .- d_hp).^3 .- (ξ .- d_hp .- κ).^3) # psi and kappa
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


    function germ_response_inhibitor_dep_inducer_signal_gh(u, W, t, ρₛ, ξ, Pₛ, K_I, n, μ_ω, σ_ω, μ_ψ, σ_ψ)
        """
        Compute the germination response for an inhibitor-dependent
        induction signal for a given set of parameters.
        Uses Gauss-Hermite approximation.
        inputs:
            u - transformed Gauss-Hermite nodes
            W - transformed Gauss-Hermite weights (matrix)
            t - time in seconds
            ρₛ - spore density in spores/um^3
            ξ - spore radius in um
            Pₛ - permeation constant for the inhibitor in um/s
            K_I - half-saturation constant for the inhibitor
            n - Hill coefficient for the inhibitor
            μ_ω - mean induction threshold
            σ_ω - standard deviation of induction threshold
            μ_ψ - mean initial concentration
            σ_ψ - standard deviation of initial concentration
        output:
            the germination response for the given parameters
        """
        
        # Transform to log-normal
        μ_ψ_log = log(μ_ψ^2 / sqrt(σ_ψ^2 + μ_ψ^2))
        σ_ψ_log = sqrt(log(σ_ψ^2 / μ_ψ^2 + 1))
        ψ = exp.(μ_ψ_log .+ σ_ψ_log .* u)

        # Distributions
        dist_ω = Normal(μ_ω, σ_ω)

        # Inhibitor
        V = 4/3 * π .* ξ.^3
        A = 4 * π .* ξ.^2
        τ = V ./ (Pₛ * A)
        ϕ = ρₛ .* V
        β = ϕ .+ (1 .- ϕ) .* exp.(-t ./ (τ .* (1 .- ϕ)))

        # Reshape
        β, ψ = meshgrid(β, ψ)

        c_in = ψ .* β
        s_mod = 1 ./ (1 .+ (c_in ./ K_I).^n)

        tail = cdf.(dist_ω, s_mod)

        return sum(W .* tail)
    end


    function germ_response_inhibitor_dep_inducer_signal_st_gh(u, W3, t, ρₛ, c₀_cs, d_hp, ξ, κ, Pₛ, Pₛ_cs, K_cs, K_I, n, μ_ω, σ_ω, μ_ψ, σ_ψ)
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
            the germination response for the given parameters
        """

        # Transform to log-normal
        μ_ψ_log = log(μ_ψ^2 / sqrt(σ_ψ^2 + μ_ψ^2))
        σ_ψ_log = sqrt(log(σ_ψ^2 / μ_ψ^2 + 1))
        ψ = exp.(μ_ψ_log .+ σ_ψ_log .* u)

        # Distributions
        dist_ω = Normal(μ_ω, σ_ω)

        # Inducer
        A = 4 * π .* ξ.^2
        V_cw = 0.32 .* π .* ((ξ .- d_hp).^3 .- (ξ .- d_hp .- κ).^3)
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

    
    function germ_response_inhibitor_dep_inducer_combined_gh(u, W, t, ρₛ, ξ, Pₛ, k, K_I, n, μ_ω, σ_ω, μ_ψ, σ_ψ)
        """
        Compute the germination response for an inhibitor-dependent
        induction threshold and signal for a given set of parameters.
        Uses Gauss-Hermite approximation.
        inputs:
            u - transformed Gauss-Hermite nodes
            W - transformed Gauss-Hermite weights (matrix)
            t - time in seconds
            ρₛ - spore density in spores/um^3
            ξ - spore radius in um
            Pₛ - permeation constant for the inhibitor in um/s
            k - inhibition strength over induction threshold
            K_I - half-saturation constant for the inhibitor
            n - Hill coefficient for the inhibitor
            μ_ω - mean induction threshold
            σ_ω - standard deviation of induction threshold
            μ_ψ - mean initial concentration
            σ_ψ - standard deviation of initial concentration
        output:
            the germination response for the given parameters
        """

        # Transform to log-normal
        μ_ψ_log = log(μ_ψ^2 / sqrt(σ_ψ^2 + μ_ψ^2))
        σ_ψ_log = sqrt(log(σ_ψ^2 / μ_ψ^2 + 1))
        ψ = exp.(μ_ψ_log .+ σ_ψ_log .* u)

        # Distributions
        dist_ω = Normal(μ_ω, σ_ω)

        # Inhibitor
        V = 4/3 * π .* ξ.^3
        A = 4 * π .* ξ.^2
        τ = V ./ (Pₛ * A)
        ϕ = ρₛ .* V
        β = ϕ .+ (1 .- ϕ) .* exp.(-t ./ (τ .* (1 .- ϕ)))

        # Reshape
        β, ψ = meshgrid(β, ψ)

        c_in = ψ .* β
        s_mod = 1 ./ (1 .+ (c_in ./ K_I).^n)

        tail = cdf.(dist_ω, s_mod .- k .* c_in)

        return sum(W .* tail)
    end


    function germ_response_inhibitor_dep_inducer_combined_st_gh(u, W3, t, ρₛ, c₀_cs, d_hp, ξ, κ, Pₛ, Pₛ_cs, k, K_cs, K_I, n, μ_ω, σ_ω, μ_ψ, σ_ψ)
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
            the germination response for the given parameters
        """

        # Transform to log-normal
        μ_ψ_log = log(μ_ψ^2 / sqrt(σ_ψ^2 + μ_ψ^2))
        σ_ψ_log = sqrt(log(σ_ψ^2 / μ_ψ^2 + 1))
        ψ = exp.(μ_ψ_log .+ σ_ψ_log .* u)

        # Distributions
        dist_ω = Normal(μ_ω, σ_ω)

        # Inducer
        A = 4 * π .* ξ.^2
        V_cw = 0.32 .* π .* ((ξ .- d_hp).^3 .- (ξ .- d_hp .- κ).^3)
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


    function germ_response_inducer_dep_inhibitor_thresh_2_factor_st_gh(u, W, t, ρₛ, c₀_cs, d_hp, ξ, κ, Pₛ, Pₛ_cs, k, K_cs, μ_γ, σ_γ, μ_ω, σ_ω)
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
            the germination response for the given parameters
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
        V_cw = 0.32 .* π .* ((ξ .- d_hp).^3 .- (ξ .- d_hp .- κ).^3)
        c_cs = inducer_concentration.(c₀_cs, t, Pₛ_cs, A, V_cw)
        s = c_cs ./ (K_cs .+ c_cs)

        tail = (1 .- cdf.(dist_γ, β .- k .* s)) .* cdf.(dist_ω, s)

        return sum(W .* tail)
    end


    function germ_response_inducer_dep_inhibitor_perm_2_factor_st_gh(u, W, t, ρₛ, c₀_cs, d_hp, ξ, κ, Pₛ, Pₛ_cs, K_cs, μ_γ, σ_γ, μ_ω, σ_ω)
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
            the germination response for the given parameters
        """

        # Distributions
        dist_γ = Normal(μ_γ, σ_γ)
        dist_ω = Normal(μ_ω, σ_ω)

        # Inducer
        A = 4 * π .* ξ.^2
        V_cw = 0.32 .* π .* ((ξ .- d_hp).^3 .- (ξ .- d_hp .- κ).^3)
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


    function germ_response_inducer_dep_inhibitor_combined_2_factor_st_gh(u, W, t, ρₛ, c₀_cs, d_hp, ξ, κ, Pₛ, Pₛ_cs, K_cs, k, μ_γ, σ_γ, μ_ω, σ_ω)
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
            the germination response for the given parameters
        """

        # Distributions
        dist_γ = Normal(μ_γ, σ_γ)
        dist_ω = Normal(μ_ω, σ_ω)

        # Inducer
        A = 4 * π .* ξ.^2
        V_cw = 0.32 .* π .* ((ξ .- d_hp).^3 .- (ξ .- d_hp .- κ).^3)
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


    function germ_response_inhibitor_dep_inducer_thresh_2_factor_st_gh(u, W3, t, ρₛ, c₀_cs, d_hp, ξ, κ, Pₛ, Pₛ_cs, K_cs, k, μ_γ, σ_γ, μ_ω, σ_ω, μ_ψ, σ_ψ)
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
            the germination response for the given parameters
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
        V_cw = 0.32 .* π .* ((ξ .- d_hp).^3 .- (ξ .- d_hp .- κ).^3) # psi and kappa
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

    function germ_response_inhibitor_dep_inducer_signal_2_factor_st_gh(u, W3, t, ρₛ, c₀_cs, d_hp, ξ, κ, Pₛ, Pₛ_cs, K_cs, K_I, n, μ_γ, σ_γ, μ_ω, σ_ω, μ_ψ, σ_ψ)
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
            the germination response for the given parameters
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
        V_cw = 0.32 .* π .* ((ξ .- d_hp).^3 .- (ξ .- d_hp .- κ).^3) # psi and kappa
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


    function germ_response_inhibitor_dep_inducer_combined_2_factor_st_gh(u, W3, t, ρₛ, c₀_cs, d_hp, ξ, κ, Pₛ, Pₛ_cs, K_cs, K_I, n, k, μ_γ, σ_γ, μ_ω, σ_ω, μ_ψ, σ_ψ)
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
            the germination response for the given parameters
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
        V_cw = 0.32 .* π .* ((ξ .- d_hp).^3 .- (ξ .- d_hp .- κ).^3) # psi and kappa
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


    function germ_response_inhibitor_var_perm_gh(u, W, t, ρₛ, ξ, μ_π, σ_π, μ_γ, σ_γ)
        """
        Compute the germination response for purely
        inhibition-dependent germination for a given set of parameters,
        whereby the permeation constant is a random variable.
        Uses Gauss-Hermite approximation.
        inputs:
            u - transformed Gauss-Hermite nodes
            W - transformed Gauss-Hermite weights (matrix)
            t - time in seconds
            ρₛ - spore density in spores/um^3
            ξ - spore radius in um
            μ_π - mean permeation constant in um/s
            σ_π - standard deviation of permeation constant
            μ_γ - mean inhibition threshold
            σ_γ - standard deviation of inhibition threshold
        output:
            the germination response for the given parameters
        """

        # Transform to log-normal
        μ_π_log = log(μ_π^2 / sqrt(σ_π^2 + μ_π^2))
        σ_π_log = sqrt(log(σ_π^2 / μ_π^2 + 1))
        πₛ = exp.(μ_π_log .+ σ_π_log .* u)

        # Distributions
        dist_γ = Normal(μ_γ, σ_γ)

        # Inhibitor
        V = 4/3 * π .* ξ.^3
        A = 4 * π .* ξ.^2

        # Reshape
        πₛ, V = meshgrid(πₛ, V)
        A = repeat(A, 1, size(πₛ, 2))

        τ = V ./ (πₛ .* A)
        ϕ = ρₛ .* V
        β = ϕ .+ (1 .- ϕ) .* exp.(-t ./ (τ .* (1 .- ϕ)))

        tail = 1 .- cdf.(dist_γ, β)

        return sum(W .* tail)
    end


    function germ_response_inducer_thresh_var_perm_st_gh(u, W4, t, ρₛ, c₀_cs, d_hp, ξ, κ, Pₛ, Pₛ_cs, K_cs, k, μ_ω, σ_ω, μ_ψ, σ_ψ, μ_α, σ_α)
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
            the germination response for the given parameters
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
        V_cw = 0.32 .* π .* ((ξ .- d_hp).^3 .- (ξ .- d_hp .- κ).^3)
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


    function germ_response_independent_factors_var_perm_st_gh(u, W3, t, ρₛ, c₀_cs, d_hp, ξ, κ, Pₛ, Pₛ_cs, K_cs, μ_γ, σ_γ, μ_ω, σ_ω, μ_α, σ_α)
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
            the germination response for the given parameters
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
        V_cw = 0.32 .* π .* ((ξ .- d_hp).^3 .- (ξ .- d_hp .- κ).^3)

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


    function germ_response_inducer_dep_inhibitor_combined_eq(ρₛ, dist_ξ, μ_γ, σ_γ; reltol=1e-4)
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
            the equilibrium germination response for the given parameters
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


    function germ_response_inducer_dep_inhibitor_combined_eq_c_ex(ρₛ, dist_ξ, c_ex, μ_γ, σ_γ, μ_ψ, σ_ψ; reltol=1e-4)
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
            the equilibrium germination response for the given parameters
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


    function germ_response_inhibitor_dep_inducer_thresh_2_factors_eq(ρₛ, dist_ξ, c₀_cs, K_cs, k, μ_γ, σ_γ, μ_ω, σ_ω, μ_ψ, σ_ψ; reltol=1e-4)
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
        output:
            the equilibrium germination response for the given parameters
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

        return hcubature(integrand, [0.0, 0.0], [quantile(dist_ξ, 1-1e-8), quantile(dist_ψ, 1-1e-9)], reltol=reltol, abstol=1e-6)[1]
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
            the equilibrium germination response for the given parameters
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
            the equilibrium germination response for the given parameters
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
            the equilibrium germination response for the given parameters
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
end