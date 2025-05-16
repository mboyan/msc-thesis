module GermStats
    """
    Contains tools for generating germination statistics
    """

    using QuadGK
    using Cubature
    using LinearAlgebra
    using MeshGrid
    using Distributions
    using SpecialFunctions
    using Revise

    include("./conversions.jl")
    Revise.includet("./conversions.jl")
    using .Conversions

    export germ_response
    export germ_response_simple
    export germ_response_equilibrium
    export germ_response_combined_independent
    export germ_response_combined_independent_eq
    export germ_response_inhibitor_dependent_inducer_thresh
    export germ_response_inhibitor_dependent_inducer_signal
    export germ_response_inhibitor_dependent_inducer_combined
    export germ_response_combined_independent_st
    export germ_response_inducer_dependent_inhibitor_thresh_st
    export germ_response_inducer_dependent_inhibitor_perm_st
    export germ_response_inducer_dependent_inhibitor_combined_st
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


    function germ_response(ρₛ, c_ex, Pₛ, μ_ψ, σ_ψ, μ_γ, σ_γ, μ_ξ, σ_ξ, t)
        """
        Compute the germination response for purely 
        inhibition-dependent germination for a given set of parameters.
        inputs:
            ρₛ - spore density in spores/mL
            c_ex - exogenously added concentration in M
            Pₛ - permeation constant in um/s
            μ_ψ - mean initial concentration
            σ_ψ - standard deviation of initial concentration
            μ_γ - mean inhibition threshold
            σ_γ - standard deviation of inhibition threshold
            μ_ξ - mean spore radius in um
            σ_ξ - standard deviation of spore radius in um
            t - time
        output:
            germ_response - the germination response for the given parameters
        """
    
        # Convert units
        ρₛ = inverse_mL_to_cubic_um(ρₛ) # Convert from spores/mL to spores/m^3

        # Distributions
        μ_ξ_log = log(μ_ξ^2 / sqrt(σ_ξ^2 + μ_ξ^2))
        σ_ξ_log = sqrt(log(σ_ξ^2 / μ_ξ^2 + 1))
        μ_ψ_log = log(μ_ψ^2 / sqrt(σ_ψ^2 + μ_ψ^2))
        σ_ψ_log = sqrt(log(σ_ψ^2 / μ_ψ^2 + 1))
        dist_ξ = LogNormal(μ_ξ_log, σ_ξ_log)
        dist_ψ = LogNormal(μ_ψ_log, σ_ψ_log)
        
        ξ_lo = quantile(dist_ξ, 1e-6)   # e.g. ~0.000001 tail
        ξ_hi = quantile(dist_ξ, 1-1e-6) # upper 0.999999 quantile
        ψ_lo = quantile(dist_ψ, 1e-6)   # e.g. ~0.000001 tail
        ψ_hi = quantile(dist_ψ, 1-1e-6) # upper 0.999999 quantile
        
        function integrand_xi(ξ)
    
            V = 4/3 * π .* ξ^3
            A = 4 * π .* ξ^2
            τ = V ./ (Pₛ * A)
            ϕ = ρₛ .* V
            β = ϕ .+ (1 .- ϕ) .* exp.(-t ./ (τ .* (1 .- ϕ)))
            χ = (1 .- ϕ) .* (1 .- exp.(-t ./ (τ .* (1 .- ϕ))))
    
            function integrand_psi(ψ)
                z = (β .+ χ .* c_ex ./ ψ .- μ_γ) ./ σ_γ
                Φ = 0.5 .* (1 .+ erf.(z ./ √2))
                return (1 .- Φ) .* pdf(dist_ψ, ψ)
            end
            
            return quadgk(x -> integrand_psi(x), ψ_lo, ψ_hi, rtol=1e-8)[1] .* pdf(dist_ξ, ξ)
        end
    
        return quadgk(x -> integrand_xi(x), ξ_lo, ξ_hi, rtol=1e-8)[1]
    end


    function germ_response_simple(ρₛ, Pₛ, μ_γ, σ_γ, μ_ξ, σ_ξ, t)
        """
        Compute the germination response for purely
        inhibition-dependent germination for a given set of parameters,
        without considering an external initial concentration.
        inputs:
            ρₛ - spore density in spores/mL
            Pₛ - permeation constant in um/s
            μ_γ - mean inhibition threshold
            σ_γ - standard deviation of inhibition threshold
            μ_ξ - mean spore radius in um
            σ_ξ - standard deviation of spore radius in um
            t - time in seconds
        output:
            germ_response - the germination response for the given parameters
        """

        # Convert units
        ρₛ = inverse_mL_to_cubic_um(ρₛ) # Convert from spores/mL to spores/m^3

        # Distributions
        μ_ξ_log = log(μ_ξ^2 / sqrt(σ_ξ^2 + μ_ξ^2))
        σ_ξ_log = sqrt(log(σ_ξ^2 / μ_ξ^2 + 1))
        dist_ξ = LogNormal(μ_ξ_log, σ_ξ_log)
        
        function integrand(ξ)
            V = 4/3 * π .* ξ^3
            A = 4 * π .* ξ^2
            τ = V ./ (Pₛ * A)
            ϕ = ρₛ .* V
            β = ϕ .+ (1 .- ϕ) .* exp.(-t ./ (τ .* (1 .- ϕ)))
            z = (β .- μ_γ) ./ σ_γ
            Φ = 0.5 .* (1 .+ erf.(z ./ √2))
            return (1 .- Φ) .* pdf(dist_ξ, ξ)
        end

        return quadgk(x -> integrand(x), 0.0, Inf, rtol=1e-8)[1]
    end


    function germ_response_equilibrium(ρₛ, μ_γ, σ_γ, μ_ξ, σ_ξ)
        """
        Compute the equilibrium germination response for a purely
        inhibition-driven germination for a given set of parameters,
        without considering an external initial concentration.
        inputs:
            ρₛ - spore density in spores/mL
            μ_γ - mean inhibition threshold
            σ_γ - standard deviation of inhibition threshold
            μ_ξ - mean spore radius in um
            σ_ξ - standard deviation of spore radius in um
        output:
            germ_response - the germination response for the given parameters
        """

        # Convert units
        ρₛ = inverse_mL_to_cubic_um(ρₛ) # Convert from spores/mL to spores/m^3
        
        # Distributions
        μ_ξ_log = log(μ_ξ^2 / sqrt(σ_ξ^2 + μ_ξ^2))
        σ_ξ_log = sqrt(log(σ_ξ^2 / μ_ξ^2 + 1))
        dist_ξ =LogNormal(μ_ξ_log, σ_ξ_log)
        
        function integrand(ξ)
            V = 4/3 * π .* ξ^3
            ϕ = ρₛ .* V
            z = (ϕ .- μ_γ) ./ σ_γ
            Φ = 0.5 .* (1 .+ erf.(z ./ √2))
            return (1 .- Φ) .* pdf(dist_ξ, ξ)
        end

        return quadgk(x -> integrand(x), 0.0, Inf, rtol=1e-8)[1]
    end


    function germ_response_combined_independent(s, μ_ω, σ_ω, ρₛ, Pₛ, μ_γ, σ_γ, μ_ξ, σ_ξ, t; c_ex=nothing, μ_ψ=nothing, σ_ψ=nothing)
        """
        Compute the germination response for independent
        inhibition and induction for a given set of parameters.
        inputs:
            s - inducer signal strength
            μ_ω - mean induction threshold
            σ_ω - standard deviation of induction threshold
            ρₛ - spore density in spores/mL
            Pₛ - permeation constant in um/s
            μ_γ - mean inhibition threshold
            σ_γ - standard deviation of inhibition threshold
            μ_ξ - mean spore radius in um
            σ_ξ - standard deviation of spore radius in um
            t - time in seconds
            c_ex - exogenously added concentration in M
            μ_ψ - mean initial concentration
            σ_ψ - standard deviation of initial concentration
        output:
            germ_response - the germination response for the given parameters
        """

        # Convert units
        ρₛ = inverse_mL_to_cubic_um(ρₛ) # Convert from spores/mL to spores/m^3
        
        z = (s .- μ_ω) ./ σ_ω
        induction_factor = 0.5 .* (1 .+ erf.(z ./ √2))
        
        if isnothing(c_ex) || isnothing(μ_ψ) || isnothing(σ_ψ)
            return induction_factor .* germ_response_simple(ρₛ, Pₛ, μ_γ, σ_γ, μ_ξ, σ_ξ, t)
        else
            return induction_factor .* germ_response(ρₛ, c_ex, Pₛ, μ_ψ, σ_ψ, μ_γ, σ_γ, μ_ξ, σ_ξ, t)
        end
    end


    function germ_response_combined_independent_eq(s, μ_ω, σ_ω, ρₛ, μ_γ, σ_γ, μ_ξ, σ_ξ)
        """
        Compute the germination response for independent
        inhibition and induction for a given set of parameters.
        inputs:
            s - inducer signal strength
            μ_ω - mean induction threshold
            σ_ω - standard deviation of induction threshold
            ρₛ - spore density in spores/mL
            μ_γ - mean inhibition threshold
            σ_γ - standard deviation of inhibition threshold
            μ_ξ - mean spore radius in um
            σ_ξ - standard deviation of spore radius in um
        output:
            germ_response - the germination response for the given parameters
        """

        # Convert units
        ρₛ = inverse_mL_to_cubic_um(ρₛ) # Convert from spores/mL to spores/m^3
        
        z = (s .- μ_ω) ./ σ_ω
        induction_factor = 0.5 .* (1 .+ erf.(z ./ √2))
        
        return induction_factor .* germ_response_equilibrium(ρₛ, μ_γ, σ_γ, μ_ξ, σ_ξ)
    end


    function germ_response_inhibitor_dependent_inducer_thresh(s, k, μ_ω, σ_ω, ρₛ, Pₛ, μ_ψ, σ_ψ, μ_ξ, σ_ξ, t)
        """
        Compute the germination response for an inhibitor-dependent
        induction threshold for a given set of parameters.
        inputs:
            s - inducer signal strength
            k - inhibition strength over induction signal
            μ_ω - mean induction threshold
            σ_ω - standard deviation of induction threshold
            ρₛ - spore density in spores/mL
            Pₛ - permeation constant in um/s
            μ_ψ - mean initial concentration
            σ_ψ - standard deviation of initial concentration
            μ_ξ - mean spore radius in um
            σ_ξ - standard deviation of spore radius in um
            t - time in seconds
        output:
            germ_response - the germination response for the given parameters
        """

        # Convert units
        ρₛ = inverse_mL_to_cubic_um(ρₛ) # Convert from spores/mL to spores/m^3
        
        # Distributions
        μ_ξ_log = log(μ_ξ^2 / sqrt(σ_ξ^2 + μ_ξ^2))
        σ_ξ_log = sqrt(log(σ_ξ^2 / μ_ξ^2 + 1))
        μ_ψ_log = log(μ_ψ^2 / sqrt(σ_ψ^2 + μ_ψ^2))
        σ_ψ_log = sqrt(log(σ_ψ^2 / μ_ψ^2 + 1))
        dist_ξ = LogNormal(μ_ξ_log, σ_ξ_log)
        dist_ψ = LogNormal(μ_ψ_log, σ_ψ_log)
        
        ξ_lo = quantile(dist_ξ, 1e-6)   # e.g. ~0.000001 tail
        ξ_hi = quantile(dist_ξ, 1-1e-6) # upper 0.999999 quantile
        ψ_lo = quantile(dist_ψ, 1e-6)   # e.g. ~0.000001 tail
        ψ_hi = quantile(dist_ψ, 1-1e-6) # upper 0.999999 quantile

        function integrand(input::AbstractVector)

            ξ, ψ = input

            A, V = compute_spore_area_and_volume_from_dia(2ξ)
            τ = V / (A * Pₛ)
            ϕ = ρₛ .* V # volume fraction
            
            c_in = ψ .* (ϕ .+ (1 .- ϕ) .* exp.(-t ./ (τ .* (1 .- ϕ))))
            z = (s .- k .* c_in .- μ_ω) ./ σ_ω
            Φ = 0.5 .* (1 .+ erf.(z ./ sqrt(2)))
            
            return Φ .* pdf(dist_ψ, ψ) .* pdf(dist_ξ, ξ)
        end

        return hcubature(integrand, [ξ_lo, ψ_lo], [ξ_hi, ψ_hi], reltol=1e-4)[1]
    end


    function germ_response_inhibitor_dependent_inducer_signal(s, K, n, μ_ω, σ_ω, ρₛ, Pₛ, μ_ψ, σ_ψ, μ_ξ, σ_ξ, t)
        """
        Compute the germination response for an inhibitor-dependent
        induction threshold for a given set of parameters.
        inputs:
            s - inducer signal strength
            K - half-saturation constant for the inhibitor
            n - Hill coefficient for the inhibitor
            μ_ω - mean induction threshold
            σ_ω - standard deviation of induction threshold
            ρₛ - spore density in spores/mL
            Pₛ - permeation constant in um/s
            μ_ψ - mean initial concentration
            σ_ψ - standard deviation of initial concentration
            μ_ξ - mean spore radius in um
            σ_ξ - standard deviation of spore radius in um
            t - time in seconds
        output:
            germ_response - the germination response for the given parameters
        """

        # Convert units
        ρₛ = inverse_mL_to_cubic_um(ρₛ) # Convert from spores/mL to spores/m^3
        
        # Distributions
        μ_ξ_log = log(μ_ξ^2 / sqrt(σ_ξ^2 + μ_ξ^2))
        σ_ξ_log = sqrt(log(σ_ξ^2 / μ_ξ^2 + 1))
        μ_ψ_log = log(μ_ψ^2 / sqrt(σ_ψ^2 + μ_ψ^2))
        σ_ψ_log = sqrt(log(σ_ψ^2 / μ_ψ^2 + 1))
        dist_ξ = LogNormal(μ_ξ_log, σ_ξ_log)
        dist_ψ = LogNormal(μ_ψ_log, σ_ψ_log)
        
        ξ_lo = quantile(dist_ξ, 1e-6)   # e.g. ~0.000001 tail
        ξ_hi = quantile(dist_ξ, 1-1e-6) # upper 0.999999 quantile
        ψ_lo = quantile(dist_ψ, 1e-6)   # e.g. ~0.000001 tail
        ψ_hi = quantile(dist_ψ, 1-1e-6) # upper 0.999999 quantile

        function integrand(input::AbstractVector)

            ξ, ψ = input

            A, V = compute_spore_area_and_volume_from_dia(2ξ)
            τ = V / (A * Pₛ)
            ϕ = ρₛ .* V # volume fraction
            
            c_in = ψ .* (ϕ .+ (1 .- ϕ) .* exp.(-t ./ (τ .* (1 .- ϕ))))
            s_mod = s ./ (1 .+ (c_in ./ K).^n)
            z = (s_mod .- μ_ω) ./ σ_ω
            Φ = 0.5 .* (1 .+ erf.(z ./ sqrt(2)))
            
            return Φ .* pdf(dist_ψ, ψ) .* pdf(dist_ξ, ξ)
        end

        return hcubature(integrand, [ξ_lo, ψ_lo], [ξ_hi, ψ_hi], reltol=1e-4)[1]
    end


    function germ_response_inhibitor_dependent_inducer_combined(s, k, K, n, μ_ω, σ_ω, ρₛ, Pₛ, μ_ψ, σ_ψ, μ_ξ, σ_ξ, t)
        """
        Compute the germination response for an inhibitor-dependent
        induction threshold for a given set of parameters.
        inputs:
            s - inducer signal strength
            k - inhibition strength over induction signal
            K - half-saturation constant for the inhibitor
            n - Hill coefficient for the inhibitor
            μ_ω - mean induction threshold
            σ_ω - standard deviation of induction threshold
            ρₛ - spore density in spores/mL
            Pₛ - permeation constant in um/s
            μ_ψ - mean initial concentration
            σ_ψ - standard deviation of initial concentration
            μ_ξ - mean spore radius in um
            σ_ξ - standard deviation of spore radius in um
            t - time in seconds
        output:
            germ_response - the germination response for the given parameters
        """

        # Convert units
        ρₛ = inverse_mL_to_cubic_um(ρₛ) # Convert from spores/mL to spores/m^3
        
        # Distributions
        μ_ξ_log = log(μ_ξ^2 / sqrt(σ_ξ^2 + μ_ξ^2))
        σ_ξ_log = sqrt(log(σ_ξ^2 / μ_ξ^2 + 1))
        μ_ψ_log = log(μ_ψ^2 / sqrt(σ_ψ^2 + μ_ψ^2))
        σ_ψ_log = sqrt(log(σ_ψ^2 / μ_ψ^2 + 1))
        dist_ξ = LogNormal(μ_ξ_log, σ_ξ_log)
        dist_ψ = LogNormal(μ_ψ_log, σ_ψ_log)
        
        ξ_lo = quantile(dist_ξ, 1e-6)   # e.g. ~0.000001 tail
        ξ_hi = quantile(dist_ξ, 1-1e-6) # upper 0.999999 quantile
        ψ_lo = quantile(dist_ψ, 1e-6)   # e.g. ~0.000001 tail
        ψ_hi = quantile(dist_ψ, 1-1e-6) # upper 0.999999 quantile

        function integrand(input::AbstractVector)

            ξ, ψ = input

            A, V = compute_spore_area_and_volume_from_dia(2ξ)
            τ = V / (A * Pₛ)
            ϕ = ρₛ .* V # volume fraction
            
            c_in = ψ .* (ϕ .+ (1 .- ϕ) .* exp.(-t ./ (τ .* (1 .- ϕ))))
            s_mod = s ./ (1 .+ (c_in ./ K).^n)
            z = (s_mod .- k .* c_in .- μ_ω) ./ σ_ω
            Φ = 0.5 .* (1 .+ erf.(z ./ sqrt(2)))
            
            return Φ .* pdf(dist_ψ, ψ) .* pdf(dist_ξ, ξ)
        end

        return hcubature(integrand, [ξ_lo, ψ_lo], [ξ_hi, ψ_hi], reltol=1e-4)[1]
    end


    function germ_response_combined_independent_st(s_max, K_cs, c₀_cs, Pₛ_cs, μ_κ, σ_κ, d_hp, μ_ω, σ_ω, ρₛ, Pₛ_inh, μ_γ, σ_γ, μ_ξ, σ_ξ, t)
        """
        Compute the germination response for independent
        inhibition and induction for a given set of parameters.
        The inducer signal is time-dependent.
        inputs:
            s_max - maximum inducer signal strength
            K_cs - half-saturation constant for the carbon source
            c₀_cs - initial concentration of carbon source in M
            Pₛ_cs - permeation constant for the carbon source in um/s
            μ_κ - mean cell wall thickness in um
            σ_κ - standard deviation of cell wall thickness in um
            d_hp - thickness of the hydrophobin layer in um
            μ_ω - mean induction threshold
            σ_ω - standard deviation of induction threshold
            ρₛ - spore density in spores/mL
            Pₛ_inh - permeation constant in um/s
            μ_γ - mean inhibition threshold
            σ_γ - standard deviation of inhibition threshold
            μ_ξ - mean spore radius in um
            σ_ξ - standard deviation of spore radius in um
            t - time in seconds
        output:
            germ_response - the germination response for the given parameters
        """

        # Convert units
        ρₛ = inverse_mL_to_cubic_um(ρₛ) # Convert from spores/mL to spores/m^3

        # Distributions
        μ_ξ_log = log(μ_ξ^2 / sqrt(σ_ξ^2 + μ_ξ^2))
        σ_ξ_log = sqrt(log(σ_ξ^2 / μ_ξ^2 + 1))
        μ_κ_log = log(μ_κ^2 / sqrt(σ_κ^2 + μ_κ^2))
        σ_κ_log = sqrt(log(σ_κ^2 / μ_κ^2 + 1))
        dist_ξ = LogNormal(μ_ξ_log, σ_ξ_log)
        dist_κ = LogNormal(μ_κ_log, σ_κ_log)

        ξ_lo = quantile(dist_ξ, 1e-6)   # e.g. ~0.000001 tail
        ξ_hi = quantile(dist_ξ, 1-1e-6) # upper 0.999999 quantile
        κ_lo = quantile(dist_κ, 1e-6)   # e.g. ~0.000001 tail
        κ_hi = quantile(dist_κ, 1-1e-6) # upper 0.999999 quantile
        
        function integrand(input::AbstractVector)

            ξ, κ = input

            # Inhibitor
            V = 4/3 * π .* ξ^3
            A = 4 * π .* ξ^2
            τ = V ./ (Pₛ_inh * A)
            ϕ = ρₛ .* V
            β = ϕ .+ (1 .- ϕ) .* exp.(-t ./ (τ .* (1 .- ϕ)))

            z₁ = (β .- μ_γ) ./ σ_γ
            Φ₁ = 0.5 .* (1 .+ erf.(z₁ ./ √2))

            # Inducer
            V_cw = 0.32 * π * ((ξ - d_hp)^3 - (ξ - d_hp - κ)^3)
            c_cs = inducer_concentration(c₀_cs, t, Pₛ_cs, A, V_cw)
            s = s_max .* c_cs ./ (K_cs .+ c_cs)

            z₂ = (s .- μ_ω) ./ σ_ω
            Φ₂ = 0.5 .* (1 .+ erf.(z₂ ./ √2))

            return (1 .- Φ₁) .* Φ₂ .* pdf(dist_ξ, ξ) .* pdf(dist_κ, κ)
        end

        return hcubature(integrand, [ξ_lo, κ_lo], [ξ_hi, κ_hi], reltol=1e-4)[1]
    end


    function germ_response_inducer_dependent_inhibitor_thresh_st(s_max, K_cs, c₀_cs, Pₛ_cs, μ_κ, σ_κ, d_hp, ρₛ, Pₛ, μ_γ, σ_γ, μ_ξ, σ_ξ, t)
        """
        Compute the germination response for purely
        inhibition-dependent germination for a given set of parameters,
        without considering an external initial concentration.
        inputs:
            s_max - maximum inducer signal strength
            K_cs - half-saturation constant for the carbon source
            c₀_cs - initial concentration of carbon source in M
            Pₛ_cs - permeation constant for the carbon source in um/s
            μ_κ - mean cell wall thickness in um
            σ_κ - standard deviation of cell wall thickness in um
            d_hp - thickness of the hydrophobin layer in um
            ρₛ - spore density in spores/mL
            Pₛ - permeation constant in um/s
            μ_γ - mean inhibition threshold
            σ_γ - standard deviation of inhibition threshold
            μ_ξ - mean spore radius in um
            σ_ξ - standard deviation of spore radius in um
            t - time in seconds
        output:
            germ_response - the germination response for the given parameters
        """

        # Convert units
        ρₛ = inverse_mL_to_cubic_um(ρₛ) # Convert from spores/mL to spores/m^3

        # Distributions
        μ_ξ_log = log(μ_ξ^2 / sqrt(σ_ξ^2 + μ_ξ^2))
        σ_ξ_log = sqrt(log(σ_ξ^2 / μ_ξ^2 + 1))
        μ_κ_log = log(μ_κ^2 / sqrt(σ_κ^2 + μ_κ^2))
        σ_κ_log = sqrt(log(σ_κ^2 / μ_κ^2 + 1))
        dist_ξ = LogNormal(μ_ξ_log, σ_ξ_log)
        dist_κ = LogNormal(μ_κ_log, σ_κ_log)
        
        ξ_lo = quantile(dist_ξ, 1e-6)   # e.g. ~0.000001 tail
        ξ_hi = quantile(dist_ξ, 1-1e-6) # upper 0.999999 quantile
        κ_lo = quantile(dist_κ, 1e-6)   # e.g. ~0.000001 tail
        κ_hi = quantile(dist_κ, 1-1e-6) # upper 0.999999 quantile
        
        function integrand(input::AbstractVector)

            ξ, κ = input

            # Inhibitor
            V = 4/3 * π .* ξ^3
            A = 4 * π .* ξ^2
            τ = V ./ (Pₛ * A)
            ϕ = ρₛ .* V
            β = ϕ .+ (1 .- ϕ) .* exp.(-t ./ (τ .* (1 .- ϕ)))

            # Inducer
            V_cw = 0.32 * π * ((ξ - d_hp)^3 - (ξ - d_hp - κ)^3)
            c_cs = inducer_concentration(c₀_cs, t, Pₛ_cs, A, V_cw)
            s = s_max .* c_cs ./ (K_cs .+ c_cs)

            z = (β .- s .- μ_γ) ./ σ_γ
            Φ = 0.5 .* (1 .+ erf.(z ./ √2))

            # Inducer
            return (1 .- Φ) .* pdf(dist_ξ, ξ) .* pdf(dist_κ, κ)
        end

        return hcubature(integrand, [ξ_lo, κ_lo], [ξ_hi, κ_hi], reltol=1e-4)[1]
    end


    function germ_response_inducer_dependent_inhibitor_perm_st(s_max, K_cs, c₀_cs, Pₛ_cs, μ_κ, σ_κ, d_hp, ρₛ, Pₛ, μ_γ, σ_γ, μ_ξ, σ_ξ, t)
        """
        Compute the germination response for purely
        inhibition-dependent germination for a given set of parameters,
        without considering an external initial concentration.
        inputs:
            s_max - maximum inducer signal strength
            K_cs - half-saturation constant for the carbon source
            c₀_cs - initial concentration of carbon source in M
            Pₛ_cs - permeation constant for the carbon source in um/s
            μ_κ - mean cell wall thickness in um
            σ_κ - standard deviation of cell wall thickness in um
            d_hp - thickness of the hydrophobin layer in um
            ρₛ - spore density in spores/mL
            Pₛ - permeation constant in um/s
            μ_γ - mean inhibition threshold
            σ_γ - standard deviation of inhibition threshold
            μ_ξ - mean spore radius in um
            σ_ξ - standard deviation of spore radius in um
            t - time in seconds
        output:
            germ_response - the germination response for the given parameters
        """

        # Convert units
        ρₛ = inverse_mL_to_cubic_um(ρₛ) # Convert from spores/mL to spores/m^3

        # Distributions
        μ_ξ_log = log(μ_ξ^2 / sqrt(σ_ξ^2 + μ_ξ^2))
        σ_ξ_log = sqrt(log(σ_ξ^2 / μ_ξ^2 + 1))
        μ_κ_log = log(μ_κ^2 / sqrt(σ_κ^2 + μ_κ^2))
        σ_κ_log = sqrt(log(σ_κ^2 / μ_κ^2 + 1))
        dist_ξ = LogNormal(μ_ξ_log, σ_ξ_log)
        dist_κ = LogNormal(μ_κ_log, σ_κ_log)

        ξ_lo = quantile(dist_ξ, 1e-6)   # e.g. ~0.000001 tail
        ξ_hi = quantile(dist_ξ, 1-1e-6) # upper 0.999999 quantile
        κ_lo = quantile(dist_κ, 1e-6)   # e.g. ~0.000001 tail
        κ_hi = quantile(dist_κ, 1-1e-6) # upper 0.999999 quantile
        
        function integrand(input::AbstractVector)

            ξ, κ = input

            V = 4/3 * π .* ξ^3
            A = 4 * π .* ξ^2

            # Inducer
            V_cw = 0.32 * π * ((ξ - d_hp)^3 - (ξ - d_hp - κ)^3)
            c_cs = inducer_concentration(c₀_cs, t, Pₛ_cs, A, V_cw)
            s = s_max .* c_cs ./ (K_cs .+ c_cs)

            # Inhibitor
            τ = V ./ (s .*Pₛ .* A)
            ϕ = ρₛ .* V
            β = ϕ .+ (1 .- ϕ) .* exp.(-t ./ (τ .* (1 .- ϕ)))

            z = (β .- μ_γ) ./ σ_γ
            Φ = 0.5 .* (1 .+ erf.(z ./ √2))

            # Inducer
            return (1 .- Φ) .* pdf(dist_ξ, ξ) .* pdf(dist_κ, κ)
        end

        return hcubature(integrand, [ξ_lo, κ_lo], [ξ_hi, κ_hi], reltol=1e-4)[1]
    end


    function germ_response_inducer_dependent_inhibitor_combined_st(s_max, k, K_cs, c₀_cs, Pₛ_cs, μ_κ, σ_κ, d_hp, ρₛ, Pₛ, μ_γ, σ_γ, μ_ξ, σ_ξ, t)
        """
        Compute the germination response for purely
        inhibition-dependent germination for a given set of parameters,
        without considering an external initial concentration.
        inputs:
            s_max - maximum inducer signal strength
            k - proportionality constant for threshold modulation vs permeability modulation
            K_cs - half-saturation constant for the carbon source
            c₀_cs - initial concentration of carbon source in M
            Pₛ_cs - permeation constant for the carbon source in um/s
            μ_κ - mean cell wall thickness in um
            σ_κ - standard deviation of cell wall thickness in um
            d_hp - thickness of the hydrophobin layer in um
            ρₛ - spore density in spores/mL
            Pₛ - permeation constant in um/s
            μ_γ - mean inhibition threshold
            σ_γ - standard deviation of inhibition threshold
            μ_ξ - mean spore radius in um
            σ_ξ - standard deviation of spore radius in um
            t - time in seconds
        output:
            germ_response - the germination response for the given parameters
        """

        # Convert units
        ρₛ = inverse_mL_to_cubic_um(ρₛ) # Convert from spores/mL to spores/m^3

        # Distributions
        μ_ξ_log = log(μ_ξ^2 / sqrt(σ_ξ^2 + μ_ξ^2))
        σ_ξ_log = sqrt(log(σ_ξ^2 / μ_ξ^2 + 1))
        μ_κ_log = log(μ_κ^2 / sqrt(σ_κ^2 + μ_κ^2))
        σ_κ_log = sqrt(log(σ_κ^2 / μ_κ^2 + 1))
        dist_ξ = LogNormal(μ_ξ_log, σ_ξ_log)
        dist_κ = LogNormal(μ_κ_log, σ_κ_log)

        ξ_lo = quantile(dist_ξ, 1e-6)   # e.g. ~0.000001 tail
        ξ_hi = quantile(dist_ξ, 1-1e-6) # upper 0.999999 quantile
        κ_lo = quantile(dist_κ, 1e-6)   # e.g. ~0.000001 tail
        κ_hi = quantile(dist_κ, 1-1e-6) # upper 0.999999 quantile
        
        function integrand(input::AbstractVector)

            ξ, κ = input

            V = 4/3 * π .* ξ^3
            A = 4 * π .* ξ^2

            # Inducer
            V_cw = 0.32 * π * ((ξ - d_hp)^3 - (ξ - d_hp - κ)^3)
            c_cs = inducer_concentration(c₀_cs, t, Pₛ_cs, A, V_cw)
            s = s_max .* c_cs ./ (K_cs .+ c_cs)

            # Inhibitor
            τ = V ./ (s .*Pₛ .* A)
            ϕ = ρₛ .* V
            β = ϕ .+ (1 .- ϕ) .* exp.(-t ./ (τ .* (1 .- ϕ)))

            z = (β .- k .* s .- μ_γ) ./ σ_γ
            Φ = 0.5 .* (1 .+ erf.(z ./ √2))

            # Inducer
            return (1 .- Φ) .* pdf(dist_ξ, ξ) .* pdf(dist_κ, κ)
        end

        return hcubature(integrand, [ξ_lo, κ_lo], [ξ_hi, κ_hi], reltol=1e-4)[1]
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
    function germ_response_inhibitor_gh(t, ρₛ, μ_ξ, σ_ξ, Pₛ, μ_γ, σ_γ, ghnodes, ghweights)
        """
        Compute the germination response for purely
        inhibition-dependent germination for a given set of parameters,
        without considering an external initial concentration.
        Uses Gauss-Hermite approximation.
        inputs:
            t - time in seconds
            ρₛ - spore density in spores/um^3
            μ_ξ - mean spore radius in um
            σ_ξ - standard deviation of spore radius in um
            Pₛ - permeation constant in um/s
            μ_γ - mean inhibition threshold
            σ_γ - standard deviation of inhibition threshold
            ghnodes - Gauss-Hermite nodes
            ghweights - Gauss-Hermite weights
        output:
            the germination response for the given parameters
        """

        # Transform to log-normal
        u = √2 .* ghnodes
        hw = ghweights ./ √π
        # ξ = μ_ξ .+ σ_ξ .* u
        μ_ξ_log = log(μ_ξ^2 / sqrt(σ_ξ^2 + μ_ξ^2))
        σ_ξ_log = sqrt(log(σ_ξ^2 / μ_ξ^2 + 1))
        ξ = exp.(μ_ξ_log .+ σ_ξ_log .* u)

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


    function germ_response_independent_factors_gh(t, ρₛ, μ_ξ, σ_ξ, Pₛ, s, μ_γ, σ_γ, μ_ω, σ_ω, ghnodes, ghweights)
        """
        Compute the germination response for independent
        inhibition and induction for a given set of parameters.
        Uses Gauss-Hermite approximation.
        inputs:
            t - time in seconds
            ρₛ - spore density in spores/um^3
            μ_ξ - mean spore radius in um
            σ_ξ - standard deviation of spore radius in um
            Pₛ - permeation constant in um/s
            s - inducer signal strength
            μ_γ - mean inhibition threshold
            σ_γ - standard deviation of inhibition threshold
            μ_ω - mean induction threshold
            σ_ω - standard deviation of induction threshold
            ghnodes - Gauss-Hermite nodes
            ghweights - Gauss-Hermite weights
        output:
            the germination response for the given parameters
        """

        # Distributions
        dist_ω = Normal(μ_ω, σ_ω)
        
        induction_factor = cdf.(dist_ω, s)
        
        return induction_factor .* germ_response_simple(ρₛ, Pₛ, μ_γ, σ_γ, μ_ξ, σ_ξ, t)
    end
    

    function germ_response_independent_factors_st_gh(t, ρₛ, c₀_cs, d_hp, μ_ξ, σ_ξ, μ_κ, σ_κ, Pₛ, Pₛ_cs, s_max, K_cs, μ_γ, σ_γ, μ_ω, σ_ω, ghnodes, ghweights)
        """
        Compute the germination response for independent
        inhibition and induction for a given set of parameters.
        The inducer signal is time-dependent.
        Uses Gauss-Hermite approximation.
        inputs:
            t - time in seconds
            ρₛ - spore density in spores/um^3
            c₀_cs - initial concentration of carbon source in M
            d_hp - thickness of the hydrophobin layer in um
            μ_ξ - mean spore radius in um
            σ_ξ - standard deviation of spore radius in um
            μ_κ - mean cell wall thickness in um
            σ_κ - standard deviation of cell wall thickness in um
            Pₛ - permeation constant for the inhibitor in um/s
            Pₛ_cs - permeation constant for the carbon source in um/s
            s_max - maximum inducer signal strength
            K_cs - half-saturation constant for the carbon source
            μ_γ - mean inhibition threshold
            σ_γ - standard deviation of inhibition threshold
            μ_ω - mean induction threshold
            σ_ω - standard deviation of induction threshold
            ghnodes - Gauss-Hermite nodes
            ghweights - Gauss-Hermite weights
        output:
            the germination response for the given parameters
        """

        # Transform to log-normal
        u = √2 .* ghnodes
        hw = ghweights ./ √π
        # ξ = μ_ξ .+ σ_ξ .* u
        μ_ξ_log = log(μ_ξ^2 / sqrt(σ_ξ^2 + μ_ξ^2))
        σ_ξ_log = sqrt(log(σ_ξ^2 / μ_ξ^2 + 1))
        μ_κ_log = log(μ_κ^2 / sqrt(σ_κ^2 + μ_κ^2))
        σ_κ_log = sqrt(log(σ_κ^2 / μ_κ^2 + 1))
        ξ = exp.(μ_ξ_log .+ σ_ξ_log .* u)
        κ = exp.(μ_κ_log .+ σ_κ_log .* u)

        # Reshape
        ξ, κ = meshgrid(ξ, κ)

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
        s = s_max .* c_cs ./ (K_cs .+ c_cs)
        
        tail = cdf.(dist_ω, s) .* (1 .- cdf.(dist_γ, β))
        W = hw * hw'

        return sum(W .* tail)
    end


    function germ_response_inducer_dep_inhibitor_thresh_st_gh(t, ρₛ, c₀_cs, d_hp, μ_ξ, σ_ξ, μ_κ, σ_κ, Pₛ, Pₛ_cs, s_max, K_cs, μ_γ, σ_γ, ghnodes, ghweights)
        """
        Compute the germination response for an inducer-dependent
        inhibitor threshold for a given set of parameters,
        without considering an external initial concentration.
        The inducer signal is time-dependent.
        Uses Gauss-Hermite approximation.
        inputs:
            t - time in seconds
            ρₛ - spore density in spores/um^3
            c₀_cs - initial concentration of carbon source in M
            d_hp - thickness of the hydrophobin layer in um
            μ_ξ - mean spore radius in um
            σ_ξ - standard deviation of spore radius in um
            μ_κ - mean cell wall thickness in um
            σ_κ - standard deviation of cell wall thickness in um
            Pₛ - permeation constant for the inhibitor in um/s
            Pₛ_cs - permeation constant for the carbon source in um/s
            s_max - maximum inducer signal strength
            K_cs - half-saturation constant for the carbon source
            μ_γ - mean inhibition threshold
            σ_γ - standard deviation of inhibition threshold
        output:
            the germination response for the given parameters
        """

        # Transform to log-normal
        u = √2 .* ghnodes
        hw = ghweights ./ √π
        μ_ξ_log = log(μ_ξ^2 / sqrt(σ_ξ^2 + μ_ξ^2))
        σ_ξ_log = sqrt(log(σ_ξ^2 / μ_ξ^2 + 1))
        μ_κ_log = log(μ_κ^2 / sqrt(σ_κ^2 + μ_κ^2))
        σ_κ_log = sqrt(log(σ_κ^2 / μ_κ^2 + 1))
        ξ = exp.(μ_ξ_log .+ σ_ξ_log .* u)
        κ = exp.(μ_κ_log .+ σ_κ_log .* u)

        # Reshape
        ξ, κ = meshgrid(ξ, κ)

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
        s = s_max .* c_cs ./ (K_cs .+ c_cs)

        tail = 1 .- cdf.(dist_γ, β .- s)
        W = hw * hw'

        return sum(W .* tail)
    end


    function germ_response_inducer_dep_inhibitor_perm_st_gh(t, ρₛ, c₀_cs, d_hp, μ_ξ, σ_ξ, μ_κ, σ_κ, Pₛ, Pₛ_cs, s_max, K_cs, μ_γ, σ_γ, ghnodes, ghweights)
        """
        Compute the germination response for an inducer-dependent
        inhibitor permeation for a given set of parameters,
        without considering an external initial concentration.
        The inducer signal is time-dependent.
        Uses Gauss-Hermite approximation.
        inputs:
            t - time in seconds
            ρₛ - spore density in spores/um^3
            c₀_cs - initial concentration of carbon source in M
            d_hp - thickness of the hydrophobin layer in um
            μ_ξ - mean spore radius in um
            σ_ξ - standard deviation of spore radius in um
            μ_κ - mean cell wall thickness in um
            σ_κ - standard deviation of cell wall thickness in um
            Pₛ - permeation constant for inhibitor in um/s
            Pₛ_cs - permeation constant for the carbon source in um/s
            s_max - maximum inducer signal strength
            K_cs - half-saturation constant for the carbon source
            μ_γ - mean inhibition threshold
            σ_γ - standard deviation of inhibition threshold
            ghnodes - Gauss-Hermite nodes
            ghweights - Gauss-Hermite weights
        output:
            the germination response for the given parameters
        """

        # Transform to log-normal
        u = √2 .* ghnodes
        hw = ghweights ./ √π
        μ_ξ_log = log(μ_ξ^2 / sqrt(σ_ξ^2 + μ_ξ^2))
        σ_ξ_log = sqrt(log(σ_ξ^2 / μ_ξ^2 + 1))
        μ_κ_log = log(μ_κ^2 / sqrt(σ_κ^2 + μ_κ^2))
        σ_κ_log = sqrt(log(σ_κ^2 / μ_κ^2 + 1))
        ξ = exp.(μ_ξ_log .+ σ_ξ_log .* u)
        κ = exp.(μ_κ_log .+ σ_κ_log .* u)

        # Reshape
        ξ, κ = meshgrid(ξ, κ)

        # Distributions
        dist_γ = Normal(μ_γ, σ_γ)

        # Inducer
        A = 4 * π .* ξ.^2
        V_cw = 0.32 .* π .* ((ξ .- d_hp).^3 .- (ξ .- d_hp .- κ).^3)
        c_cs = inducer_concentration.(c₀_cs, t, Pₛ_cs, A, V_cw)
        s = s_max .* c_cs ./ (K_cs .+ c_cs)

        # Inhibitor
        V = 4/3 * π .* ξ.^3
        τ = V ./ (s .* Pₛ .* A)
        ϕ = ρₛ .* V
        β = ϕ .+ (1 .- ϕ) .* exp.(-t ./ (τ .* (1 .- ϕ)))

        tail = 1 .- cdf.(dist_γ, β)
        W = hw * hw'

        return sum(W .* tail)
    end


    function germ_response_inducer_dep_inhibitor_combined_st_gh(t, ρₛ, c₀_cs, d_hp, μ_ξ, σ_ξ, μ_κ, σ_κ, Pₛ, Pₛ_cs, s_max, K_cs, k, μ_γ, σ_γ, ghnodes, ghweights)
        """
        Compute the germination response for an inducer-dependent
        inhibitor threshold and permeation for a given set of parameters,
        without considering an external initial concentration.
        The inducer signal is time-dependent.
        Uses Gauss-Hermite approximation.
        inputs:
            t - time in seconds
            ρₛ - spore density in spores/um^3
            c₀_cs - initial concentration of carbon source in M
            d_hp - thickness of the hydrophobin layer in um
            μ_ξ - mean spore radius in um
            σ_ξ - standard deviation of spore radius in um
            μ_κ - mean cell wall thickness in um
            σ_κ - standard deviation of cell wall thickness in um
            Pₛ - permeation constant for the inhibitor in um/s
            Pₛ_cs - permeation constant for the carbon source in um/s
            s_max - maximum inducer signal strength
            K_cs - half-saturation constant for the carbon source
            k - proportionality constant for threshold modulation vs permeability modulation
            μ_γ - mean inhibition threshold
            σ_γ - standard deviation of inhibition threshold
        output:
            the germination response for the given parameters
        """

        # Transform to log-normal
        u = √2 .* ghnodes
        hw = ghweights ./ √π
        μ_ξ_log = log(μ_ξ^2 / sqrt(σ_ξ^2 + μ_ξ^2))
        σ_ξ_log = sqrt(log(σ_ξ^2 / μ_ξ^2 + 1))
        μ_κ_log = log(μ_κ^2 / sqrt(σ_κ^2 + μ_κ^2))
        σ_κ_log = sqrt(log(σ_κ^2 / μ_κ^2 + 1))
        ξ = exp.(μ_ξ_log .+ σ_ξ_log .* u)
        κ = exp.(μ_κ_log .+ σ_κ_log .* u)

        # Reshape
        ξ, κ = meshgrid(ξ, κ)

        # Distributions
        dist_γ = Normal(μ_γ, σ_γ)

        # Inducer
        A = 4 * π .* ξ.^2
        V_cw = 0.32 .* π .* ((ξ .- d_hp).^3 .- (ξ .- d_hp .- κ).^3)
        c_cs = inducer_concentration.(c₀_cs, t, Pₛ_cs, A, V_cw)
        s = s_max .* c_cs ./ (K_cs .+ c_cs)

        # Inhibitor
        V = 4/3 * π .* ξ.^3
        τ = V ./ (s .* Pₛ .* A)
        ϕ = ρₛ .* V
        β = ϕ .+ (1 .- ϕ) .* exp.(-t ./ (τ .* (1 .- ϕ)))

        tail = 1 .- cdf.(dist_γ, β .- k .* s)
        W = hw * hw'

        return sum(W .* tail)
    end


    function germ_response_inhibitor_dep_inducer_thresh_gh(t, ρₛ, μ_ξ, σ_ξ, Pₛ, s, k, μ_ω, σ_ω, μ_ψ, σ_ψ, ghnodes, ghweights)
        """
        Compute the germination response for an inhibitor-dependent
        induction threshold for a given set of parameters.
        Uses Gauss-Hermite approximation.
        inputs:
            t - time in seconds
            ρₛ - spore density in spores/um^3
            μ_ξ - mean spore radius in um
            σ_ξ - standard deviation of spore radius in um
            Pₛ - permeation constant for the inhibitor in um/s
            s - inducer signal strength
            k - inhibition strength over induction threshold
            μ_ω - mean induction threshold
            σ_ω - standard deviation of induction threshold
            μ_ψ - mean initial concentration
            σ_ψ - standard deviation of initial concentration
            ghnodes - Gauss-Hermite nodes
            ghweights - Gauss-Hermite weights
        output:
            the germination response for the given parameters
        """

        # Transform to log-normal
        u = √2 .* ghnodes
        hw = ghweights ./ √π
        μ_ξ_log = log(μ_ξ^2 / sqrt(σ_ξ^2 + μ_ξ^2))
        σ_ξ_log = sqrt(log(σ_ξ^2 / μ_ξ^2 + 1))
        μ_ψ_log = log(μ_ψ^2 / sqrt(σ_ψ^2 + μ_ψ^2))
        σ_ψ_log = sqrt(log(σ_ψ^2 / μ_ψ^2 + 1))
        ξ = exp.(μ_ξ_log .+ σ_ξ_log .* u)
        ψ = exp.(μ_ψ_log .+ σ_ψ_log .* u)

        # Reshape
        ξ, ψ = meshgrid(ξ, ψ)

        # Distributions
        dist_ω = Normal(μ_ω, σ_ω)

        # Inhibitor
        V = 4/3 * π .* ξ.^3
        A = 4 * π .* ξ.^2
        τ = V ./ (Pₛ * A)
        ϕ = ρₛ .* V

        c_in = ψ .* (ϕ .+ (1 .- ϕ) .* exp.(-t ./ (τ .* (1 .- ϕ))))

        tail = cdf.(dist_ω, s .- k .* c_in)
        W = hw * hw'

        return sum(W .* tail)
    end


    function germ_response_inhibitor_dep_inducer_thresh_st_gh(t, ρₛ, c₀_cs, d_hp, μ_ξ, σ_ξ, μ_κ, σ_κ, Pₛ, Pₛ_cs, s_max, K_cs, k, μ_ω, σ_ω, μ_ψ, σ_ψ, ghnodes, ghweights)
        """
        Compute the germination response for an inhibitor-dependent
        induction threshold for a given set of parameters.
        The inducer signal is time-dependent.
        Uses Gauss-Hermite approximation.
        inputs:
            t - time in seconds
            ρₛ - spore density in spores/um^3
            c₀_cs - initial concentration of carbon source in M
            d_hp - thickness of the hydrophobin layer in um
            μ_ξ - mean spore radius in um
            σ_ξ - standard deviation of spore radius in um
            μ_κ - mean cell wall thickness in um
            σ_κ - standard deviation of cell wall thickness in um
            Pₛ - permeation constant for the inhibitor in um/s
            Pₛ_cs - permeation constant for the carbon source in um/s
            s_max - inducer signal strength
            K_cs - half-saturation constant for the carbon source
            k - inhibition strength over induction threshold
            μ_ω - mean induction threshold
            σ_ω - standard deviation of induction threshold
            μ_ψ - mean initial concentration
            σ_ψ - standard deviation of initial concentration
            ghnodes - Gauss-Hermite nodes
            ghweights - Gauss-Hermite weights
        output:
            the germination response for the given parameters
        """

        # Transform to log-normal
        u = √2 .* ghnodes
        hw = ghweights ./ √π
        μ_ξ_log = log(μ_ξ^2 / sqrt(σ_ξ^2 + μ_ξ^2))
        σ_ξ_log = sqrt(log(σ_ξ^2 / μ_ξ^2 + 1))
        μ_κ_log = log(μ_κ^2 / sqrt(σ_κ^2 + μ_κ^2))
        σ_κ_log = sqrt(log(σ_κ^2 / μ_κ^2 + 1))
        μ_ψ_log = log(μ_ψ^2 / sqrt(σ_ψ^2 + μ_ψ^2))
        σ_ψ_log = sqrt(log(σ_ψ^2 / μ_ψ^2 + 1))
        ξ = exp.(μ_ξ_log .+ σ_ξ_log .* u)
        κ = exp.(μ_κ_log .+ σ_κ_log .* u)
        ψ = exp.(μ_ψ_log .+ σ_ψ_log .* u)

        # Reshape
        ξ, κ, ψ = meshgrid(ξ, κ, ψ)

        # Distributions
        dist_ω = Normal(μ_ω, σ_ω)

        # Inducer
        A = 4 * π .* ξ.^2
        V_cw = 0.32 .* π .* ((ξ .- d_hp).^3 .- (ξ .- d_hp .- κ).^3)
        c_cs = inducer_concentration.(c₀_cs, t, Pₛ_cs, A, V_cw)
        s = s_max .* c_cs ./ (K_cs .+ c_cs)

        # Inhibitor
        V = 4/3 * π .* ξ.^3
        τ = V ./ (Pₛ * A)
        ϕ = ρₛ .* V

        c_in = ψ .* (ϕ .+ (1 .- ϕ) .* exp.(-t ./ (τ .* (1 .- ϕ))))

        tail = cdf.(dist_ω, s .- k .* c_in)
        N = length(ghnodes)
        W = reshape(hw, N,1,1) .* reshape(hw, 1,N,1) .* reshape(hw, 1,1,N)

        return sum(W .* tail)
    end


    function germ_response_inhibitor_dep_inducer_signal_gh(t, ρₛ, μ_ξ, σ_ξ, Pₛ, s, K_I, n, μ_ω, σ_ω, μ_ψ, σ_ψ, ghnodes, ghweights)
        """
        Compute the germination response for an inhibitor-dependent
        induction signal for a given set of parameters.
        Uses Gauss-Hermite approximation.
        inputs:
            t - time in seconds
            ρₛ - spore density in spores/um^3
            μ_ξ - mean spore radius in um
            σ_ξ - standard deviation of spore radius in um
            Pₛ - permeation constant for the inhibitor in um/s
            s - inducer signal strength
            K_I - half-saturation constant for the inhibitor
            n - Hill coefficient for the inhibitor
            μ_ω - mean induction threshold
            σ_ω - standard deviation of induction threshold
            μ_ψ - mean initial concentration
            σ_ψ - standard deviation of initial concentration
            ghnodes - Gauss-Hermite nodes
            ghweights - Gauss-Hermite weights
        output:
            the germination response for the given parameters
        """
        
        # Transform to log-normal
        u = √2 .* ghnodes
        hw = ghweights ./ √π
        μ_ξ_log = log(μ_ξ^2 / sqrt(σ_ξ^2 + μ_ξ^2))
        σ_ξ_log = sqrt(log(σ_ξ^2 / μ_ξ^2 + 1))
        μ_ψ_log = log(μ_ψ^2 / sqrt(σ_ψ^2 + μ_ψ^2))
        σ_ψ_log = sqrt(log(σ_ψ^2 / μ_ψ^2 + 1))
        ξ = exp.(μ_ξ_log .+ σ_ξ_log .* u)
        ψ = exp.(μ_ψ_log .+ σ_ψ_log .* u)

        # Reshape
        ξ, ψ = meshgrid(ξ, ψ)

        # Distributions
        dist_ω = Normal(μ_ω, σ_ω)

        # Inhibitor
        V = 4/3 * π .* ξ.^3
        A = 4 * π .* ξ.^2
        τ = V ./ (Pₛ * A)
        ϕ = ρₛ .* V

        c_in = ψ .* (ϕ .+ (1 .- ϕ) .* exp.(-t ./ (τ .* (1 .- ϕ))))
        s_mod = s ./ (1 .+ (c_in ./ K_I).^n)

        tail = cdf.(dist_ω, s_mod)
        W = hw * hw'

        return sum(W .* tail)
    end


    function germ_response_inhibitor_dep_inducer_signal_st_gh(t, ρₛ, c₀_cs, d_hp, μ_ξ, σ_ξ, μ_κ, σ_κ, Pₛ, Pₛ_cs, s_max, K_cs, K_I, n, μ_ω, σ_ω, μ_ψ, σ_ψ, ghnodes, ghweights)
        """
        Compute the germination response for an inhibitor-dependent
        induction signal for a given set of parameters.
        The inducer signal is time-dependent.
        Uses Gauss-Hermite approximation.
        inputs:
            t - time in seconds
            ρₛ - spore density in spores/um^3
            c₀_cs - initial concentration of carbon source in M
            d_hp - thickness of the hydrophobin layer in um
            μ_ξ - mean spore radius in um
            σ_ξ - standard deviation of spore radius in um
            μ_κ - mean cell wall thickness in um
            σ_κ - standard deviation of cell wall thickness in um
            Pₛ - permeation constant for the inhibitor in um/s
            Pₛ_cs - permeation constant for the carbon source in um/s
            s_max - inducer signal strength
            K_cs - half-saturation constant for the carbon source
            K_I - half-saturation constant for the inhibitor
            n - Hill coefficient for the inhibitor
            k - inhibition strength over induction threshold
            μ_ω - mean induction threshold
            σ_ω - standard deviation of induction threshold
            μ_ψ - mean initial concentration
            σ_ψ - standard deviation of initial concentration
            ghnodes - Gauss-Hermite nodes
            ghweights - Gauss-Hermite weights
        output:
            the germination response for the given parameters
        """

        # Transform to log-normal
        u = √2 .* ghnodes
        hw = ghweights ./ √π
        μ_ξ_log = log(μ_ξ^2 / sqrt(σ_ξ^2 + μ_ξ^2))
        σ_ξ_log = sqrt(log(σ_ξ^2 / μ_ξ^2 + 1))
        μ_κ_log = log(μ_κ^2 / sqrt(σ_κ^2 + μ_κ^2))
        σ_κ_log = sqrt(log(σ_κ^2 / μ_κ^2 + 1))
        μ_ψ_log = log(μ_ψ^2 / sqrt(σ_ψ^2 + μ_ψ^2))
        σ_ψ_log = sqrt(log(σ_ψ^2 / μ_ψ^2 + 1))
        ξ = exp.(μ_ξ_log .+ σ_ξ_log .* u)
        κ = exp.(μ_κ_log .+ σ_κ_log .* u)
        ψ = exp.(μ_ψ_log .+ σ_ψ_log .* u)

        # Reshape
        ξ, κ, ψ = meshgrid(ξ, κ, ψ)

        # Distributions
        dist_ω = Normal(μ_ω, σ_ω)

        # Inducer
        A = 4 * π .* ξ.^2
        V_cw = 0.32 .* π .* ((ξ .- d_hp).^3 .- (ξ .- d_hp .- κ).^3)
        c_cs = inducer_concentration.(c₀_cs, t, Pₛ_cs, A, V_cw)
        s = s_max .* c_cs ./ (K_cs .+ c_cs)

        # Inhibitor
        V = 4/3 * π .* ξ.^3
        τ = V ./ (Pₛ * A)
        ϕ = ρₛ .* V

        c_in = ψ .* (ϕ .+ (1 .- ϕ) .* exp.(-t ./ (τ .* (1 .- ϕ))))
        s_mod = s ./ (1 .+ (c_in ./ K_I).^n)

        tail = cdf.(dist_ω, s_mod)
        N = length(ghnodes)
        W = reshape(hw, N,1,1) .* reshape(hw, 1,N,1) .* reshape(hw, 1,1,N)

        return sum(W .* tail)
    end

    
    function germ_response_inhibitor_dep_inducer_combined_gh(t, ρₛ, μ_ξ, σ_ξ, Pₛ, s, k, K_I, n, μ_ω, σ_ω, μ_ψ, σ_ψ, ghnodes, ghweights)
        """
        Compute the germination response for an inhibitor-dependent
        induction threshold and signal for a given set of parameters.
        Uses Gauss-Hermite approximation.
        inputs:
            t - time in seconds
            ρₛ - spore density in spores/um^3
            μ_ξ - mean spore radius in um
            σ_ξ - standard deviation of spore radius in um
            Pₛ - permeation constant for the inhibitor in um/s
            s - inducer signal strength
            k - inhibition strength over induction threshold
            K_I - half-saturation constant for the inhibitor
            n - Hill coefficient for the inhibitor
            μ_ω - mean induction threshold
            σ_ω - standard deviation of induction threshold
            μ_ψ - mean initial concentration
            σ_ψ - standard deviation of initial concentration
            ghnodes - Gauss-Hermite nodes
            ghweights - Gauss-Hermite weights
        output:
            the germination response for the given parameters
        """

        # Transform to log-normal
        u = √2 .* ghnodes
        hw = ghweights ./ √π
        μ_ξ_log = log(μ_ξ^2 / sqrt(σ_ξ^2 + μ_ξ^2))
        σ_ξ_log = sqrt(log(σ_ξ^2 / μ_ξ^2 + 1))
        μ_ψ_log = log(μ_ψ^2 / sqrt(σ_ψ^2 + μ_ψ^2))
        σ_ψ_log = sqrt(log(σ_ψ^2 / μ_ψ^2 + 1))
        ξ = exp.(μ_ξ_log .+ σ_ξ_log .* u)
        ψ = exp.(μ_ψ_log .+ σ_ψ_log .* u)

        # Reshape
        ξ, ψ = meshgrid(ξ, ψ)

        # Distributions
        dist_ω = Normal(μ_ω, σ_ω)

        # Inhibitor
        V = 4/3 * π .* ξ.^3
        A = 4 * π .* ξ.^2
        τ = V ./ (Pₛ * A)
        ϕ = ρₛ .* V

        c_in = ψ .* (ϕ .+ (1 .- ϕ) .* exp.(-t ./ (τ .* (1 .- ϕ))))
        s_mod = s ./ (1 .+ (c_in ./ K_I).^n)

        tail = cdf.(dist_ω, s_mod .- k .* c_in)
        W = hw * hw'

        return sum(W .* tail)
    end


    function germ_response_inhibitor_dep_inducer_combined_st_gh(t, ρₛ, c₀_cs, d_hp, μ_ξ, σ_ξ, μ_κ, σ_κ, Pₛ, Pₛ_cs, s_max, k, K_cs, K_I, n, μ_ω, σ_ω, μ_ψ, σ_ψ, ghnodes, ghweights)
        """
        Compute the germination response for an inhibitor-dependent
        induction threshold and signal for a given set of parameters.
        The inducer signal is time-dependent.
        Uses Gauss-Hermite approximation.
        inputs:
            t - time in seconds
            ρₛ - spore density in spores/um^3
            c₀_cs - initial concentration of carbon source in M
            d_hp - thickness of the hydrophobin layer in um
            μ_ξ - mean spore radius in um
            σ_ξ - standard deviation of spore radius in um
            μ_κ - mean cell wall thickness in um
            σ_κ - standard deviation of cell wall thickness in um
            Pₛ - permeation constant for the inhibitor in um/s
            Pₛ_cs - permeation constant for the carbon source in um/s
            s_max - inducer signal strength
            k - proportionality constant for threshold modulation vs signal modulation
            K_cs - half-saturation constant for the carbon source
            K_I - half-saturation constant for the inhibitor
            n - Hill coefficient for the inhibitor
            k - inhibition strength over induction threshold
            μ_ω - mean induction threshold
            σ_ω - standard deviation of induction threshold
            μ_ψ - mean initial concentration
            σ_ψ - standard deviation of initial concentration
            ghnodes - Gauss-Hermite nodes
            ghweights - Gauss-Hermite weights
        output:
            the germination response for the given parameters
        """

        # Transform to log-normal
        u = √2 .* ghnodes
        hw = ghweights ./ √π
        μ_ξ_log = log(μ_ξ^2 / sqrt(σ_ξ^2 + μ_ξ^2))
        σ_ξ_log = sqrt(log(σ_ξ^2 / μ_ξ^2 + 1))
        μ_κ_log = log(μ_κ^2 / sqrt(σ_κ^2 + μ_κ^2))
        σ_κ_log = sqrt(log(σ_κ^2 / μ_κ^2 + 1))
        μ_ψ_log = log(μ_ψ^2 / sqrt(σ_ψ^2 + μ_ψ^2))
        σ_ψ_log = sqrt(log(σ_ψ^2 / μ_ψ^2 + 1))
        ξ = exp.(μ_ξ_log .+ σ_ξ_log .* u)
        κ = exp.(μ_κ_log .+ σ_κ_log .* u)
        ψ = exp.(μ_ψ_log .+ σ_ψ_log .* u)

        # Reshape
        ξ, κ, ψ = meshgrid(ξ, κ, ψ)

        # Distributions
        dist_ω = Normal(μ_ω, σ_ω)

        # Inducer
        A = 4 * π .* ξ.^2
        V_cw = 0.32 .* π .* ((ξ .- d_hp).^3 .- (ξ .- d_hp .- κ).^3)
        c_cs = inducer_concentration.(c₀_cs, t, Pₛ_cs, A, V_cw)
        s = s_max .* c_cs ./ (K_cs .+ c_cs)

        # Inhibitor
        V = 4/3 * π .* ξ.^3
        τ = V ./ (Pₛ * A)
        ϕ = ρₛ .* V

        c_in = ψ .* (ϕ .+ (1 .- ϕ) .* exp.(-t ./ (τ .* (1 .- ϕ))))
        s_mod = s ./ (1 .+ (c_in ./ K_I).^n)

        tail = cdf.(dist_ω, s_mod .- k .* c_in)
        N = length(ghnodes)
        W = reshape(hw, N,1,1) .* reshape(hw, 1,N,1) .* reshape(hw, 1,1,N)

        return sum(W .* tail)
    end
end