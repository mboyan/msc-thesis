module DataUtils
    """
    Contains utility functions.
    """

    # using Distributed
    # addprocs(4)

    using DataFrames
    using CSV
    using FastGaussQuadrature
    using BlackBoxOptim
    using Optim
    using NLopt
    using MeshGrid
    using ArgCheck
    using Revise
    
    include("./conversions.jl")
    include("./germstats.jl")
    Revise.includet("./conversions.jl")
    Revise.includet("./germstats.jl")
    using .Conversions
    using .GermStats

    export parse_ijadpanahsaravi_data
    export dantigny
    export generate_dantigny_dataset
    export fit_model_to_data
    export get_params_for_idx
    

    function parse_ijadpanahsaravi_data()
        """
        Parses the dataset from Ijadpanahsaravi et al. (2023)
        with multiple incolulum densities and returns a DataFrame.
        """

        df_germination = DataFrame(CSV.File("Data/swelling_germination_results.csv"; header=true))

        # Filter the data to only include swelling
        df_germination_swelling = filter(row -> row[1] == "Swelling", df_germination)

        # Expression for parsing
        regex_triplet = r"(-?\d+\.\d+)\[(-?\d+\.\d+);(-?\d+\.\d+)\]*.*"

        # Function to parse a string and return three numbers
        function parse_numbers(s)
            m = match(regex_triplet, s)
            if m !== nothing
                num1 = parse(Float64, m.captures[1])
                num2 = parse(Float64, m.captures[2])
                num3 = parse(Float64, m.captures[3])
                return num1, num2, num3
            else
                return missing, missing, missing
            end
        end

        # Parse Pmax and its confidence intervals
        Pmax_parsed = [parse_numbers(row[1]) for row in eachrow(df_germination_swelling[!, 4])]
        Pmax_vals = [x[1] for x in Pmax_parsed]
        PmaxCIlow_vals = [x[2] for x in Pmax_parsed]
        PmaxCIhigh_vals = [x[3] for x in Pmax_parsed]

        # Parse tau and its confidence intervals
        tau_parsed = [parse_numbers(row[1]) for row in eachrow(df_germination_swelling[!, 5])]
        tau_vals = [x[1] for x in tau_parsed]
        tauCIlow_vals = [x[2] for x in tau_parsed]
        tauCIhigh_vals = [x[3] for x in tau_parsed]

        # Parse d and its confidence intervals
        d_parsed = [parse_numbers(row[1]) for row in eachrow(df_germination_swelling[!, 6])]
        d_vals = [x[1] for x in d_parsed]
        dCIlow_vals = [x[2] for x in d_parsed]
        dCIhigh_vals = [x[3] for x in d_parsed]

        # Reconstruct the DataFrame with the parsed values
        df_germination_rebuilt = DataFrame(
            :CarbonSource => df_germination_swelling[!, 2],
            :Density => inverse_um_to_mL(df_germination_swelling[!, 3] / 150),
            :Pmax => Pmax_vals,
            :Pmax_CI_Lower => PmaxCIlow_vals,
            :Pmax_CI_Upper => PmaxCIhigh_vals,
            :tau => tau_vals,
            :tau_CI_Lower => tauCIlow_vals,
            :tau_CI_Upper => tauCIhigh_vals,
            :d => d_vals,
            :d_CI_Lower => dCIlow_vals,
            :d_CI_Upper => dCIhigh_vals,
            :N => df_germination_swelling[!, 8],
            :M => df_germination_swelling[!, 9]
        )

        df_germination_rebuilt
    end


    function dantigny(t, p_max, τ, ν)
        """
        Dantigny model for the germination of a fungal culture.
        inputs:
            t: time (in hours)
            p_max: maximum germination rate
            τ: time constant (in hours)
            ν: design parameter (dimensionless)
        outputs:
            p: germination rate (dimensionless)
        """
        p = p_max * (1 - 1 / (1 + (t / τ)^ν))
        return p
    end


    function generate_dantigny_dataset(df_germination, t_max, n_pts=1000)
        """
        Generate time-dependent germination data using the Dantigny model
        and parameters from a dictionary.
        inputs:
            df_germination (DataFrame): Dantigny model parameters
            t_max (float): maximum time point in seconds
            n_pts (int): number of time points to generate
        outputs:
            times (Vector): vector of time points
            dantigny_data (Matrix): matrix of time-dependent germination data
        """
        
        sources = unique(df_germination[!, :CarbonSource])
        densities = unique(df_germination[!, :Density])
        
        times = collect(LinRange(0, t_max, n_pts))
        dantigny_data = zeros(length(sources), length(densities), n_pts)
        errs = zeros(length(sources), length(densities), 2)
        p_maxs = zeros(length(sources), length(densities))
        taus = zeros(length(sources), length(densities))
        nus = zeros(length(sources), length(densities))
        for (i, source) in enumerate(sources)
            for (j, density) in enumerate(densities)
                # Get the parameters for the current source and density
                params = df_germination[(df_germination[!, :CarbonSource] .== source) .& (df_germination[!, :Density] .== density), :]
                
                if nrow(params) == 0
                    continue
                end
                
                p_max = params[1, :Pmax]
                τ = params[1, :tau]
                ν = params[1, :d]
                
                # Generate the time course using the Dantigny model
                dantigny_data[i, j, :] = dantigny.(times, p_max, τ, ν)

                # Confidence_intervals
                errs[i, j, 1] = max(p_max - params[1, :Pmax_CI_Lower], 0)
                errs[i, j, 2] = params[1, :Pmax_CI_Upper] - p_max

                # Dantigny parameters
                p_maxs[i, j] = p_max
                taus[i, j] = τ
                nus[i, j] = ν
            end
        end

        return dantigny_data * 0.01, times * 3600, sources, densities, errs, p_maxs, taus, nus
    end


    function fit_model_to_data(model_type, def_params, dantigny_data, times, sources, densities, bounds_dict, p_maxs; st=false, max_steps=10000)
        """
        Fit a selected germination model to the data.
        inputs:
            model_type (String): model type to fit
                ("independent", "inhibitor", "inhibitor_thresh", "inhibitor_perm", "inducer", "inducer_thresh", "inducer_signal")
            def_params (Dict): default parameter values
            dantigny_data (Matrix): time-dependent data from varying inducers and spore densities to fit
            times (Vector): time points
            sources (Vector): carbon sources
            densities (Vector): spore densities
            bounds_dict (Dict): bounds for the free parameters
            p_maxs (Matrix): saturation fractions for the Dantigny model
            st (Bool): whether to use a time-dependent inducer
            max_steps (int): maximum number of steps for the optimization
        outputs:
            params_out (Dict): optimized parameters
        """
        @argcheck model_type in ["independent",
                                "inhibitor", "inhibitor_thresh", "inhibitor_perm",
                                "inducer", "inducer_thresh", "inducer_signal",
                                "combined", "combined_thresh", "combined_signal",
                                "special_inhibitor", "special_independent"]

        # println("Running ", model_type, " model fitting with ", st ? "time-dependent inducer" : "static inducer")

        # Reshape input
        densities_tile = repeat(densities, outer=[1, length(sources), length(times)])
        densities_tile = permutedims(densities_tile, (2, 1, 3))
        times_tile = repeat(times, outer=[1, length(sources), length(densities)])
        times_tile = permutedims(times_tile, (2, 3, 1))

        # Determine number of nodes depending on the integral dimension
        if model_type in ["independent", "inhibitor", "inhibitor_thresh", "inhibitor_perm"] && !st
            n_nodes = 216 # 1D integral
        elseif (model_type in ["inducer", "inducer_thresh", "inducer_signal"] && !st) ||
                (model_type in ["independent", "inhibitor", "inhibitor_thresh", "inhibitor_perm"] && st) ||
                model_type in ["special_inhibitor"]
            n_nodes = 36 # 2D integral
        elseif (model_type in ["inducer", "inducer_thresh", "inducer_signal"] && st) ||
                model_type in ["combined", "combined_thresh", "combined_signal", "special_independent"]
            n_nodes = 10 # 3D integral
        end
        println("Number of nodes: ", n_nodes)

        # Gauss-Hermite nodes
        ghnodes, ghweights = gausshermite(n_nodes)
        u = √2 .* ghnodes
        hw = ghweights ./ √π

        # Unpack means and stds and weight samples
        μ_ξ = def_params[:μ_ξ]
        σ_ξ = def_params[:σ_ξ]
        μ_ξ_log = log(μ_ξ^2 / sqrt(σ_ξ^2 + μ_ξ^2))
        σ_ξ_log = sqrt(log(σ_ξ^2 / μ_ξ^2 + 1))
        ξ = exp.(μ_ξ_log .+ σ_ξ_log .* u)

        if haskey(def_params, :μ_κ)
            μ_κ = def_params[:μ_κ]
            σ_κ = def_params[:σ_κ]
            μ_κ_log = log(μ_κ^2 / sqrt(σ_κ^2 + μ_κ^2))
            σ_κ_log = sqrt(log(σ_κ^2 / μ_κ^2 + 1))
            κ = exp.(μ_κ_log .+ σ_κ_log .* u)

            ξ2, κ2 = meshgrid(ξ, κ)
        end

        W = hw * hw'
        W3 = reshape(hw, n_nodes,1,1) .* reshape(hw, 1,n_nodes,1) .* reshape(hw, 1,1,n_nodes)

        n_src = length(sources)

        model_type_split = split(model_type, "_")
        
        if model_type == "independent"
            # Independent inducer/inhibitor
            if st
                println("Model: independent factors with time-dependent inducer")
                wrapper = (inputs, params) -> germ_response_independent_factors_st_gh(
                    u, W,
                    inputs[1], #t
                    inputs[2], #ρₛ
                    def_params[:c₀_cs],
                    def_params[:d_hp],
                    ξ2,
                    κ2,
                    params[1], #Pₛ
                    params[2], #Pₛ_cs
                    params[3], #K_cs
                    params[4], #μ_γ
                    params[4] * exp(params[5]), # σ_γ = μ_γ * exp(δ_γ)
                    params[6], #μ_ω
                    params[6] * exp(params[7]) # σ_ω = μ_ω * exp(δ_ω)
                )
                param_keys = [:Pₛ, :Pₛ_cs, :K_cs, :μ_γ, :δ_γ, :μ_ω, :δ_ω]
                param_occurrences = [1, n_src, n_src, 1, 1, n_src, n_src]
            else
                println("Model: independent factors with static inducer")
                wrapper = (inputs, params) -> Main.germ_response_independent_factors_gh(
                    u, hw,
                    inputs[1], #t
                    inputs[2], #ρₛ
                    ξ,
                    params[1], #Pₛ
                    params[2], #μ_γ
                    params[2] * exp(params[3]), # σ_γ = μ_γ * exp(δ_γ)
                    params[4], #μ_ω
                    params[4] * exp(params[5]) # σ_ω = μ_ω * exp(δ_ω)
                )
                param_keys = [:Pₛ, :μ_γ, :δ_γ, :μ_ω, :δ_ω]
                param_occurrences = [1, 1, 1, n_src, n_src]
            end

        elseif model_type_split[1] == "inhibitor"

            if model_type == "inhibitor" # Inducer shifts inhibition threshold and modulates inhibitor permeability
                if st # Time-dependent inducer
                    println("Model: inducer-modulated inhibitor (combined) with time-dependent inducer")
                    wrapper = (inputs, params) -> Main.germ_response_inducer_dep_inhibitor_combined_st_gh(
                        u, W,
                        inputs[1], #t
                        inputs[2], #ρₛ
                        def_params[:c₀_cs],
                        def_params[:d_hp],
                        ξ2,
                        κ2,
                        params[1], #Pₛ,
                        params[2], #Pₛ_cs,
                        params[3], #K_cs,
                        exp(params[4]), #k,
                        params[5], #μ_γ,
                        params[5] * exp(params[6]), # σ_γ = μ_γ * exp(δ_γ)
                    )
                    param_keys = [:Pₛ, :Pₛ_cs, :K_cs, :k, :μ_γ, :δ_γ]
                    param_occurrences = [1, n_src, n_src, n_src, 1, 1]
                else
                    println("Model: inducer-modulated inhibitor (combined) with static inducer")
                    wrapper = (inputs, params) -> Main.germ_response_inhibitor_gh(
                        u, hw,
                        inputs[1], #t
                        inputs[2], #ρₛ
                        ξ,
                        params[1], #Pₛ
                        params[2], #μ_γ
                        params[2] * exp(params[3]) # σ_γ = μ_γ * exp(δ_γ)
                    )
                    param_keys = [:Pₛ, :μ_γ, :δ_γ]
                    param_occurrences = [n_src, n_src, n_src]
                end

            elseif model_type_split[2] == "thresh" # Inducer shifts inhibition threshold
                if st # Time-dependent inducer
                    println("Model: inducer-modulated inhibitor (threshold) with time-dependent inducer")
                    wrapper = (inputs, params) -> Main.germ_response_inducer_dep_inhibitor_thresh_st_gh(
                        u, W,
                        inputs[1], #t
                        inputs[2], #ρₛ
                        def_params[:c₀_cs],
                        def_params[:d_hp],
                        ξ2,
                        κ2,
                        params[1], #Pₛ
                        params[2], #Pₛ_cs
                        exp(params[3]), #k
                        params[4], #K_cs
                        params[5], #μ_γ
                        params[5] * exp(params[6]) # σ_γ = μ_γ * exp(δ_γ)
                    )
                    param_keys = [:Pₛ, :Pₛ_cs, :k, :K_cs, :μ_γ, :δ_γ]
                    param_occurrences = [1, n_src, n_src, n_src, 1, 1]
                else
                    println("Model: inducer-modulated inhibitor (threshold) with static inducer")
                    wrapper = (inputs, params) -> Main.germ_response_inhibitor_gh(
                        u, hw,
                        inputs[1], #t
                        inputs[2], #ρₛ
                        ξ,
                        params[1], #Pₛ
                        params[2], #μ_γ
                        params[2] * exp(params[3]) # σ_γ = μ_γ * exp(δ_γ)
                    )
                    param_keys = [:Pₛ, :μ_γ, :δ_γ]
                    param_occurrences = [1, n_src, n_src]
                end
            elseif model_type_split[2] == "perm" # Inducer modulates inhibitor permeability
                if st # Time-dependent inducer
                    println("Model: inducer-modulated inhibitor (permeability) with time-dependent inducer")
                    wrapper = (inputs, params) -> Main.germ_response_inducer_dep_inhibitor_perm_st_gh(
                        u, W,
                        inputs[1], #t
                        inputs[2], #ρₛ
                        def_params[:c₀_cs],
                        def_params[:d_hp],
                        ξ2,
                        κ2,
                        params[1], #Pₛ
                        params[2], #Pₛ_cs
                        params[3], #K_cs
                        params[4], #μ_γ
                        params[4] * exp(params[5]) # σ_γ = μ_γ * exp(δ_γ)
                    )
                    param_keys = [:Pₛ, :Pₛ_cs, :K_cs, :μ_γ, :δ_γ]
                    param_occurrences = [1, n_src, n_src, 1, 1]
                else
                    println("Model: inducer-modulated inhibitor (permeability) with static inducer")
                    wrapper = (inputs, params) -> Main.germ_response_inhibitor_gh(
                        u, hw,
                        inputs[1], #t
                        inputs[2], #ρₛ
                        ξ,
                        params[1], #Pₛ
                        params[2], #μ_γ
                        params[2] * exp(params[3]) # σ_γ = μ_γ * exp(δ_γ)
                    )
                    param_keys = [:Pₛ, :μ_γ, :δ_γ]
                    param_occurrences = [n_src, 1, 1]
                end            
            end

        elseif model_type_split[1] == "inducer"

            if model_type == "inducer" # Inhibitor shifts induction threshold and modulates inducer signal strength
                if st
                    println("Model: inhibitor-modulated inducer (combined) with time-dependent inducer")
                    wrapper = (inputs, params) -> Main.germ_response_inhibitor_dep_inducer_combined_st_gh(
                        u, W3,
                        inputs[1], #t
                        inputs[2], #ρₛ
                        def_params[:c₀_cs],
                        def_params[:d_hp],
                        ξ2,
                        κ2,
                        params[1], #Pₛ
                        params[2], #Pₛ_cs
                        exp(params[3]), #k
                        params[4], #K_cs
                        params[5], #K_I
                        params[6], #n
                        params[7], #μ_ω
                        params[7] * exp(params[8]), # σ_ω = μ_ω * exp(δ_ω)
                        params[9], #μ_ψ
                        params[9] * exp(params[10]) # σ_ψ = μ_ψ * exp(δ_ψ)
                    )
                    param_keys = [:Pₛ, :Pₛ_cs, :k, :K_cs, :K_I, :n, :μ_ω, :δ_ω, :μ_ψ, :δ_ψ]
                    param_occurrences = [1, n_src, n_src, n_src, n_src, n_src, n_src, n_src, 1, 1]
                else
                    println("Model: inhibitor-modulated inducer (combined) with static inducer")
                    # Reconstruct standard deviation
                    wrapper = (inputs, params) -> Main.germ_response_inhibitor_dep_inducer_combined_gh(
                        u, W,
                        inputs[1], #t
                        inputs[2], #ρₛ
                        ξ,
                        params[1], #Pₛ
                        exp(params[2]), #k
                        params[3], #K_I
                        params[4], #n
                        params[5], #μ_ω
                        params[5] * exp(params[6]), # σ_ω = μ_ω * exp(δ_ω)
                        params[7], #μ_ψ
                        params[7] * exp(params[8]) # σ_ψ = μ_ψ * exp(δ_ψ)
                    )
                    param_keys = [:Pₛ, :k, :K_I, :n, :μ_ω, :δ_ω, :μ_ψ, :δ_ψ]
                    param_occurrences = [1, n_src, n_src, n_src, n_src, n_src, 1, 1]
                end

            elseif model_type_split[2] == "thresh" # Inhibitor shifts induction threshold
                if st
                    println("Model: inhibitor-modulated inducer (threshold) with time-dependent inducer")
                    wrapper = (inputs, params) -> Main.germ_response_inhibitor_dep_inducer_thresh_st_gh(
                        u, W3,
                        inputs[1], #t
                        inputs[2], #ρₛ
                        def_params[:c₀_cs],
                        def_params[:d_hp],
                        ξ2,
                        κ2,
                        params[1], #Pₛ
                        params[2], #Pₛ_cs
                        params[3], #K_cs
                        exp(params[4]), #k
                        params[5], #μ_ω
                        params[5] * exp(params[6]), # σ_ω = μ_ω * exp(δ_ω)
                        params[7], #μ_ψ
                        params[7] * exp(params[8]) # σ_ψ = μ_ψ * exp(δ_ψ)
                    )
                    param_keys = [:Pₛ, :Pₛ_cs, :K_cs, :k, :μ_ω, :δ_ω, :μ_ψ, :δ_ψ]
                    param_occurrences = [1, n_src, n_src, n_src, n_src, n_src, 1, 1]
                else
                    println("Model: inhibitor-modulated inducer (threshold) with static inducer")
                    wrapper = (inputs, params) -> Main.germ_response_inhibitor_dep_inducer_thresh_gh(
                        u, W,
                        inputs[1], #t
                        inputs[2], #ρₛ
                        ξ,
                        params[1], #Pₛ
                        exp(params[2]), #k
                        params[3], #μ_ω
                        params[3] * exp(params[4]), # σ_ω = μ_ω * exp(δ_ω)
                        params[5], #μ_ψ
                        params[5] * exp(params[6]) # σ_ψ = μ_ψ * exp(δ_ψ)
                    )
                    param_keys = [:Pₛ, :k, :μ_ω, :δ_ω, :μ_ψ, :δ_ψ]
                    param_occurrences = [1, n_src, n_src, n_src, 1, 1]
                end

            elseif model_type_split[2] == "signal" # Inhibitor shifts induction threshold
                if st
                    println("Model: inhibitor-modulated inducer (signal) with time-dependent inducer")
                    wrapper = (inputs, params) -> Main.germ_response_inhibitor_dep_inducer_signal_st_gh(
                        u, W3,
                        inputs[1], #t
                        inputs[2], #ρₛ
                        def_params[:c₀_cs],
                        def_params[:d_hp],
                        ξ2,
                        κ2,
                        params[1], #Pₛ
                        params[2], #Pₛ_cs
                        params[3], #K_cs
                        params[4], #K_I
                        params[5], #n
                        params[6], #μ_ω
                        params[6] * exp(params[7]), # σ_ω = μ_ω * exp(δ_ω)
                        params[8], #μ_ψ
                        params[8] * exp(params[9]) # σ_ψ = μ_ψ * exp(δ_ψ)
                    )
                    param_keys = [:Pₛ, :Pₛ_cs, :K_cs, :K_I, :n, :μ_ω, :δ_ω, :μ_ψ, :δ_ψ]
                    param_occurrences = [1, n_src, n_src, n_src, n_src, n_src, n_src, 1, 1]
                else
                    println("Model: inhibitor-modulated inducer (signal) with static inducer")
                    wrapper = (inputs, params) -> Main.germ_response_inhibitor_dep_inducer_signal_gh(
                        u, W,
                        inputs[1], #t
                        inputs[2], #ρₛ
                        ξ,
                        params[1], #Pₛ
                        params[2], #K_I
                        params[3], #n
                        params[4], #μ_ω
                        params[4] * exp(params[5]), # σ_ω = μ_ω * exp(δ_ω)
                        params[6], #μ_ψ
                        params[6] * exp(params[7]) # σ_ψ = μ_ψ * exp(δ_ψ)
                    )
                    param_keys = [:Pₛ, :K_I, :n, :μ_ω, :δ_ω, :μ_ψ, :δ_ψ]
                    param_occurrences = [1, n_src, n_src, n_src, n_src, 1, 1]
                end
            end

        elseif model_type_split[1] == "combined" 

            if model_type == "combined"
                println("Model: inhibitor-modulated inducer (combined) with time-dependent inducer")
                wrapper = (inputs, params) -> Main.germ_response_inhibitor_dep_inducer_combined_2_factor_st_gh(
                    u, W3,
                    inputs[1], #t
                    inputs[2], #ρₛ
                    def_params[:c₀_cs],
                    def_params[:d_hp],
                    ξ2,
                    κ2,
                    params[1], #Pₛ
                    params[2], #Pₛ_cs
                    params[3], #K_cs
                    params[4], #K_I
                    params[5], #n
                    exp(params[6]), #k
                    params[7], #μ_γ
                    params[7] * exp(params[8]), # σ_γ = μ_γ * exp(δ_γ)
                    params[9], #μ_ω
                    params[9] * exp(params[10]), # σ_ω = μ_ω * exp(δ_ω)
                    params[11], #μ_ψ
                    params[11] * exp(params[12]) # σ_ψ = μ_ψ * exp(δ_ψ)
                )
                param_keys = [:Pₛ, :Pₛ_cs, :K_cs, :K_I, :n, :k, :μ_γ, :δ_γ, :μ_ω, :δ_ω, :μ_ψ, :δ_ψ]
                param_occurrences = [1, n_src, n_src, n_src, n_src, n_src, 1, 1, n_src, n_src, 1, 1]
            elseif model_type_split[2] == "thresh"
                println("Model: inhibitor-modulated inducer (threshold) with time-dependent inducer")
                wrapper = (inputs, params) -> Main.germ_response_inhibitor_dep_inducer_thresh_2_factor_st_gh(
                    u, W3,
                    inputs[1], #t
                    inputs[2], #ρₛ
                    def_params[:c₀_cs],
                    def_params[:d_hp],
                    ξ2,
                    κ2,
                    params[1], #Pₛ
                    params[2], #Pₛ_cs
                    params[3], #K_cs
                    exp(params[4]), #k
                    params[5], #μ_γ
                    params[5] * exp(params[6]), # σ_γ = μ_γ * exp(δ_γ)
                    params[7], #μ_ω
                    params[7] * exp(params[8]), # σ_ω = μ_ω * exp(δ_ω)
                    params[9], #μ_ψ
                    params[9] * exp(params[10]) # σ_ψ = μ_ψ * exp(δ_ψ)
                )
                param_keys = [:Pₛ, :Pₛ_cs, :K_cs, :k, :μ_γ, :δ_γ, :μ_ω, :δ_ω, :μ_ψ, :δ_ψ]
                param_occurrences = [1, n_src, n_src, n_src, 1, 1, n_src, n_src, 1, 1]
            elseif model_type_split[2] == "signal"
                println("Model: inhibitor-modulated inducer (signal) with time-dependent inducer")
                wrapper = (inputs, params) -> Main.germ_response_inhibitor_dep_inducer_signal_2_factor_st_gh(
                    u, W3,
                    inputs[1], #t
                    inputs[2], #ρₛ
                    def_params[:c₀_cs],
                    def_params[:d_hp],
                    ξ2,
                    κ2,
                    params[1], #Pₛ
                    params[2], #Pₛ_cs
                    params[3], #K_cs
                    params[4], #K_I
                    params[5], #n
                    params[6], #μ_γ
                    params[6] * exp(params[7]), # σ_γ = μ_γ * exp(δ_γ)
                    params[8], #μ_ω
                    params[8] * exp(params[9]), # σ_ω = μ_ω * exp(δ_ω)
                    params[10], #μ_ψ
                    params[10] * exp(params[11]) # σ_ψ = μ_ψ * exp(δ_ψ)
                )
                param_keys = [:Pₛ, :Pₛ_cs, :K_cs, :K_I, :n, :μ_γ, :δ_γ, :μ_ω, :δ_ω, :μ_ψ, :δ_ψ]
                param_occurrences = [1, n_src, n_src, n_src, n_src, 1, 1, n_src, n_src, 1, 1]
            end
        elseif model_type_split[1] == "special" 
            if model_type_split[2] == "inhibitor"
                println("Model: inducer-modulated inhibitor (threshold) with static inducer and varying permeability")
                wrapper = (inputs, params) -> Main.germ_response_inhibitor_var_perm_gh(
                    u, W,
                    inputs[1], #t
                    inputs[2], #ρₛ
                    ξ,
                    params[1], #μ_π
                    params[1] * exp(params[2]), # σ_π = μ_π * exp(δ_π)
                    params[3], #μ_γ
                    params[3] * exp(params[4]) # σ_γ = μ_γ * exp(δ_γ)
                )
                param_keys = [:μ_π, :δ_π, :μ_γ, :δ_γ]
                param_occurrences = [1, 1, n_src, n_src]
            elseif model_type_split[2] == "independent"
                println("Model: independent factors with static inducer and varying permeability")
                wrapper = (inputs, params) -> Main.germ_response_independent_factors_var_perm_st_gh(
                    u, W3,
                    inputs[1], #t
                    inputs[2], #ρₛ
                    def_params[:c₀_cs],
                    def_params[:d_hp],
                    ξ2,
                    κ2,
                    params[1], #Pₛ
                    params[2], #Pₛ_cs
                    params[3], #K_cs
                    params[4], #μ_γ
                    params[4] * exp(params[5]), # σ_γ = μ_γ * exp(δ_γ)
                    params[6], #μ_ω
                    params[6] * exp(params[7]), # σ_ω = μ_ω * exp(δ_ω)
                    params[8], #μ_α
                    params[8] * exp(params[9]) # σ_α = μ_α * exp(δ_α)
                )
                param_keys = [:Pₛ, :Pₛ_cs, :K_cs, :μ_γ, :δ_γ, :μ_ω, :δ_ω, :μ_α, :δ_α]
                param_occurrences = [1, n_src, n_src, 1, 1, n_src, n_src, 1, 1]
            end
        else
            error("Model type not recognized.")
        end

        # Duplicate bounds/parameters for each source
        param_starts = cumsum(param_occurrences) .- param_occurrences .+ 1
        param_keys_dup = vcat([key for key in param_keys for _ in 1:param_occurrences[param_keys .== key][1]]...)
        param_starts = cumsum(param_occurrences) .- param_occurrences .+ 1
        bounds = [bounds_dict[key] for key in param_keys_dup]
        
        # Objective function
        input_tuples =  [tuple.(times_tile[i, :, :], inverse_mL_to_cubic_um.(densities_tile[i, :, :])) for i in 1:length(sources)]
        param_indices_per_src = [param_starts .+ ((i - 1) .% param_occurrences) for i in 1:length(sources)]
        dantigny_data_flat = [collect(dantigny_data[i, :, :]) for i in 1:length(sources)]
        error_weights = collect(LinRange(1, 10, length(times))) # increasing weights towards the end
        error_weights = repeat(error_weights, inner=[1, length(sources), length(densities)])
        error_weights = permutedims(error_weights, (2, 3, 1))
        obj = params -> begin
            err = 0
            @inbounds for i in eachindex(sources)
                params_select = view(params, param_indices_per_src[i])
                ŷ = [wrapper(inputs, params_select) for inputs in input_tuples[i]]
                err += sum(abs2, ŷ .- dantigny_data_flat[i]) #+ 10 * (sum(abs2, ŷ[end] .- dantigny_data_flat[i][end])) # weight last point
                # err += sum(abs2, (ŷ .- dantigny_data_flat[i]) .* error_weights[i]) # increasing weights towards the end
            end
            return err
        end
        objgrad = (params,_) -> begin
            err = 0
            @inbounds for i in eachindex(sources)
                params_select = view(params, param_indices_per_src[i])
                ŷ = [wrapper(inputs, params_select) for inputs in input_tuples[i]]
                err += sum(abs2, ŷ .- dantigny_data_flat[i]) #+ 10 * (sum(abs2, ŷ[end] .- dantigny_data_flat[i][end])) # weight last point
                # err += sum(abs2, (ŷ .- dantigny_data_flat[i]) .* error_weights[i]) # increasing weights towards the end
            end
            return err
        end

        # Pre-warm
        # dummy_params = [ bnd[1] for bnd in bounds ]
        # for _ in 1:10
        #     obj(dummy_params)
        # end
        
        # Fit model
        # if model_type_split[1] == "inducer"
        #     res = bboptimize(params -> obj(params);
        #                 SearchRange = bounds,
        #                 MaxSteps = max_steps,
        #                 PopulationSize = 10)
        # else
        println("Running first optimisation stage")
        res = bboptimize(params -> obj(params);
                    SearchRange = bounds,
                    MaxSteps = max_steps,
                    Method = :adaptive_de_rand_1_bin_radiuslimited)
                    # Method = :adaptive_de_rand_1_bin)
        p_opt = best_candidate(res)
        best_fit = best_fitness(res)

        println("Running second optimisation stage")
        opt = Opt(:LN_COBYLA, length(bounds))
        lower_bounds!(opt, [bnd[1] for bnd in bounds])
        upper_bounds!(opt, [bnd[2] for bnd in bounds])
        xtol_rel!(opt, 1e-4)
        maxeval!(opt, 2000)
        min_objective!(opt, objgrad)

        (best_fit, res, code) = NLopt.optimize(opt, p_opt)
        p_opt = res
        println("Final fitness: ", best_fit)
        

        # Compute rmse
        rmse = sqrt(best_fit / length(dantigny_data))

        # Create a dictionary for the optimized parameters
        params_out = Dict()
        for (i, key) in enumerate(param_keys)
            for j in 1:param_occurrences[i]
                # Transform parameters back to original scale
                key_split = split(string(key), "_")
                if key_split[1] == "δ"
                    key_new = Symbol(:σ_, key_split[2])
                    val = p_opt[param_starts[i] + j - 1 - param_occurrences[i]] * exp(p_opt[param_starts[i] + j - 1])
                elseif key == :k
                    println("Converting k to original scale")
                    key_new = key
                    val = exp(p_opt[param_starts[i] + j - 1])
                else
                    key_new = key
                    val = p_opt[param_starts[i] + j - 1]
                end
                if haskey(params_out, key_new)
                    push!(params_out[key_new], val)
                else
                    params_out[key_new] = [val]
                end
            end
        end

        return params_out, rmse
    end


    function get_params_for_idx(params, idx)
        """
        Get the parameters for a specific index.
        inputs:
            params (Dict): dictionary of parameters
            idx (Int): index to get the parameters for
        outputs:
            params_out (Dict): dictionary of parameters for the specified index
        """

        params_out = Dict()

        for (key, value) in params
            params_out[key] = value[mod1(idx, length(value))]
        end

        return params_out
    end
end