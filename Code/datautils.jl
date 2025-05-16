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
        @argcheck model_type in ["independent", "inhibitor", "inhibitor_thresh", "inhibitor_perm", "inducer", "inducer_thresh", "inducer_signal"]

        # println("Running ", model_type, " model fitting with ", st ? "time-dependent inducer" : "static inducer")

        # Reshape input
        densities_tile = repeat(densities, outer=[1, length(sources), length(times)])
        densities_tile = permutedims(densities_tile, (2, 1, 3))
        times_tile = repeat(times, outer=[1, length(sources), length(densities)])
        times_tile = permutedims(times_tile, (2, 3, 1))

        model_type_split = split(model_type, "_")

        # Gauss-Hermite nodes
        n_nodes = 100
        ghnodes, ghweights = gausshermite(n_nodes)

        n_src = length(sources)
        
        if model_type == "independent"
            # Independent inducer/inhibitor
            if st
                println("Model: independent factors with time-dependent inducer")
                wrapper = (inputs, params) -> germ_response_independent_factors_st_gh(
                    inputs[1], #t
                    inputs[2], #ρₛ
                    def_params[:c₀_cs],
                    def_params[:d_hp],
                    def_params[:μ_ξ],
                    def_params[:σ_ξ],
                    def_params[:μ_κ],
                    def_params[:σ_κ],
                    params[1], #Pₛ
                    params[2], #Pₛ_cs
                    params[3], #s_max
                    params[4], #K_cs
                    params[5], #μ_γ
                    params[6], #σ_γ
                    params[7], #μ_ω
                    params[8], #σ_ω
                    ghnodes, ghweights
                )
                param_keys = [:Pₛ, :Pₛ_cs, :s_max, :K_cs, :μ_γ, :σ_γ, :μ_ω, :σ_ω]
                param_occurrences = [1, n_src, n_src, n_src, 1, 1, 1, 1]
            else
                println("Model: independent factors with static inducer")
                wrapper = (inputs, params) -> Main.germ_response_independent_factors_gh(
                    inputs[1], #t
                    inputs[2], #ρₛ
                    def_params[:μ_ξ],
                    def_params[:σ_ξ],
                    params[1], #Pₛ
                    params[2], #s
                    params[3], #μ_γ
                    params[4], #σ_γ
                    params[5], #μ_ω
                    params[6], #σ_ω
                    ghnodes, ghweights
                )
                param_keys = [:Pₛ, :s, :μ_γ, :σ_γ, :μ_ω, :σ_ω]
                param_occurrences = [1, n_src, 1, 1, 1, 1]
            end

        elseif model_type_split[1] == "inhibitor"

            if model_type == "inhibitor" # Inducer shifts inhibition threshold and modulates inhibitor permeability
                if st # Time-dependent inducer
                    println("Model: inducer-modulated inhibitor (combined) with time-dependent inducer")
                    wrapper = (inputs, params) -> Main.germ_response_inducer_dep_inhibitor_combined_st_gh(
                        inputs[1], #t
                        inputs[2], #ρₛ
                        def_params[:c₀_cs],
                        def_params[:d_hp],
                        def_params[:μ_ξ],
                        def_params[:σ_ξ],
                        def_params[:μ_κ],
                        def_params[:σ_κ],
                        params[1], #Pₛ,
                        params[2], #Pₛ_cs,
                        params[3], #s_max,
                        params[4], #K_cs,
                        params[5], #k,
                        params[6], #μ_γ,
                        params[7], #σ_γ
                        ghnodes, ghweights
                    )
                    param_keys = [:Pₛ, :Pₛ_cs, :s_max, :K_cs, :k, :μ_γ, :σ_γ]
                    param_occurrences = [1, n_src, n_src, n_src, n_src, 1, 1]
                else
                    println("Model: inducer-modulated inhibitor (combined) with static inducer")
                    wrapper = (inputs, params) -> Main.germ_response_inhibitor_gh(
                        inputs[1], #t
                        inputs[2], #ρₛ
                        def_params[:μ_ξ],
                        def_params[:σ_ξ],
                        params[1], #Pₛ
                        params[2], #μ_γ
                        params[3], #σ_γ
                        ghnodes, ghweights
                    )
                    param_keys = [:Pₛ, :μ_γ, :σ_γ]
                    param_occurrences = [n_src, n_src, n_src]
                end

            elseif model_type_split[2] == "thresh" # Inducer shifts inhibition threshold
                if st # Time-dependent inducer
                    println("Model: inducer-modulated inhibitor (threshold) with time-dependent inducer")
                    wrapper = (inputs, params) -> Main.germ_response_inducer_dep_inhibitor_thresh_st_gh(
                        inputs[1], #t
                        inputs[2], #ρₛ
                        def_params[:c₀_cs],
                        def_params[:d_hp],
                        def_params[:μ_ξ],
                        def_params[:σ_ξ],
                        def_params[:μ_κ],
                        def_params[:σ_κ],
                        params[1], #Pₛ
                        params[2], #Pₛ_cs
                        params[3], #s_max
                        params[4], #K_cs
                        params[5], #μ_γ
                        params[6], #σ_γ
                        ghnodes, ghweights
                    )
                    param_keys = [:Pₛ, :Pₛ_cs, :s_max, :K_cs, :μ_γ, :σ_γ]
                    param_occurrences = [1, n_src, n_src, n_src, n_src, n_src]
                else
                    println("Model: inducer-modulated inhibitor (threshold) with static inducer")
                    wrapper = (inputs, params) -> Main.germ_response_inhibitor_gh(
                        inputs[1], #t
                        inputs[2], #ρₛ
                        def_params[:μ_ξ],
                        def_params[:σ_ξ],
                        params[1], #Pₛ
                        params[2], #μ_γ
                        params[3], #σ_γ
                        ghnodes, ghweights
                    )
                    param_keys = [:Pₛ, :μ_γ, :σ_γ]
                    param_occurrences = [1, n_src, n_src]
                end
            elseif model_type_split[2] == "perm" # Inducer modulates inhibitor permeability
                if st # Time-dependent inducer
                    println("Model: inducer-modulated inhibitor (permeability) with time-dependent inducer")
                    wrapper = (inputs, params) -> Main.germ_response_inducer_dep_inhibitor_perm_st_gh(
                        inputs[1], #t
                        inputs[2], #ρₛ
                        def_params[:c₀_cs],
                        def_params[:d_hp],
                        def_params[:μ_ξ],
                        def_params[:σ_ξ],
                        def_params[:μ_κ],
                        def_params[:σ_κ],
                        params[1], #Pₛ
                        params[2], #Pₛ_cs
                        params[3], #s_max
                        params[4], #K_cs
                        params[5], #μ_γ
                        params[6], #σ_γ
                        ghnodes, ghweights
                    )
                    param_keys = [:Pₛ, :Pₛ_cs, :s_max, :K_cs, :μ_γ, :σ_γ]
                    param_occurrences = [1, n_src, n_src, n_src, 1, 1]
                else
                    println("Model: inducer-modulated inhibitor (permeability) with static inducer")
                    wrapper = (inputs, params) -> Main.germ_response_inhibitor_gh(
                        inputs[1], #t
                        inputs[2], #ρₛ
                        def_params[:μ_ξ],
                        def_params[:σ_ξ],
                        params[1], #Pₛ
                        params[2], #μ_γ
                        params[3], #σ_γ
                        ghnodes, ghweights
                    )
                    param_keys = [:Pₛ, :μ_γ, :σ_γ]
                    param_occurrences = [n_src, 1, 1]
                end            
            end

        elseif model_type_split[1] == "inducer"

            if model_type == "inducer" # Inhibitor shifts induction threshold and modulates inducer signal strength
                if st
                    println("Model: inhibitor-modulated inducer (combined) with time-dependent inducer")
                    wrapper = (inputs, params) -> Main.germ_response_inhibitor_dep_inducer_combined_st_gh(
                        inputs[1], #t
                        inputs[2], #ρₛ
                        def_params[:c₀_cs],
                        def_params[:d_hp],
                        def_params[:μ_ξ],
                        def_params[:σ_ξ],
                        def_params[:μ_κ],
                        def_params[:σ_κ],
                        params[1], #Pₛ
                        params[2], #Pₛ_cs
                        params[3], #s_max
                        params[4], #k
                        params[5], #K_cs
                        params[6], #K_I
                        params[7], #n
                        params[8], #μ_ω
                        params[9], #σ_ω
                        params[10], #μ_ψ
                        params[11], #σ_ψ
                        ghnodes, ghweights
                    )
                    param_keys = [:Pₛ, :Pₛ_cs, :s_max, :k, :K_cs, :K_I, :n, :μ_ω, :σ_ω, :μ_ψ, :σ_ψ]
                    param_occurrences = [1, n_src, n_src, n_src, n_src, n_src, n_src, 1, 1, 1, 1]
                else
                    println("Model: inhibitor-modulated inducer (combined) with static inducer")
                    wrapper = (inputs, params) -> Main.germ_response_inhibitor_dep_inducer_combined_gh(
                        inputs[1], #t
                        inputs[2], #ρₛ
                        def_params[:μ_ξ],
                        def_params[:σ_ξ],
                        params[1], #Pₛ
                        params[2], #s
                        params[3], #k
                        params[4], #K_I
                        params[5], #n
                        params[6], #μ_ω
                        params[7], #σ_ω
                        params[8], #μ_ψ
                        params[9], #σ_ψ
                        ghnodes, ghweights
                    )
                    param_keys = [:Pₛ, :s, :k, :K_I, :n, :μ_ω, :σ_ω, :μ_ψ, :σ_ψ]
                    param_occurrences = [1, n_src, n_src, n_src, n_src, 1, 1, 1, 1]
                end

            elseif model_type_split[2] == "thresh" # Inhibitor shifts induction threshold
                if st
                    println("Model: inhibitor-modulated inducer (threshold) with time-dependent inducer")
                    wrapper = (inputs, params) -> Main.germ_response_inhibitor_dep_inducer_thresh_st_gh(
                        inputs[1], #t
                        inputs[2], #ρₛ
                        def_params[:c₀_cs],
                        def_params[:d_hp],
                        def_params[:μ_ξ],
                        def_params[:σ_ξ],
                        def_params[:μ_κ],
                        def_params[:σ_κ],
                        params[1], #Pₛ
                        params[2], #Pₛ_cs
                        params[3], #s_max
                        params[4], #K_cs
                        params[5], #k
                        params[6], #μ_ω
                        params[7], #σ_ω
                        params[8], #μ_ψ
                        params[9], #σ_ψ
                        ghnodes, ghweights
                    )
                    param_keys = [:Pₛ, :Pₛ_cs, :s_max, :K_cs, :k, :μ_ω, :σ_ω, :μ_ψ, :σ_ψ]
                    param_occurrences = [1, n_src, n_src, n_src, n_src, 1, 1, 1, 1]
                else
                    println("Model: inhibitor-modulated inducer (threshold) with static inducer")
                    wrapper = (inputs, params) -> Main.germ_response_inhibitor_dep_inducer_thresh_gh(
                        inputs[1], #t
                        inputs[2], #ρₛ
                        def_params[:μ_ξ],
                        def_params[:σ_ξ],
                        params[1], #Pₛ
                        params[2], #s
                        params[3], #k
                        params[4], #μ_ω
                        params[5], #σ_ω
                        params[6], #μ_ψ
                        params[7], #σ_ψ
                        ghnodes, ghweights
                    )
                    param_keys = [:Pₛ, :s, :k, :μ_ω, :σ_ω, :μ_ψ, :σ_ψ]
                    param_occurrences = [1, n_src, n_src, 1, 1, 1, 1]
                end

            elseif model_type_split[2] == "signal" # Inhibitor shifts induction threshold
                if st
                    println("Model: inhibitor-modulated inducer (signal) with time-dependent inducer")
                    wrapper = (inputs, params) -> Main.germ_response_inhibitor_dep_inducer_signal_st_gh(
                        inputs[1], #t
                        inputs[2], #ρₛ
                        def_params[:c₀_cs],
                        def_params[:d_hp],
                        def_params[:μ_ξ],
                        def_params[:σ_ξ],
                        def_params[:μ_κ],
                        def_params[:σ_κ],
                        params[1], #Pₛ
                        params[2], #Pₛ_cs
                        params[3], #s_max
                        params[4], #K_cs
                        params[5], #K_I
                        params[6], #n
                        params[7], #μ_ω
                        params[8], #σ_ω
                        params[9], #μ_ψ
                        params[10], #σ_ψ
                        ghnodes, ghweights
                    )
                    param_keys = [:Pₛ, :Pₛ_cs, :s_max, :K_cs, :K_I, :n, :μ_ω, :σ_ω, :μ_ψ, :σ_ψ]
                    param_occurrences = [1, n_src, n_src, n_src, n_src, n_src, 1, 1, 1, 1]
                else
                    println("Model: inhibitor-modulated inducer (signal) with static inducer")
                    wrapper = (inputs, params) -> Main.germ_response_inhibitor_dep_inducer_signal_gh(
                        inputs[1], #t
                        inputs[2], #ρₛ
                        def_params[:μ_ξ],
                        def_params[:σ_ξ],
                        params[1], #Pₛ
                        params[2], #s
                        params[3], #K_I
                        params[4], #n
                        params[5], #μ_ω
                        params[6], #σ_ω
                        params[7], #μ_ψ
                        params[8], #σ_ψ
                        ghnodes, ghweights
                    )
                    param_keys = [:Pₛ, :s, :K_I, :n, :μ_ω, :σ_ω, :μ_ψ, :σ_ψ]
                    param_occurrences = [1, n_src, n_src, n_src, 1, 1, 1, 1]
                end
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
        # input_tuples =  [tuple.(inverse_mL_to_cubic_um.(densities_tile[i, :, :]), times_tile[i, :, :]) for i in 1:length(sources)]
        input_tuples =  [tuple.(times_tile[i, :, :], inverse_mL_to_cubic_um.(densities_tile[i, :, :])) for i in 1:length(sources)]
        param_indices_per_src = [param_starts .+ ((i - 1) .% param_occurrences) for i in 1:length(sources)]
        dantigny_data_flat = [collect(dantigny_data[i, :, :]) for i in 1:length(sources)]
        obj = params -> begin
            err = 0
            @inbounds for i in eachindex(sources)
                params_select = view(params, param_indices_per_src[i])
                ŷ = [wrapper(inputs, params_select) for inputs in input_tuples[i]]
                err += sum(abs2, ŷ .- dantigny_data_flat[i])
            end
            return err
        end
        objgrad = (params,_) -> begin
            err = 0
            @inbounds for i in eachindex(sources)
                params_select = view(params, param_indices_per_src[i])
                ŷ = [wrapper(inputs, params_select) for inputs in input_tuples[i]]
                err += sum(abs2, ŷ .- dantigny_data_flat[i])
            end
            return err
        end
        
        
        # Fit model
        if model_type_split[1] == "inducer"
            res = bboptimize(params -> obj(params);
                        SearchRange = bounds,
                        MaxSteps = max_steps,
                        PopulationSize = 10)
        else
            println("Running first optimisation stage")
            res = bboptimize(params -> obj(params);
                        SearchRange = bounds,
                        MaxSteps = max_steps,
                        Method = :adaptive_de_rand_1_bin_radiuslimited)
                        # Method = :adaptive_de_rand_1_bin)
            p_opt_temp = best_candidate(res)

            println("Running second optimisation stage")
            # lower_bounds = [bnd[1] for bnd in bounds]
            # upper_bounds = [bnd[2] for bnd in bounds]
            opt = Opt(:LN_COBYLA, length(bounds))
            println(length(bounds))
            lower_bounds!(opt, [bnd[1] for bnd in bounds])
            upper_bounds!(opt, [bnd[2] for bnd in bounds])
            xtol_rel!(opt, 1e-4)
            maxeval!(opt, 1000)
            min_objective!(opt, objgrad)

            (best_fit, res, code) = NLopt.optimize(opt, p_opt_temp)
            println("Final fitness: ", best_fit)
            # res = bboptimize(params -> obj(params), p_opt_temp;
            #             SearchRange = bounds,
            #             MaxSteps = max_steps)
        end

        # p_opt = best_candidate(res)
        p_opt = res

        # Compute rmse
        # best_fit = best_fitness(res)
        rmse = sqrt(best_fit / length(dantigny_data))

        # Create a dictionary for the optimized parameters
        params_out = Dict()
        for (i, key) in enumerate(param_keys)
            params_out[key] = []
            for j in 1:param_occurrences[i]
                push!(params_out[key], p_opt[param_starts[i] + j - 1])
            end
        end

        return params_out, rmse
    end
end