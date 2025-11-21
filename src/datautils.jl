module DataUtils
    """
    Contains utility functions.
    """

    using DataFrames
    using CSV
    using FastGaussQuadrature
    using BlackBoxOptim
    using Optim
    using NLopt
    using LsqFit
    using MeshGrid
    using Distributions
    using ArgCheck
    using QuasiMonteCarlo
    
    include("./conversions.jl")
    include("./germstats.jl")
    using .Conversions
    using .GermStats

    export parse_ijadpanahsaravi_data
    export dantigny
    export generate_dantigny_dataset
    export fit_dantigny_to_germination_curve
    export fit_model_to_data
    export get_params_for_idx
    export fit_model_to_data_equilibrium
    

    function parse_ijadpanahsaravi_data()
        """
        Parses the dataset from Ijadpanahsaravi et al. (2023)
        with multiple incolulum densities and returns a DataFrame.
        """

        df_germination = DataFrame(CSV.File("../src/Data/swelling_germination_results.csv"; header=true))

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
            dantigny_data (Matrix): matrix of time-dependent germination data
            times (Vector): vector of time points
            sources (Vector): unique string identifiers of carbon sources
            densities (Vector): unique spore densities used in the data
            errs (Matrix): negative and positive offsets of p_max CIs from the mean
            p_maxs (Matrix): maximum germination percentages
            taus (Matrix): characteristic germination times
            nus (Matrix): design parameters
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


    function fit_dantigny_to_germination_curve(germ_response, times)#; check_stderr=false)
        """
        Fit Dantigny model to simulated germination curve
        from a mechanistic model.
        inputs:
            germ_response (Vector): germination fractions per time
            times (Vector): time frames for germination fractions
            check_stderr (Bool): whether to perform a standard error check
        outputs:
            p_max (Float): maximum germination percentage
            τ (Float): half-saturation time for germination
            ν (Float): design parameter
        """

        dantigny_wrapper(t, p) = dantigny.(t, 1 / (1 + exp(-p[1])), exp(p[2]), exp(p[3])) # [p_max, τ, ν] after transformations to become strictly positive

        # Initial guesses
        p0 = [0.5, times[Int(0.5 * length(times))], 2.0]

        # Transform to strictly positive scales
        p0[1] = log(p0[1]) - log1p(-p0[1])
        p0[2] = log(p0[2])
        p0[3] = log(p0[3])

        fit = curve_fit(dantigny_wrapper, times, germ_response, p0)
        params = coef(fit)
        rmse = sqrt(mean(residuals(fit) .^ 2))
        
        if rmse > 0.03 # average deviation > 3 percentage points in germination
            println("High RMSE: $rmse")
        end

        # Transform parameters
        params[1] = 1 / (1 + exp(-params[1]))
        params[2] = exp(params[2])
        params[3] = exp(params[3])

        return params
    end


    function fit_model_to_data(model_type, def_params, dantigny_data, times, sources, densities, bounds_dict; max_steps=10000, debug=false)
        """
        Fit a selected germination model to the data.
        inputs:
            model_type (String): model type to fit
            def_params (Dict): default parameter values
            dantigny_data (Matrix): time-dependent data from varying inducers and spore densities to fit
            times (Vector): time points
            sources (Vector): carbon sources
            densities (Vector): spore densities
            bounds_dict (Dict): bounds for the free parameters
            max_steps (int): maximum number of steps for the optimization
            debug (bool) - whether to print additional debugging messages
        outputs:
            params_out (Dict): optimized parameters
        """
        @argcheck model_type in load_model_collection()[1]

        # Reshape input
        densities_tile = repeat(densities, outer=[1, length(sources), length(times)])
        densities_tile = permutedims(densities_tile, (2, 1, 3))
        times_tile = repeat(times, outer=[1, length(sources), length(densities)])
        times_tile = permutedims(times_tile, (2, 3, 1))

        # Determine number of nodes depending on the integral dimension
        if model_type in ["independent", "inhibitor", "inhibitor_thresh", "inhibitor_perm", 
                            "combined_inhibitor", "combined_inhibitor_thresh", "combined_inhibitor_perm"]
            n_nodes = 36 # 2D integral
        elseif model_type in ["inducer", "inducer_thresh", "inducer_signal",
                            "combined_inducer", "combined_inducer_thresh", "combined_inducer_signal", "special_independent",
                            "inhibitor_thresh_inducer_signal", "combined_inhibitor_thresh_inducer_signal", "combined_inhibitor_thresh_inducer_thresh", "combined_inhibitor_thresh_signal_inducer_thresh"]
            n_nodes = 10 # 3D integral
        elseif model_type in ["special_inducer", "special_combined", "special_thresh", "special_signal"]
            n_nodes = 6 # 4D integral
        elseif startswith(model_type, "feedback")
            n_nodes = 512#1024
        end
        println("Number of nodes/samples: ", n_nodes)

        model_type_split = split(model_type, "_")

        gh_integral = false
        if model_type_split[1] == "feedback"
            if (haskey(bounds_dict, :μ_γ) && haskey(bounds_dict, :μ_ω))
                sample_dim = 5
            else
                sample_dim = 4
            end
            sobol_pts = QuasiMonteCarlo.sample(n_nodes, sample_dim, SobolSample())
        else
            gh_integral = true

            # Gauss-Hermite nodes
            ghnodes, ghweights = gausshermite(n_nodes)
            u = √2 .* ghnodes
            hw = ghweights ./ √π
        end

        # Unpack means and stds and weight samples
        μ_ξ = def_params[:μ_ξ]
        σ_ξ = def_params[:σ_ξ]
        μ_ξ_log = log(μ_ξ^2 / sqrt(σ_ξ^2 + μ_ξ^2))
        σ_ξ_log = sqrt(log(σ_ξ^2 / μ_ξ^2 + 1))
        if gh_integral ξ = exp.(μ_ξ_log .+ σ_ξ_log .* u) end

        if haskey(def_params, :μ_κ)
            μ_κ = def_params[:μ_κ]
            σ_κ = def_params[:σ_κ]
            μ_κ_log = log(μ_κ^2 / sqrt(σ_κ^2 + μ_κ^2))
            σ_κ_log = sqrt(log(σ_κ^2 / μ_κ^2 + 1))
            if gh_integral
                κ = exp.(μ_κ_log .+ σ_κ_log .* u)

                ξ2, κ2 = meshgrid(ξ, κ)
            end
        end

        # Multi-dimensional Gauss-Hermite weights
        if gh_integral
            W = hw * hw'
            W3 = reshape(hw, n_nodes,1,1) .* reshape(hw, 1,n_nodes,1) .* reshape(hw, 1,1,n_nodes)
            W4 = reshape(hw, n_nodes,1,1,1) .* reshape(hw, 1,n_nodes,1,1) .* reshape(hw, 1,1,n_nodes,1) .* reshape(hw, 1,1,1,n_nodes)
        end

        # Construct distributions and geometric samples
        if !gh_integral
            dist_ξ = LogNormal(μ_ξ_log, σ_ξ_log)
            dist_κ = LogNormal(μ_κ_log, σ_κ_log)

            samples_ξ = clamp_inplace!(quantile(dist_ξ, sobol_pts[1,:]))
            samples_κ = clamp_inplace!(quantile(dist_κ, sobol_pts[2,:]))

            samples_AV = compute_spore_area_and_volume_from_dia.(2 .* samples_ξ)
            samples_A, samples_Vₛ = (getindex.(samples_AV, 1), getindex.(samples_AV, 2))
            # samples_V_out = 1.0/ρₛ .- samples_Vₛ
            samples_V_ps = compute_ps_layer_volume.(samples_ξ, def_params[:d_hp], samples_κ)
        end

        # Define number of specific parameter occurrences (general or per carbon source)
        n_src = length(sources)
        param_occurrences_dict = Dict(
            :s_max => n_src,
            :b_max => 1,
            :Pₛ => 1,
            :Pₛ_cs => n_src,
            :k_C => n_src,
            :k_I => n_src,
            :K_I => n_src,
            :K_cC => n_src,
            :K_cI => n_src,
            :n => n_src,
            :μ_γ => 1,
            :δ_γ => 1,
            :μ_ω => n_src,
            :δ_ω => n_src,
            :μ_ψ => 1,	
            :δ_ψ => 1,
            :μ_α => 1,
            :δ_α => 1
        )
        
        if model_type == "independent"
            # Independent inducer/inhibitor
            println("Model: independent factors")
            wrapper = (inputs, params) -> germ_response_independent_factors_gh(
                u, W,
                inputs[1], #t
                inputs[2], #ρₛ
                def_params[:c₀_cs],
                def_params[:d_hp],
                ξ2,
                κ2,
                params[1], #Pₛ
                params[2], #Pₛ_cs
                params[3], #K_cC
                params[4], #μ_γ
                params[4] * exp(params[5]), # σ_γ = μ_γ * exp(δ_γ)
                params[6], #μ_ω
                params[6] * exp(params[7]) # σ_ω = μ_ω * exp(δ_ω)
            )
            param_keys = [:Pₛ, :Pₛ_cs, :K_cC, :μ_γ, :δ_γ, :μ_ω, :δ_ω]
            # param_occurrences = [1, n_src, n_src, 1, 1, n_src, n_src]

        # elseif model_type == "inhibitor" # Inducer shifts inhibition threshold and modulates inhibitor permeability
        #         println("Model: inducer-modulated inhibitor (combined)")
        #         wrapper = (inputs, params) -> Main.germ_response_inducer_dep_inhibitor_gh(
        #             u, W,
        #             inputs[1], #t
        #             inputs[2], #ρₛ
        #             def_params[:c₀_cs],
        #             def_params[:d_hp],
        #             ξ2,
        #             κ2,
        #             params[1], #Pₛ,
        #             params[2], #Pₛ_cs,
        #             params[3], #K_cC,
        #             exp(params[4]), #k_C,
        #             params[5], #μ_γ,
        #             params[5] * exp(params[6]), # σ_γ = μ_γ * exp(δ_γ)
        #         )
        #         param_keys = [:Pₛ, :Pₛ_cs, :K_cC, :k_C, :μ_γ, :δ_γ]
        #         param_occurrences = [1, n_src, n_src, n_src, 1, 1]

        elseif model_type == "inhibitor_thresh" # Inducer shifts inhibition threshold
            println("Model: inducer-modulated inhibitor (threshold)")
            wrapper = (inputs, params) -> Main.germ_response_inducer_dep_inhibitor_thresh_gh(
                u, W,
                inputs[1], #t
                inputs[2], #ρₛ
                def_params[:c₀_cs],
                def_params[:d_hp],
                ξ2,
                κ2,
                params[1], #Pₛ
                params[2], #Pₛ_cs
                exp(params[3]), #k_C
                params[4], #K_cC
                params[5], #μ_γ
                params[5] * exp(params[6]) # σ_γ = μ_γ * exp(δ_γ)
            )
            param_keys = [:Pₛ, :Pₛ_cs, :k_C, :K_cC, :μ_γ, :δ_γ]
            # param_occurrences = [1, n_src, n_src, n_src, 1, 1]
            
            # elseif model_type_split[2] == "perm" # Inducer modulates inhibitor permeability
            #     println("Model: inducer-modulated inhibitor (permeability)")
            #     wrapper = (inputs, params) -> Main.germ_response_inducer_dep_inhibitor_perm_gh(
            #         u, W,
            #         inputs[1], #t
            #         inputs[2], #ρₛ
            #         def_params[:c₀_cs],
            #         def_params[:d_hp],
            #         ξ2,
            #         κ2,
            #         params[1], #Pₛ
            #         params[2], #Pₛ_cs
            #         params[3], #K_cC
            #         params[4], #μ_γ
            #         params[4] * exp(params[5]) # σ_γ = μ_γ * exp(δ_γ)
            #     )
            #     param_keys = [:Pₛ, :Pₛ_cs, :K_cC, :μ_γ, :δ_γ]
            #     param_occurrences = [1, n_src, n_src, 1, 1]
                
            # end

        elseif model_type == "inducer" # Inhibitor shifts induction threshold and modulates inducer signal strength
            println("Model: inhibitor-modulated inducer (combined)")
            wrapper = (inputs, params) -> Main.germ_response_inhibitor_dep_inducer_gh(
                u, W3,
                inputs[1], #t
                inputs[2], #ρₛ
                def_params[:c₀_cs],
                def_params[:d_hp],
                ξ2,
                κ2,
                params[1], #Pₛ
                params[2], #Pₛ_cs
                exp(params[3]), #k_I
                params[4], #K_cI
                params[5], #K_cC
                params[6], #K_I
                params[7], #n
                params[8], #μ_ω
                params[8] * exp(params[9]), # σ_ω = μ_ω * exp(δ_ω)
                params[10], #μ_ψ
                params[10] * exp(params[11]) # σ_ψ = μ_ψ * exp(δ_ψ)
            )
            param_keys = [:Pₛ, :Pₛ_cs, :k_I, :K_cI, :K_cC, :K_I, :n, :μ_ω, :δ_ω, :μ_ψ, :δ_ψ]
            # param_occurrences = [1, n_src, n_src, n_src, n_src, n_src, n_src, n_src, 1, 1]
                
        elseif model_type == "inducer_thresh" # Inhibitor shifts induction threshold
            println("Model: inhibitor-modulated inducer (threshold)")
            wrapper = (inputs, params) -> Main.germ_response_inhibitor_dep_inducer_thresh_gh(
                u, W3,
                inputs[1], #t
                inputs[2], #ρₛ
                def_params[:c₀_cs],
                def_params[:d_hp],
                ξ2,
                κ2,
                params[1], #Pₛ
                params[2], #Pₛ_cs
                params[3], #K_cI
                params[4], #K_cC
                exp(params[5]), #k_I
                params[6], #μ_ω
                params[6] * exp(params[7]), # σ_ω = μ_ω * exp(δ_ω)
                params[8], #μ_ψ
                params[8] * exp(params[9]) # σ_ψ = μ_ψ * exp(δ_ψ)
            )
            param_keys = [:Pₛ, :Pₛ_cs, :K_cI, :K_cC, :k_I, :μ_ω, :δ_ω, :μ_ψ, :δ_ψ]
            # param_occurrences = [1, n_src, n_src, n_src, n_src, n_src, 1, 1]
            
        elseif model_type == "inducer_signal" # Inhibitor shifts induction threshold
            println("Model: inhibitor-modulated inducer (signal)")
            wrapper = (inputs, params) -> Main.germ_response_inhibitor_dep_inducer_signal_gh(
                u, W3,
                inputs[1], #t
                inputs[2], #ρₛ
                def_params[:c₀_cs],
                def_params[:d_hp],
                ξ2,
                κ2,
                params[1], #Pₛ
                params[2], #Pₛ_cs
                params[3], #K_cC
                params[4], #K_I
                params[5], #n
                params[6], #μ_ω
                params[6] * exp(params[7]), # σ_ω = μ_ω * exp(δ_ω)
                params[8], #μ_ψ
                params[8] * exp(params[9]) # σ_ψ = μ_ψ * exp(δ_ψ)
            )
            param_keys = [:Pₛ, :Pₛ_cs, :K_cC, :K_I, :n, :μ_ω, :δ_ω, :μ_ψ, :δ_ψ]
            # param_occurrences = [1, n_src, n_src, n_src, n_src, n_src, n_src, 1, 1]

        elseif model_type == "combined_inhibitor"
            println("Model: inducer-modulated inhibitor (combined)")
            wrapper = (inputs, params) -> Main.germ_response_inducer_dep_inhibitor_2_factor_gh(
                u, W,
                inputs[1], #t
                inputs[2], #ρₛ
                def_params[:c₀_cs],
                def_params[:d_hp],
                ξ2,
                κ2,
                params[1], #Pₛ,
                params[2], #Pₛ_cs,
                params[3], #K_cC,
                exp(params[4]), #k_C,
                params[5], #μ_γ,
                params[5] * exp(params[6]), # σ_γ = μ_γ * exp(δ_γ)
                params[7], #μ_ω,
                params[7] * exp(params[8]) # σ_ω = μ_ω * exp(δ_ω)
            )
            param_keys = [:Pₛ, :Pₛ_cs, :K_cC, :k_C, :μ_γ, :δ_γ, :μ_ω, :δ_ω]
            # param_occurrences = [1, n_src, n_src, n_src, 1, 1, n_src, n_src]

        elseif model_type == "combined_inhibitor_thresh"
            println("Model: inducer-modulated inhibitor (combined)")
            wrapper = (inputs, params) -> Main.germ_response_inducer_dep_inhibitor_thresh_2_factor_gh(
                u, W,
                inputs[1], #t
                inputs[2], #ρₛ
                def_params[:c₀_cs],
                def_params[:d_hp],
                ξ2,
                κ2,
                params[1], #Pₛ
                params[2], #Pₛ_cs
                exp(params[3]), #k_C
                params[4], #K_cC
                params[5], #μ_γ
                params[5] * exp(params[6]), # σ_γ = μ_γ * exp(δ_γ)
                params[7], #μ_ω
                params[7] * exp(params[8]) # σ_ω = μ_ω * exp(δ_ω)
            )
            param_keys = [:Pₛ, :Pₛ_cs, :k_C, :K_cC, :μ_γ, :δ_γ, :μ_ω, :δ_ω]
            # param_occurrences = [1, n_src, n_src, n_src, 1, 1, n_src, n_src]

            # elseif model_type_split[3] == "perm"
            #     println("Model: inducer-modulated inhibitor (release)")
            #     wrapper = (inputs, params) -> Main.germ_response_inducer_dep_inhibitor_perm_2_factor_gh(
            #         u, W,
            #         inputs[1], #t
            #         inputs[2], #ρₛ
            #         def_params[:c₀_cs],
            #         def_params[:d_hp],
            #         ξ2,
            #         κ2,
            #         params[1], #Pₛ
            #         params[2], #Pₛ_cs
            #         params[3], #K_cC
            #         params[4], #μ_γ
            #         params[4] * exp(params[5]), # σ_γ = μ_γ * exp(δ_γ)
            #         params[6], #μ_ω
            #         params[6] * exp(params[7]) # σ_ω = μ_ω * exp(δ_ω)
            #     )
            #     param_keys = [:Pₛ, :Pₛ_cs, :K_cC, :μ_γ, :δ_γ, :μ_ω, :δ_ω]
            #     param_occurrences = [1, n_src, n_src, 1, 1, n_src, n_src]
            # end

        elseif model_type == "combined_inducer"
            println("Model: inhibitor-modulated inducer (combined)")
            wrapper = (inputs, params) -> Main.germ_response_inhibitor_dep_inducer_2_factor_gh(
                u, W3,
                inputs[1], #t
                inputs[2], #ρₛ
                def_params[:c₀_cs],
                def_params[:d_hp],
                ξ2,
                κ2,
                params[1], #Pₛ
                params[2], #Pₛ_cs
                params[3], #K_cI
                params[4], #K_cC
                params[5], #K_I
                params[6], #n
                exp(params[7]), #k_I
                params[8], #μ_γ
                params[8] * exp(params[9]), # σ_γ = μ_γ * exp(δ_γ)
                params[10], #μ_ω
                params[10] * exp(params[11]), # σ_ω = μ_ω * exp(δ_ω)
                params[12], #μ_ψ
                params[12] * exp(params[13]) # σ_ψ = μ_ψ * exp(δ_ψ)
            )
            param_keys = [:Pₛ, :Pₛ_cs, :K_cI, :K_cC, :K_I, :n, :k_I, :μ_γ, :δ_γ, :μ_ω, :δ_ω, :μ_ψ, :δ_ψ]
            # param_occurrences = [1, n_src, n_src, n_src, n_src, n_src, 1, 1, n_src, n_src, 1, 1]

        elseif model_type == "combined_inducer_thresh"
            println("Model: inhibitor-modulated inducer (threshold)")
            wrapper = (inputs, params) -> Main.germ_response_inhibitor_dep_inducer_thresh_2_factor_gh(
                u, W3,
                inputs[1], #t
                inputs[2], #ρₛ
                def_params[:c₀_cs],
                def_params[:d_hp],
                ξ2,
                κ2,
                params[1], #Pₛ
                params[2], #Pₛ_cs
                params[3], #K_cI
                params[4], #K_cC
                exp(params[5]), #k_I
                params[6], #μ_γ
                params[6] * exp(params[7]), # σ_γ = μ_γ * exp(δ_γ)
                params[8], #μ_ω
                params[8] * exp(params[9]), # σ_ω = μ_ω * exp(δ_ω)
                params[10], #μ_ψ
                params[10] * exp(params[11]) # σ_ψ = μ_ψ * exp(δ_ψ)
            )
            param_keys = [:Pₛ, :Pₛ_cs, :K_cI, :K_cC, :k_I, :μ_γ, :δ_γ, :μ_ω, :δ_ω, :μ_ψ, :δ_ψ]
            # param_occurrences = [1, n_src, n_src, n_src, 1, 1, n_src, n_src, 1, 1]

        elseif model_type == "combined_inducer_signal"
            println("Model: inhibitor-modulated inducer (signal)")
            wrapper = (inputs, params) -> Main.germ_response_inhibitor_dep_inducer_signal_2_factor_gh(
                u, W3,
                inputs[1], #t
                inputs[2], #ρₛ
                def_params[:c₀_cs],
                def_params[:d_hp],
                ξ2,
                κ2,
                params[1], #Pₛ
                params[2], #Pₛ_cs
                params[3], #K_cC
                params[4], #K_I
                params[5], #n
                params[6], #μ_γ
                params[6] * exp(params[7]), # σ_γ = μ_γ * exp(δ_γ)
                params[8], #μ_ω
                params[8] * exp(params[9]), # σ_ω = μ_ω * exp(δ_ω)
                params[10], #μ_ψ
                params[10] * exp(params[11]) # σ_ψ = μ_ψ * exp(δ_ψ)
            )
            param_keys = [:Pₛ, :Pₛ_cs, :K_cC, :K_I, :n, :μ_γ, :δ_γ, :μ_ω, :δ_ω, :μ_ψ, :δ_ψ]
            # param_occurrences = [1, n_src, n_src, n_src, n_src, 1, 1, n_src, n_src, 1, 1]

        elseif model_type == "special_inducer"
            println("Model: inhibitor-modulated inducer (combined) with varying permeability")
            wrapper = (inputs, params) -> Main.germ_response_inducer_var_perm_gh(
                u, W4,
                inputs[1], #t
                inputs[2], #ρₛ
                def_params[:c₀_cs],
                def_params[:d_hp],
                ξ2,
                κ2,
                params[1], #Pₛ
                params[2], #Pₛ_cs
                params[3], #K_cC
                params[4], #K_I
                exp(params[5]), #k_I
                params[6], #n
                params[7], #μ_ω
                params[7] * exp(params[8]), # σ_ω = μ_ω * exp(δ_ω)
                params[9], #μ_ψ
                params[9] * exp(params[10]), # σ_ψ = μ_ψ * exp(δ_ψ)
                params[11], #μ_α
                params[11] * exp(params[12]) # σ_α = μ_α * exp(δ_α)
            )
            param_keys = [:Pₛ, :Pₛ_cs, :K_cC, :K_I, :k_I, :n, :μ_ω, :δ_ω, :μ_ψ, :δ_ψ, :μ_α, :δ_α]
            # param_occurrences = [1, n_src, n_src, n_src, n_src, n_src, n_src, n_src, 1, 1, 1, 1]

        elseif model_type == "special_independent"
            println("Model: independent factors with varying permeability")
            wrapper = (inputs, params) -> Main.germ_response_independent_factors_var_perm_gh(
                u, W3,
                inputs[1], #t
                inputs[2], #ρₛ
                def_params[:c₀_cs],
                def_params[:d_hp],
                ξ2,
                κ2,
                params[1], #Pₛ
                params[2], #Pₛ_cs
                params[3], #K_cC
                params[4], #μ_γ
                params[4] * exp(params[5]), # σ_γ = μ_γ * exp(δ_γ)
                params[6], #μ_ω
                params[6] * exp(params[7]), # σ_ω = μ_ω * exp(δ_ω)
                params[8], #μ_α
                params[8] * exp(params[9]) # σ_α = μ_α * exp(δ_α)
            )
            param_keys = [:Pₛ, :Pₛ_cs, :K_cC, :μ_γ, :δ_γ, :μ_ω, :δ_ω, :μ_α, :δ_α]
            # param_occurrences = [1, n_src, n_src, 1, 1, n_src, n_src, 1, 1]

        elseif model_type == "special_combined"
            println("Model: 2-factor germination with inhibitor-modulated inducer (combined) and varying permeability")
            wrapper = (inputs, params) -> Main.germ_response_inducer_2_factors_var_perm_gh(
                u, W4,
                inputs[1], #t
                inputs[2], #ρₛ
                def_params[:c₀_cs],
                def_params[:d_hp],
                ξ2,
                κ2,
                params[1], #Pₛ
                params[2], #Pₛ_cs
                params[3], #K_cC
                params[4], #K_I
                exp(params[5]), #k_I
                params[6], #n
                params[7], #μ_γ
                params[7] * exp(params[8]), # σ_γ = μ_γ * exp(δ_γ)
                params[9], #μ_ω
                params[9] * exp(params[10]), # σ_ω = μ_ω * exp(δ_ω)
                params[11], #μ_ψ
                params[11] * exp(params[12]), # σ_ψ = μ_ψ * exp(δ_ψ)
                params[13], #μ_α
                params[13] * exp(params[14]) # σ_α = μ_α * exp(δ_α)
            )
            param_keys = [:Pₛ, :Pₛ_cs, :K_cC, :K_I, :k_I, :n, :μ_γ, :δ_γ, :μ_ω, :δ_ω, :μ_ψ, :δ_ψ, :μ_α, :δ_α]
            # param_occurrences = [1, n_src, n_src, n_src, n_src, n_src, 1, 1, n_src, n_src, 1, 1, 1, 1]

        elseif model_type == "special_combined_thresh"
            println("Model: 2-factor germination with inhibitor-modulated inducer (threshold) and varying permeability")
            wrapper = (inputs, params) -> Main.germ_response_inducer_thresh_2_factors_var_perm_gh(
                u, W4,
                inputs[1], #t
                inputs[2], #ρₛ
                def_params[:c₀_cs],
                def_params[:d_hp],
                ξ2,
                κ2,
                params[1], #Pₛ
                params[2], #Pₛ_cs
                params[3], #K_cC
                exp(params[4]), #k_I
                params[5], #μ_γ
                params[5] * exp(params[6]), # σ_γ = μ_γ * exp(δ_γ)
                params[7], #μ_ω
                params[7] * exp(params[8]), # σ_ω = μ_ω * exp(δ_ω)
                params[9], #μ_ψ
                params[9] * exp(params[10]), # σ_ψ = μ_ψ * exp(δ_ψ)
                params[11], #μ_α
                params[11] * exp(params[12]) # σ_α = μ_α * exp(δ_α)
            )
            param_keys = [:Pₛ, :Pₛ_cs, :K_cC, :k_I, :μ_γ, :δ_γ, :μ_ω, :δ_ω, :μ_ψ, :δ_ψ, :μ_α, :δ_α]
            # param_occurrences = [1, n_src, n_src, n_src, 1, 1, n_src, n_src, 1, 1, 1, 1]

        elseif model_type == "special_combined_signal"
            println("Model: 2-factor germination with inhibitor-modulated inducer (signal) and varying permeability")
            wrapper = (inputs, params) -> Main.germ_response_inducer_signal_2_factors_var_perm_gh(
                u, W4,
                inputs[1], #t
                inputs[2], #ρₛ
                def_params[:c₀_cs],
                def_params[:d_hp],
                ξ2,
                κ2,
                params[1], #Pₛ
                params[2], #Pₛ_cs
                params[3], #K_cC
                params[4], #K_I
                params[5], #n
                params[6], #μ_γ
                params[6] * exp(params[7]), # σ_γ = μ_γ * exp(δ_γ)
                params[8], #μ_ω
                params[8] * exp(params[9]), # σ_ω = μ_ω * exp(δ_ω)
                params[10], #μ_ψ
                params[10] * exp(params[11]), # σ_ψ = μ_ψ * exp(δ_ψ)
                params[12], #μ_α
                params[12] * exp(params[13]) # σ_α = μ_α * exp(δ_α)
            )
            param_keys = [:Pₛ, :Pₛ_cs, :K_cC, :K_I, :n, :μ_γ, :δ_γ, :μ_ω, :δ_ω, :μ_ψ, :δ_ψ, :μ_α, :δ_α]
            # param_occurrences = [1, n_src, n_src, n_src, n_src, 1, 1, n_src, n_src, 1, 1, 1, 1]
        
            # ==================== FEEDBACK MODELS ========================
        elseif model_type == "feedback_inhibitor_inducer_perm" # A
            println("Model: inhibitor-dependent germination with inducer-dependent inhibitor/inducer permeability")
            wrapper = (V_out, params) -> Main.germ_response_feedback(
                Main.ode_inducer_dependent_perm!,
                Main.thresh_criterion_inhibitor,
                sobol_pts,
                times,
                [samples_A, samples_Vₛ, V_out, samples_V_ps],
                def_params[:c₀_cs],
                [params[1]], # s_max
                params[2], # Pₛ_I
                params[3], # Pₛ_C
                [nothing, params[4]], # K_cC
                params[5], # μ_ψ
                params[5] * exp(params[6]), # σ_ψ = μ_ψ * exp(δ_ψ)
                [params[7]], # μ_γ
                [params[7] * exp(params[8])] # σ_γ = μ_γ * exp(δ_γ)
            )
            param_keys = [:s_max, :Pₛ, :Pₛ_cs, :K_cC, :μ_ψ, :δ_ψ, :μ_γ, :δ_γ]
            # param_occurrences = [n_src, 1, n_src, n_src, 1, 1, 1, 1]

        elseif model_type == "feedback_combined_inducer_perm" # A
            println("Model: 2-factor germination with inducer-dependent inhibitor/inducer permeability")
            wrapper = (V_out, params) -> Main.germ_response_feedback(
                Main.ode_inducer_dependent_perm!,
                Main.thresh_criterion_combined,
                sobol_pts,
                times,
                [samples_A, samples_Vₛ, V_out, samples_V_ps],
                def_params[:c₀_cs],
                [params[1]], # s_max
                params[2], # Pₛ_I
                params[3], # Pₛ_C
                [nothing, params[4]], # K_cC
                params[5], # μ_ψ
                params[5] * exp(params[6]), # σ_ψ = μ_ψ * exp(δ_ψ)
                [params[7], params[9]], # [μ_γ, μ_ω]
                [params[7] * exp(params[8]), params[9] * exp(params[10])] # [σ_γ = μ_γ * exp(δ_γ), σ_ω = μ_ω * exp(δ_ω)]
            )
            param_keys = [:s_max, :Pₛ, :Pₛ_cs, :K_cC, :μ_ψ, :δ_ψ, :μ_γ, :δ_γ, :μ_ω, :δ_ω]
            # param_occurrences = [n_src, 1, n_src, n_src, 1, 1, 1, 1, n_src, n_src]
        
        elseif model_type == "feedback_inducer_inhibitor_perm" # D
            println("Model: Inducer-dependent germination with inhibitor-dependent inhibitor/inducer permeability")
            wrapper = (V_out, params) -> Main.germ_response_feedback(
                Main.ode_inhibitor_dependent_perm!,
                Main.thresh_criterion_inducer,
                sobol_pts,
                times,
                [samples_A, samples_Vₛ, V_out, samples_V_ps],
                def_params[:c₀_cs],
                [params[1]], # b_max
                params[2], # Pₛ_I
                params[3], # Pₛ_C
                [params[4], nothing], # K_cI
                params[5], # μ_ψ
                params[5] * exp(params[6]), # σ_ψ = μ_ψ * exp(δ_ψ)
                [params[7]], # [μ_ω]
                [params[7] * exp(params[8])] # [σ_ω = μ_ω * exp(δ_ω)]
            )
            param_keys = [:b_max, :Pₛ, :Pₛ_cs, :K_cI, :μ_ψ, :δ_ψ, :μ_ω, :δ_ω]
            # param_occurrences = [1, 1, n_src, 1, 1, 1, n_src, n_src]
        
        elseif model_type == "feedback_combined_inhibitor_perm" # D
            println("Model: 2-factor germination with inhibitor-dependent inhibitor/inducer permeability")
            wrapper = (V_out, params) -> Main.germ_response_feedback(
                Main.ode_inhibitor_dependent_perm!,
                Main.thresh_criterion_combined,
                sobol_pts,
                times,
                [samples_A, samples_Vₛ, V_out, samples_V_ps],
                def_params[:c₀_cs],
                [params[1]], # b_max
                params[2], # Pₛ_I
                params[3], # Pₛ_C
                [params[4], nothing], # K_cI
                params[5], # μ_ψ
                params[5] * exp(params[6]), # σ_ψ = μ_ψ * exp(δ_ψ)
                [params[7], params[9]], # [μ_γ, μ_ω]
                [params[7] * exp(params[8]), params[9] * exp(params[10])] # [σ_ω = μ_ω * exp(δ_ω)]
            )
            param_keys = [:b_max, :Pₛ, :Pₛ_cs, :K_cI, :μ_ψ, :δ_ψ, :μ_γ, :δ_γ, :μ_ω, :δ_ω]
            # param_occurrences = [1, 1, n_src, 1, 1, 1, 1, 1, n_src, n_src]
        
        elseif model_type == "feedback_inhibitor_inducer_perm_thresh" # AB
            println("Model: Inhibitor-dependent germination with inducer-dependent permeability and inhibition threshold")
            wrapper = (V_out, params) -> Main.germ_response_feedback(
                Main.ode_inducer_dependent_perm!,
                Main.thresh_criterion_inhibitor_shift,
                sobol_pts,
                times,
                [samples_A, samples_Vₛ, V_out, samples_V_ps],
                def_params[:c₀_cs],
                [params[1]], # s_max
                params[2], # Pₛ_I
                params[3], # Pₛ_C
                [nothing, params[4]], # K_cC
                params[5], # μ_ψ
                params[5] * exp(params[6]), # σ_ψ = μ_ψ * exp(δ_ψ)
                [params[7]], # μ_γ
                [params[7] * exp(params[8])]; # σ_γ = μ_γ * exp(δ_γ)
                ks=[exp(params[9])] # k_C
            )
            param_keys = [:s_max, :Pₛ, :Pₛ_cs, :K_cC, :μ_ψ, :δ_ψ, :μ_γ, :δ_γ, :k_C]
            # param_occurrences = [n_src, 1, n_src, n_src, 1, 1, 1, 1, n_src]

        elseif model_type == "feedback_combined_inducer_perm_thresh" # AB
            println("Model: 2-factor germination with inducer-dependent permeability and inhibition threshold")
            wrapper = (V_out, params) -> Main.germ_response_feedback(
                Main.ode_inducer_dependent_perm!,
                Main.thresh_criterion_combined_inhibitor_shift,
                sobol_pts,
                times,
                [samples_A, samples_Vₛ, V_out, samples_V_ps],
                def_params[:c₀_cs],
                [params[1]], # s_max
                params[2], # Pₛ_I
                params[3], # Pₛ_C
                [nothing, params[4]], # K_cC
                params[5], # μ_ψ
                params[5] * exp(params[6]), # σ_ψ = μ_ψ * exp(δ_ψ)
                [params[7], params[9]], # [μ_γ, μ_ω]
                [params[7] * exp(params[8]), params[9] * exp(params[10])]; # [σ_γ = μ_γ * exp(δ_γ), σ_ω = μ_ω * exp(δ_ω)]
                ks=[exp(params[11])] # k_C
            )
            param_keys = [:s_max, :Pₛ, :Pₛ_cs, :K_cC, :μ_ψ, :δ_ψ, :μ_γ, :δ_γ, :μ_ω, :δ_ω, :k_C]
            # param_occurrences = [n_src, 1, n_src, n_src, 1, 1, 1, 1, n_src, n_src, n_src]
        
        elseif model_type == "feedback_inhibitor_inducer_perm_inhibitor_signal" # AC
            println("Model: Inhibitor-dependent germination with inducer-dependent permeability and inhibitor-dependent induction signal")
            wrapper = (V_out, params) -> Main.germ_response_feedback(
                Main.ode_inducer_dependent_perm_inhibitor_dependent_signal!,
                Main.thresh_criterion_inhibitor,
                sobol_pts,
                times,
                [samples_A, samples_Vₛ, V_out, samples_V_ps],
                def_params[:c₀_cs],
                [params[1]], # s_max
                params[2], # Pₛ_I
                params[3], # Pₛ_C
                [nothing, params[4], params[5]], # K_cC, K_I
                params[6], # μ_ψ
                params[6] * exp(params[7]), # σ_ψ = μ_ψ * exp(δ_ψ)
                [params[8]], # μ_γ
                [params[8] * exp(params[9])]; # σ_γ = μ_γ * exp(δ_γ)
                n=params[10] # n
            )
            param_keys = [:s_max, :Pₛ, :Pₛ_cs, :K_cC, :K_I, :μ_ψ, :δ_ψ, :μ_γ, :δ_γ, :n]
            # param_occurrences = [n_src, 1, n_src, 1, n_src, 1, 1, 1, 1, n_src]

        elseif model_type == "feedback_inducer_inducer_perm_inhibitor_signal" # AC
            println("Model: Inducer-dependent germination with inducer-dependent permeability and inhibitor-dependent induction signal")
            wrapper = (V_out, params) -> Main.germ_response_feedback(
                Main.ode_inducer_dependent_perm_inhibitor_dependent_signal!,
                Main.thresh_criterion_inducer,
                sobol_pts,
                times,
                [samples_A, samples_Vₛ, V_out, samples_V_ps],
                def_params[:c₀_cs],
                [params[1]], # s_max
                params[2], # Pₛ_I
                params[3], # Pₛ_C
                [nothing, params[4], params[5]], # K_cC, K_I
                params[6], # μ_ψ
                params[6] * exp(params[7]), # σ_ψ = μ_ψ * exp(δ_ψ)
                [params[8]], # μ_ω
                [params[8] * exp(params[9])]; # σ_ω = μ_ω * exp(δ_ω)
                n=params[10] # n
            )
            param_keys = [:s_max, :Pₛ, :Pₛ_cs, :K_cC, :K_I, :μ_ψ, :δ_ψ, :μ_ω, :δ_ω, :n]
            # param_occurrences = [n_src, 1, n_src, 1, n_src, 1, 1, n_src, n_src, n_src]
            
        elseif model_type == "feedback_combined_inducer_perm_inhibitor_signal" # AC
            println("Model: 2-factor germination with inducer-dependent permeability and inhibitor-dependent induction signal")
            wrapper = (V_out, params) -> Main.germ_response_feedback(
                Main.ode_inducer_dependent_perm_inhibitor_dependent_signal!,
                Main.thresh_criterion_combined,
                sobol_pts,
                times,
                [samples_A, samples_Vₛ, V_out, samples_V_ps],
                def_params[:c₀_cs],
                [params[1]], # s_max
                params[2], # Pₛ_I
                params[3], # Pₛ_C
                [nothing, params[4], params[5]], # K_cC, K_I
                params[6], # μ_ψ
                params[6] * exp(params[7]), # σ_ψ = μ_ψ * exp(δ_ψ)
                [params[8], params[10]], # [μ_γ, μ_ω]
                [params[8] * exp(params[9]), params[10] * exp(params[11])]; # [σ_γ = μ_γ * exp(δ_γ), σ_ω = μ_ω * exp(δ_ω)]
                n=params[12] # n
            )
            param_keys = [:s_max, :Pₛ, :Pₛ_cs, :K_cC, :K_I, :μ_ψ, :δ_ψ, :μ_γ, :δ_γ, :μ_ω, :δ_ω, :n]
            # param_occurrences = [n_src, 1, n_src, 1, n_src, 1, 1, 1, 1, n_src, n_src, n_src]
            
        elseif model_type == "feedback_inhibitor_inhibitor_inducer_perm" # AD
            println("Model: Inhibitor-dependent germination with inhibitor- and inducer-dependent inhibitor/inducer permeability")
            wrapper = (V_out, params) -> Main.germ_response_feedback(
                Main.ode_inducer_and_inhibitor_dependent_perm!,
                Main.thresh_criterion_inhibitor,
                sobol_pts,
                times,
                [samples_A, samples_Vₛ, V_out, samples_V_ps],
                def_params[:c₀_cs],
                [params[1], params[2]], # b_max, s_max
                params[3], # Pₛ_I
                params[4], # Pₛ_C
                [params[5], params[6]], # K_cI, K_cC
                params[7], # μ_ψ
                params[7] * exp(params[8]), # σ_ψ = μ_ψ * exp(δ_ψ)
                [params[9]], # μ_γ
                [params[9] * exp(params[10])] # σ_γ = μ_γ * exp(δ_γ)
            )
            param_keys = [:b_max, :s_max, :Pₛ, :Pₛ_cs, :K_cI, :K_cC, :μ_ψ, :δ_ψ, :μ_γ, :δ_γ]
            # param_occurrences = [1, n_src, 1, n_src, 1, n_src, 1, 1, 1, 1]
        
        elseif model_type == "feedback_inducer_inhibitor_inducer_perm" # AD
            println("Model: Inducer-dependent germination with inhibitor- and inducer-dependent inhibitor/inducer permeability")
            wrapper = (V_out, params) -> Main.germ_response_feedback(
                Main.ode_inducer_and_inhibitor_dependent_perm!,
                Main.thresh_criterion_inducer,
                sobol_pts,
                times,
                [samples_A, samples_Vₛ, V_out, samples_V_ps],
                def_params[:c₀_cs],
                [params[1], params[2]], # b_max, s_max
                params[3], # Pₛ_I
                params[4], # Pₛ_C
                [params[5], params[6]], # K_cI, K_cC
                params[7], # μ_ψ
                params[7] * exp(params[8]), # σ_ψ = μ_ψ * exp(δ_ψ)
                [params[9]], # μ_ω
                [params[9] * exp(params[10])] # σ_ω = μ_ω * exp(δ_ω)
            )
            param_keys = [:b_max, :s_max, :Pₛ, :Pₛ_cs, :K_cI, :K_cC, :μ_ψ, :δ_ψ, :μ_ω, :δ_ω]
            # param_occurrences = [1, n_src, 1, n_src, 1, n_src, 1, 1, n_src, n_src]
        
        elseif model_type == "feedback_combined_inhibitor_inducer_perm" # AD
            println("Model: 2-factor germination with inhibitor- and inducer-dependent inhibitor/inducer permeability")
            wrapper = (V_out, params) -> Main.germ_response_feedback(
                Main.ode_inducer_and_inhibitor_dependent_perm!,
                Main.thresh_criterion_combined,
                sobol_pts,
                times,
                [samples_A, samples_Vₛ, V_out, samples_V_ps],
                def_params[:c₀_cs],
                [params[1], params[2]], # b_max, s_max
                params[3], # Pₛ_I
                params[4], # Pₛ_C
                [params[5], params[6]], # K_cI, K_cC
                params[7], # μ_ψ
                params[7] * exp(params[8]), # σ_ψ = μ_ψ * exp(δ_ψ)
                [params[9], params[11]], # μ_γ, μ_ω
                [params[9] * exp(params[10]), params[11] * exp(params[12])] # σ_γ = μ_γ * exp(δ_γ), σ_ω = μ_ω * exp(δ_ω)
            )
            param_keys = [:b_max, :s_max, :Pₛ, :Pₛ_cs, :K_cI, :K_cC, :μ_ψ, :δ_ψ, :μ_γ, :δ_γ, :μ_ω, :δ_ω]
            # param_occurrences = [1, n_src, 1, n_src, 1, n_src, 1, 1, 1, 1, n_src, n_src]
        
        elseif model_type == "feedback_inducer_inhibitor_thresh_inducer_perm" # AE
            println("Model: Inducer-dependent germination with inhibitor-dependent induction threshold and inducer-dependent inhibitor/inducer permeability")
            wrapper = (V_out, params) -> Main.germ_response_feedback(
                Main.ode_inducer_dependent_perm!,
                Main.thresh_criterion_inducer_shift,
                sobol_pts,
                times,
                [samples_A, samples_Vₛ, V_out, samples_V_ps],
                def_params[:c₀_cs],
                [params[1]], # s_max
                params[2], # Pₛ_I
                params[3], # Pₛ_C
                [params[4], params[5]], # K_cI, K_cC
                params[6], # μ_ψ
                params[6] * exp(params[7]), # σ_ψ = μ_ψ * exp(δ_ψ)
                [params[8]], # μ_ω
                [params[8] * exp(params[9])]; # σ_ω = μ_ω * exp(δ_ω)
                ks=[exp(params[10])] # k_I
            )
            param_keys = [:s_max, :Pₛ, :Pₛ_cs, :K_cI, :K_cC, :μ_ψ, :δ_ψ, :μ_ω, :δ_ω, :k_I]
            # param_occurrences = [n_src, 1, n_src, 1, n_src, 1, 1, n_src, n_src, n_src]

        elseif model_type == "feedback_combined_inhibitor_thresh_inducer_perm" # AE
            println("Model: 2-factor germination with inhibitor-dependent induction threshold and inducer-dependent inhibitor/inducer permeability")
            wrapper = (V_out, params) -> Main.germ_response_feedback(
                Main.ode_inducer_dependent_perm!,
                Main.thresh_criterion_combined_inducer_shift,
                sobol_pts,
                times,
                [samples_A, samples_Vₛ, V_out, samples_V_ps],
                def_params[:c₀_cs],
                [params[1]], # s_max
                params[2], # Pₛ_I
                params[3], # Pₛ_C
                [params[4], params[5]], # K_cI, K_cC
                params[6], # μ_ψ
                params[6] * exp(params[7]), # σ_ψ = μ_ψ * exp(δ_ψ)
                [params[8], params[10]], # μ_ω
                [params[8] * exp(params[9]), params[10] * exp(params[11])]; # σ_ω = μ_ω * exp(δ_ω)
                ks=[exp(params[12])] # k_I
            )
            param_keys = [:s_max, :Pₛ, :Pₛ_cs, :K_cI, :K_cC, :μ_ψ, :δ_ψ, :μ_γ, :δ_γ, :μ_ω, :δ_ω, :k_I]
            # param_occurrences = [n_src, 1, n_src, 1, n_src, 1, 1, 1, 1, n_src, n_src, n_src]
            
        elseif model_type == "inhibitor_thresh_inducer_signal" # BC
            println("Model: Inhibitor-dependent germination with inducer-dependent inhibition threshold and inhibitor-dependent induction signal")
            wrapper = (inputs, params) -> Main.germ_response_inh_dep_ind_signal_ind_dep_inh_thresh_gh(
                u, W3,
                inputs[1], #t
                inputs[2], #ρₛ
                def_params[:c₀_cs],
                def_params[:d_hp],
                ξ2,
                κ2,
                params[1], #Pₛ
                params[2], #Pₛ_cs
                exp(params[3]), #k_C
                params[4], #K_cC
                params[5], #K_I
                params[6], #n
                params[7], #μ_γ
                params[7] * exp(params[8]), #σ_γ
                params[9], #μ_ψ
                params[9] * exp(params[10]) #σ_ψ
            )
            param_keys = [:Pₛ, :Pₛ_cs, :k_C, :K_cC, :K_I, :n, :μ_γ, :δ_γ, :μ_ψ, :δ_ψ]
            # param_occurrences = [1, n_src, n_src, n_src, 1, n_src, 1, 1, 1, 1]

        elseif model_type == "combined_inhibitor_thresh_inducer_signal" # BC
            println("Model: 2-factor germination with inducer-dependent inhibition threshold and inhibitor-dependent induction signal")
            wrapper = (inputs, params) -> Main.germ_response_inh_dep_ind_signal_ind_dep_inh_thresh_2_factor_gh(
                u, W3,
                inputs[1], #t
                inputs[2], #ρₛ
                def_params[:c₀_cs],
                def_params[:d_hp],
                ξ2,
                κ2,
                params[1], #Pₛ
                params[2], #Pₛ_cs
                exp(params[3]), #k_C
                params[4], #K_cC
                params[5], #K_I
                params[6], #n
                params[7], #μ_γ
                params[7] * exp(params[8]), #σ_γ
                params[9], #μ_ω
                params[9] * exp(params[10]), #σ_ω
                params[11], #μ_ψ
                params[11] * exp(params[12]) #σ_ψ
            )
            param_keys = [:Pₛ, :Pₛ_cs, :k_C, :K_cC, :K_I, :n, :μ_γ, :δ_γ, :μ_ω, :δ_ω, :μ_ψ, :δ_ψ]
            # param_occurrences = [1, n_src, n_src, n_src, 1, n_src, 1, 1, n_src, n_src, 1, 1]
            
        elseif model_type == "feedback_inhibitor_inducer_thresh_inhibitor_perm" # BD
            println("Model: Inhibitor-dependent germination with inducer-dependent inhibition threshold and inhibitor-dependent inhibitor/inducer permeability")
            wrapper = (V_out, params) -> Main.germ_response_feedback(
                Main.ode_inhibitor_dependent_perm!,
                Main.thresh_criterion_inhibitor_shift,
                sobol_pts,
                times,
                [samples_A, samples_Vₛ, V_out, samples_V_ps],
                def_params[:c₀_cs],
                [params[1]], # b_max
                params[2], # Pₛ_I
                params[3], # Pₛ_C
                [params[4], params[5]], # K_cI, K_cC
                params[6], # μ_ψ
                params[6] * exp(params[7]), # σ_ψ = μ_ψ * exp(δ_ψ)
                [params[8]], # [μ_γ]
                [params[8] * exp(params[9])], # [σ_γ = μ_γ * exp(δ_γ)]
                ks=[exp(params[10])] # k_C
            )
            param_keys = [:b_max, :Pₛ, :Pₛ_cs, :K_cI, :K_cC, :μ_ψ, :δ_ψ, :μ_γ, :δ_γ, :k_C]
            # param_occurrences = [1, 1, n_src, 1, n_src, 1, 1, 1, 1, n_src]
        
        elseif model_type == "feedback_combined_inducer_thresh_inhibitor_perm" # BD
            println("Model: Inhibitor-dependent germination with inducer-dependent inhibition threshold and inhibitor-dependent inhibitor/inducer permeability")
            wrapper = (V_out, params) -> Main.germ_response_feedback(
                Main.ode_inhibitor_dependent_perm!,
                Main.thresh_criterion_combined_inhibitor_shift,
                sobol_pts,
                times,
                [samples_A, samples_Vₛ, V_out, samples_V_ps],
                def_params[:c₀_cs],
                [params[1]], # b_max
                params[2], # Pₛ_I
                params[3], # Pₛ_C
                [params[4], params[5]], # K_cI, K_cC
                params[6], # μ_ψ
                params[6] * exp(params[7]), # σ_ψ = μ_ψ * exp(δ_ψ)
                [params[8], params[10]], # [μ_γ, μ_ω]
                [params[8] * exp(params[9]), params[10] * exp(params[11])], # [σ_γ = μ_γ * exp(δ_γ), σ_ω = μ_ω * exp(δ_ω)]
                ks=[exp(params[12])] # k_C
            )
            param_keys = [:b_max, :Pₛ, :Pₛ_cs, :K_cI, :K_cC, :μ_ψ, :δ_ψ, :μ_γ, :δ_γ, :μ_ω, :δ_ω, :k_C]
            # param_occurrences = [1, 1, n_src, 1, n_src, 1, 1, 1, 1, n_src, n_src, n_src]
        
        elseif model_type == "combined_inhibitor_thresh_inducer_thresh" # BE
            println("Model: inhibitor-modulated induction threshold / inducer-modulated inhibition threshold")
            wrapper = (inputs, params) -> Main.germ_response_inh_dep_ind_thresh_ind_dep_inh_thresh_2_factor_gh(
                u, W3,
                inputs[1], #t
                inputs[2], #ρₛ
                def_params[:c₀_cs],
                def_params[:d_hp],
                ξ2,
                κ2,
                params[1], #Pₛ
                params[2], #Pₛ_cs
                params[3], #K_cI
                params[4], #K_cC
                exp(params[5]), #k_I
                exp(params[6]), #k_C
                params[7], #μ_γ
                params[7] * exp(params[8]), # σ_γ = μ_γ * exp(δ_γ)
                params[9], #μ_ω
                params[9] * exp(params[10]), # σ_ω = μ_ω * exp(δ_ω)
                params[11], #μ_ψ
                params[11] * exp(params[12]) # σ_ψ = μ_ψ * exp(δ_ψ)
            )
            param_keys = [:Pₛ, :Pₛ_cs, :K_cI, :K_cC, :k_I, :k_C, :μ_γ, :δ_γ, :μ_ω, :δ_ω, :μ_ψ, :δ_ψ]
            # param_occurrences = [1, n_src, n_src, n_src, n_src, 1, 1, n_src, n_src, 1, 1]

        elseif model_type == "feedback_inducer_inhibitor_perm_signal" # CD
            println("Model: Inhibitor-dependent germination with inducer-dependent inhibition threshold and inhibitor-dependent inhibitor/inducer permeability")
            wrapper = (V_out, params) -> Main.germ_response_feedback(
                Main.ode_inhibitor_dependent_perm!,
                Main.thresh_criterion_inducer_signal,
                sobol_pts,
                times,
                [samples_A, samples_Vₛ, V_out, samples_V_ps],
                def_params[:c₀_cs],
                [params[1]], # b_max
                params[2], # Pₛ_I
                params[3], # Pₛ_C
                [params[4], params[5], params[6]], # K_cI, K_cC, K_I
                params[7], # μ_ψ
                params[7] * exp(params[8]), # σ_ψ = μ_ψ * exp(δ_ψ)
                [params[9]], # [μ_ω]
                [params[9] * exp(params[10])] # [σ_ω = μ_ω * exp(δ_ω)]
            )
            param_keys = [:b_max, :Pₛ, :Pₛ_cs, :K_cI, :K_cC, :K_I, :μ_ψ, :δ_ψ, :μ_ω, :δ_ω]
            # param_occurrences = [1, 1, n_src, n_src, 1, 1, 1, n_src, n_src]
        
        elseif model_type == "feedback_combined_inhibitor_perm_signal" # CD
            println("Model: 2-factor germination with inducer-dependent inhibition threshold and inhibitor-dependent inhibitor/inducer permeability")
            wrapper = (V_out, params) -> Main.germ_response_feedback(
                Main.ode_inhibitor_dependent_perm!,
                Main.thresh_criterion_combined_inducer_signal,
                sobol_pts,
                times,
                [samples_A, samples_Vₛ, V_out, samples_V_ps],
                def_params[:c₀_cs],
                [params[1]], # b_max
                params[2], # Pₛ_I
                params[3], # Pₛ_C
                [params[4], params[5], params[6]], # K_cI, K_cC, K_I
                params[7], # μ_ψ
                params[7] * exp(params[8]), # σ_ψ = μ_ψ * exp(δ_ψ)
                [params[9], params[11]], # [μ_γ, μ_ω]
                [params[9] * exp(params[10]), params[11] * exp(params[12])] # [σ_γ = μ_γ * exp(δ_γ), σ_ω = μ_ω * exp(δ_ω)]
            )
            param_keys = [:b_max, :Pₛ, :Pₛ_cs, :K_cI, :K_cC, :K_I, :μ_ψ, :δ_ψ, :μ_γ, :δ_γ, :μ_ω, :δ_ω]
            # param_occurrences = [1, 1, n_src, n_src, 1, 1, 1, 1, 1, n_src, n_src]
        
        elseif model_type == "feedback_inducer_inhibitor_perm_thresh" # DE
            println("Model: Inhibitor-dependent germination with inhibitor-dependent inhibitor/inducer permeability and induction threshold")
            wrapper = (V_out, params) -> Main.germ_response_feedback(
                Main.ode_inhibitor_dependent_perm!,
                Main.thresh_criterion_inducer_shift,
                sobol_pts,
                times,
                [samples_A, samples_Vₛ, V_out, samples_V_ps],
                def_params[:c₀_cs],
                [params[1]], # b_max
                params[2], # Pₛ_I
                params[3], # Pₛ_C
                [params[4], params[5]], # K_cI, K_cC
                params[6], # μ_ψ
                params[6] * exp(params[7]), # σ_ψ = μ_ψ * exp(δ_ψ)
                [params[8]], # [μ_ω]
                [params[8] * exp(params[9])], # [σ_ω = μ_ω * exp(δ_ω)]
                ks=[exp(params[10])] # k_I
            )
            param_keys = [:b_max, :Pₛ, :Pₛ_cs, :K_cI, :K_cC, :μ_ψ, :δ_ψ, :μ_ω, :δ_ω, :k_I]
            # param_occurrences = [1, 1, n_src, 1, n_src, 1, 1, n_src, n_src, n_src]
        
        elseif model_type == "feedback_combined_inhibitor_perm_thresh" # DE
            println("Model: Inhibitor-dependent germination with inhibitor-dependent inhibitor/inducer permeability and induction threshold")
            wrapper = (V_out, params) -> Main.germ_response_feedback(
                Main.ode_inhibitor_dependent_perm!,
                Main.thresh_criterion_combined_inducer_shift,
                sobol_pts,
                times,
                [samples_A, samples_Vₛ, V_out, samples_V_ps],
                def_params[:c₀_cs],
                [params[1]], # b_max
                params[2], # Pₛ_I
                params[3], # Pₛ_C
                [params[4], params[5]], # K_cI, K_cC
                params[6], # μ_ψ
                params[6] * exp(params[7]), # σ_ψ = μ_ψ * exp(δ_ψ)
                [params[8], params[10]], # [μ_γ, μ_ω]
                [params[8] * exp(params[9]), params[10] * exp(params[11])], # [σ_γ = μ_γ * exp(δ_γ), σ_ω = μ_ω * exp(δ_ω)]
                ks=[exp(params[12])] # k_I
            )
            param_keys = [:b_max, :Pₛ, :Pₛ_cs, :K_cI, :K_cC, :μ_ψ, :δ_ψ, :μ_γ, :δ_γ, :μ_ω, :δ_ω, :k_I]
            # param_occurrences = [1, 1, n_src, 1, n_src, 1, 1, 1, 1, n_src, n_src, n_src]
        
        elseif model_type == "feedback_inhibitor_inducer_perm_thresh_inhibitor_signal" # ABC
            println("Model: Inducer-dependent germination with inducer-dependent permeability and inhibitor-dependent induction signal")
            wrapper = (V_out, params) -> Main.germ_response_feedback(
                Main.ode_inducer_dependent_perm_inhibitor_dependent_signal!,
                Main.thresh_criterion_inhibitor_shift,
                sobol_pts,
                times,
                [samples_A, samples_Vₛ, V_out, samples_V_ps],
                def_params[:c₀_cs],
                [params[1]], # s_max
                params[2], # Pₛ_I
                params[3], # Pₛ_C
                [nothing, params[4], params[5]], # K_cC, K_I
                params[6], # μ_ψ
                params[6] * exp(params[7]), # σ_ψ = μ_ψ * exp(δ_ψ)
                [params[8]], # μ_γ
                [params[8] * exp(params[9])]; # σ_γ = μ_γ * exp(δ_γ)
                ks=[exp(params[10])], # k_C
                n=params[11] # n
            )
            param_keys = [:s_max, :Pₛ, :Pₛ_cs, :K_cC, :K_I, :μ_ψ, :δ_ψ, :μ_γ, :δ_γ, :k_C, :n]
            # param_occurrences = [n_src, 1, n_src, 1, n_src, 1, 1, n_src, 1, 1, n_src]
            
        elseif model_type == "feedback_combined_inducer_perm_thresh_inhibitor_signal" # ABC
            println("Model: 2-factor germination with inducer-dependent permeability and inhibitor-dependent induction signal")
            wrapper = (V_out, params) -> Main.germ_response_feedback(
                Main.ode_inducer_dependent_perm_inhibitor_dependent_signal!,
                Main.thresh_criterion_combined_inhibitor_shift,
                sobol_pts,
                times,
                [samples_A, samples_Vₛ, V_out, samples_V_ps],
                def_params[:c₀_cs],
                [params[1]], # s_max
                params[2], # Pₛ_I
                params[3], # Pₛ_C
                [nothing, params[4], params[5]], # K_cC, K_I
                params[6], # μ_ψ
                params[6] * exp(params[7]), # σ_ψ = μ_ψ * exp(δ_ψ)
                [params[8], params[11]], # [μ_γ, μ_ω]
                [params[8] * exp(params[9]), params[10] * exp(params[11])]; # [σ_γ = μ_γ * exp(δ_γ), σ_ω = μ_ω * exp(δ_ω)]
                ks=[exp(params[12])], # k_C
                n=params[13] # n
            )
            param_keys = [:s_max, :Pₛ, :Pₛ_cs, :K_cC, :K_I, :μ_ψ, :δ_ψ, :μ_γ, :δ_γ, :μ_ω, :δ_ω, :k_C, :n]
            # param_occurrences = [n_src, 1, n_src, 1, n_src, 1, 1, 1, 1, n_src, n_src, n_src, n_src]
            
        elseif model_type == "feedback_inhibitor_inducer_perm_thresh_inhibitor_perm" # ABD
            println("Model: Inhibitor-dependent germination with inducer-dependent inhibition threshold and inhibitor- and inducer-dependent inhibitor/inducer permeability")
            wrapper = (V_out, params) -> Main.germ_response_feedback(
                Main.ode_inducer_and_inhibitor_dependent_perm!,
                Main.thresh_criterion_inhibitor_shift,
                sobol_pts,
                times,
                [samples_A, samples_Vₛ, V_out, samples_V_ps],
                def_params[:c₀_cs],
                [params[1], params[2]], # b_max, s_max
                params[3], # Pₛ_I
                params[4], # Pₛ_C
                [params[5], params[6]], # K_cI, K_cC
                params[7], # μ_ψ
                params[7] * exp(params[8]), # σ_ψ = μ_ψ * exp(δ_ψ)
                [params[9]], # μ_γ
                [params[9] * exp(params[10])]; # σ_γ = μ_γ * exp(δ_γ)
                ks=[exp(params[11])] # k_C
            )
            param_keys = [:b_max, :s_max, :Pₛ, :Pₛ_cs, :K_cI, :K_cC, :μ_ψ, :δ_ψ, :μ_γ, :δ_γ, :k_C]
            # param_occurrences = [1, n_src, 1, n_src, 1, n_src, 1, 1, 1, 1, n_src]
        
        elseif model_type == "feedback_combined_inducer_perm_thresh_inhibitor_perm" # ABD
            println("Model: 2-factor germination with inducer-dependent inhibition threshold and inhibitor- and inducer-dependent inhibitor/inducer permeability")
            wrapper = (V_out, params) -> Main.germ_response_feedback(
                Main.ode_inducer_and_inhibitor_dependent_perm!,
                Main.thresh_criterion_combined_inhibitor_shift,
                sobol_pts,
                times,
                [samples_A, samples_Vₛ, V_out, samples_V_ps],
                def_params[:c₀_cs],
                [params[1], params[2]], # b_max, s_max
                params[3], # Pₛ_I
                params[4], # Pₛ_C
                [params[5], params[6]], # K_cI, K_cC
                params[7], # μ_ψ
                params[7] * exp(params[8]), # σ_ψ = μ_ψ * exp(δ_ψ)
                [params[9], params[11]], # μ_γ, μ_ω
                [params[9] * exp(params[10]), params[11] * exp(params[12])]; # σ_γ = μ_γ * exp(δ_γ), σ_ω = μ_ω * exp(δ_ω)
                ks=[exp(params[13])] # k_C
            )
            param_keys = [:b_max, :s_max, :Pₛ, :Pₛ_cs, :K_cI, :K_cC, :μ_ψ, :δ_ψ, :μ_γ, :δ_γ, :μ_ω, :δ_ω, :k_C]
            # param_occurrences = [1, n_src, 1, n_src, 1, n_src, 1, 1, 1, 1, n_src, n_src, n_src]
        
        elseif model_type == "feedback_combined_inducer_perm_thresh_inhibitor_thresh" # ABE
            println("Model: 2-factor germination with inducer-dependent inhibitor/inducer permeability and inhibition threshold and inhibitor-dependent induction threshold")
            wrapper = (V_out, params) -> Main.germ_response_feedback(
                Main.ode_inducer_dependent_perm!,
                Main.thresh_criterion_combined_shift,
                sobol_pts,
                times,
                [samples_A, samples_Vₛ, V_out, samples_V_ps],
                def_params[:c₀_cs],
                [params[1]], # s_max
                params[2], # Pₛ_I
                params[3], # Pₛ_C
                [params[4], params[5]], # :K_cI, K_cC
                params[6], # μ_ψ
                params[6] * exp(params[7]), # σ_ψ = μ_ψ * exp(δ_ψ)
                [params[8], params[10]], # [μ_γ, μ_ω]
                [params[8] * exp(params[9]), params[10] * exp(params[11])]; # [σ_γ = μ_γ * exp(δ_γ), σ_ω = μ_ω * exp(δ_ω)]
                ks=[exp(params[12]), exp(params[13])] # k_I, k_C
            )
            param_keys = [:s_max, :Pₛ, :Pₛ_cs, :K_cI, :K_cC, :μ_ψ, :δ_ψ, :μ_γ, :δ_γ, :μ_ω, :δ_ω, :k_I, :k_C]
            # param_occurrences = [n_src, 1, n_src, 1, n_src, 1, 1, 1, 1, n_src, n_src, n_src, n_src]
        
        elseif model_type == "feedback_inhibitor_inhibitor_inducer_perm_inhibitor_signal" # ACD
            println("Model: Inhibitor-dependent germination with inducer/inhibitor-dependent inhibitor/inducer permeability and inhibitor-dependent induction signal")
            wrapper = (V_out, params) -> Main.germ_response_feedback(
                Main.ode_inducer_and_inhibitor_dependent_perm_inhibitor_dependent_signal!,
                Main.thresh_criterion_inhibitor,
                sobol_pts,
                times,
                [samples_A, samples_Vₛ, V_out, samples_V_ps],
                def_params[:c₀_cs],
                [params[1], params[2]], # b_max, s_max
                params[3], # Pₛ_I
                params[4], # Pₛ_C
                [params[5], params[6], params[7]], # K_cI, K_cC, K_I
                params[8], # μ_ψ
                params[8] * exp(params[9]), # σ_ψ = μ_ψ * exp(δ_ψ)
                [params[10]], # μ_γ
                [params[10] * exp(params[11])]; # σ_γ = μ_γ * exp(δ_γ)
                n=params[12] # n
            )
            param_keys = [:b_max, :s_max, :Pₛ, :Pₛ_cs, :K_cI, :K_cC, :K_I, :μ_ψ, :δ_ψ, :μ_γ, :δ_γ, :n]
            # param_occurrences = [1, n_src, 1, n_src, 1, n_src, n_src, 1, 1, 1, 1, n_src]
        
        elseif model_type == "feedback_inducer_inhibitor_inducer_perm_inhibitor_signal" # ACD
            println("Model: Inhibitor-dependent germination with inducer/inhibitor-dependent inhibitor/inducer permeability and inhibitor-dependent induction signal")
            wrapper = (V_out, params) -> Main.germ_response_feedback(
                Main.ode_inducer_and_inhibitor_dependent_perm_inhibitor_dependent_signal!,
                Main.thresh_criterion_inducer,
                sobol_pts,
                times,
                [samples_A, samples_Vₛ, V_out, samples_V_ps],
                def_params[:c₀_cs],
                [params[1], params[2]], # b_max, s_max
                params[3], # Pₛ_I
                params[4], # Pₛ_C
                [params[5], params[6], params[7]], # K_cI, K_cC, K_I
                params[8], # μ_ψ
                params[8] * exp(params[9]), # σ_ψ = μ_ψ * exp(δ_ψ)
                [params[10]], # μ_ω
                [params[10] * exp(params[11])]; # σ_ω = μ_ω * exp(δ_ω)
                n=params[12] # n
            )
            param_keys = [:b_max, :s_max, :Pₛ, :Pₛ_cs, :K_cI, :K_cC, :K_I, :μ_ψ, :δ_ψ, :μ_ω, :δ_ω, :n]
            # param_occurrences = [1, n_src, 1, n_src, 1, n_src, 1, 1, n_src, n_src, n_src]
        
        elseif model_type == "feedback_combined_inhibitor_inducer_perm_inhibitor_signal" # ACD
            println("Model: 2-factor germination with inducer/inhibitor-dependent inhibitor/inducer permeability and inhibitor-dependent induction signal")
            wrapper = (V_out, params) -> Main.germ_response_feedback(
                Main.ode_inducer_and_inhibitor_dependent_perm_inhibitor_dependent_signal!,
                Main.thresh_criterion_combined,
                sobol_pts,
                times,
                [samples_A, samples_Vₛ, V_out, samples_V_ps],
                def_params[:c₀_cs],
                [params[1], params[2]], # b_max, s_max
                params[3], # Pₛ_I
                params[4], # Pₛ_C
                [params[5], params[6], params[7]], # K_cI, K_cC, K_I
                params[8], # μ_ψ
                params[8] * exp(params[9]), # σ_ψ = μ_ψ * exp(δ_ψ)
                [params[10], params[12]], # μ_γ, μ_ω
                [params[10] * exp(params[11]), params[12] * exp(params[13])]; # σ_γ = μ_γ * exp(δ_γ), σ_ω = μ_ω * exp(δ_ω)
                n=params[14] # n
            )
            param_keys = [:b_max, :s_max, :Pₛ, :Pₛ_cs, :K_cI, :K_cC, :K_I, :μ_ψ, :δ_ψ, :μ_γ, :δ_γ, :μ_ω, :δ_ω, :n]
            # param_occurrences = [1, n_src, 1, n_src, 1, n_src, 1, 1, 1, 1, n_src, n_src, n_src]
        
        elseif model_type == "feedback_inducer_inducer_perm_inhibitor_thresh_signal" # ACE
            println("Model: Inducer-dependent germination with inducer-dependent permeability and inhibitor-dependent induction threshold and signal")
            wrapper = (V_out, params) -> Main.germ_response_feedback(
                Main.ode_inducer_dependent_perm_inhibitor_dependent_signal!,
                Main.thresh_criterion_inducer_signal_shift,
                sobol_pts,
                times,
                [samples_A, samples_Vₛ, V_out, samples_V_ps],
                def_params[:c₀_cs],
                [params[1]], # s_max
                params[2], # Pₛ_I
                params[3], # Pₛ_C
                [params[4], params[5], params[6]], # K_cI, K_cC, K_I
                params[7], # μ_ψ
                params[7] * exp(params[8]), # σ_ψ = μ_ψ * exp(δ_ψ)
                [params[9]], # μ_ω
                [params[9] * exp(params[10])]; # σ_ω = μ_ω * exp(δ_ω)
                ks=[exp(params[11])], # k_I
                n=params[12] # n
            )
            param_keys = [:s_max, :Pₛ, :Pₛ_cs, :K_cI, :K_cC, :K_I, :μ_ψ, :δ_ψ, :μ_ω, :δ_ω, :k_I, :n]
            # param_occurrences = [n_src, 1, n_src, 1, n_src, 1, 1, n_src, n_src, n_src, n_src]
            
        elseif model_type == "feedback_combined_inducer_perm_inhibitor_thresh_signal" # ACE
            println("Model: 2-factor germination with inducer-dependent permeability and inhibitor-dependent induction threshold and signal")
            wrapper = (V_out, params) -> Main.germ_response_feedback(
                Main.ode_inducer_dependent_perm_inhibitor_dependent_signal!,
                Main.thresh_criterion_combined_inducer_signal_shift,
                sobol_pts,
                times,
                [samples_A, samples_Vₛ, V_out, samples_V_ps],
                def_params[:c₀_cs],
                [params[1]], # s_max
                params[2], # Pₛ_I
                params[3], # Pₛ_C
                [params[4], params[5], params[6]], # K_cI, K_cC, K_I
                params[7], # μ_ψ
                params[7] * exp(params[8]), # σ_ψ = μ_ψ * exp(δ_ψ)
                [params[9], params[11]], # [μ_γ, μ_ω]
                [params[9] * exp(params[10]), params[11] * exp(params[12])]; # [σ_γ = μ_γ * exp(δ_γ), σ_ω = μ_ω * exp(δ_ω)]
                ks=[exp(params[13])], # k_I
                n=params[14] # n
            )
            param_keys = [:s_max, :Pₛ, :Pₛ_cs, :K_cI, :K_cC, :K_I, :μ_ψ, :δ_ψ, :μ_γ, :δ_γ, :μ_ω, :δ_ω, :k_I, :n]
            # param_occurrences = [n_src, 1, n_src, 1, n_src, 1, 1, 1, 1, n_src, n_src, n_src, n_src]
            
        elseif model_type == "feedback_inducer_inhibitor_inducer_perm_inhibitor_thresh" # ADE
            println("Model: Inhibitor-dependent germination with inhibitor- and inducer-dependent inhibitor/inducer permeability, inhibitor-dependent induction threshold")
            wrapper = (V_out, params) -> Main.germ_response_feedback(
                Main.ode_inducer_and_inhibitor_dependent_perm!,
                Main.thresh_criterion_inducer_shift,
                sobol_pts,
                times,
                [samples_A, samples_Vₛ, V_out, samples_V_ps],
                def_params[:c₀_cs],
                [params[1], params[2]], # b_max, s_max
                params[3], # Pₛ_I
                params[4], # Pₛ_C
                [params[5], params[6]], # K_cI, K_cC
                params[7], # μ_ψ
                params[7] * exp(params[8]), # σ_ψ = μ_ψ * exp(δ_ψ)
                [params[9]], # μ_γ
                [params[9] * exp(params[10])]; # σ_γ = μ_γ * exp(δ_γ)
                ks=[exp(params[11])] # k_I
            )
            param_keys = [:b_max, :s_max, :Pₛ, :Pₛ_cs, :K_cI, :K_cC, :μ_ψ, :δ_ψ, :μ_γ, :δ_γ, :k_I]
            # param_occurrences = [1, n_src, 1, n_src, 1, n_src, 1, 1, 1, 1, n_src]
        
        elseif model_type == "feedback_combined_inhibitor_inducer_perm_inhibitor_thresh" # ADE
            println("Model: 2-factor germination with inhibitor- and inducer-dependent inhibitor/inducer permeability, inhibitor-dependent induction threshold")
            wrapper = (V_out, params) -> Main.germ_response_feedback(
                Main.ode_inducer_and_inhibitor_dependent_perm!,
                Main.thresh_criterion_combined_inducer_shift,
                sobol_pts,
                times,
                [samples_A, samples_Vₛ, V_out, samples_V_ps],
                def_params[:c₀_cs],
                [params[1], params[2]], # b_max, s_max
                params[3], # Pₛ_I
                params[4], # Pₛ_C
                [params[5], params[6]], # K_cI, K_cC
                params[7], # μ_ψ
                params[7] * exp(params[8]), # σ_ψ = μ_ψ * exp(δ_ψ)
                [params[9], params[11]], # μ_γ, μ_ω
                [params[9] * exp(params[10]), params[11] * exp(params[12])]; # σ_γ = μ_γ * exp(δ_γ), σ_ω = μ_ω * exp(δ_ω)
                ks=[exp(params[13])] # k_I
            )
            param_keys = [:b_max, :s_max, :Pₛ, :Pₛ_cs, :K_cI, :K_cC, :μ_ψ, :δ_ψ, :μ_γ, :δ_γ, :μ_ω, :δ_ω, :k_I]
            # param_occurrences = [1, n_src, 1, n_src, 1, n_src, 1, 1, 1, 1, n_src, n_src, n_src]
        
        elseif model_type == "feedback_inhibitor_inducer_thresh_inhibitor_perm_signal" # BCD
            println("Model: Inhibitor-dependent germination with inhibitor-dependent inhibitor/inducer permeability and inducer-dependent inhibition threshold")
            wrapper = (V_out, params) -> Main.germ_response_feedback(
                Main.ode_inhibitor_dependent_perm!,
                Main.thresh_criterion_inhibitor_signal_shift,
                sobol_pts,
                times,
                [samples_A, samples_Vₛ, V_out, samples_V_ps],
                def_params[:c₀_cs],
                [params[1]], # b_max
                params[2], # Pₛ_I
                params[3], # Pₛ_C
                [params[4], params[5], params[6]], # K_cI, K_cC, K_I
                params[7], # μ_ψ
                params[7] * exp(params[8]), # σ_ψ = μ_ψ * exp(δ_ψ)
                [params[9]], # [μ_γ]
                [params[9] * exp(params[10])]; # [σ_γ = μ_γ * exp(δ_γ)]
                ks=params[11], # k_C
                n=params[12] # n
            )
            param_keys = [:b_max, :Pₛ, :Pₛ_cs, :K_cI, :K_cC, :K_I, :μ_ψ, :δ_ψ, :μ_γ, :δ_γ, :k_C, :n]
            # param_occurrences = [1, 1, n_src, n_src, 1, 1, 1, 1, 1, n_src, n_src]
        
        elseif model_type == "feedback_combined_inducer_thresh_inhibitor_perm_signal" # BCD
            println("Model: 2-factor germination with inhibitor-dependent inhibitor/inducer permeability and inducer-dependent inhibition threshold")
            wrapper = (V_out, params) -> Main.germ_response_feedback(
                Main.ode_inhibitor_dependent_perm!,
                Main.thresh_criterion_combined_inhibitor_signal_shift,
                sobol_pts,
                times,
                [samples_A, samples_Vₛ, V_out, samples_V_ps],
                def_params[:c₀_cs],
                [params[1]], # b_max
                params[2], # Pₛ_I
                params[3], # Pₛ_C
                [params[4], params[5], params[6]], # K_cI, K_cC, K_I
                params[7], # μ_ψ
                params[7] * exp(params[8]), # σ_ψ = μ_ψ * exp(δ_ψ)
                [params[9], params[11]], # [μ_γ, μ_ω]
                [params[9] * exp(params[10]), params[11] * exp(params[12])]; # [σ_ω = μ_ω * exp(δ_ω)]
                ks=params[13], # k_C
                n=params[14] # n
            )
            param_keys = [:b_max, :Pₛ, :Pₛ_cs, :K_cI, :K_cC, :K_I, :μ_ψ, :δ_ψ, :μ_γ, :δ_γ, :μ_ω, :δ_ω, :k_C, :n]
            # param_occurrences = [1, 1, n_src, n_src, 1, 1, 1, 1, 1, n_src, n_src, n_src, n_src]
        
        elseif model_type == "combined_inhibitor_thresh_signal_inducer_thresh" # BCE
            println("Model: inhibitor-modulated induction threshold and signal / inducer-modulated inhibition threshold")
            wrapper = (inputs, params) -> Main.germ_response_inh_dep_ind_thresh_signal_ind_dep_inh_thresh_2_factor_gh(
                u, W3,
                inputs[1], #t
                inputs[2], #ρₛ
                def_params[:c₀_cs],
                def_params[:d_hp],
                ξ2,
                κ2,
                params[1], #Pₛ
                params[2], #Pₛ_cs
                params[3], #K_cI
                params[4], #K_cC
                params[5], #K_I
                params[6], #n
                exp(params[7]), #k_I
                exp(params[8]), #k_C
                params[9], #μ_γ
                params[9] * exp(params[10]), # σ_γ = μ_γ * exp(δ_γ)
                params[11], #μ_ω
                params[11] * exp(params[12]), # σ_ω = μ_ω * exp(δ_ω)
                params[13], #μ_ψ
                params[13] * exp(params[14]) # σ_ψ = μ_ψ * exp(δ_ψ)
            )
            param_keys = [:Pₛ, :Pₛ_cs, :K_cI, :K_cC, :K_I, :n, :k_I, :k_C, :μ_γ, :δ_γ, :μ_ω, :δ_ω, :μ_ψ, :δ_ψ]
            # param_occurrences = [1, n_src, n_src, 1, n_src, n_src, n_src, 1, 1, n_src, n_src, 1, 1]

        elseif model_type == "feedback_combined_inhibitor_perm_thresh_inducer_thresh" # BDE
            println("Model: 2-factor germination with inhibitor-dependent inhibitor/inducer permeability and inducer threshold, inducer-dependent inhibition threshold")
            wrapper = (V_out, params) -> Main.germ_response_feedback(
                Main.ode_inhibitor_dependent_perm!,
                Main.thresh_criterion_combined_shift,
                sobol_pts,
                times,
                [samples_A, samples_Vₛ, V_out, samples_V_ps],
                def_params[:c₀_cs],
                [params[1]], # b_max
                params[2], # Pₛ_I
                params[3], # Pₛ_C
                [params[4], params[5]], # K_cI, K_cC
                params[6], # μ_ψ
                params[6] * exp(params[7]), # σ_ψ = μ_ψ * exp(δ_ψ)
                [params[8], params[10]], # [μ_γ, μ_ω]
                [params[8] * exp(params[9]), params[10] * exp(params[11])]; # [σ_ω = μ_ω * exp(δ_ω)]
                ks=[params[12], params[13]] # k_I, k_C
            )
            param_keys = [:b_max, :Pₛ, :Pₛ_cs, :K_cI, :K_cC, :μ_ψ, :δ_ψ, :μ_γ, :δ_γ, :μ_ω, :δ_ω, :k_I, :k_C]
            # param_occurrences = [1, 1, n_src, 1, n_src, 1, 1, 1, 1, n_src, n_src, n_src, n_src]
        
        elseif model_type == "feedback_inducer_inhibitor_perm_thresh_signal" # CDE
            println("Model: Inducer-dependent germination with inhibitor-dependent inhibitor/inducer permeability, induction threshold and signal")
            wrapper = (V_out, params) -> Main.germ_response_feedback(
                Main.ode_inhibitor_dependent_perm!,
                Main.thresh_criterion_inducer_signal_shift,
                sobol_pts,
                times,
                [samples_A, samples_Vₛ, V_out, samples_V_ps],
                def_params[:c₀_cs],
                [params[1]], # b_max
                params[2], # Pₛ_I
                params[3], # Pₛ_C
                [params[4], params[5], params[6]], # K_cI, K_cC, K_I
                params[7], # μ_ψ
                params[7] * exp(params[8]), # σ_ψ = μ_ψ * exp(δ_ψ)
                [params[9]], # [μ_ω]
                [params[9] * exp(params[10])]; # [σ_ω = μ_ω * exp(δ_ω)]
                ks=[params[11]], # k_I
                n=params[12] # n
            )
            param_keys = [:b_max, :Pₛ, :Pₛ_cs, :K_cI, :K_cC, :K_I, :μ_ψ, :δ_ψ, :μ_ω, :δ_ω, :k_I, :n]
            # param_occurrences = [1, 1, n_src, 1, n_src, 1, 1, 1, n_src, n_src, n_src, n_src]
        
        elseif model_type == "feedback_combined_inhibitor_perm_thresh_signal" # CDE
            println("Model: 2-factor germination with inhibitor-dependent inhibitor/inducer permeability, induction threshold and signal")
            wrapper = (V_out, params) -> Main.germ_response_feedback(
                Main.ode_inhibitor_dependent_perm!,
                Main.thresh_criterion_combined_inducer_signal_shift,
                sobol_pts,
                times,
                [samples_A, samples_Vₛ, V_out, samples_V_ps],
                def_params[:c₀_cs],
                [params[1]], # b_max
                params[2], # Pₛ_I
                params[3], # Pₛ_C
                [params[4], params[5], params[6]], # K_cI, K_cC, K_I
                params[7], # μ_ψ
                params[7] * exp(params[8]), # σ_ψ = μ_ψ * exp(δ_ψ)
                [params[9], params[11]], # [μ_γ, μ_ω]
                [params[9] * exp(params[10]), params[11] * exp(params[12])]; # [σ_ω = μ_ω * exp(δ_ω)]
                ks=[params[13]], # k_I
                n=params[14] # n
            )
            param_keys = [:b_max, :Pₛ, :Pₛ_cs, :K_cI, :K_cC, :K_I, :μ_ψ, :δ_ψ, :μ_γ, :δ_γ, :μ_ω, :δ_ω, :k_I, :n]
            # param_occurrences = [1, 1, n_src, 1, n_src, 1, 1, 1, 1, 1, n_src, n_src, n_src, n_src]
        
        elseif model_type == "feedback_inhibitor_inducer_perm_thresh_inhibitor_perm_signal" # ABCD
            println("Model: Inhibitor-dependent germination with inducer/inhibitor-dependent inhibitor/inducer permeability, inducer_dependent inhibition threshold and inhibitor-dependent induction signal")
            wrapper = (V_out, params) -> Main.germ_response_feedback(
                Main.ode_inducer_and_inhibitor_dependent_perm_inhibitor_dependent_signal!,
                Main.thresh_criterion_inhibitor_shift,
                sobol_pts,
                times,
                [samples_A, samples_Vₛ, V_out, samples_V_ps],
                def_params[:c₀_cs],
                [params[1], params[2]], # b_max, s_max
                params[3], # Pₛ_I
                params[4], # Pₛ_C
                [params[5], params[6], params[7]], # K_cI, K_cC, K_I
                params[8], # μ_ψ
                params[8] * exp(params[9]), # σ_ψ = μ_ψ * exp(δ_ψ)
                [params[10]], # μ_γ
                [params[10] * exp(params[11])]; # σ_γ = μ_γ * exp(δ_γ)
                ks=[params[12]], # k_C
                n=params[13] # n
            )
            param_keys = [:b_max, :s_max, :Pₛ, :Pₛ_cs, :K_cI, :K_cC, :K_I, :μ_ψ, :δ_ψ, :μ_γ, :δ_γ, :k_C, :n]
            # param_occurrences = [1, n_src, 1, n_src, 1, n_src, 1, 1, 1, 1, n_src, n_src]
        
        elseif model_type == "feedback_combined_inducer_perm_thresh_inhibitor_perm_signal" # ABCD
            println("Model: 2-factor germination with inducer/inhibitor-dependent inhibitor/inducer permeability, inducer_dependent inhibition threshold and inhibitor-dependent induction signal")
            wrapper = (V_out, params) -> Main.germ_response_feedback(
                Main.ode_inducer_and_inhibitor_dependent_perm_inhibitor_dependent_signal!,
                Main.thresh_criterion_combined_inhibitor_shift,
                sobol_pts,
                times,
                [samples_A, samples_Vₛ, V_out, samples_V_ps],
                def_params[:c₀_cs],
                [params[1], params[2]], # b_max, s_max
                params[3], # Pₛ_I
                params[4], # Pₛ_C
                [params[5], params[6], params[7]], # K_cI, K_cC, K_I
                params[8], # μ_ψ
                params[8] * exp(params[9]), # σ_ψ = μ_ψ * exp(δ_ψ)
                [params[10], params[12]], # μ_γ, μ_ω
                [params[10] * exp(params[11]), params[12] * exp(params[13])]; # σ_γ = μ_γ * exp(δ_γ), σ_ω = μ_ω * exp(δ_ω)
                ks=[params[14]], # k_C
                n=params[15] # n
            )
            param_keys = [:b_max, :s_max, :Pₛ, :Pₛ_cs, :K_cI, :K_cC, :K_I, :μ_ψ, :δ_ψ, :μ_γ, :δ_γ, :μ_ω, :δ_ω, :k_C, :n]
            # param_occurrences = [1, n_src, 1, n_src, 1, n_src, 1, 1, 1, 1, n_src, n_src, n_src, n_src]
        
        elseif model_type == "feedback_combined_inhibitor_thresh_signal_inducer_perm_thresh" # ABCE
            println("Model: 2-factor germination with inducer-dependent permeability/inhibition threshold and inhibitor-dependent induction signal and threshold")
            wrapper = (V_out, params) -> Main.germ_response_feedback(
                Main.ode_inducer_dependent_perm_inhibitor_dependent_signal!,
                Main.thresh_criterion_combined_inhibitor_shift_inducer_signal_shift,
                sobol_pts,
                times,
                [samples_A, samples_Vₛ, V_out, samples_V_ps],
                def_params[:c₀_cs],
                [params[1]], # s_max
                params[2], # Pₛ_I
                params[3], # Pₛ_C
                [params[4], params[5], params[6]], # K_cI, K_cC, K_I
                params[7], # μ_ψ
                params[7] * exp(params[8]), # σ_ψ = μ_ψ * exp(δ_ψ)
                [params[9], params[11]], # [μ_γ, μ_ω]
                [params[9] * exp(params[10]), params[11] * exp(params[12])]; # [σ_γ = μ_γ * exp(δ_γ), σ_ω = μ_ω * exp(δ_ω)]
                ks=[params[13], params[14]],
                n=params[15] # n
            )
            param_keys = [:s_max, :Pₛ, :Pₛ_cs, :K_cI, :K_cC, :K_I, :μ_ψ, :δ_ψ, :μ_γ, :δ_γ, :μ_ω, :δ_ω, :k_I, :k_C, :n]
            # param_occurrences = [n_src, 1, n_src, 1, n_src, 1, 1, 1, 1, n_src, n_src, n_src]
            
        elseif model_type == "feedback_combined_inhibitor_perm_thresh_inducer_perm_thresh" # ABDE
            println("Model: 2-factor germination with inhibitor- and inducer-dependent inhibitor/inducer permeability, inhibitor- and inducer-dependent thresholds")
            wrapper = (V_out, params) -> Main.germ_response_feedback(
                Main.ode_inducer_and_inhibitor_dependent_perm!,
                Main.thresh_criterion_combined_shift,
                sobol_pts,
                times,
                [samples_A, samples_Vₛ, V_out, samples_V_ps],
                def_params[:c₀_cs],
                [params[1], params[2]], # b_max, s_max
                params[3], # Pₛ_I
                params[4], # Pₛ_C
                [params[5], params[6]], # K_cI, K_cC
                params[7], # μ_ψ
                params[7] * exp(params[8]), # σ_ψ = μ_ψ * exp(δ_ψ)
                [params[9], params[11]], # μ_γ, μ_ω
                [params[9] * exp(params[10]), params[11] * exp(params[12])]; # σ_γ = μ_γ * exp(δ_γ), σ_ω = μ_ω * exp(δ_ω)
                ks=[params[13], params[14]] # k_I, k_C
            )
            param_keys = [:b_max, :s_max, :Pₛ, :Pₛ_cs, :K_cI, :K_cC, :μ_ψ, :δ_ψ, :μ_γ, :δ_γ, :μ_ω, :δ_ω, :k_I, :k_C]
            # param_occurrences = [1, n_src, 1, n_src, 1, n_src, 1, 1, 1, 1, n_src, n_src]
        
        elseif model_type == "feedback_inducer_inhibitor_perm_thresh_signal_inducer_perm" # ACDE
            println("Model: Inhibitor-dependent germination with inducer/inhibitor-dependent inhibitor/inducer permeability and inhibitor-dependent induction signal and threshold")
            wrapper = (V_out, params) -> Main.germ_response_feedback(
                Main.ode_inducer_and_inhibitor_dependent_perm_inhibitor_dependent_signal!,
                Main.thresh_criterion_inducer_signal_shift,
                sobol_pts,
                times,
                [samples_A, samples_Vₛ, V_out, samples_V_ps],
                def_params[:c₀_cs],
                [params[1], params[2]], # b_max, s_max
                params[3], # Pₛ_I
                params[4], # Pₛ_C
                [params[5], params[6], params[7]], # K_cI, K_cC, K_I
                params[8], # μ_ψ
                params[8] * exp(params[9]), # σ_ψ = μ_ψ * exp(δ_ψ)
                [params[10]], # μ_ω
                [params[10] * exp(params[11])]; # σ_ω = μ_ω * exp(δ_ω)
                ks=[params[12]], # k_I
                n=params[13] # n
            )
            param_keys = [:b_max, :s_max, :Pₛ, :Pₛ_cs, :K_cI, :K_cC, :K_I, :μ_ψ, :δ_ψ, :μ_ω, :δ_ω, :k_I, :n]
        
        elseif model_type == "feedback_combined_inhibitor_perm_thresh_signal_inducer_perm" # ACDE
            println("Model: 2-factor germination with inducer/inhibitor-dependent inhibitor/inducer permeability and inhibitor-dependent induction signal and threshold")
            wrapper = (V_out, params) -> Main.germ_response_feedback(
                Main.ode_inducer_and_inhibitor_dependent_perm_inhibitor_dependent_signal!,
                Main.thresh_criterion_combined_inducer_signal_shift,
                sobol_pts,
                times,
                [samples_A, samples_Vₛ, V_out, samples_V_ps],
                def_params[:c₀_cs],
                [params[1], params[2]], # b_max, s_max
                params[3], # Pₛ_I
                params[4], # Pₛ_C
                [params[5], params[6], params[7]], # K_cI, K_cC, K_I
                params[8], # μ_ψ
                params[8] * exp(params[9]), # σ_ψ = μ_ψ * exp(δ_ψ)
                [params[10], params[12]], # μ_γ, μ_ω
                [params[10] * exp(params[11]), params[12] * exp(params[13])]; # σ_γ = μ_γ * exp(δ_γ), σ_ω = μ_ω * exp(δ_ω)
                ks=[params[14]], # k_I
                n=params[15] # n
            )
            param_keys = [:b_max, :s_max, :Pₛ, :Pₛ_cs, :K_cI, :K_cC, :K_I, :μ_ψ, :δ_ψ, :μ_γ, :δ_γ, :μ_ω, :δ_ω, :k_I, :n]
        
        elseif model_type == "feedback_combined_inhibitor_perm_thresh_signal_inducer_thresh" # BCDE
            println("Model: 2-factor germination with inhibitor-dependent inhibitor/inducer permeability, induction threshold and signal, and inducer-dependent inhibition threshold")
            wrapper = (V_out, params) -> Main.germ_response_feedback(
                Main.ode_inhibitor_dependent_perm!,
                Main.thresh_criterion_combined_inhibitor_shift_inducer_signal_shift,
                sobol_pts,
                times,
                [samples_A, samples_Vₛ, V_out, samples_V_ps],
                def_params[:c₀_cs],
                [params[1]], # b_max
                params[2], # Pₛ_I
                params[3], # Pₛ_C
                [params[4], params[5], params[6]], # K_cI, K_cC, K_I
                params[7], # μ_ψ
                params[7] * exp(params[8]), # σ_ψ = μ_ψ * exp(δ_ψ)
                [params[9], params[11]], # [μ_γ, μ_ω]
                [params[9] * exp(params[10]), params[11] * exp(params[12])]; # [σ_ω = μ_ω * exp(δ_ω)]
                ks=[params[13], params[14]],
                n=params[15]
            )
            param_keys = [:b_max, :Pₛ, :Pₛ_cs, :K_cI, :K_cC, :K_I, :μ_ψ, :δ_ψ, :μ_γ, :δ_γ, :μ_ω, :δ_ω, :k_I, :k_C, :n]
        
        elseif model_type == "feedback_combined_inhibitor_perm_thresh_signal_inducer_perm_thresh" # ABCDE
            println("Model: 2-factor germination with inducer/inhibitor-dependent inhibitor/inducer permeability, inducer-dependent inhibition threshold and inhibitor-dependent induction threshold/signal")
            wrapper = (V_out, params) -> Main.germ_response_feedback(
                Main.ode_inducer_and_inhibitor_dependent_perm_inhibitor_dependent_signal!,
                Main.thresh_criterion_combined_inhibitor_shift_inducer_signal_shift,
                sobol_pts,
                times,
                [samples_A, samples_Vₛ, V_out, samples_V_ps],
                def_params[:c₀_cs],
                [params[1], params[2]], # b_max, s_max
                params[3], # Pₛ_I
                params[4], # Pₛ_C
                [params[5], params[6], params[7]], # K_cI, K_cC, K_I
                params[8], # μ_ψ
                params[8] * exp(params[9]), # σ_ψ = μ_ψ * exp(δ_ψ)
                [params[10], params[12]], # μ_γ, μ_ω
                [params[10] * exp(params[11]), params[12] * exp(params[13])]; # σ_γ = μ_γ * exp(δ_γ), σ_ω = μ_ω * exp(δ_ω)
                ks=[params[14], params[15]], # k_I, k_C
                n=params[16] # n
            )
            param_keys = [:b_max, :s_max, :Pₛ, :Pₛ_cs, :K_cI, :K_cC, :K_I, :μ_ψ, :δ_ψ, :μ_γ, :δ_γ, :μ_ω, :δ_ω, :k_I, :k_C, :n]
        
        else
            error("Model type not recognized.")
        end

        if debug
            # Simply print wrapper function and parameters
            
            # println("Parameters: ", sort(vcat(param_keys, collect(keys(def_params)))))
            println("Parameters: ", sort(param_keys))
            params_out = nothing
            rmse = nothing
        
        else
            # Model fitting

            param_occurrences = [param_occurrences_dict[pkey] for pkey in param_keys]

            # Duplicate bounds/parameters for each source
            param_starts = cumsum(param_occurrences) .- param_occurrences .+ 1
            param_keys_dup = vcat([key for key in param_keys for _ in 1:param_occurrences[param_keys .== key][1]]...)
            param_starts = cumsum(param_occurrences) .- param_occurrences .+ 1
            bounds = [bounds_dict[key] for key in param_keys_dup]

            param_indices_per_src = [param_starts .+ ((i - 1) .% param_occurrences) for i in 1:length(sources)]
            
            if gh_integral

                # Objective function
                input_tuples =  [tuple.(times_tile[i, :, :], inverse_mL_to_cubic_um.(densities_tile[i, :, :])) for i in 1:length(sources)]
                dantigny_data_flat = [collect(dantigny_data[i, :, :]) for i in 1:length(sources)]

                # println(size(times_tile))
                # println(size(densities_tile))
                # println(size(input_tuples))
                # println(size(dantigny_data))
                # println(size(dantigny_data_flat))

                obj = params -> begin
                    err = 0.0
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
            else

                # Objective function (feedback model)
                input_densities = inverse_mL_to_cubic_um.(densities)
                samples_V_out = 1 ./ input_densities .- samples_Vₛ'

                params = zeros(length(keys(bounds_dict)))
                for (i, key) in enumerate(keys(bounds_dict))
                    params[i] = mean(bounds_dict[key])
                end

                obj = params -> begin
                    err = 0.0
                    @inbounds for i in eachindex(sources)
                        params_select = view(params, param_indices_per_src[i])

                        # run a single simulation at all times
                        ŷ = reduce(vcat, [wrapper(samples_V_out[j, :], params_select) for j in eachindex(input_densities)]')

                        err += sum(abs2, ŷ .- dantigny_data[i, :, :])
                    end
                    return err
                end
                objgrad = (params,_) -> begin
                    err = 0.0
                    @inbounds for i in eachindex(sources)
                        params_select = view(params, param_indices_per_src[i])
                        
                        # run a single simulation at all times
                        ŷ = reduce(vcat, [wrapper(samples_V_out[j, :], params_select) for j in eachindex(input_densities)]')

                        err += sum(abs2, ŷ .- dantigny_data[i, :, :])
                    end
                    return err
                end
            end
            
            # Fit model
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
                        val = p_opt[param_starts[i - 1] + j - 1] * exp(p_opt[param_starts[i] + j - 1])
                    elseif (key == :k_I || key == :k_C)
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


    function fit_model_to_data_equilibrium(model_type, def_params, germ_data, densities, bounds_dict; c_ex_vals=nothing, ref_density=nothing, max_steps=10000)
        """
        Fit a selected equilibrium germination model to the data.
        inputs:
            model_type (String): model type to fit
            def_params (Dict): default parameter values
            germ_data (Array): germination data (concatenated datesets if both exogenous and endogenous models are fitted)
            densities (Vector): spore densities in spores/mL
            bounds_dict (Dict): bounds for the free parameters
            c_ex_vals (Vector): exogenous inhibitor concentrations (optional)
            ref_density (Float): reference density for exogenous inhibitor (optional, only used when fitting both exogenous and endogenous models)
            max_steps (int): maximum number of steps for the optimization
        outputs:
            params_out (Dict): optimized parameters
        """

        @argcheck model_type in ["inhibitor", "combined_inducer", "combined_inducer_thresh", "combined_inducer_signal", "independent",
                                "inhibitor_ex", "combined_inducer_ex", "combined_inducer_thresh_ex", "combined_inducer_signal_ex", "independent_ex"]

        if isnothing(ref_density)
            ref_density = inverse_mL_to_cubic_um(densities[1]) # Use first density as reference if not provided
        elseif !isnothing(c_ex_vals)
            ref_density = inverse_mL_to_cubic_um(ref_density) # Convert to cubic micrometers if provided
        end

        # Unpack radius distribution
        μ_ξ = def_params[:μ_ξ]
        σ_ξ = def_params[:σ_ξ]
        μ_ξ_log = log(μ_ξ^2 / sqrt(σ_ξ^2 + μ_ξ^2))
        σ_ξ_log = sqrt(log(σ_ξ^2 / μ_ξ^2 + 1))
        dist_ξ = LogNormal(μ_ξ_log, σ_ξ_log)

        if model_type in ["inhibitor", "inhibitor_ex"]
            println("Model: Inducer-dependent inhibitor threshold and release")
            wrapper = (ρₛ, params) -> Main.germ_response_inducer_dep_inhibitor_eq(
                ρₛ,
                dist_ξ,
                params[1], #μ_γ
                params[1] * exp(params[2]) # σ_γ = μ_γ * exp(δ_γ)
            )
            param_keys = [:μ_γ, :δ_γ]
        elseif model_type in ["combined_inducer", "combined_inducer_ex"]
            println("Model: Two-factor germination with inhibitor-dependent induction threshold and signal")
            wrapper = (ρₛ, params) -> Main.germ_response_inhibitor_dep_inducer_2_factors_eq(
                ρₛ,
                dist_ξ,
                def_params[:c₀_cs],
                params[1], #K_cI
                params[2], #K_cC
                params[3], #K_I
                exp(params[4]), #k_I
                params[5], #n
                params[6], #μ_γ
                params[6] * exp(params[7]), # σ_γ = μ_γ * exp(δ_γ)
                params[8], #μ_ω
                params[8] * exp(params[9]), # σ_ω = μ_ω * exp(δ_ω)
                params[10], #μ_ψ
                params[10] * exp(params[11]) # σ_ψ = μ_ψ * exp(δ_ψ)
            )
            param_keys = [:K_cI, :K_cC, :K_I, :k_I, :n, :μ_γ, :δ_γ, :μ_ω, :δ_ω, :μ_ψ, :δ_ψ]
        elseif model_type in ["combined_inducer_thresh", "combined_inducer_thresh_ex"]
            println("Model: Two-factor germination with inhibitor-dependent induction threshold")
            wrapper = (ρₛ, params) -> Main.germ_response_inhibitor_dep_inducer_thresh_2_factors_eq(
                ρₛ,
                dist_ξ,
                def_params[:c₀_cs],
                params[1], #K_cC
                exp(params[2]), #k_I
                params[3], #μ_γ
                params[3] * exp(params[4]), # σ_γ = μ_γ * exp(δ_γ)
                params[5], #μ_ω
                params[5] * exp(params[6]), # σ_ω = μ_ω * exp(δ_ω)
                params[7], #μ_ψ
                params[7] * exp(params[8]) # σ_ψ = μ_ψ * exp(δ_ψ)
            )
            param_keys = [:K_cC, :k_I, :μ_γ, :δ_γ, :μ_ω, :δ_ω, :μ_ψ, :δ_ψ]
        elseif model_type in ["combined_inducer_signal", "combined_inducer_signal_ex"]
            println("Model: Two-factor germination with inhibitor-dependent induction signal")
            wrapper = (ρₛ, params) -> Main.germ_response_inhibitor_dep_inducer_signal_2_factors_eq(
                ρₛ,
                dist_ξ,
                def_params[:c₀_cs],
                params[1], #K_cC
                params[2], #K_I
                params[3], #n
                params[4], #μ_γ
                params[4] * exp(params[5]), # σ_γ = μ_γ * exp(δ_γ)
                params[6], #μ_ω
                params[6] * exp(params[7]), # σ_ω = μ_ω * exp(δ_ω)
                params[8], #μ_ψ
                params[8] * exp(params[9]) # σ_ψ = μ_ψ * exp(δ_ψ)
            )
            param_keys = [:K_cC, :K_I, :n, :μ_γ, :δ_γ, :μ_ω, :δ_ω, :μ_ψ, :δ_ψ]
        elseif model_type in ["independent", "independent_ex"]
            println("Model: Independent factors")
            wrapper = (ρₛ, params) -> Main.germ_response_independent_eq(
                ρₛ,
                dist_ξ,
                def_params[:c₀_cs],
                params[1], #K_cC
                params[2], #μ_γ
                params[2] * exp(params[3]), # σ_γ = μ_γ * exp(δ_γ)
                params[4], #μ_ω
                params[4] * exp(params[5]) # σ_ω = μ_ω * exp(δ_ω)
            )
            param_keys = [:K_cC, :μ_γ, :δ_γ, :μ_ω, :δ_ω]
        end
        if model_type == "inhibitor_ex"
            println("Model: Inducer-dependent inhibitor threshold and release (exogenous inhibitor)")
            wrapper_ex = (c_ex, params) -> Main.germ_response_inducer_dep_inhibitor_eq_c_ex(
                ref_density, # Assuming single density for exogenous inhibitor
                dist_ξ,
                c_ex,
                params[1], #μ_γ
                params[1] * exp(params[2]), # σ_γ = μ_γ * exp(δ_γ)
                params[3], #μ_ψ
                params[3] * exp(params[4]) # σ_ψ = μ_ψ * exp(δ_ψ)
            )
            param_keys = [:μ_γ, :δ_γ, :μ_ψ, :δ_ψ]
        elseif model_type == "combined_inducer_ex"
            println("Model: Two-factor germination with inhibitor-dependent induction threshold and signal (exogenous inhibitor)")
            wrapper_ex = (c_ex, params) -> Main.germ_response_inhibitor_dep_inducer_2_factors_eq_c_ex(
                ref_density, # Assuming single density for exogenous inhibitor
                dist_ξ,
                c_ex,
                def_params[:c₀_cs],
                params[1], #K_cI
                params[2], #K_cC
                params[3], #K_I
                exp(params[4]), #k_I
                params[5], #n
                params[6], #μ_γ
                params[6] * exp(params[7]), # σ_γ = μ_γ * exp(δ_γ)
                params[8], #μ_ω
                params[8] * exp(params[9]), # σ_ω = μ_ω * exp(δ_ω)
                params[10], #μ_ψ
                params[10] * exp(params[11]) # σ_ψ = μ_ψ * exp(δ_ψ)
            )
            param_keys = [:K_cI, :K_cC, :K_I, :k_I, :n, :μ_γ, :δ_γ, :μ_ω, :δ_ω, :μ_ψ, :δ_ψ]
        elseif model_type == "combined_inducer_thresh_ex"
            println("Model: Two-factor germination with inhibitor-dependent induction threshold (exogenous inhibitor)")
            wrapper_ex = (c_ex, params) -> Main.germ_response_inhibitor_dep_inducer_thresh_2_factors_eq_c_ex(
                ref_density, # Assuming single density for exogenous inhibitor
                dist_ξ,
                c_ex,
                def_params[:c₀_cs],
                params[1], #K_cI
                params[2], #K_cC
                exp(params[3]), #k_I
                params[4], #μ_γ
                params[4] * exp(params[5]), # σ_γ = μ_γ * exp(δ_γ)
                params[6], #μ_ω
                params[6] * exp(params[7]), # σ_ω = μ_ω * exp(δ_ω)
                params[8], #μ_ψ
                params[8] * exp(params[9]) # σ_ψ = μ_ψ * exp(δ_ψ)
            )
            param_keys = [:K_cI, :K_cC, :k_I, :μ_γ, :δ_γ, :μ_ω, :δ_ω, :μ_ψ, :δ_ψ]
        elseif model_type == "combined_inducer_signal_ex"
            println("Model: Two-factor germination with inhibitor-dependent induction signal (exogenous inhibitor)")
            wrapper_ex = (c_ex, params) -> Main.germ_response_inhibitor_dep_inducer_signal_2_factors_eq_c_ex(
                ref_density, # Assuming single density for exogenous inhibitor
                dist_ξ,
                c_ex,
                def_params[:c₀_cs],
                params[1], #K_cC
                params[2], #K_I
                params[3], #n
                params[4], #μ_γ
                params[4] * exp(params[5]), # σ_γ = μ_γ * exp(δ_γ)
                params[6], #μ_ω
                params[6] * exp(params[7]), # σ_ω = μ_ω * exp(δ_ω)
                params[8], #μ_ψ
                params[8] * exp(params[9]) # σ_ψ = μ_ψ * exp(δ_ψ)
            )
            param_keys = [:K_cC, :K_I, :n, :μ_γ, :δ_γ, :μ_ω, :δ_ω, :μ_ψ, :δ_ψ]
        elseif model_type == "independent_ex"
            println("Model: Independent factors (exogenous inhibitor)")
            wrapper_ex = (c_ex, params) -> Main.germ_response_independent_eq_c_ex(
                ref_density, # Assuming single density for exogenous inhibitor
                dist_ξ,
                c_ex,
                def_params[:c₀_cs],
                params[1], #K_cC
                params[2], #μ_γ
                params[2] * exp(params[3]), # σ_γ = μ_γ * exp(δ_γ)
                params[4], #μ_ω
                params[4] * exp(params[5]), # σ_ω = μ_ω * exp(δ_ω)
                params[6], #μ_ψ
                params[6] * exp(params[7]) # σ_ψ = μ_ψ * exp(δ_ψ)
            )
            param_keys = [:K_cC, :μ_γ, :δ_γ, :μ_ω, :δ_ω, :μ_ψ, :δ_ψ]
        end

        bounds = [bounds_dict[key] for key in param_keys]
        densities_input = inverse_mL_to_cubic_um.(densities)

        if isnothing(c_ex_vals) # No exogenous inhibitor concentrations provided
            n_inputs = length(densities_input)
            # Objective function
            obj = params -> begin
                ŷ = [wrapper(ρₛ, params) for ρₛ in densities_input]
                err = sum(abs2, ŷ .- germ_data) 
                return err
            end
            objgrad = (params, _) -> begin
                ŷ = [wrapper(ρₛ, params) for ρₛ in densities_input]
                err = sum(abs2, ŷ .- germ_data) 
                return err
            end
        elseif length(densities) == 1 # Exogenous inhibitor concentrations provided with single density
            n_inputs = length(c_ex_vals)
            # Objective function
            obj = params -> begin
                ŷ = [wrapper_ex(c_ex, params) for c_ex in c_ex_vals]
                err = sum(abs2, ŷ .- germ_data) 
                return err
            end
            objgrad = (params, _) -> begin
                ŷ = [wrapper_ex(c_ex, params) for c_ex in c_ex_vals]
                err = sum(abs2, ŷ .- germ_data) 
                return err
            end
        elseif !isnothing(ref_density) && length(densities) > 1 # Exogenous inhibitor concentrations provided with multiple densities
            n_inputs = length(c_ex_vals) + length(densities)
            # Objective function
            obj = params -> begin
                ŷ = vcat([wrapper(ρₛ, params) for ρₛ in densities_input], [wrapper_ex(c_ex, params) for c_ex in c_ex_vals])
                err = sum(abs2, ŷ .- germ_data) 
                return err
            end
            objgrad = (params, _) -> begin
                ŷ = vcat([wrapper(ρₛ, params) for ρₛ in densities_input], [wrapper_ex(c_ex, params) for c_ex in c_ex_vals])
                err = sum(abs2, ŷ .- germ_data) 
                return err
            end
        else
            error("Invalid input: either provide c_ex_vals or a non-singular density array.")
        end
            

        # Fit model
        println("Running first optimisation stage")
        res = bboptimize(params -> obj(params);
                    SearchRange = bounds,
                    MaxSteps = max_steps,
                    Method = :adaptive_de_rand_1_bin_radiuslimited)
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
        rmse = sqrt(best_fit / n_inputs)

        # Create a dictionary for the optimized parameters
        params_out = Dict()
        for (i, key) in enumerate(param_keys)
            # Transform parameters back to original scale
            key_split = split(string(key), "_")
            if key_split[1] == "δ"
                println("Converting δ = $(p_opt[i]) to σ scale")
                key_new = Symbol(:σ_, key_split[2])
                val = p_opt[i - 1] * exp(p_opt[i])
            elseif (key == :k_I || key == :k_C)
                println("Converting k to original scale")
                key_new = key
                val = exp(p_opt[i])
            else
                key_new = key
                val = p_opt[i]
            end
            if haskey(params_out, key_new)
                push!(params_out[key_new], val)
            else
                params_out[key_new] = [val]
            end
        end

        return params_out, rmse
    end

    function fit_feedback_model_to_data(model_type, def_params, dantigny_data, times, sources, densities, bounds_dict, max_steps=10000)

    end

end