module DataUtils
    """
    Contains utility functions.
    """

    using DataFrames
    using CSV
    using BlackBoxOptim
    
    include("./conversions.jl")
    using .Conversions

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
            end
        end

        return dantigny_data * 0.01, times * 3600, sources, densities
    end


    function fit_model_to_data(model_type, def_params, dantigny_data, times, sources, densities, bounds_dict, max_steps=10000)
        """
        Fit a selected germination model to the data.
        inputs:
            model_type (String): model type to fit
                ("independent", "inhibitor", "inhibitor_thresh", "inhibitor_signal", "inducer", "inducer_thresh", "inducer_perm")
            def_params (Dict): default parameter values
            dantigny_data (Matrix): time-dependent data from varying inducers and spore densities to fit
            times (Vector): time points
            sources (Vector): carbon sources
            densities (Vector): spore densities
            bounds_dict (Dict): bounds for the free parameters
            max_steps (int): maximum number of steps for the optimization
        """
        
        # Estimator and bounds
        ŷ = nothing
        bounds = nothing

        if model_type == "independent"
            # Independent inducer/inhibitor
            wrapper(inputs, params) = Main.germination_response_combined_independent.(
                params[1], # s
                params[2], # μ_ω
                params[3], # σ_ω	
                inputs[1], # ρₛ,
                params[4], # Pₛ
                params[5], # μ_γ,
                params[6], # σ_γ,
                def_params[:μ_ξ],
                def_params[:σ_ξ],
                inputs[2], # ρₛ
            )

            # Reshape input
            densities_tile = repeat(densities, outer=[1, length(sources), length(times)])
            densities_tile = permutedims(densities_tile, (2, 1, 3))
            times_tile = repeat(times, outer=[1, length(sources), length(densities)])
            times_tile = permutedims(times_tile, (2, 3, 1))

            bounds = [
                bounds_dict[:s],
                bounds_dict[:μ_ω],
                bounds_dict[:σ_ω],
                bounds_dict[:Pₛ],
                bounds_dict[:μ_γ],
                bounds_dict[:σ_γ]
            ]
        end

        function obj(params)
            """
            Objective function
            """
            ŷ = [wrapper(inputs, params) for inputs in tuple.(densities_tile, times_tile)]
            ŷ = reshape(ŷ, size(densities_tile))
            return sum(abs2, ŷ .- dantigny_data)
        end
        
        # Fit model
        res = bboptimize(params -> obj(params);
                    SearchRange = bounds,
                    NumDimensions = 3,
                    MaxSteps = max_steps)
        p_opt = best_candidate(res)
    end
end