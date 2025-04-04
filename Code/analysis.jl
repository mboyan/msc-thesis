module Analysis
__precompile__(false)
    """
    Functions for analysing model results.
    """

    using JLD2
    using Revise

    include("./conversions.jl")
    Revise.includet("./conversions.jl")
    using .Conversions

    export get_concentration_evolution_from_file
    export get_coverage_and_exponent_from_files

    function parse_parameters!(exp_ID, sim_ID, param_dict, extra_params=nothing)
        """
        Retrieve parameters names and their respective values
        for a simulation with a given ID.
        inputs:
            exp_ID (string): experiment ID
            sim_ID (string): simulation ID
            param_dict (Dict): dictionary for storing parameter values
            extra_params (Vector{Symbol}): optional extra parameters to extract
        """
        path = @__DIR__
        path = joinpath(path, "Data", exp_ID)

        # Load corresponding simulation parameters
        file_param = sim_ID * "_params.jld2"
        filepath_param = joinpath(path, file_param)
        sim_params = load(filepath_param, "parameters")

        # Parse parameter names
        file_chars = collect(sim_ID)
        numeric_mask = isnumeric.(file_chars)
        numeric_indices = findall(x -> x == 1, numeric_mask)
        pushfirst!(numeric_indices, -1)
        parameters = [join(file_chars[(2+numeric_indices[i]):(numeric_indices[i+1]-2)]) for i in 1:(length(numeric_indices)-1)]
        
        # Append parameter values
        for param in parameters
            if length(param) > 0
                param = Symbol(param)
                if haskey(param_dict, param)
                    push!(param_dict[param], sim_params[param])
                else
                    param_dict[param] = [sim_params[param]]
                end
            end
        end

        # Append extra parameters
        if !isnothing(extra_params)
            for param in extra_params
                param = Symbol(param)
                if haskey(sim_params, param)
                    if haskey(param_dict, param)
                        push!(param_dict[param], sim_params[param])
                    else
                        param_dict[param] = [sim_params[param]]
                    end
                else
                    print("Parameter $(string(param)) not found.\r")
                end
            end
        end
    end

    
    function get_concentration_evolution_from_file(exp_ID, sim_ID; get_frames=false)
        """
        Retrieve times and concentrations
        from a simulation with a given ID.
        inputs:
            exp_ID (string): experiment ID
            sim_ID (string): simulation ID
            get_frames (bool): if true, returns snapshots of the lattice
        outputs:
            times (Vector{Float64}): the meaurement times
            concentrations (Vector{Float64}): the concentration measurements
            param_dict (Dict): dictionary for storing parameter values
        """
        path = @__DIR__

        # Parse parameters
        param_dict = Dict()
        parse_parameters!(exp_ID, sim_ID, param_dict, [:dx])

        # Get results
        file = joinpath(path, "Data", exp_ID, "$(sim_ID)_results.jld2")
        times = load(file, "times")
        if get_frames
            concentrations = load(file, "frame_samples")
        else
            concentrations = load(file, "c_solutions")
        end
        
        return  times, concentrations, param_dict
    end

    
    function get_coverage_and_exponent_from_files(exp_ID; norm_exponent=true)
        """
        Retrieve the measured spore coverage and the
        fitted decay exponent from the simulation files
        of an experiment and identify the varying parameters
        related to them.
        inputs:
            exp_ID (string): experiment ID
            normalize_exponent (bool): whether to normalise the decay exponent given the Ps parameter
        """
        path = @__DIR__
        path = joinpath(path, "Data", exp_ID)
        
        # Find all result files
        result_dict = Dict(:coverage => [], :exponent => [])
        for file in readdir(path)
            if endswith(file, "results.jld2")
                # println(occursin("cluster_size", file))

                filepath_res = joinpath(path, file)
                coverage = load(filepath_res, "coverage")
                exponent = load(filepath_res, "exponent")

                # Find corresponding simulation parameters
                sim_ID = join(split(file, '_')[1:(end-1)], '_')
                if norm_exponent
                    parse_parameters!(exp_ID, sim_ID, result_dict, [:Ps, :spore_diameter])
                    if haskey(result_dict, :spore_diameter)
                        spore_diameter = result_dict[:spore_diameter][1]
                    else
                        spore_diameter = 5.0 # default value in microns
                    end
                    A_spore, V_spore = compute_spore_area_and_volume(spore_diameter)
                    tau = result_dict[:Ps][1] .* A_spore / V_spore
                    exponent = -exponent / tau
                    delete!(result_dict, :Ps)
                    delete!(result_dict, :spore_diameter)
                else
                    parse_parameters!(exp_ID, sim_ID, result_dict)
                end

                push!(result_dict[:coverage], coverage)
                push!(result_dict[:exponent], exponent)
            end
        end

        return result_dict
    end

end