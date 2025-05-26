module Analysis
__precompile__(false)
    """
    Functions for analysing model results.
    """

    using JLD2
    using LaTeXTabulars, LaTeXStrings
    using Revise

    include("./conversions.jl")
    Revise.includet("./conversions.jl")
    using .Conversions

    export get_concentration_evolution_from_file
    export get_coverage_and_exponent_from_files
    export get_density_and_exponent_from_files

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
                    A_spore, V_spore = compute_spore_area_and_volume_from_dia(spore_diameter)
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


    function get_density_and_exponent_from_files(exp_ID; norm_exponent=true)
        """
        Infer the spore density from the grid size and retrieve the
        permeation constant and fitted decay exponent from the simulation files
        of an experiment and identify the varying parameters
        related to them.
        inputs:
            exp_ID (string): experiment ID
            normalize_exponent (bool): whether to normalise the decay exponent given the Ps parameter
        """
        path = @__DIR__
        path = joinpath(path, "Data", exp_ID)
        
        # Find all result files
        result_dict = Dict(:spore_density => [], :spore_spacing => [], :Ps => [], :exponent => [])
        for file in readdir(path)
            if endswith(file, "results.jld2")
                # println(occursin("cluster_size", file))

                filepath_res = joinpath(path, file)
                exponent = load(filepath_res, "exponent")

                # Find corresponding simulation parameters
                sim_ID = join(split(file, '_')[1:(end-1)], '_')
                if norm_exponent
                    parse_parameters!(exp_ID, sim_ID, result_dict, [:N, :H, :dx, :spore_diameter])
                    if haskey(result_dict, :spore_diameter)
                        spore_diameter = result_dict[:spore_diameter][end]
                    else
                        spore_diameter = 5.0 # default value in microns
                    end
                    A_spore, V_spore = compute_spore_area_and_volume_from_dia(spore_diameter)
                    tau = V_spore ./ (result_dict[:Ps][end] .* A_spore)
                    exponent = -exponent * tau
                    delete!(result_dict, :spore_diameter)
                else
                    parse_parameters!(exp_ID, sim_ID, result_dict, [:N, :H, :dx])
                end

                if haskey(result_dict, :H)
                    H = result_dict[:H][end]
                else
                    H = result_dict[:N][end]
                end

                spore_density = 1 / (result_dict[:N][end] * H * result_dict[:dx][end]^3)
                # spore_spacing = result_dict[:N][1] * result_dict[:dx][1]
                spore_spacing = 1 / cbrt(spore_density)

                if norm_exponent
                    ϕ = spore_density * V_spore
                    exponent = exponent * (1 - ϕ)
                    # println("Exponent: $(exponent)")
                end

                spore_density = inverse_cubic_um_to_mL(spore_density)

                push!(result_dict[:spore_density], spore_density)
                push!(result_dict[:spore_spacing], spore_spacing)
                push!(result_dict[:exponent], exponent)
                delete!(result_dict, :N)
                delete!(result_dict, :H)
                delete!(result_dict, :dx)
            end
        end

        return result_dict
    end


    function summarise_fitted_parameters(exp_ID)
        """
        Summarise fitted parameters into a LaTeX table.
        inputs:
            exp_ID (string): experiment ID
        """

        path = @__DIR__
        path = joinpath(path, "Data", exp_ID)

        model_labels = Dict(
            "independent" => "Independent",
            "inhibitor" => "Inducer " * L"\rightarrow" * " inhibitor threshold and release",
            "inhibitor_thresh" => "Inducer " * L"\rightarrow" * " inhibition threshold",
            "inhibitor_perm" => "Inducer " * L"\rightarrow" * " inhibition threshold and release",
            "inducer" => "Inhibitor " * L"\rightarrow" * " induction threshold and signal",
            "inducer_thresh" => "Inhibitor " * L"\rightarrow" * " induction threshold",
            "inducer_signal" => "Inhibitor " * L"\rightarrow" * " induction signal",
            "combined" => "2 factors, inhibitor " * L"\rightarrow" * " induction threshold and signal",
            "combined_thresh" => "2 factors, inhibitor " * L"\rightarrow" * " induction threshold",
            "combined_signal" => "2 factors, inhibitor " * L"\rightarrow" * " induction signal",
            "special_inhibitor" => "Inducer " * L"\rightarrow" * " inhibitor threshold and release (var. permeability)",
            "special_independent" => "Independent (var. permeability)"
        )

        units = Dict(
            :Pₛ => L"\si{\micro\meter\per\second}",
            :Pₛ_cs => L"\si{\micro\meter\per\second}",
            :μ_γ => L"-",
            :σ_γ => L"-",
            :k => L"-",
            :K_I => L"\si{M}",
            :K_cs => L"\si{M}",
            :n => L"-",
            :s => L"-",
            :μ_ω => L"-",
            :σ_ω => L"-",
            :μ_ψ => L"\si{M}",	
            :σ_ψ => L"\si{M}",
            :μ_π => L"\si{\micro\meter\per\second}",
            :σ_π => L"\si{\micro\meter\per\second}",
            :μ_α => L"-",
            :σ_α => L"-"
        )

        n_params = length(keys(units))
        tabular = Tabular("|l"^(n_params+1) * "|")
        header = [L"\textbf{Model}"] ++ [L"\textbf{$(string(key))}" for key in keys(units)] ++ [L"\textbf{Units}"]

        # Find all result files
        rows = []
        for file in readdir(path)
            if beginswith(file, "fit")
                model_type = join(split(file, '_')[2:(end-1)], '_')
                params_opt = jldopen("Data/fit_inducer_thresh.jld2", "r") do file
                    return file["params_opt"]
                end
                row = [model_labels[model_type]] ++ [haskey(params_opt, key) ? params_opt[key] : " " for key in keys(units)]
                push!(rows, row)
            end
        end

        # Create LaTeX table
        return latex_tabular(String, tabular, [Rule(:top), header, Rule(:midrule)], rows, [Rule(:bottom)]...)
        
    end
end