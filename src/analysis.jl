module Analysis
__precompile__(false)
    """
    Functions for analysing model results.
    """

    using JLD2
    using LaTeXTabulars, LaTeXStrings
    using Printf
    using ArgCheck

    include("./conversions.jl")
    using .Conversions

    export get_concentration_evolution_from_file
    export get_coverage_and_exponent_from_files
    export get_density_and_exponent_from_files
    export summarise_fitted_parameters

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

                spore_density = 1 / (result_dict[:N][end]^2 * H * result_dict[:dx][end]^3)
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


    function summarise_fitted_parameters(exp_ID; model_types=nothing, extra_tags=["st"])
        """
        Summarise fitted parameters into a LaTeX table.
        inputs:
            exp_ID (string): experiment ID
            model_types (Vector{String}): optional list of model types to include in the table
            extra_tag (String): optional extra tags of result file
        """

        @argcheck any(in.(extra_tags, Ref([nothing, "st", "ex", "total"])))

        path = @__DIR__
        path = joinpath(path, "Data", exp_ID)

        model_labels = Dict(
            "independent" => "\\textbf{2 factors,}\\newline \\textbf{independent}",
            "inhibitor" => "\\textbf{Inducer →  }\\newline \\textbf{inhibitor}\\newline \\textbf{thresh. + release}",
            "inhibitor_thresh" => "\\textbf{Inducer →  }\\newline \\textbf{inhibitor}\\newline \\textbf{thresh.}",
            "inhibitor_perm" => "\\textbf{Inducer →  }\\newline \\textbf{inhibitor}\\newline \\textbf{release}",
            "inducer" => "\\textbf{Inhibitor →  }\\newline \\textbf{inducer}\\newline \\textbf{thresh. + signal}",
            "inducer_thresh" => "\\textbf{Inhibitor →  }\\newline \\textbf{inducer}\\newline \\textbf{threshold}",
            "inducer_signal" => "\\textbf{Inhibitor →  }\\newline \\textbf{inducer}\\newline \\textbf{signal}",
            "combined_inhibitor" => "\\textbf{2 factors,}\\newline \\textbf{inducer →  }\\newline \\textbf{inhibitor}\\newline \\textbf{thresh. + release}",
            "combined_inhibitor_thresh" => "\\textbf{2 factors,}\\newline \\textbf{inducer →  }\\newline \\textbf{inhibitor}\\newline \\textbf{threshold}",
            "combined_inhibitor_perm" => "\\textbf{2 factors,}\\newline \\textbf{inducer →  }\\newline \\textbf{inhibitor}\\newline \\textbf{release}",
            "combined_inducer" => "\\textbf{2 factors,}\\newline \\textbf{inhibitor →  }\\newline \\textbf{inducer}\\newline \\textbf{thresh. + signal}",
            "combined_inducer_thresh" => "\\textbf{2 factors,}\\newline \\textbf{inhibitor →  }\\newline \\textbf{inducer}\\newline \\textbf{threshold}",
            "combined_inducer_signal" => "\\textbf{2 factors,}\\newline \\textbf{inhibitor →  }\\newline \\textbf{inducer}\\newline \\textbf{signal}",
            "special_inhibitor" => "\\textbf{Inducer →  }\\newline \\textbf{inhibitor}\\newline \\textbf{thresh. + release}\\newline \\textbf{(var. perm.)}",
            "special_inducer" => "\\textbf{Inhibitor →  }\\newline \\textbf{inducer}\\newline \\textbf{thresh.}\\newline \\textbf{(var. perm.)}",
            "special_independent" => "\\textbf{2 factors,}\\newline \\textbf{independent}\\newline \\textbf{(var. perm.)}",
            "special_combined" => "\\textbf{2 factors,}\\newline \\textbf{Inhibitor →  }\\newline \\textbf{inducer}\\newline \\textbf{thresh. + signal}\\newline \\textbf{(var. perm.)}",
            "special_combined_thresh" => "\\textbf{2 factors,}\\newline \\textbf{Inhibitor →  }\\newline \\textbf{inducer}\\newline \\textbf{threshold}\\newline \\textbf{(var. perm.)}",
            "special_combined_signal" => "\\textbf{2 factors,}\\newline \\textbf{Inhibitor →  }\\newline \\textbf{inducer}\\newline \\textbf{signal}\\newline \\textbf{(var. perm.)}"
        )

        if isnothing(model_types)
            label_keys_sort = ["independent", "inhibitor", "inhibitor_thresh", "inhibitor_perm", "inducer", "inducer_thresh", "inducer_signal",
                                "combined_inhibitor", "combined_inhibitor_thresh", "combined_inhibitor_perm",
                                "combined_inducer", "combined_inducer_thresh", "combined_inducer_signal",
                                "special_inhibitor", "special_inducer", "special_independent", "special_combined"]
        else
            label_keys_sort = model_types
        end

        params_dict = Dict(
            :Pₛ => (L"P_s^{\textrm{inh}}", L"\si{\micro\meter\per\second}"),
            :Pₛ_cs => (L"P_s^{\textrm{cs}}", L"\si{\micro\meter\per\second}"),
            :μ_γ => (L"\mu_\gamma", L"-"),
            :σ_γ => (L"\sigma_\gamma", L"-"),
            :k => (L"k", L"-"),
            :K_I => (L"K_I", L"\si{M}"),
            :K_cs => (L"K_{\textrm{cs}}", L"\si{M}"),
            :n => (L"n", L"-"),
            :s => (L"s", L"-"),
            :μ_ω => (L"\mu_\omega", L"-"),
            :σ_ω => (L"\sigma_\omega", L"-"),
            :μ_ψ => (L"\mu_\psi", L"\si{M}"),	
            :σ_ψ => (L"\sigma_\psi", L"\si{M}"),
            :μ_π => (L"\mu_\pi", L"\si{\micro\meter\per\second}"),
            :σ_π => (L"\sigma_\pi", L"\si{\micro\meter\per\second}"),
            :μ_α => (L"\mu_\alpha", L"-"),
            :σ_α => (L"\sigma_\alpha", L"-")
        )

        keys_sort = [:Pₛ, :Pₛ_cs, :k, :K_I, :K_cs, :n, :s, :μ_γ, :σ_γ, :μ_ω, :σ_ω, :μ_ψ, :σ_ψ, :μ_π, :σ_π, :μ_α, :σ_α]

        n_cols = length(label_keys_sort)
        n_rows = length(keys_sort)
        # tabular = Tabular("|p{1.5cm}" * "|p{4cm}"^(n_cols) * "|p{1.5cm}|")
        tabular = Tabular("p{2cm}" * "p{3.5cm}"^(n_cols) * "p{1.5cm}")

        # Find all result files
        cells = Array{LaTeXString}(undef, n_rows+1, n_cols+2)
        param_present = zeros(Bool, n_rows)
        for file in readdir(path)
            if startswith(file, "fit")
                file_name = splitext(file)[1]
                type_split = split(file_name, '_')

                # println("Processing file: $(file_name)")
                # println("Type split: $(type_split)")

                # Check if the file matches the extra tag
                if !isnothing(extra_tags) && !any(in.(extra_tags, Ref(type_split))) && !any(in.(["combined", "special", "independent", "inhibitor", "inducer"], Ref(type_split)))#!("combined" in type_split) && !("special" in type_split)
                    continue
                end

                # Remove suffixes
                if !isnothing(extra_tags)
                    type_split = filter(x -> !(x in extra_tags), type_split)
                end 
                # for extra_tag in extra_tags
                #     if extra_tag in type_split
                #         deleteat!(type_split, findfirst(x -> x == extra_tag, type_split))
                #     end
                # end
                
                model_type = join(type_split[2:end], '_')
                
                # Skip if model type is not in the specified list
                if !(model_type in label_keys_sort)
                    # println(model_type, " not in label_keys_sort, skipping...")
                    continue
                end

                params_opt = jldopen(joinpath(path, file), "r") do f
                    return f["params_opt"]
                end

                println("Processing model: $(model_type)")
                # println("Parameters: $(params_opt)")

                mt_idx = findfirst(x -> x == model_type, label_keys_sort)
                # cells[1, mt_idx+1] = latexstring(model_labels[model_type] * type_add)
                cells[1, mt_idx+1] = "Model: \\newline"  * latexstring(model_labels[model_type])

                for (i, key) in enumerate(keys_sort)
                    if haskey(params_opt, key)
                        value = params_opt[key]#round.(params_opt[key], digits=6)
                        val_str = [@sprintf("%.4e", val) for val in value]
                        val_str = ["\\num{$(val)}" for val in val_str]
                        if length(value) > 1
                            val_str[1] = val_str[1] * "\\,^*"
                            val_str[2] = val_str[2] * "\\,^{**}"
                            val_joint = join(val_str, ", \\newline")
                            val_joint = latexstring(val_joint)
                        else
                            val_joint = latexstring(val_str[1])
                        end
                        cells[i+1, mt_idx + 1] = val_joint
                        param_present[i] = true
                    else
                        cells[i+1, mt_idx + 1] = L" "
                    end
                end
            end
        end

        # Add parameter names to first column and units to the last column
        for (i, key) in enumerate(keys_sort)
            if haskey(params_dict, key)
                param = params_dict[key][1]
                cells[i+1, 1] = latexstring(param)
                unit = params_dict[key][2]
                cells[i+1, end] = latexstring(unit)
            else
                cells[i+1, end] = L" "
            end
        end

        cells[1, 1] = latexstring("\\textbf{Parameter}")
        cells[1, end] = latexstring("\\textbf{Units}")

        header = cells[1, :]
        rows = []
        for i in 2:(n_rows+1)
            if param_present[i-1]
                push!(rows, Rule(:mid))
                push!(rows, cells[i, :])
            end
        end

        # Create LaTeX table
        return latex_tabular(String, tabular, [Rule(:top), header, rows..., Rule(:bottom)])
        
    end
end