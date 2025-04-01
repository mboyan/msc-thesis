module Setup
__precompile__(false)
    """
    Contains setup functions
    """

    using ArgCheck
    using IterTools
    using JLD2
    using SHA
    using Serialization
    using CurveFit
    using Revise

    include("./conversions.jl")
    include("./diffusion.jl")
    Revise.includet("./conversions.jl")
    Revise.includet("./diffusion.jl")
    using .Conversions
    using .Diffusion

    export setup_spore_cluster
    export run_simulation
    export run_simulations
    export setup_model_comparison

    function setup_spore_cluster(n_nbrs::Int, L, spore_rad::Float64, cut_half::Bool=false)
        """
        Compute the centers of a cluster of spheres,
        one placed in the center of the lattice and the rest
        placed at the vertices of a regular n_nbrs-gon.
        inputs:
            n_nbrs (int): number of neighbors
            L (int): size of the lattice
            spore_rad (float): radius of the spores
            cut_half (bool): whether to cut the cluster in half
        outputs:
            spore_centers (Array): centers of the spores
        """

        @argcheck n_nbrs in [0, 2, 3, 4, 6, 8, 12] "n_nbrs must be in [2, 3, 4, 6, 8, 12]"
        @argcheck L > 0 "N must be positive"

        # Safety distance
        # spore_rad += 0.5

        spore_dia = 2 * spore_rad
        center = [0.5 * L, 0.5 * L, 0.5 * L]

        spore_centers = zeros(n_nbrs + 1, 3)
        spore_centers[1, :] = center

        if n_nbrs == 2
            # Two neighbours at poles
            spore_centers[2, :] = [center[1], center[2], center[3] + spore_dia]
            spore_centers[3, :] = [center[1], center[2], center[3] - spore_dia]
        elseif n_nbrs == 3
            # Three neighbours at equilateral triangle
            for i in 1:3
                spore_centers[i + 1, :] = [center[1] + spore_dia * cos(2 * π * i / 3),
                                           center[2],
                                           center[3] + spore_dia * sin(2 * π * i / 3)]
            end
        elseif n_nbrs == 4
            # Four neighbours at tetrahedron vertices
            cp = spore_dia/sqrt(3)
            cm = -cp
            spore_centers[2, :] = [center[1] + cp, center[2] + cp, center[2] + cp]
            spore_centers[3, :] = [center[1] + cm, center[2] + cm, center[2] + cp]
            spore_centers[4, :] = [center[1] + cp, center[2] + cm, center[2] + cm]
            spore_centers[5, :] = [center[1] + cm, center[2] + cp, center[2] + cm]
        elseif n_nbrs == 6
            # Six neighbours at octahedron vertices
            coords = [spore_dia, 0, 0]
            for i in 1:3
                spore_centers[2*i, :] = center .+ coords
                spore_centers[2*i + 1, :] = center .- coords
                coords = circshift(coords, 1)
            end
        elseif n_nbrs == 8
            # Eight neighbours at cube vertices
            for i in 1:4
                spore_centers[2*i, :] = [center[1] + spore_dia * sqrt(2/3) * cos(2 * π * i / 4),
                                         center[2] + spore_dia * sqrt(2/3) * sin(2 * π * i / 4),
                                         center[3] + spore_dia * sqrt(1/3)]
                spore_centers[2*i + 1, :] = [center[1] + spore_dia * sqrt(2/3) * cos(2 * π * i / 4),
                                             center[2] + spore_dia * sqrt(2/3) * sin(2 * π * i / 4),
                                             center[3] - spore_dia * sqrt(1/3)]
            end
        elseif n_nbrs == 12
            # Twelve neighbours at icosahedron vertices
            phi = (1 + sqrt(5)) / 2
            norm_coeff = spore_dia / sqrt(phi + 2)
            coords = [0, norm_coeff, phi * norm_coeff]
            for i in 1:12
                coords[1 + mod1(i, 2)] = -coords[1 + mod1(i, 2)]
                coordshift = circshift(coords, floor(Int, i/4))
                spore_centers[i + 1, :] = center .+ coordshift
            end
        end

        # Take only the upper half of the cluster`
        if cut_half
            mask = spore_centers[:, 3] .≤ center[3]
            spore_centers = spore_centers[mask, :]
        end

        # Convert to list of tuples
        if typeof(L) == Int
            spore_centers = [(Int(round(x)), Int(round(y)), Int(round(z))) for (x, y, z) in eachrow(spore_centers)]
        else
            spore_centers = [(x, y, z) for (x, y, z) in eachrow(spore_centers)]
        end

        return spore_centers
    end


    function run_simulation(t_max, sim_params::Dict)
        """
        Run a diffusion simulation with the given parameters.
        inputs:
            t_max (float): maximum time
            sim_params (Dict): simulation parameters
        outputs:
            c_solutions (Array): concentration solutions
            c_frames (Array): concentration frames
            times (Array): time points
            coverage (float): coverage
            exponent (Array): fitted exponents
        """

        # Load simulation parameters
        N = sim_params[:N]
        dx = sim_params[:dx]
        dt = sim_params[:dt]
        t_max = sim_params[:t_max]
        D = sim_params[:D]
        Pₛ = sim_params[:Ps]
        c₀ = sim_params[:c0]
        sim_res = sim_params[:sim_res]
        n_save_frames = sim_params[:n_save_frames]

        @argcheck sim_res in ["low", "medium", "high"] "sim_res must be 'low', 'medium', or 'high'"

        # Spore diameter
        if haskey(sim_params, :spore_diameter)
            spore_diameter = sim_params[:spore_diameter]
        else
            spore_diameter = 5.0 # Default value in microns
        end
        spore_rad = spore_diameter / 2.0
        A_spore = 4 * π * spore_rad^2
        V_spore = 4/3 * π * spore_rad^3

        # Lattice height
        if haskey(sim_params, :H)
            H = sim_params[:H]
        else
            H = N # Default value
        end

        # Partition coefficient
        if haskey(sim_params, :K)
            K = sim_params[:K]
        else
            K = 1.0 # Default value
        end

        # Absorbing boundary
        if haskey(sim_params, :abs_bndry)
            abs_bndry = sim_params[:abs_bndry]
        else
            abs_bndry = false # Default value
        end

        # Cluster size
        if haskey(sim_params, :cluster_size)
            cluster_size = 1 # Default value
            cluster_params = (0, false)
        else
            cluster_size = sim_params[:cluster_size]
            if sim_res == :high
                # Translate cluster size to neighbour arrangement parameters
                if cluster_size == 1
                    cluster_params = (0, false)
                elseif cluster_size == 2
                    cluster_params = (2, true)
                elseif cluster_size == 3
                    cluster_params = (2, false)
                elseif cluster_size == 6
                    cluster_params = (6, true)
                elseif cluster_size == 7
                    cluster_params = (6, false)
                else
                    error("Invalid cluster size for high resolution")
                end
            end
        end

        # Cluster center-to-center distance
        if haskey(sim_params, :dist)
            dist = sim_params[:dist]
        else
            dist = spore_diameter # Default value
        end

        # Run simulation
        c_init = zeros(Float64, N, N, H)
        if sim_res == "low"
            sp_cen_indices = setup_spore_cluster(2, N, 0.5 * dist / dx, true) # additive single neighbour contributions
            coverage = cluster_size * measure_coverage(sp_cen_indices[1], sp_cen_indices[2:end], rad=spore_rad, dx=dx)
            c_med_evolution, c_solutions, times, _ = diffusion_time_dependent_GPU_low_res(c_init, c₀, t_max; D=D, Pₛ=Pₛ, A=A_spore, V=V_spore, dt=dt, dx=dx,
                                                                                            n_save_frames=n_save_frames, spore_vol_idx=sp_cen_indices[1],
                                                                                            cluster_size=cluster_size, cluster_spacing=dist)
            frame_samples = c_med_evolution[:, :, N ÷ 2, :]
        elseif sim_res == "medium"
            sp_cen_indices = setup_spore_cluster(2, N, 0.5 * dist / dx, true) # additive single neighbour contributions
            coverage = cluster_size * measure_coverage(sp_cen_indices[1], sp_cen_indices[2:end], rad=spore_rad, dx=dx)
            c_init[sp_cen_indices[1]...] = c₀
            c_evolution, times = diffusion_time_dependent_GPU!(c_init, t_max; D=D, Ps=Pₛ, dt=dt, dx=dx,
                                                                n_save_frames=n_save_frames, spore_idx=sp_cen_indices[1],
                                                                cluster_size=cluster_size, cluster_spacing=dist)
            c_solutions = c_evolution[:, sp_cen_indices[1]...]
            frame_samples = c_evolution[:, :, N ÷ 2, :]
        elseif sim_res == "high"
            Db = Pₛ * dx / K
            sp_cen_indices = setup_spore_cluster(cluster_params[1], N, 0.5 * dist / dx + 0.5, cluster_params[2]) # with safety radius of 0.5
            coverage = measure_coverage(sp_cen_indices[1], sp_cen_indices[2:end], rad=spore_rad, dx=dx)
            frame_samples, c_solutions, times, region_ids, _ = diffusion_time_dependent_GPU_hi_res_implicit(c_init, c₀, sp_cen_indices, spore_rad, t_max;
                                                                                                            D=D, Db=Db, dt=dt, dx=dx, n_save_frames=n_save_frames,
                                                                                                            crank_nicolson=false, abs_bndry=abs_bndry)
        end

        fit = exp_fit(times, c_solutions)
        exponent = fit[2]

        return c_solutions, frame_samples, times, coverage, exponent
    end

    
    function run_simulations(exp_ID, t_max, sim_params::Dict)
        """
        Recognizes which parameter in the dictionary contains 
        a range of value and runs simulations for all combinations
        of parameter values.
        inputs:
            expID (int): experiment ID
            t_max (float): maximum time
            sim_params (Dict): simulation parameters
        """

        path = @__DIR__

        # Create experiment directory if it doesn't exist
        if !isdir("$(path)/Data/$(exp_ID)")
            mkdir("$(path)/Data/$(exp_ID)")
        end
        
        # Gather varying parameters
        param_keys = []
        param_values = []
        for (param, values) in sim_params
            if !(typeof(values) == String) && length(values) > 1
                push!(param_keys, param)
                push!(param_values, values)
            end
        end

        # Create all combinations of parameters
        param_combos = Base.Iterators.product(param_values...)
        param_combos = collect(param_combos)
        param_indices = [collect(1:length(vals)) for vals in param_values]
        param_IDs = [string(param_keys[i]) .* "_" .* string.(param_indices[i]) for i in eachindex(param_keys)]
        param_ID_combos = Base.Iterators.product(param_IDs...)
        param_ID_combos = collect(param_ID_combos)

        # Run simulations for each combination of parameters
        for (i, param_combo) in enumerate(param_combos)

            # Create a unique simulation ID for each combination of parameters
            sim_ID = join(param_ID_combos[i], "_")

            # Create a new dictionary for the current combination of parameters
            sim_params_comb = deepcopy(sim_params)
            for (j, param) in enumerate(param_keys)
                sim_params_comb[param] = param_combo[j]
                println("$(param_keys[j]) = $(param_combo[j])")
            end

            # Create a hash string from the parameters
            buffer = IOBuffer()
            serialize(buffer, sim_params_comb)
            params_hash = bytes2hex(sha256(take!(buffer)))

            # Check if the parameter combination already exists
            if isfile(joinpath(path, "Data", exp_ID, "$(exp_ID)_sims.jld2"))
                # Load existing simulation data
                jldopen(joinpath(path, "Data", exp_ID, "$(exp_ID)_sims.jld2"), "r+") do file
                    sim_data = file["simulations"]
                    if params_hash in sim_data["sim_hashes"]
                        # Find index of the existing simulation
                        idx = findfirst(x -> x == params_hash, sim_data["sim_hashes"])
                        sim_ID_found = sim_data["sim_IDs"][idx]
                        println("Simulation with parameters $sim_params_comb already exists with simID $sim_ID_found. Overwriting it.")
                        # continue
                    else
                        # Append new simulation data
                        push!(file["simulations/sim_IDs"], sim_ID)
                        push!(file["simulations/sim_hashes"], params_hash)
                        println("New simulation with parameters $sim_params_comb added with simID $sim_ID.")
                    end
                end
            else
                # Create a new file for simulation data
                jldopen(joinpath(path, "Data", exp_ID, "$(exp_ID)_sims.jld2"), "w") do file
                    simulations = JLD2.Group(file, "simulations")
                    simulations["sim_IDs"] = [sim_ID]
                    simulations["sim_hashes"] = [params_hash]
                end
            end

            # Save parameters to a file
            jldopen(joinpath(path, "Data", exp_ID, "$(sim_ID)_params.jld2"), "w") do file
                file["parameters"] = sim_params_comb
            end

            # Run the simulation with the current combination of parameters
            c_solutions, frame_samples, times, coverage, exponent = run_simulation(t_max, sim_params_comb)

            # Save the results to a file
            jldopen(joinpath(path, "Data", exp_ID, "$(sim_ID)_results.jld2"), "w") do file
                file["c_solutions"] = c_solutions
                file["frame_samples"] = frame_samples
                file["times"] = times
                file["coverage"] = coverage
                file["exponent"] = exponent
            end
        end
    end


    function setup_model_comparison(exp_ID, t_max, sim_params::Union{Vector{Dict{Symbol, Any}}, Array{Dict{Symbol, Any}}})
        """
        Set up and run a model comparison experiment.
        inputs:
            exp_ID (int): experiment ID
            t_max (float): maximum time
            sim_params (Array): array of simulation parameters for each model
        """

        # Run simulations for each set of parameters
        for (i, params) in enumerate(sim_params)
            exp_ID_extended = exp_ID * "_model_$(i)"
            run_simulations(exp_ID_extended, t_max, params)
        end
    end

end