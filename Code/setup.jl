module Setup
__precompile__(false)
    """
    Contains setup functions
    """

    using ArgCheck
    using IterTools
    using JLD2

    # using .Conversions
    # using .Diffusion

    export setup_spore_cluster

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

        @argcheck n_nbrs in [2, 3, 4, 6, 8, 12] "n_nbrs must be in [2, 3, 4, 6, 8, 12]"
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

    function run_simulation(sim_ID, max_time; sim_params::Dict)
        """
        Run a diffusion simulation with the given parameters.
        inputs:
            sim_ID (int): simulation ID
            max_time (float): maximum time
            sim_params (Dict): simulation parameters
        outputs:
            c_solutions (Array): concentration solutions
            c_frames (Array): concentration frames
            exponents (Array): fitted exponents
        """

        # Load simulation parameters
        N = sim_params["N"]
        H = sim_params["H"]
        dx = sim_params["dx"]
        dt = sim_params["dt"]
        t_max = sim_params["t_max"]
        D = sim_params["D"]
        Pₛ = sim_params["Ps"]
        c₀ = sim_params["c0"]
        sim_res = sim_params["sim_res"]

        @argcheck sim_res in ["low", "medium", "high"] "sim_res must be in [\"low\", \"medium\", \"high\"]"

        if isnothing(sim_params["spore_diameter"])
            spore_diameter = 5.0 # Default value in microns
        else
            spore_diameter = sim_params["spore_diameter"]
        end

        if isnothing(sim_params["K"])
            K = 1.0 # Default value
        else
            K = sim_params["K"]
        end

        if !isnothing(sim_params["cluster_size"])
            cluster_size = sim_params["cluster_size"]
            if sim_res == "high"
                # Translate cluster size to neighbour arrangement parameters
                if cluster_size == 1
                    cluster_params = (2, true)
                elseif cluster_size == 2
                    cluster_params = (2, false)
                elseif cluster_size == 5
                    cluster_params = (6, true)
                elseif cluster_size == 6
                    cluster_params = (6, false)
                else
                    error("Invalid cluster size for high resolution")
                end
            end
        end
    end

    function setup_simulation_segment(exp_ID, sim_ID, max_time, exp_params=nothing, sim_params=nothing)
        """
        Set up and run a time segment of a simulation and append
        the results to the simulation data.
        inputs:
            exp_ID (int): experiment ID
            sim_ID (int): simulation ID
            max_time (float): maximum time
            exp_params (Dict): experiment parameters
            sim_params (Dict): simulation parameters
        """

        path = @__DIR__

        if isdir("$(path)/Data/$(exp_ID)")
            if isfile("$(path)/Data/$(exp_ID)/$(sim_ID).jld2")
                # Load simulation data
                jldopen("$(path)/Data/$(exp_ID)/$(sim_ID).jld2", "r+") do file
                    times = file["times"]
                end
            else
                if isnothing(sim_params)
                    error("File not found, simulation parameters must be supplied.")
                end
            end
        else
            if isnothing(exp_params) || isnothing(sim_params)
                error("Directory not found, experiment and simulation parameters must be supplied.")
            else
                # Create directory and save parameters
                mkdir("$(path)/Data/$(exp_ID)")
                jldsave("$(path)/Data/$(exp_ID)/$(sim_ID).jld2", Dict("exp_params" => exp_params, "sim_params" => sim_params))
            end
        end
    end

end