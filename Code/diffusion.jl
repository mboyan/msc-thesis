module Diffusion
__precompile__(false)
    """
    Functions for solving the diffusion equation.
    """

    using ArgCheck
    using CUDA
    using CUDA.CUSPARSE
    using CUDA.CUSOLVER
    using SparseArrays
    using Krylov
    using LinearAlgebra
    using SpecialFunctions
    using QuadGK
    using IterTools
    using StatsBase
    using Revise

    include("./solver.jl")
    include("./conversions.jl")
    Revise.includet("./solver.jl")
    Revise.includet("./conversions.jl")
    using .Solver
    using .Conversions

    export permeation_time_dependent_analytical
    export diffusion_time_dependent_analytical_src
    export slow_release_src_grid
    export slow_release_src_grid_src
    export compute_permeation_constant
    export diffusion_time_dependent_GPU!
    export diffusion_time_dependent_GPU_low_res
    export diffusion_time_dependent_GPU_hi_res!
    export diffusion_time_dependent_GPU_hi_res_implicit
    
    # ===== ANALYTICAL SOLUTIONS =====
    function permeation_time_dependent_analytical(c_in, c_out, t, Ps, A, V; alpha=1.0)
        """
        Compute the concentration of a solute in a spore given the initial and external concentrations.
        inputs:
            c_in (float) - the initial concentration at the spore
            c_out (float) - the initial external concentration
            t (float) - time
            Ps (float) - the spore membrane permeation constant
            A (float) - the surface area of the spore
            V (float) - the volume of the spore
            alpha (float) - permeable fraction of the area; defaults to 1
        """
        tau = V / (alpha * A * Ps)
        c = c_out .- (c_out .- c_in) .* exp.(-t / tau)
        return c
    end


    function diffusion_time_dependent_analytical_src(c₀, D, time, vol; dims=3)
        """
        Compute the analytical solution of the time-dependent diffusion equation at the source.
        inputs:
            c₀ (float) - the initial concentration
            D (float) - the diffusion constant
            time (float) - the time at which the concentration is to be computed
            vol (float) - the volume of the initial concentration source
        """
        if dims == 2
            result = (vol * c₀) ^ (2/3) / (4*π*D*time)
        elseif dims == 3
            result = vol * c₀ / (4*π*D.*time)^(1.5)
        end
        result = [isnan(x) || isinf(x) ? c₀[i] : x for (i, x) in enumerate(result)]
        return result
    end


    # function aggregation_time_integral(t, τ, D, r)
    #     return t^(-3/2) * exp(t/τ - r^2/(4 * D * t))
    # end

    function aggregation_time_integral(r, t, Ps, A, V, D)
        ϵ = 1e-12
        if t < ϵ
            val = 1.0
        else
            integrand(τ) = τ^(-3/2) * exp((Ps * A / V) * τ - (r^2) / (4 * D * τ))
            # Perform integration over τ from 0 to t
            val, err = quadgk.(integrand, ϵ, t, rtol=1e-8)
        end
        return val
    end
    
    function aggregation_space_integral(t, Ps, A, V, D, ρₛ, R_diff)
        integrand_r(r) = r^2 * aggregation_time_integral(r, t, Ps, A, V, D)
        # Integrate over r from 0 to R_diff
        val, err = quadgk(integrand_r, 0, R_diff, rtol=1e-8)
        return 4 * π * ρₛ * val
    end


    function slow_release_src_grid(x, src_density, c₀, times, D, Pₛ, A, V; discrete=true)
        """
        Compute the concentration at sampling points
        due to periodically repeating permeating sources within an
        effective diffusion radius.
        inputs:
            x (array of tuples): 3D positions of the observation points
            src_density (float): number density of sources in spores/mL
            times (array of floats): times at which to compute the concentration
            c₀ (float): initial concentration at the sources in mol/L
            D (float): diffusion coefficient in um^2/s
            Pₛ (float): source release rate in um/s
            A (float): source area in um^2
            V (float): source volume in um^3
            dt (float): time step size in seconds
            discrete (bool): whether to compute the neighbour contributions discretely or continuously
        """

        # Conversions
        src_density = inverse_mL_to_cubic_um(src_density)

        # Constants
        τ = V / (A * Pₛ)
        ϵ = 1e-12
        prefactors = c₀ * Pₛ * A / (4 * π * D)^(3/2) .* exp.(-times./τ)
        prefactors = repeat(prefactors, 1, size(x)[1])
        prefactors = permutedims(prefactors)
        # println(prefactors)

        # Find source grid spacing
        vol_src_cell = 1 / src_density
        dx = vol_src_cell^(1/3)
        R_max = sqrt(6 * D * maximum(times))
        # println("Maximum radius: $R_max, dx: $dx")
        if R_max / dx > 50 && discrete
            println("Warning: large number of subdivisions. Switching to continuous mode")
            discrete = false
        end

        # Iterate over time frames
        src_sums = zeros(size(x)[1], size(times)[1])
        # n_nbrs = zeros(size(times)[1])
        for (i, t) in enumerate(times)
            # Find diffusion radius
            R = sqrt(6 * D * t)
            
            if discrete

                # Generate relevant source positions
                if R > dx
                    src_x = vec(0:dx:R)
                    src_x = [reverse(-src_x[2:end]); src_x]
                    src_pts = vec([(x, y, z) for x in src_x, y in src_x, z in src_x])
                else
                    src_pts = [(0, 0, 0)]
                end
                src_pts = src_pts[norm.(src_pts) .≤ R]
                # println("Number of source points at R=$R: ", length(src_pts))

                # Compute distances
                src_distances = [map(x_pt -> norm(x_pt .- src_pt), x) for src_pt in src_pts]

                # Extract unique distances and their counts
                counts = countmap(src_distances)
                src_dist_unique = collect(keys(counts))
                unique_counts = collect(values(counts))

                # Reshape unique counts to match the size of `x`
                unique_counts = permutedims(repeat(unique_counts, 1, size(x, 1)))

                # Convert unique distances to a matrix
                src_dist_unique = hcat(src_dist_unique...)

                # Sum source contributions
                if t < ϵ
                    time_integral = 1.0
                else
                    time_integral = aggregation_time_integral.(src_dist_unique, t, Pₛ, A, V, D)
                end
                contributions = unique_counts .* sum((A * Pₛ * c₀ / (4π * D)^(3/2)) * exp(-t/τ) * time_integral)
                src_sums[:, i] = sum(contributions, dims=2)
            else
                # Compute the integral over the source grid
                src_sums[:, i] .= aggregation_space_integral(t, Pₛ, A, V, D, src_density, R)
            end
        end

        results = prefactors .* src_sums
        
        return results
    end


    function slow_release_src_grid_src(src_density, c₀, times, D, Pₛ, A, V; discrete=true)
        """
        Compute the concentration inside a spore source
        due to periodically repeating permeating sources within an
        effective diffusion radius.
        inputs:
            x (array of tuples): 3D positions of the observation points
            t_max (float): maximum time of observation in seconcs
            src_density (float): number density of sources in spores/mL
            c0 (float): initial concentration at the sources in mol/L
            D (float): diffusion coefficient in um^2/s
            Ps (float): source release rate in um/s
            A (float): source area in um^2
            V (float): source volume in um^3
            dt (float): time step size in seconds
            discrete (bool): whether to compute the neighbour contributions discretely or continuously
        """

        # Constants
        τ = V / (A * Pₛ)

        # c_ins = zeros(size(times, 1))
        decay = exp.(-times./τ)
        c_out = slow_release_src_grid([(0, 0, 0)], src_density, c₀, times, D, Pₛ, A, V, discrete=discrete)[1, :]
        c_ins = decay .* (c₀ .+ c_out)

        return c_ins
    end

    function compute_permeation_constant(c_in_target, c_out, c_0, t, A, V; alpha=1.0)
        """
        Compute the permeation constant P_s given the constant external concentration,
        the initial concentration of the solute, the concentration of the solute at time t
        and the surface area and volume of the spore.
        inputs:
            c_in (float) - the concentration at the spore at time t
            c_out (float) - the constant external concentration
            c_0 (float) - the initial concentration of the solute at the spore
            t (float) - time
            A (float) - the surface area of the spore
            V (float) - the volume of the spore
            alpha (float) - permeable fraction of the area; defaults to 1
        """
        Ps = V / (alpha * A * t) * log((c_out - c_0) / (c_out - c_in_target))
        return Ps
    end

    # ===== NUMERICAL SOLUTIONS =====
    function diffusion_time_dependent_GPU!(c_init, t_max; D=1.0, Db=nothing, Ps=1.0, dt=0.005, dx=5, n_save_frames=100,
        spore_idx=nothing, c_thresholds=nothing, abs_bndry=false, neumann_z=false, cluster_size=1, cluster_spacing=10, nondim=false)
        """
        Compute the evolution of a square lattice of concentration scalars
        based on the time-dependent diffusion equation.
        inputs:
            c_init (vector of float) - the initial state of the lattice
            t_max (int) - a maximum number of iterations
            D (float) - the diffusion constant; defaults to 1
            Db (float) - the diffusion constant through the spore; defaults to nothing
            Ps (float) - the permeation constant through the spore barrier; defaults to 1
            dt (float) - timestep; defaults to 0.001
            dx (float) - spatial increment; defaults to 0.005
            n_save_frames (int) - determines the number of frames to save during the simulation; detaults to 100
            spore_idx (tuple) - the indices of the spore location; defaults to nothing
            spore_spacing (int) - the spacing between spore indices along each dimension; defaults to nothing; if used, spore_idx is ignored
            c_thresholds (vector of float) - threshold values for the concentration; defaults to nothing
            abs_bndry (bool) - whether to use absorbing boundary conditions; defaults to false
            neumann_z (bool) - whether to use Neumann boundary conditions in the z-direction; defaults to false
            cluster_size (int) - if provided, creates an orthogonal neighbour cluster with the given size
            cluster_spacing (int) - spacing between the spores in the cluster
            nondim (bool) - whether to use nondimensionalisation; defaults to false
        outputs:
            c_evolotion (array) - the states of the lattice at all moments in time
            times (array) - the times at which the states were saved
            t_thresholds (array) - the times at which the concentration crossed the threshold
        """

        @argcheck ndims(c_init) == 3 "c_init must be a 3D array"
        @argcheck 0 < cluster_size ≤ 7 "Cluster size must be less than or equal to 7"

        GC.gc()

        # Determine number of lattice rows/columns
        N = size(c_init)[1]
        H = size(c_init)[3]

        # Set spore index
        if isnothing(spore_idx)
            spore_idx = (N ÷ 2, N ÷ 2, H ÷ 2)
        end

        # Set up spore cluster
        if cluster_size > 1
            # Generate possible combinations
            cluster_sp_lattice = cluster_spacing ÷ dx
            spacing_combos = [-cluster_sp_lattice, 0, cluster_sp_lattice]
            spore_cluster = vec([(i, j, k) for i in spacing_combos, j in spacing_combos, k in spacing_combos])
            spore_cluster = spore_cluster[norm.(spore_cluster) .== cluster_sp_lattice]
            spore_cluster = [Int.(round.(spore_idx .+ spore)) for spore in spore_cluster]

            # Filter out relevant neighbours
            order_indices = [1, 6, 2, 5, 3, 4]
            order_indices = order_indices[1:(cluster_size-1)]
            spore_cluster = spore_cluster[order_indices]
            println("Generating $(length(order_indices)) neighbours")
            for spc in spore_cluster
                c_init[spc...] = c_init[spore_idx...]
            end
        end

        # Non-dimensionalise
        if nondim
            lengthscale = dx
            timescale = lengthscale / Ps
            D = D / (Ps * lengthscale)
            dx = 1.0
            dt = dt / timescale
            t_max = t_max / timescale
            Ps = 1.0
            println("Parameters rescaled to D = $D, Ps = $Ps, dt = $dt, dx = $dx, t_max = $t_max")
        end

        # Save update factor
        # dtdx2 = Float32(dt / (dx^2))
        dtdx2 = dt / (dx^2)

        # Correction factor for permeation
        if isnothing(Db)
            Db = Ps * dx
        end
        println("Using D = $D, Db = $Db, Ps = $Ps")

        # Check stability
        if  D * dtdx2 ≥ 0.2
            println("Warning: inappropriate scaling of dx and dt due to D, may result in an unstable simulation; Ddt/dx2 = $(D*dtdx2).")
        end
        if  Db * dtdx2 ≥ 0.2
            println("Warning: inappropriate scaling of dx and dt due to Db, may result in an unstable simulation; Dbdt/dx2 = $(Db*dtdx2).")
        end

        # Determine number of frames
        n_frames = Int(floor(t_max / dt))

        # Correct number of frames to save
        if n_frames < n_save_frames
            println("Correcting number of frames to save.")
            n_save_frames = n_frames
        end

        # Allocate arrays for saving data
        c_evolution = zeros(n_save_frames + 1, N, N, H)
        times = zeros(n_save_frames + 1)
        println("Storage arrays allocated.")
        save_interval = floor(n_frames / n_save_frames)
        save_ct = 1

        # Allocate arrays for saving threshold crossing times
        if isnothing(c_thresholds)
            t_thresholds = nothing
        else
            t_thresholds = zeros(length(c_thresholds))
        end
        thresh_ct = 0

        # Initialise lattice states on GPU
        c_A_gpu = cu(c_init)
        c_B_gpu = CUDA.zeros(N, N, H)

        kernel_blocks, kernel_threads = invoke_smart_kernel_3D(size(c_init))

        # Run the simulation
        for t in 1:n_frames
            
            # Save frame
            if (t - 1) % save_interval == 0 && save_ct ≤ n_save_frames
                c_evolution[save_ct, :, :, :] .= Array(c_A_gpu)
                times[save_ct] = (t - 1) * dt
                # print("\rFrame $save_ct saved.")
                # println(maximum(c_evolution[save_ct, :, :, :]))
                # println("Concentration at center: ", c_evolution[save_ct, spore_idx...])
                # if cluster_size > 1
                #     for spc in spore_cluster
                #         println("Concentration at neighbour $(spc): ", c_evolution[save_ct, spc...])
                #     end
                #     println("======")
                # end
                save_ct += 1
            end

            # Update the lattice
            if cluster_size == 1
                @cuda threads=kernel_threads blocks=kernel_blocks update_GPU!(c_A_gpu, c_B_gpu, N, H, dtdx2, D, Db, spore_idx, abs_bndry, neumann_z)
            else
                @cuda threads=kernel_threads blocks=kernel_blocks update_GPU_spore_cluster!(c_A_gpu, c_B_gpu, N, H, dtdx2, D, Db, spore_idx, cluster_sp_lattice, cluster_size, abs_bndry, neumann_z)
            end
            c_A_gpu, c_B_gpu = c_B_gpu, c_A_gpu

            # Check for threshold crossing
            if !isnothing(c_thresholds)
                if thresh_ct < length(t_thresholds) && t_thresholds[thresh_ct] == 0 && CUDA.reduce(max_reduce_kernel, c_A_gpu) < c_thresholds[thresh_ct]
                    t_thresholds[thresh_ct] = t * dt
                    thresh_ct += 1
                end
            end
        end

        # Save final frame
        c_evolution[save_ct, :, :, :] .= Array(c_A_gpu)
        times[save_ct] = t_max

        # Re-dimensionalise
        if nondim
            times .= times .* timescale
        end

        return c_evolution, times, t_thresholds
    end

    function diffusion_time_dependent_GPU_low_res(c_init, c₀, t_max; D=1.0, Pₛ=1.0, A=150, V=125, dt=150, dx=25, n_save_frames=100,
                                                    spore_vol_idx=nothing, c_thresholds=nothing, neumann_z=false, cluster_size=1, cluster_spacing=10, nondim=false)
        """
        Compute the evolution of a square lattice of concentration scalars
        based on the time-dependent diffusion equation. A concentration source
        smaller than the lattice resolution is injecting new concentration
        according to a slow permeation scheme.
        inputs:
            c_init (vector of float) - the initial state of the lattice
            c0 (float) - the initial concentration at the spore
            t_max (int) - a maximum number of iterations
            D (float) - the diffusion constant; defaults to 1
            Ps (float) - the permeation constant through the spore barrier; defaults to 1
            A (float) - the surface area of the spore; defaults to 150
            V (float) - the volume of the spore; defaults to 125
            dt (float) - timestep; defaults to 0.001
            dx (float) - spatial increment; defaults to 0.005
            n_save_frames (int) - determines the number of frames to save during the simulation; detaults to 100
            spore_vol_idx (tuple) - the indices of the volume containing the spore location; defaults to nothing
            c_thresholds (vector of float) - threshold values for the concentration; defaults to nothing
            neumann_z (bool) - whether to use Neumann boundary conditions in the z-direction; defaults to false
            cluster_size (int) - if provided, creates an orthogonal neighbour cluster with the given size
            cluster_spacing (int) - spacing between the spores in the cluster
            nondim (bool) - whether to use nondimensionalisation; defaults to false
        outputs:
            c_evolotion (array) - the states of the lattice at all moments in time
            times (array) - the times at which the states were saved
            t_thresholds (array) - the times at which the concentration crossed the threshold
        """

        @argcheck ndims(c_init) == 3 "c_init must be a 3D array"
        @argcheck cluster_size ≤ 7 "Cluster size must be less than or equal to 6"

        GC.gc()

        # Determine number of lattice rows/columns
        N = size(c_init)[1]
        H = size(c_init)[3]

        # Set spore volume index
        if isnothing(spore_vol_idx)
            spore_vol_idx = (N ÷ 2, N ÷ 2, H ÷ 2)
        end

        # Non-dimensionalise
        if nondim
            lengthscale = dx
            timescale = lengthscale / Ps
            D = D / (Pₛ * lengthscale)
            dx = 1.0
            dt = dt / timescale
            t_max = t_max / timescale
            Pₛ = 1.0
            println("Parameters rescaled to D = $D, Pₛ = $Pₛ, dt = $dt, dx = $dx, t_max = $t_max")
        end

        # Save update factors
        inv_dx2 = 1 / (dx^2)
        inv_tau = A * Pₛ / V
        println("Using D = $D, Ps = $Pₛ")

        # Check stability
        if  D * dt * inv_dx2 ≥ 0.2
            println("Warning: inappropriate scaling of dx and dt due to D, may result in an unstable simulation; Ddt/dx2 = $(D*dt*inv_dx2).")
        end

        # Determine number of frames
        n_frames = Int(floor(t_max / dt))

        # Correct number of frames to save
        if n_frames < n_save_frames
            println("Correcting number of frames to save.")
            n_save_frames = n_frames
        end

        # Allocate arrays for saving data
        c_spore_array = zeros(N, N, H)
        c_spore_array[spore_vol_idx...] = c₀
        c_med_evolution = zeros(n_save_frames + 1, N, N, H)
        c_spore_evolution = zeros(n_save_frames + 1)
        times = zeros(n_save_frames + 1)
        println("Storage arrays allocated.")
        save_interval = floor(n_frames / n_save_frames)
        save_ct = 1

        # Set up spore cluster
        if cluster_size > 1
            # Generate possible combinations
            cluster_sp_lattice = cluster_spacing ÷ dx
            spacing_combos = [-cluster_sp_lattice, 0, cluster_sp_lattice]
            spore_cluster = vec([(i, j, k) for i in spacing_combos, j in spacing_combos, k in spacing_combos])
            spore_cluster = spore_cluster[norm.(spore_cluster) .== cluster_sp_lattice]
            spore_cluster = [Int.(round.(spore_vol_idx .+ spore)) for spore in spore_cluster]

            # Filter out relevant neighbours
            order_indices = [1, 6, 2, 5, 3, 4]
            order_indices = order_indices[1:(cluster_size-1)]
            spore_cluster = spore_cluster[order_indices]
            println("Generating $(length(order_indices)) neighbours")
            for spc in spore_cluster
                c_spore_array[spc...] = c₀
            end
        end

        # Allocate arrays for saving threshold crossing times
        if isnothing(c_thresholds)
            t_thresholds = nothing
        else
            t_thresholds = zeros(length(c_thresholds))
        end
        thresh_ct = 0

        # Initialise lattice states on GPU
        c_A_gpu = cu(c_init)
        c_B_gpu = CUDA.zeros(N, N, H)
        c_spore_gpu = cu(c_spore_array)
        # c_spore_B_gpu = CUDA.zeros(N, N, H)

        kernel_blocks, kernel_threads = invoke_smart_kernel_3D(size(c_init))

        # Run the simulation
        for t in 1:n_frames
            
            # Save frame
            if (t - 1) % save_interval == 0 && save_ct ≤ n_save_frames
                c_med_evolution[save_ct, :, :, :] .= Array(c_A_gpu)
                c_spore_evolution[save_ct] = CUDA.reduce(max_reduce_kernel, c_spore_gpu, init=-Inf)
                times[save_ct] = (t - 1) * dt
                print("\rFrame $save_ct saved.")
                # println(c_spore_evolution[save_ct])
                save_ct += 1
            end

            # Update the lattice
            if cluster_size == 1
                @cuda threads=kernel_threads blocks=kernel_blocks update_GPU_low_res!(c_A_gpu, c_B_gpu, N, H, inv_dx2, D, spore_vol_idx, c_spore_gpu, inv_tau, dt, neumann_z)
            else
                @cuda threads=kernel_threads blocks=kernel_blocks update_GPU_low_res_spore_cluster!(c_A_gpu, c_B_gpu, N, H, inv_dx2, D, spore_vol_idx, c_spore_gpu, inv_tau, dt, cluster_sp_lattice, cluster_size, neumann_z)
            end
            c_A_gpu, c_B_gpu = c_B_gpu, c_A_gpu

            # Check for threshold crossing
            if !isnothing(c_thresholds)
                if thresh_ct < length(t_thresholds) && t_thresholds[thresh_ct] == 0 && CUDA.reduce(max_reduce_kernel, c_spore_gpu, init=-Inf) < c_thresholds[thresh_ct]
                    t_thresholds[thresh_ct] = t * dt
                    thresh_ct += 1
                end
            end
        end

        # Save final frame
        c_med_evolution[save_ct, :, :, :] .= Array(c_A_gpu)
        c_spore_evolution[save_ct] = CUDA.reduce(max_reduce_kernel, c_spore_gpu, init=-Inf)
        times[save_ct] = t_max

        # Re-dimensionalise
        if nondim
            times .= times .* timescale
        end

        return c_med_evolution, c_spore_evolution, times, t_thresholds
    end

    function diffusion_time_dependent_GPU_hi_res!(c_init, c₀, sp_cen_indices, spore_rad, t_max; D=1.0, Db=1.0, dt=0.005, dx=0.2, n_save_frames=100,
        c_thresholds=nothing, abs_bndry=false, neumann_z=false, corr_factor=1.1, nondim=false, empty_interior=true)
        """
        Compute the evolution of a square lattice of concentration scalars
        based on the time-dependent diffusion equation.
        inputs:
            c_init (vector of float) - the initial state of the lattice
            c₀ (float) - the initial concentration at the spore
            sp_cen_indices (array of tuples) - the indices of the spore locations
            spore_rad (float) - the radius of the spore
            t_max (int) - a maximum number of iterations
            D (float) - the diffusion constant; defaults to 1
            Db (float) - the diffusion constant through the spore cell wall; defaults to 1
            dt (float) - timestep; defaults to 0.001
            dx (float) - spatial increment; defaults to 0.005
            n_save_frames (int) - determines the number of frames to save during the simulation; detaults to 100
            c_thresholds (vector of float) - threshold values for the concentration; defaults to nothing
            abs_bndry (bool) - whether to use absorbing boundary conditions; defaults to false
            neumann_z (bool) - whether to use Neumann boundary conditions in the z-direction; defaults to false
            corr_factor (float) - correction factor for calculating the cell wall thickness
            nondim (bool) - whether to use nondimensionalisation; defaults to false
            empty_interior (bool) - whether to make interior beyond the cell wall inaccessible for diffusion
        outputs:
            c_evolotion (array) - the states of the lattice at all moments in time
            times (array) - the times at which the states were saved
            t_thresholds (array) - the times at which the concentration crossed the threshold
        """

        @assert length(sp_cen_indices[1]) == 3 "spore_idx must be a 3D array"
        @assert typeof(sp_cen_indices[1]) == Tuple{Int, Int, Int} "spore_idx must be an array of tuples"

        GC.gc()

        # Determine number of lattice rows/columns
        N = size(c_init)[1]
        H = size(c_init)[3]

        @assert spore_rad < N * dx && spore_rad < H * dx "spore_rad must be less than N and H"

        # Radus in lattice units
        spore_rad_lattice = spore_rad / dx
        println("Spore radius in lattice units: ", spore_rad_lattice)

        # Non-dimensionalise
        if nondim
            lengthscale = spore_rad * 2
            timescale = lengthscale^2 / Db
            D = D / Db
            dx = dx / lengthscale
            dt = dt / timescale
            t_max = t_max / timescale
            Db = 1.0
            println("Parameters rescaled to D = $D, Db = $Db, dt = $dt, dx = $dx, t_max = $t_max")
        else
            lengthscale = 1.0
        end

        # Save update factor
        dtdx2 = dt / (dx^2)

        # Compute internal cell wall diffusion constant from effective diffusion constant
        Dcw = Db * D / (2 * D - Db)
        println("Using D = $D, Db = $Db, Dcw = $Dcw")
        println("D*dt/dx2 = $(D*dtdx2), Db*dt/dx2 = $(Db*dtdx2), Dcw*dt/dx2 = $(Dcw*dtdx2)")

        # Check stability
        if D * dtdx2 ≥ 0.2
            println("Warning: inappropriate scaling of dx and dt due to D, may result in an unstable simulation; Ddt/dx2 = $(D*dtdx2).")
        end
        if Db * dtdx2 ≥ 0.2
            println("Warning: inappropriate scaling of dx and dt due to Db, may result in an unstable simulation; Dbdt/dx2 = $(Db*dtdx2).")
        end
        if Dcw * dtdx2 ≥ 0.2
            println("Warning: inappropriate scaling of dx and dt due to Dcw, may result in an unstable simulation; Dcwdt/dx2 = $(Dcw*dtdx2).")
        end

        # Construct cell wall index map
        steps = [0, -1, 1]
        moore_nbrs = vec([(di, dj, dk) for di in steps, dj in steps, dk in steps])
        # println("Moore neighbors: ", moore_nbrs)
        region_ids = zeros(Int, N, N, H)

        # Initialise concentrations in cell wall
        cw_thickness = corr_factor*sqrt(3)
        for i in 1:N, j in 1:N, k in 1:H
            for sp_cen_idx in sp_cen_indices
                if empty_interior
                    if spore_rad_lattice - cw_thickness ≤ sqrt((i - sp_cen_idx[1])^2 + (j - sp_cen_idx[2])^2 + (k - sp_cen_idx[3])^2) ≤ spore_rad_lattice
                        # Cell wall
                        c_init[i, j, k] += 1.0#c₀
                        region_ids[i, j, k] = 1
                    elseif sqrt((i - sp_cen_idx[1])^2 + (j - sp_cen_idx[2])^2 + (k - sp_cen_idx[3])^2) ≤ spore_rad_lattice - cw_thickness
                        # Interior
                        region_ids[i, j, k] = 2
                    end
                else
                    if sqrt((i - sp_cen_idx[1])^2 + (j - sp_cen_idx[2])^2 + (k - sp_cen_idx[3])^2) ≤ spore_rad_lattice
                        # Spore interior
                        c_init[i, j, k] += 1.0
                        region_ids[i, j, k] = 1
                    end
                end
            end
        end
        # Count concentration sites and distribute spore concentration over volume
        n_spore_sites = sum(region_ids .> 0.0)
        n_cw_sites = sum(region_ids .== 1.0)
        c_init .= c_init .* c₀ .* n_cw_sites / n_spore_sites
        println("Concentrations initialised.")

        # println("Ideal cell wall volume: ", 4/3 * π * (spore_rad^3 - (spore_rad - dx * 2)^3))
        # println("Discretised cell wall volume: ", sum(region_ids .== 1) * dx^3)
        # println("Ideal spore volume: ", 4/3 * π * spore_rad^3)
        # println("Discretised spore volume: ", sum(region_ids .≥ 1) * dx^3)

        # Determine number of frames
        n_frames = Int(floor(t_max / dt))

        # Correct number of frames to save
        if n_frames < n_save_frames
            println("Correcting number of frames to save.")
            n_save_frames = n_frames
        end

        # Allocate arrays for saving data
        c_frames = zeros(n_save_frames + 1, N, H) # Only a cross-section is saved
        c_spore = zeros(n_save_frames + 1)
        times = zeros(n_save_frames + 1)
        println("Storage arrays allocated.")
        save_interval = floor(n_frames / n_save_frames)
        save_ct = 1

        # Allocate arrays for saving threshold crossing times
        if isnothing(c_thresholds)
            t_thresholds = nothing
        else
            t_thresholds = zeros(length(c_thresholds))
        end
        thresh_ct = 0

        # Initialise arrays on GPU
        c_A_gpu = cu(c_init)
        c_B_gpu = CUDA.zeros(Float64, N, N, H)
        region_ids_gpu = cu(region_ids)

        kernel_blocks, kernel_threads = invoke_smart_kernel_3D(size(c_init))

        # Run the simulation
        for t in 1:n_frames

            # println("Frame $t")

            # Save frame
            if (t - 1) % save_interval == 0 && save_ct ≤ n_save_frames
                CUDA.synchronize()
                c_frames[save_ct, :, :] .= Array(c_A_gpu)[:, N ÷ 2, :]
                c_spore[save_ct] = compute_spore_concentration(reshape(Array(c_A_gpu), (1, N, N, H)), Array(region_ids_gpu), spore_rad, dx * lengthscale)[1]
                # println(c_spore[save_ct])
                # println(maximum(c_frames[save_ct, :, :]))
                times[save_ct] = (t - 1) * dt
                print("\rFrame $save_ct saved.")
                save_ct += 1
            end

            # Update the lattice
            @cuda threads=kernel_threads blocks=kernel_blocks update_GPU_hi_res!(c_A_gpu, c_B_gpu, N, H, dtdx2, D, Dcw, Db, region_ids_gpu, abs_bndry, neumann_z)
            c_A_gpu, c_B_gpu = c_B_gpu, c_A_gpu

            # Check for threshold crossing
            if !isnothing(c_thresholds)
                if thresh_ct < length(t_thresholds) && t_thresholds[thresh_ct] == 0 && CUDA.reduce(max_reduce_kernel, c_A_gpu) < c_thresholds[thresh_ct]
                    t_thresholds[thresh_ct] = t * dt
                    thresh_ct += 1
                end
            end
        end

        # Save final frame
        CUDA.synchronize()
        c_frames[save_ct, :, :] .= Array(c_A_gpu)[:, N ÷ 2, :]
        c_spore[save_ct] = compute_spore_concentration(reshape(Array(c_A_gpu), (1, N, N, H)), Array(region_ids_gpu), spore_rad, dx * lengthscale)[1]
        times[save_ct] = t_max

        # Re-dimensionalise
        if nondim
            times .= times .* timescale
        end

        return c_frames, c_spore, times, region_ids[:, N ÷ 2, :], t_thresholds
    end


    function diffusion_time_dependent_GPU_hi_res_implicit(c_init, c₀, sp_cen_indices, spore_rad, t_max; D=1.0, Db=1.0, dt=0.005, dx=0.2, n_save_frames=100,
        c_thresholds=nothing, crank_nicolson=true, abs_bndry=false, neumann_z=false, corr_factor=1.1, nondim=false, empty_interior=true)
        """
        Compute the evolution of a square lattice of concentration scalars
        based on the time-dependent diffusion equation using the Crank–Nicolson
        or the Backward Euler method.
        inputs:
            c_init (vector of float) - the initial state of the lattice
            c₀ (float) - the initial concentration at the spore
            sp_cen_indices (array of tuples) - the indices of the spore locations
            spore_rad (float) - the radius of the spore
            t_max (int) - a maximum number of iterations
            D (float) - the diffusion constant; defaults to 1
            Db (float) - the diffusion constant through the spore cell wall; defaults to 1
            dt (float) - timestep; defaults to 0.001
            dx (float) - spatial increment; defaults to 0.005
            n_save_frames (int) - determines the number of frames to save during the simulation; detaults to 100
            c_thresholds (vector of float) - threshold values for the concentration; defaults to nothing
            crank_nicolson (bool) - whether to use the Crank–Nicolson method, else uses Backward Euler; defaults to true
            abs_bndry (bool) - whether to use absorbing boundary conditions; defaults to false
            neumann_z (bool) - whether to use Neumann boundary conditions in the z-direction; defaults to false
            corr_factor (float) - correction factor for calculating the cell wall thickness
            nondim (bool) - whether to use nondimensionalisation; defaults to false
            empty_interior (bool) - whether to make interior beyond the cell wall inaccessible for diffusion
        outputs:
            c_evolotion (array) - the states of the lattice at all moments in time
            times (array) - the times at which the states were saved
            t_thresholds (array) - the times at which the concentration crossed the threshold
        """

        @assert length(sp_cen_indices[1]) == 3 "spore_idx must be a 3D array"
        @assert typeof(sp_cen_indices[1]) == Tuple{Int, Int, Int} "spore_idx must be an array of tuples"

        GC.gc()

        # Determine number of lattice rows/columns
        N = size(c_init)[1]
        H = size(c_init)[3]

        @assert spore_rad < N * dx && spore_rad < H * dx "spore_rad must be less than N and H"

        # Radus in lattice units
        spore_rad_lattice = spore_rad / dx
        println("Spore radius in lattice units: ", spore_rad_lattice)

        # Non-dimensionalise
        if nondim
            lengthscale = spore_rad * 2
            timescale = lengthscale^2 / D
            dx = dx / lengthscale
            dt = dt / timescale
            t_max = t_max / timescale
            Db = Db / D
            # D = 1.0
            # lengthscale = spore_rad * 2
            # timescale = lengthscale^2 / Db
            # D = D / Db
            # dx = dx / lengthscale
            # dt = dt / timescale
            # t_max = t_max / timescale
            # Db = 1.0
            println("Parameters rescaled to D = $D, Db = $Db, dt = $dt, dx = $dx, t_max = $t_max")
        else
            lengthscale = 1.0
        end

        # Save update factor
        dtdx2 = dt / (dx^2)

        # Cell wall thickness
        cw_thickness = corr_factor*sqrt(3)

        # Compute internal cell wall diffusion constant from effective diffusion constant
        Dcw = Db * D / (2 * D - Db)
        println("Using D = $D, Db = $Db, Dcw = $Dcw")
        println("D*dt/dx2 = $(D*dtdx2), Db*dt/dx2 = $(Db*dtdx2), Dcw*dt/dx2 = $(Dcw*dtdx2)")
        println("Timescale for accuracy: ", dx^2 / max([D, Db, Dcw]...))

        kernel_blocks, kernel_threads = invoke_smart_kernel_3D(size(c_init))
        
        # Initialise sparse matrix elements
        Nt = N * N * H
        nnz = Nt * 7
        valsA_gpu = CUDA.zeros(Float32, nnz)
        valsB_gpu = CUDA.zeros(Float32, nnz)
        colidx_gpu = CUDA.zeros(Int, nnz)
        rowptr = 1:7:nnz+1
        rowptr_gpu = cu(collect(rowptr))

        # Initialise concentrations and operators on GPU
        c_gpu = cu(vec(c_init))
        region_ids_gpu = CUDA.zeros(Int, N, N, H)
        # debugger_gpu = CUDA.zeros(Int, N, N, H)
        # pc_vals_gpu = CUDA.zeros(Float32, Nt)
        # pc_rowptr = 1:Nt+1
        # pc_rowptr_gpu = cu(collect(pc_rowptr))
        # pc_colidx_gpu = cu(collect(1:Nt))
        # if abs_bndry
        #     for sp_cen_idx in sp_cen_indices
        #         @cuda threads=kernel_threads blocks=kernel_blocks initialise_lattice_and_operator_GPU_abs_bndry!(c_gpu, colidx_gpu, valsA_gpu, valsB_gpu, region_ids_gpu,
        #                                                                                                         c₀, sp_cen_idx[1], sp_cen_idx[2], sp_cen_idx[3],
        #                                                                                                         spore_rad_lattice, cw_thickness, D, Dcw, Db, dtdx2, N, H, crank_nicolson)
        #     end
        # else
        #     for sp_cen_idx in sp_cen_indices
        #         @cuda threads=kernel_threads blocks=kernel_blocks initialise_lattice_and_operator_GPU!(c_gpu, colidx_gpu, valsA_gpu, valsB_gpu, region_ids_gpu, #pc_vals_gpu, debugger_gpu,
        #                                                                                                 c₀, sp_cen_idx[1], sp_cen_idx[2], sp_cen_idx[3],
        #                                                                                                 spore_rad_lattice, cw_thickness, D, Dcw, Db, dtdx2, N, H, crank_nicolson)
        #     end
        # end
        if abs_bndry
            for sp_cen_idx in sp_cen_indices
                @cuda threads=kernel_threads blocks=kernel_blocks initialise_lattice_and_operator_GPU_abs_bndry!(c_gpu, colidx_gpu, valsA_gpu, valsB_gpu, region_ids_gpu,
                                                                                                                1f0, sp_cen_idx[1], sp_cen_idx[2], sp_cen_idx[3],
                                                                                                                spore_rad_lattice, cw_thickness, D, Dcw, Db, dtdx2, N, H, crank_nicolson, empty_interior)
            end
        else
            for sp_cen_idx in sp_cen_indices
                @cuda threads=kernel_threads blocks=kernel_blocks initialise_lattice_and_operator_GPU!(c_gpu, colidx_gpu, valsA_gpu, valsB_gpu, region_ids_gpu, #pc_vals_gpu, debugger_gpu,
                                                                                                        1f0, sp_cen_idx[1], sp_cen_idx[2], sp_cen_idx[3],
                                                                                                        spore_rad_lattice, cw_thickness, D, Dcw, Db, dtdx2, N, H, crank_nicolson, empty_interior)
            end
        end
        CUDA.synchronize()
        # println(unique(Array(valsA_gpu)))
        # regids = Array(region_ids_gpu)
        # println("Sum of region ids: ", sum(regids))
        op_A_gpu = CuSparseMatrixCSR(rowptr_gpu, colidx_gpu, valsA_gpu, (Nt, Nt))
        op_B_gpu = CuSparseMatrixCSR(rowptr_gpu, colidx_gpu, valsB_gpu, (Nt, Nt))
        # precond = CuSparseMatrixCSR(pc_rowptr_gpu, pc_colidx_gpu, pc_vals_gpu, (Nt, Nt))
        
        # debugger = sparse(op_A_gpu)

        # println("Ideal cell wall volume: ", 4/3 * π * (spore_rad^3 - (spore_rad - dx * 2)^3))
        # println("Discretised cell wall volume: ", sum(region_ids .== 1) * dx^3)
        # println("Ideal spore volume: ", 4/3 * π * spore_rad^3)
        # println("Discretised spore volume: ", sum(region_ids .≥ 1) * dx^3)

        # println(size(reshape(Array(c_gpu), (1, N, N, H))))
        # c_temp = compute_spore_concentration(reshape(Array(c_gpu), (1, N, N, H)), reshape(Array(region_ids_gpu), (N, N, H)), spore_rad, dx*sqrt(2), dx * lengthscale)
        # println(size(c_temp))

        # Distribute concentration
        region_ids = Array(region_ids_gpu)
        n_spore_sites = sum(region_ids .> 0.0)
        n_cw_sites = sum(region_ids .== 1.0)
        # println(n_spore_sites, " spore sites")
        # println(n_cw_sites, " cell wall sites")
        # println("Rescaled concentration: ", c₀ * n_spore_sites / n_cw_sites)
        c_gpu .= c_gpu .* c₀ .* (n_spore_sites / n_cw_sites)
        
        # Determine number of frames
        n_frames = Int(floor(t_max / dt))

        # Correct number of frames to save
        if n_frames < n_save_frames
            println("Correcting number of frames to save.")
            n_save_frames = n_frames
        end

        # Allocate arrays for saving data
        c_frames = zeros(n_save_frames + 1, N, H) # Only a cross-section is saved
        c_spore = zeros(n_save_frames + 1)
        times = zeros(n_save_frames + 1)
        println("Storage arrays allocated.")
        save_interval = floor(n_frames / n_save_frames)
        save_ct = 1

        # Allocate arrays for saving threshold crossing times
        if isnothing(c_thresholds)
            t_thresholds = nothing
        else
            t_thresholds = zeros(length(c_thresholds))
        end
        thresh_ct = 0

        # Run the simulation
        for t in 1:n_frames

            # println("Frame $t")

            # Save frame
            if (t - 1) % save_interval == 0 && save_ct ≤ n_save_frames
                CUDA.synchronize()
                c_frames[save_ct, :, :] .= reshape(Array(c_gpu), (N, N, H))[:, N ÷ 2, :]
                # println("Spore radius: ", spore_rad)
                # println("Correction factor: ", corr_factor)
                c_spore[save_ct] = compute_spore_concentration(reshape(Array(c_gpu), (1, N, N, H)),
                                                                reshape(Array(region_ids_gpu), (N, N, H)), spore_rad, dx * lengthscale)[1]
                times[save_ct] = (t - 1) * dt
                # println(maximum(c_frames[save_ct, :, :]))
                println(c_spore[save_ct])
                print("\rFrame $save_ct saved.")
                save_ct += 1
            end

            # Update the lattice
            b_gpu = crank_nicolson ? op_B_gpu * c_gpu : c_gpu  # Right-hand side
            c_gpu, stats = Krylov.cg(op_A_gpu, b_gpu; atol=Float32(1e-12), itmax=1000)

            # Check for threshold crossing
            if !isnothing(c_thresholds)
                if thresh_ct < length(t_thresholds) && t_thresholds[thresh_ct] == 0 && CUDA.reduce(max_reduce_kernel, c_gpu) < c_thresholds[thresh_ct]
                    t_thresholds[thresh_ct] = t * dt
                    thresh_ct += 1
                end
            end
        end

        # Save final frame
        CUDA.synchronize()
        c_frames[save_ct, :, :] .= reshape(Array(c_gpu), (N, N, H))[:, N ÷ 2, :]
        c_spore[save_ct] = compute_spore_concentration(reshape(Array(c_gpu), (1, N, N, H)),
                                                        reshape(Array(region_ids_gpu), (N, N, H)), spore_rad, dx * lengthscale)[1]
        times[save_ct] = t_max
        # println(maximum(c_frames[save_ct, :, :]))

        # Re-dimensionalise
        if nondim
            times .= times .* timescale
        end

        # region_ids = Array(region_ids_gpu)[:, N ÷ 2, :]

        return c_frames, c_spore, times, region_ids[:, N ÷ 2, :], t_thresholds#, Array(debugger_gpu)[:, N ÷ 2, :]
    end
end