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
    # using IterativeSolvers
    # using LinearMaps
    using Krylov
    # using KrylovPreconditioners
    # using AMGX
    using LinearAlgebra
    using IterTools
    using Revise

    include("./solver.jl")
    include("./conversions.jl")
    Revise.includet("./solver.jl")
    Revise.includet("./conversions.jl")
    using .Solver
    using .Conversions

    export permeation_time_dependent_analytical
    export diffusion_time_dependent_analytical_src
    export compute_permeation_constant
    export diffusion_time_dependent_GPU
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


    function diffusion_time_dependent_analytical_src(c_init, D, time, vol; dims=3)
        """
        Compute the analytical solution of the time-dependent diffusion equation at the source.
        inputs:
            c_init (float) - the initial concentration
            D (float) - the diffusion constant
            time (float) - the time at which the concentration is to be computed
            vol (float) - the volume of the initial concentration source
        """
        if dims == 2
            result = (vol * c_init) ^ (2/3) / (4*π*D*time)
        elseif dims == 3
            result = vol * c_init / (4*π*D.*time)^(1.5)
        end
        result = [isnan(x) || isinf(x) ? c_init[i] : x for (i, x) in enumerate(result)]
        return result
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
    function diffusion_time_dependent_GPU(c_init, t_max; D=1.0, Db=nothing, Ps=1.0, dt=0.005, dx=5, n_save_frames=100,
        spore_idx=nothing, spore_idx_spacing=nothing, c_thresholds=nothing, neumann_z=false)
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
            neumann_z (bool) - whether to use Neumann boundary conditions in the z-direction; defaults to false
        outputs:
            c_evolotion (array) - the states of the lattice at all moments in time
            times (array) - the times at which the states were saved
            t_thresholds (array) - the times at which the concentration crossed the threshold
        """

        @argcheck ndims(c_init) == 3 "c_init must be a 3D array"

        GC.gc()

        # Set the spore index reference
        if !isnothing(spore_idx_spacing)
            spore_idx_ref = spore_idx_spacing
        elseif !isnothing(spore_idx)
            spore_idx_ref = spore_idx
            println("3D simulation")
        else
            println("Either spore_idx or spore_idx_spacing must be provided.")
            return
        end

        # Determine number of lattice rows/columns
        N = size(c_init)[1]
        H = size(c_init)[3]

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
                # println("Frame $save_ct saved.")
                save_ct += 1
            end

            # Update the lattice
            @cuda threads=kernel_threads blocks=kernel_blocks update_GPU!(c_A_gpu, c_B_gpu, N, H, dtdx2, D, Db, spore_idx_ref, neumann_z)
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

        return c_evolution, times, t_thresholds
    end

    function diffusion_time_dependent_GPU_low_res(c_init, c₀, t_max; D=1.0, Pₛ=1.0, A=150, V=125, dt=150, dx=25, n_save_frames=100,
        spore_vol_idx=nothing, spore_vol_idx_spacing=nothing, c_thresholds=nothing, neumann_z=false)
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
            spore_vol_spacing (int) - the spacing between spore volume indices along each dimension; defaults to nothing; if used, spore_vol_idx is ignored
            c_thresholds (vector of float) - threshold values for the concentration; defaults to nothing
            neumann_z (bool) - whether to use Neumann boundary conditions in the z-direction; defaults to false
        outputs:
            c_evolotion (array) - the states of the lattice at all moments in time
            times (array) - the times at which the states were saved
            t_thresholds (array) - the times at which the concentration crossed the threshold
        """

        @argcheck ndims(c_init) == 3 "c_init must be a 3D array"

        GC.gc()

        # Set the spore index reference
        if !isnothing(spore_vol_idx_spacing)
            spore_vol_idx_ref = spore_vol_idx_spacing
        elseif !isnothing(spore_vol_idx)
            spore_vol_idx_ref = spore_vol_idx
            println("3D simulation")
        else
            println("Either spore_vol_idx or spore_vol_idx_spacing must be provided.")
            return
        end

        # Determine number of lattice rows/columns
        N = size(c_init)[1]
        H = size(c_init)[3]

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
        c_spore_array[spore_vol_idx_ref...] = c₀
        c_med_evolution = zeros(n_save_frames + 1, N, N, H)
        c_spore_evolution = zeros(n_save_frames + 1)
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
                # println("Frame $save_ct saved.")
                save_ct += 1
            end

            # Update the lattice
            @cuda threads=kernel_threads blocks=kernel_blocks update_GPU_low_res!(c_A_gpu, c_B_gpu, N, H, inv_dx2, D, spore_vol_idx_ref, c_spore_gpu, inv_tau, dt, neumann_z)
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

        return c_med_evolution, c_spore_evolution, times, t_thresholds
    end

    function diffusion_time_dependent_GPU_hi_res!(c_init, c₀, sp_cen_indices, spore_rad, t_max; D=1.0, Db=1.0, dt=0.005, dx=0.2, n_save_frames=100,
        c_thresholds=nothing, abs_bndry=false, neumann_z=false)
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

        # Save update factor
        dtdx2 = dt / (dx^2)

        # # Compute effective diffusion constant at interface
        # Deff = 2 * D * Db / (D + Db)
        # println("Using D = $D, Db = $Db, Deff = $Deff")

        # Compute internal cell wall diffusion constant from effective diffusion constant
        Dcw = Db * D / (2 * D - Db)
        println("Using D = $D, Db = $Db, Dcw = $Dcw")
        println("D*dt/dx2 = $(D*dtdx2), Db*dt/dx2 = $(Db*dtdx2), Dcw*dt/dx2 = $(Dcw*dtdx2)")

        # Convert to Float32
        # dtdx2 = Float32(dtdx2)
        # D = Float32(D)
        # Db = Float32(Db)
        # Deff = Float32(Deff)
        # println("D*dt/dx2 = $(D*dtdx2), Db*dt/dx2 = $(Db*dtdx2), Deff*dt/dx2 = $(Deff*dtdx2)")

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

        # Radus in lattice units
        spore_rad_lattice = spore_rad / dx
        println("Spore radius in lattice units: ", spore_rad_lattice)

        # Construct cell wall index map
        steps = [0, -1, 1]
        moore_nbrs = vec([(di, dj, dk) for di in steps, dj in steps, dk in steps])
        # println("Moore neighbors: ", moore_nbrs)
        region_ids = zeros(Int, N, N, H)

        # Initialise concentrations in cell wall
        corr_factor = 1.45
        cw_thickness = corr_factor*sqrt(3)
        for i in 1:N, j in 1:N, k in 1:H
            for sp_cen_idx in sp_cen_indices
                if spore_rad_lattice - cw_thickness ≤ sqrt((i - sp_cen_idx[1] - 1)^2 + (j - sp_cen_idx[2] - 1)^2 + (k - sp_cen_idx[3] - 1)^2) ≤ spore_rad_lattice
                    # Cell wall
                    c_init[i, j, k] += c₀
                    region_ids[i, j, k] = 1
                elseif sqrt((i - sp_cen_idx[1] - 1)^2 + (j - sp_cen_idx[2] - 1)^2 + (k - sp_cen_idx[3] - 1)^2) ≤ spore_rad_lattice - cw_thickness
                    # Interior
                    c_init[i, j, k] += 0.0
                    region_ids[i, j, k] = 2
                # else
                #     # Exterior
                #     c_init[i, j, k] .+ 0.0
                end
            end
        end
        println("Concentrations initialised.")

        println("Ideal cell wall volume: ", 4/3 * π * (spore_rad^3 - (spore_rad - dx * 2)^3))
        println("Discretised cell wall volume: ", sum(region_ids .== 1) * dx^3)
        println("Ideal spore volume: ", 4/3 * π * spore_rad^3)
        println("Discretised spore volume: ", sum(region_ids .≥ 1) * dx^3)

        # Get number of interfaces between cell wall and exterior
        # regions_int_ext = region_ids .≠ 2
        # intf_x = sum(abs.(diff(regions_int_ext, dims=1)))
        # intf_y = sum(abs.(diff(regions_int_ext, dims=2)))
        # intf_z = sum(abs.(diff(regions_int_ext, dims=3)))
        # println("Number of interfaces in x: ", sum(intf_x))

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
                c_spore[save_ct] = compute_spore_concentration(reshape(Array(c_A_gpu), (1, N, N, H)), Array(region_ids_gpu), spore_rad, dx)[1]
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
        c_spore[save_ct] = compute_spore_concentration(reshape(Array(c_A_gpu), (1, N, N, H)), Array(region_ids_gpu), spore_rad, dx)[1]
        times[save_ct] = t_max

        return c_frames, c_spore, times, region_ids[:, N ÷ 2, :], t_thresholds
    end


    function diffusion_time_dependent_GPU_hi_res_implicit(c_init, c₀, sp_cen_indices, spore_rad, t_max; D=1.0, Db=1.0, dt=0.005, dx=0.2, n_save_frames=100,
        c_thresholds=nothing, crank_nicolson=true, abs_bndry=false, neumann_z=false)
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

        # Save update factor
        dtdx2 = dt / (dx^2)

        # Cell wall thickness
        corr_factor = 1.45
        cw_thickness = corr_factor*sqrt(3)

        # # Compute effective diffusion constant at interface
        # Deff = 2 * D * Db / (D + Db)
        # println("Using D = $D, Db = $Db, Deff = $Deff")
        # println("D*dt/dx2 = $(D*dtdx2), Db*dt/dx2 = $(Db*dtdx2), Deff*dt/dx2 = $(Deff*dtdx2)")
        # println("Timescale for accuracy: ", dx^2 / max([D, Db]...))

        # Compute internal cell wall diffusion constant from effective diffusion constant
        Dcw = Db * D / (2 * D - Db)
        println("Using D = $D, Db = $Db, Dcw = $Dcw")
        println("D*dt/dx2 = $(D*dtdx2), Db*dt/dx2 = $(Db*dtdx2), Dcw*dt/dx2 = $(Dcw*dtdx2)")
        println("Timescale for accuracy: ", dx^2 / max([D, Db, Dcw]...))

        # Radus in lattice units
        spore_rad_lattice = spore_rad / dx
        println("Spore radius in lattice units: ", spore_rad_lattice)

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
        if abs_bndry
            for sp_cen_idx in sp_cen_indices
                @cuda threads=kernel_threads blocks=kernel_blocks initialise_lattice_and_operator_GPU_abs_bndry!(c_gpu, colidx_gpu, valsA_gpu, valsB_gpu, region_ids_gpu,
                                                                                                        c₀, sp_cen_idx[1], sp_cen_idx[2], sp_cen_idx[3],
                                                                                                        spore_rad_lattice, cw_thickness, D, Dcw, Db, dtdx2, N, H, crank_nicolson)
            end
        else
            for sp_cen_idx in sp_cen_indices
                @cuda threads=kernel_threads blocks=kernel_blocks initialise_lattice_and_operator_GPU!(c_gpu, colidx_gpu, valsA_gpu, valsB_gpu, region_ids_gpu, #pc_vals_gpu, debugger_gpu,
                                                                                                        c₀, sp_cen_idx[1], sp_cen_idx[2], sp_cen_idx[3],
                                                                                                        spore_rad_lattice, cw_thickness, D, Dcw, Db, dtdx2, N, H, crank_nicolson)
            end
        end
        CUDA.synchronize()
        # println(unique(Array(valsA_gpu)))
        op_A_gpu = CuSparseMatrixCSR(rowptr_gpu, colidx_gpu, valsA_gpu, (Nt, Nt))
        op_B_gpu = CuSparseMatrixCSR(rowptr_gpu, colidx_gpu, valsB_gpu, (Nt, Nt))
        # precond = CuSparseMatrixCSR(pc_rowptr_gpu, pc_colidx_gpu, pc_vals_gpu, (Nt, Nt))

        # println("Ideal cell wall volume: ", 4/3 * π * (spore_rad^3 - (spore_rad - dx * 2)^3))
        # println("Discretised cell wall volume: ", sum(region_ids .== 1) * dx^3)
        # println("Ideal spore volume: ", 4/3 * π * spore_rad^3)
        # println("Discretised spore volume: ", sum(region_ids .≥ 1) * dx^3)

        # println(size(reshape(Array(c_gpu), (1, N, N, H))))
        # c_temp = compute_spore_concentration(reshape(Array(c_gpu), (1, N, N, H)), reshape(Array(region_ids_gpu), (N, N, H)), spore_rad, dx*sqrt(2), dx)
        # println(size(c_temp))
        
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
                c_spore[save_ct] = compute_spore_concentration(reshape(Array(c_gpu), (1, N, N, H)),
                                                                reshape(Array(region_ids_gpu), (N, N, H)), spore_rad, dx)[1]
                times[save_ct] = (t - 1) * dt
                println(maximum(c_frames[save_ct, :, :]))
                print("\rFrame $save_ct saved.")
                save_ct += 1
            end

            # Update the lattice
            b_gpu = crank_nicolson ? op_B_gpu * c_gpu : c_gpu#  # Right-hand side
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
                                                        reshape(Array(region_ids_gpu), (N, N, H)), spore_rad, dx)[1]
        times[save_ct] = t_max
        # println(maximum(c_frames[save_ct, :, :]))

        region_ids = Array(region_ids_gpu)[:, N ÷ 2, :]

        return c_frames, c_spore, times, region_ids, t_thresholds#, Array(debugger_gpu)[:, N ÷ 2, :]
    end
end