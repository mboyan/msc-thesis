module Diffusion
__precompile__(false)
    """
    Functions for solving the diffusion equation.
    """

    using ArgCheck
    using CUDA
    using IterTools
    # using StaticArrays

    export permeation_time_dependent_analytical
    export diffusion_time_dependent_analytical_src
    export compute_permeation_constant
    export diffusion_time_dependent_GPU
    export diffusion_time_dependent_GPU_low_res
    export diffusion_time_dependent_GPU_hi_res
    
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
    function invoke_smart_kernel_3D(size, threads_per_block=(8, 8, 8))
        """
        Invoke a kernel with the appropriate number of blocks and threads per block.
        """
        blocks_per_grid = (Int(round(size[1] / threads_per_block[1])),
                           Int(round(size[2] / threads_per_block[2])),
                           Int(round(size[3] / threads_per_block[3])))
        return blocks_per_grid, threads_per_block
    end

    function max_reduce_kernel(a, b)
        return max(a, b)
    end

    function update_GPU!(c_old, c_new, N, H, dtdx2, D, Db, spore_idx, neumann_z)
        """
        Update the concentration values on the lattice
        using the time-dependent diffusion equation.
        inputs:
            c_old (array) - the current state of the lattice
            c_new (array) - the updated state of the lattice
            N (int) - the number of lattice rows/columns
            H (int) - the number of lattice layers
            dtdx2 (float) - the update factor
            D (float) - the diffusion constant
            Db (float) - the diffusion constant through the spore
            spore_idx (tuple) - the indices of the spore location
            neumann_z (bool) - whether to use Neumann boundary conditions in the z-direction
        """
        i, j, k = CUDA.blockIdx().x, CUDA.blockIdx().y, CUDA.blockIdx().z
        ti, tj, tk = CUDA.threadIdx().x, CUDA.threadIdx().y, CUDA.threadIdx().z
    
        # Determine the indices of the current cell
        idx = ((i - 1) * blockDim().x + ti, (j - 1) * blockDim().y + tj, (k - 1) * blockDim().z + tk)
    
        # Update the concentration value
        if 1 ≤ idx[1] ≤ N && 1 ≤ idx[2] ≤ N && 1 ≤ idx[3] ≤ H
            
            center = c_old[idx...]
            bottom = c_old[idx[1], idx[2], mod1(idx[3] - 1, H)]
            top = c_old[idx[1], idx[2], mod1(idx[3] + 1, H)]
            left = c_old[idx[1], mod1(idx[2] - 1, N), idx[3]]
            right = c_old[idx[1], mod1(idx[2] + 1, N), idx[3]]
            front = c_old[mod1(idx[1] - 1, N), idx[2], idx[3]]
            back = c_old[mod1(idx[1] + 1, N), idx[2], idx[3]]

            # Neumann boundary conditions in the z-direction
            if neumann_z
                if idx[3] == 1
                    bottom = center
                elseif idx[3] == H
                    top = center
                end
            end

            spore_dist_vec = (idx[1] - spore_idx[1], idx[2] - spore_idx[2], idx[3] - spore_idx[3])
            at_spore = spore_dist_vec == (0, 0, 0)
            Ddtdx2bottom = spore_dist_vec == (0, 0, 1) || at_spore ? Db * dtdx2 : D * dtdx2
            Ddtdx2top = spore_dist_vec == (0, 0, -1) || at_spore ? Db * dtdx2 : D * dtdx2
            Ddtdx2left = spore_dist_vec == (0, 1, 0) || at_spore ? Db * dtdx2 : D * dtdx2
            Ddtdx2right = spore_dist_vec == (0, -1, 0) || at_spore ? Db * dtdx2 : D * dtdx2
            Ddtdx2front = spore_dist_vec == (1, 0, 0) || at_spore ? Db * dtdx2 : D * dtdx2
            Ddtdx2back = spore_dist_vec == (-1, 0, 0) || at_spore ? Db * dtdx2 : D * dtdx2

            weighted_nbrs = Ddtdx2bottom * bottom + Ddtdx2top * top +
                Ddtdx2left * left + Ddtdx2right * right +
                Ddtdx2front * front + Ddtdx2back * back

            c_new[idx...] = center + weighted_nbrs - (Ddtdx2bottom + Ddtdx2top + Ddtdx2left + Ddtdx2right + Ddtdx2front + Ddtdx2back) * center
        end
    
        return nothing
    end

    function update_GPU_low_res!(c_old, c_new, N, H, inv_dx2, D, spore_vol_idx, c_spore, inv_tau, dt, neumann_z)
        """
        Update the concentration values on the lattice
        using the time-dependent diffusion equation.
        inputs:
            c_old (array) - the current state of the lattice
            c_new (array) - the updated state of the lattice
            N (int) - the number of lattice rows/columns
            H (int) - the number of lattice layers
            inv_dx2 (float) - inverse of the squared spatial increment
            D (float) - the diffusion constant
            spore_vol_idx (tuple) - the indices of the volume containing the spore location
            c_spore (array) - the concentration at the spore (only nonzero at the spore location)
            inv_tau (float) - reciprocal of the characteristic time for permeation
            dt (float) - timestep
            neumann_z (bool) - whether to use Neumann boundary conditions in the z-direction
        """
        i, j, k = CUDA.blockIdx().x, CUDA.blockIdx().y, CUDA.blockIdx().z
        ti, tj, tk = CUDA.threadIdx().x, CUDA.threadIdx().y, CUDA.threadIdx().z
    
        # Determine the indices of the current cell
        idx = ((i - 1) * blockDim().x + ti, (j - 1) * blockDim().y + tj, (k - 1) * blockDim().z + tk)
        
        # Factors
        dtdx2 = dt * inv_dx2
        dttau = dt * inv_tau

        # Update the concentration value
        if 1 ≤ idx[1] ≤ N && 1 ≤ idx[2] ≤ N && 1 ≤ idx[3] ≤ H

            center = c_old[idx...]
            bottom = c_old[idx[1], idx[2], mod1(idx[3] - 1, H)]
            top = c_old[idx[1], idx[2], mod1(idx[3] + 1, H)]
            left = c_old[idx[1], mod1(idx[2] - 1, N), idx[3]]
            right = c_old[idx[1], mod1(idx[2] + 1, N), idx[3]]
            front = c_old[mod1(idx[1] - 1, N), idx[2], idx[3]]
            back = c_old[mod1(idx[1] + 1, N), idx[2], idx[3]]

            # Neumann boundary conditions in the z-direction
            if neumann_z
                if idx[3] == 1
                    bottom = center
                elseif idx[3] == H
                    top = center
                end
            end

            # Special handling of spore-containing volume
            if idx[1] == spore_vol_idx[1] && idx[2] == spore_vol_idx[2] && idx[3] == spore_vol_idx[3]
                delta_c_half = (c_spore[idx...] - center) * (1 - 0.5 * dttau)
                c_new[idx...] = center + D * dtdx2 * (bottom + top + left + right + front + back - 6 * center) + dttau * delta_c_half
                c_spore[idx...] = c_new[idx...] + delta_c_half * (1 - 0.5 * dttau)
            else
                c_new[idx...] = center + D * dtdx2 * (bottom + top + left + right + front + back - 6 * center)
            end
        end

        return nothing
    end

    @inline function device_norm3_sq(v::NTuple{3, Int})
        s = v[1]*v[1] + v[2]*v[2] + v[3]*v[3]
        return s
    end

    function update_GPU_hi_res!(lattice_old, lattice_new, N, H, dtdx2, D, Db, Deff, neumann_z)
        """
        Update the concentration values on the lattice
        using the time-dependent diffusion equation.
        inputs:
            lattice_old (array) - the current state of the lattice (concentrations + region IDs)
            lattice_new (array) - the updated state of the lattice (concentrations + region IDs)
            N (int) - the number of lattice rows/columns
            H (int) - the number of lattice layers
            dtdx2 (float) - the update factor
            D (float) - the diffusion constant
            Db (float) - the diffusion constant through the spore
            Deff (float) - the effective diffusion constant at the spore interface
            neumann_z (bool) - whether to use Neumann boundary conditions in the z-direction
        """
        i, j, k = CUDA.blockIdx().x, CUDA.blockIdx().y, CUDA.blockIdx().z
        ti, tj, tk = CUDA.threadIdx().x, CUDA.threadIdx().y, CUDA.threadIdx().z
    
        # Determine the indices of the current cell
        idx = ((i - 1) * blockDim().x + ti, (j - 1) * blockDim().y + tj, (k - 1) * blockDim().z + tk)

        # Update the concentration value
        if 1 ≤ idx[1] ≤ N && 1 ≤ idx[2] ≤ N && 1 ≤ idx[3] ≤ H

            # Decode spore indices
            if lattice_old[idx...] < 10
                # Exterior site
                region_id = 0
                center = lattice_old[idx...]
            elseif lattice_old[idx...] < 100
                # Cell wall
                region_id = 1
                center = rem(lattice_old[idx...], 10)
            else
                # Interior site
                region_id = 2
                center = rem(lattice_old[idx...], 100)
            end

            # vneum_nbrs = ((idx[1], idx[2], mod1(idx[3] - 1, H)), (idx[1], idx[2], mod1(idx[3] + 1, H)),
            #             (idx[1], mod1(idx[2] - 1, N), idx[3]), (idx[1], mod1(idx[2] + 1, N), idx[3]),
            #             (mod1(idx[1] - 1, N), idx[2], idx[3]), (mod1(idx[1] + 1, N), idx[2], idx[3]))
            
            bottom = lattice_old[idx[1], idx[2], mod1(idx[3] - 1, H)]
            top = lattice_old[idx[1], idx[2], mod1(idx[3] + 1, H)]
            left = lattice_old[idx[1], mod1(idx[2] - 1, N), idx[3]]
            right = lattice_old[idx[1], mod1(idx[2] + 1, N), idx[3]]
            front = lattice_old[mod1(idx[1] - 1, N), idx[2], idx[3]]
            back = lattice_old[mod1(idx[1] + 1, N), idx[2], idx[3]]

            diff_bottom, diff_top, diff_left, diff_right, diff_front, diff_back = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

            if region_id == 0 # Exterior site
                # Check bottom neighbour
                if bottom < 10 # Exterior - exterior
                    diff_bottom = D * (bottom - center)
                elseif bottom < 100 # Exterior - cell wall
                    diff_bottom = Deff * (rem(bottom, 10) - center)
                end
                # Check top neighbour
                if top < 10 # Exterior - exterior
                    diff_top = D * (top - center)
                elseif top < 100 # Exterior - cell wall
                    diff_top = Deff * (rem(top, 10) - center)
                end
                # Check left neighbour
                if left < 10 # Exterior - exterior
                    diff_left = D * (left - center)
                elseif left < 100 # Exterior - cell wall
                    diff_left = Deff * (rem(left, 10) - center)
                end
                # Check right neighbour
                if right < 10 # Exterior - exterior
                    diff_right = D * (right - center)
                elseif right < 100 # Exterior - cell wall
                    diff_right = Deff * (rem(right, 10) - center)
                end
                # Check front neighbour
                if front < 10 # Exterior - exterior
                    diff_front = D * (front - center)
                elseif front < 100 # Exterior - cell wall
                    diff_front = Deff * (rem(front, 10) - center)
                end
                # Check back neighbour
                if back < 10 # Exterior - exterior
                    diff_back = D * (back - center)
                elseif back < 100 # Exterior - cell wall
                    diff_back = Deff * (rem(back, 10) - center)
                end
            elseif region_id == 1 # Cell wall site
                # Check bottom neighbour
                if bottom < 10 # Cell wall - exterior
                    diff_bottom = Deff * (bottom - center)
                elseif bottom < 100 # Cell wall - cell wall
                    diff_bottom = Db * (rem(bottom, 10) - center)
                end
                # Check top neighbour
                if top < 10 # Cell wall - exterior
                    diff_top = Deff * (top - center)
                elseif top < 100 # Cell wall - cell wall
                    diff_top = Db * (rem(top, 10) - center)
                end
                # Check left neighbour
                if left < 10 # Cell wall - exterior
                    diff_left = Deff * (left - center)
                elseif left < 100 # Cell wall - cell wall
                    diff_left = Db * (rem(left, 10) - center)
                end
                # Check right neighbour
                if right < 10 # Cell wall - exterior
                    diff_right = Deff * (right - center)
                elseif right < 100 # Cell wall - cell wall
                    diff_right = Db * (rem(right, 10) - center)
                end
                # Check front neighbour
                if front < 10 # Cell wall - exterior
                    diff_front = Deff * (front - center)
                elseif front < 100 # Cell wall - cell wall
                    diff_front = Db * (rem(front, 10) - center)
                end
                # Check back neighbour
                if back < 10 # Cell wall - exterior
                    diff_back = Deff * (back - center)
                elseif back < 100 # Cell wall - cell wall
                    diff_back = Db * (rem(back, 10) - center)
                end
            else
                return nothing
            end

            center_f = Float32(center)
            diff_bottom_f = Float32(diff_bottom)
            diff_top_f = Float32(diff_top)
            diff_left_f = Float32(diff_left)
            diff_right_f = Float32(diff_right)
            diff_front_f = Float32(diff_front)
            diff_back_f = Float32(diff_back)

            c_new = center_f + dtdx2 * (diff_bottom_f + diff_top_f + diff_left_f + diff_right_f + diff_front_f + diff_back_f)

            # c_new = center + dtdx2 * (diff_bottom + diff_top + diff_left + diff_right + diff_front + diff_back)
            if region_id == 0
                lattice_new[idx...] = c_new
            elseif region_id == 1
                lattice_new[idx...] = 10f0 + c_new
            else
                lattice_new[idx...] = 100f0 + c_new
            end
        end
        
        return nothing
    end


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
        dtdx2 = Float32(dt / (dx^2))

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
                times[save_ct] = t * dt
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
                times[save_ct] = t * dt
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

    function diffusion_time_dependent_GPU_hi_res(c_init, c₀, sp_cen_indices, spore_rad, t_max; D=1.0, Db=1.0, dt=0.005, dx=0.2, n_save_frames=100,
        c_thresholds=nothing, neumann_z=false)
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

        # Compute effective diffusion constant at interface
        Deff = 2 * D * Db / (D + Db)
        println("Using D = $D, Db = $Db, Deff = $Deff")

        # Check stability
        if D * dtdx2 ≥ 0.2
            println("Warning: inappropriate scaling of dx and dt due to D, may result in an unstable simulation; Ddt/dx2 = $(D*dtdx2).")
        end
        if Db * dtdx2 ≥ 0.2
            println("Warning: inappropriate scaling of dx and dt due to Db, may result in an unstable simulation; Dbdt/dx2 = $(Db*dtdx2).")
        end
        if Deff * dtdx2 ≥ 0.2
            println("Warning: inappropriate scaling of dx and dt due to Deff, may result in an unstable simulation; Deffdt/dx2 = $(Deff*dtdx2).")
        end

        # Radus in lattice units
        spore_rad_lattice = spore_rad / dx
        println("Spore radius in lattice units: ", spore_rad_lattice)

        # Construct cell wall index map
        steps = [0, -1, 1]
        moore_nbrs = vec([(di, dj, dk) for di in steps, dj in steps, dk in steps])
        # println("Moore neighbors: ", moore_nbrs)

        # Initialise concentrations in cell wall
        for i in 1:N, j in 1:N, k in 1:H
            included = false
            included_nbrs = 0
            for (di, dj, dk) in moore_nbrs
                for sp_cen_idx in sp_cen_indices
                    if (i + di - sp_cen_idx[1] - 1)^2 + (j + dj - sp_cen_idx[2] - 1)^2 + (k + dk - sp_cen_idx[3] - 1)^2 ≤ spore_rad_lattice^2
                        if (di, dj, dk) == (0, 0, 0)
                            included = true
                        else
                            included_nbrs += 1
                        end
                    end
                end
            end
            if included && included_nbrs < 26
                c_init[i, j, k] = 11.0 # Encode cell wall 1.0 + 10
            elseif included && included_nbrs == 26
                c_init[i, j, k] = 100.0 # Encode spore 0.0 + 100
            else
                c_init[i, j, k] = 0.0 # Encode exterior 0.0 + 0
            end
        end
        println("Concentrations initialised.")

        # Determine number of frames
        n_frames = Int(floor(t_max / dt))

        # Allocate arrays for saving data
        c_evolution = zeros(n_save_frames + 1, N, H) # Only a cross-section is saved
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
        c_B_gpu = CUDA.zeros(N, N, H)

        kernel_blocks, kernel_threads = invoke_smart_kernel_3D(size(c_init))
        println("Kernel blocks: $kernel_blocks, kernel threads: $kernel_threads")

        # Run the simulation
        for t in 1:n_frames

            # println("Frame $t")

            # Save frame
            if (t - 1) % save_interval == 0 && save_ct ≤ n_save_frames
                c_A_temp = Array(c_A_gpu)
                # println(size(rem.(c_A_temp[:, N ÷ 2, :], floor.(Int, log10.(c_A_temp.+1e-6)) )))
                c_evolution[save_ct, :, :] .= rem.(c_A_temp[:, N ÷ 2, :], floor.(Int, log10.(c_A_temp[:, N ÷ 2, :].+1e-6))).*c₀
                println(maximum(c_evolution[save_ct, :, :]))
                times[save_ct] = t * dt
                # println("Frame $save_ct saved.")
                save_ct += 1
            end

            # Update the lattice
            @cuda threads=kernel_threads blocks=kernel_blocks update_GPU_hi_res!(c_A_gpu, c_B_gpu, N, H, dtdx2, D, Db, Deff, neumann_z)
            CUDA.synchronize()
            c_A_gpu, c_B_gpu = c_B_gpu, c_A_gpu
            
            # println("Kernel execution completed for Frame $t")

            # Check for threshold crossing
            if !isnothing(c_thresholds)
                if thresh_ct < length(t_thresholds) && t_thresholds[thresh_ct] == 0 && CUDA.reduce(max_reduce_kernel, c_A_gpu) < c_thresholds[thresh_ct]
                    t_thresholds[thresh_ct] = t * dt
                    thresh_ct += 1
                end
            end
        end

        # Save final frame
        c_A_temp = Array(c_A_gpu)
        c_evolution[save_ct, :, :] .= rem.(c_A_temp[:, N ÷ 2, :], floor.(Int, log10.(c_A_temp[:, N ÷ 2, :].+1e-6))).*c₀

        # c_test = zeros(N, H)
        # cw_indices_2D_ne = [(i + sp_cen_indices[1][1], k + sp_cen_indices[1][3]) for (i, j, k) in cw_idx_map if j == 1]
        # cw_indices_2D_se = [(i + sp_cen_indices[1][1], -k + sp_cen_indices[1][3]) for (i, j, k) in cw_idx_map if i > 0 && j == 1 && k > 0]
        # cw_indices_2D_sw = [(-i + sp_cen_indices[1][1], -k + sp_cen_indices[1][3]) for (i, j, k) in cw_idx_map if j == 1]
        # cw_indices_2D_nw = [(-i + sp_cen_indices[1][1], k + sp_cen_indices[1][3]) for (i, j, k) in cw_idx_map if i > 0 && j == 1 && k > 0]
        # cw_indices_2D = vcat(cw_indices_2D_ne, cw_indices_2D_se, cw_indices_2D_sw, cw_indices_2D_nw)
        # cw_indices_cartesian = CartesianIndex.(cw_indices_2D)
        # println("Number of cell wall indices: ", length(cw_indices_cartesian))
        # c_test[cw_indices_cartesian] .= c₀
        # # count nonzero elements
        # println("Nonzero elements: ", count(!iszero, c_test))
        # c_evolution[1, :, :] .= c_test

        return c_evolution, times, t_thresholds
    end
end