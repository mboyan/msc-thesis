module Diffusion
__precompile__(false)
    """
    Functions for solving the diffusion equation.
    """

    using ArgCheck
    using CUDA
    using IterTools

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


    function compare_indices(indices1, indices2)
        """
        Utility function for comparing tuples of indices.
        inputs:
            indices1 (tuple) - the first tuple of indices
            indices2 (tuple) - the second tuple of indices
        """
        return indices1[1] == indices2[1] && indices1[2] == indices2[2] && indices1[3] == indices2[3]
    end


    function update_GPU_hi_res!(c_old, c_new, N, H, dtdx2, D, Db, spore_center_indices, cw_index_map, spore_rad_lattice, neumann_z)
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
            spore_center_indices (array of tuples) - the indices of the spore locations
            cw_index_map (array of tuples) - zero-based indices of the cell wall locations in 1 quadrant
            spore_rad_lattice (float) - the radius of the spore in lattice units
            neumann_z (bool) - whether to use Neumann boundary conditions in the z-direction
        """
        i, j, k = CUDA.blockIdx().x, CUDA.blockIdx().y, CUDA.blockIdx().z
        ti, tj, tk = CUDA.threadIdx().x, CUDA.threadIdx().y, CUDA.threadIdx().z
    
        # Determine the indices of the current cell
        idx = ((i - 1) * blockDim().x + ti, (j - 1) * blockDim().y + tj, (k - 1) * blockDim().z + tk)
        
        # Check if in spore
        in_spore = false
        sp_cw_index = 0
        max_dist = norm([N, N, H])
        closest_spore = 0
        for (n, spore_center_idx) in enumerate(spore_center_indices)
            dist = norm([idx[x] - spore_center_idx[x] for x in 1:3])
            # Record closest spore
            if dist < max_dist
                max_dist = dist
                closest_spore = n
            end
            if dist ≤ spore_rad_lattice
                in_spore = true
                # Check if in cell wall
                for cw_idx in cw_index_map
                    # Get all symmetries
                    cw_idx_neu = (cw_idx[1] + spore_center_idx[1], cw_idx[2] + spore_center_idx[2], cw_idx[3] + spore_center_idx[3])
                    cw_idx_seu = (cw_idx_neu[1], -cw_idx[2] + spore_center_idx[2], cw_idx_neu[3])
                    cw_idx_nwu = (-cw_idx[1] + spore_center_idx[1], cw_idx_neu[2], cw_idx_neu[3])
                    cw_idx_swu = (cw_idx_nwu[1], cw_idx_seu[2], cw_idx_neu[3])
                    cw_idx_ned = (cw_idx_neu[1], cw_idx_ne[2], -cw_idx[3] + spore_center_idx[3])
                    cw_idx_sed = (cw_idx_seu[1], cw_idx_seu[2], cw_idx_ned[3])
                    cw_idx_nwd = (cw_idx_nwu[1], cw_idx_nwu[2], cw_idx_ned[3])
                    cw_idx_swd = (cw_idx_swu[1], cw_idx_swu[2], cw_idx_ned[3])
                    # if idx[1] == cw_idx_neu[1] && idx[2] == cw_idx_neu[2] && idx[3] == cw_idx_neu[3] ||
                    #     idx[1] == cw_idx_seu[1] && idx[2] == cw_idx_seu[2] && idx[3] == cw_idx_seu[3] ||
                    #     idx[1] == cw_idx_nwu[1] && idx[2] == cw_idx_nwu[2] && idx[3] == cw_idx_nwu[3] ||
                    #     idx[1] == cw_idx_swu[1] && idx[2] == cw_idx_swu[2] && idx[3] == cw_idx_swu[3] ||
                    #     idx[1] == cw_idx_ned[1] && idx[2] == cw_idx_ned[2] && idx[3] == cw_idx_ned[3] ||
                    #     idx[1] == cw_idx_sed[1] && idx[2] == cw_idx_sed[2] && idx[3] == cw_idx_sed[3] ||
                    #     idx[1] == cw_idx_nwd[1] && idx[2] == cw_idx_nwd[2] && idx[3] == cw_idx_nwd[3] ||
                    #     idx[1] == cw_idx_swd[1] && idx[2] == cw_idx_swd[2] && idx[3] == cw_idx_swd[3]
                    #     # Save the index of the spore
                    #     sp_cw_index = n
                    #     break
                    # end
                    if compare_indices(idx, cw_idx_neu) || compare_indices(idx, cw_idx_seu) ||
                        compare_indices(idx, cw_idx_nwu) || compare_indices(idx, cw_idx_swu) ||
                        compare_indices(idx, cw_idx_ned) || compare_indices(idx, cw_idx_sed) ||
                        compare_indices(idx, cw_idx_nwd) || compare_indices(idx, cw_idx_swd)
                        # Save the index of the spore
                        sp_cw_index = n
                        break
                    end
                end
                break
            end
        end

        # Update the concentration value
        if 1 ≤ idx[1] ≤ N && 1 ≤ idx[2] ≤ N && 1 ≤ idx[3] ≤ H
            
            von_neumann_nbrs = [(idx[1], idx[2], mod1(idx[3] - 1, H)), (idx[1], idx[2], mod1(idx[3] + 1, H)),
                                (idx[1], mod1(idx[2] - 1, N), idx[3]), (idx[1], mod1(idx[2] + 1, N), idx[3]),
                                (mod1(idx[1] - 1, N), idx[2], idx[3]), (mod1(idx[1] + 1, N), idx[2], idx[3])]

            # Take absolute of relative coordinates
            von_neumann_abs = [[abs(von_neumann_nbrs[x][1] - spore_center_indices[sp_cw_index]),
                                abs(von_neumann_nbrs[x][2] - spore_center_indices[sp_cw_index]),
                                abs(von_neumann_nbrs[x][3] - spore_center_indices[sp_cw_index])] for x in 1:6]

            # Cell wall site
            if sp_cw_index > 0
                # Check bottom neighbour
                if norm(von_neumann_abs[1]) ≤ spore_rad_lattice
                    if von_neumann_abs[1] in cw_index_map
                        diff_bottom = Db * dtdx2 * (c_old[von_neumann_nbrs[1]...] - c_old[idx...])
                    else
                        diff_bottom = 0
                    end
                else
                    diff_bottom = D * dtdx2 * (c_old[von_neumann_nbrs[1]...] - c_old[idx...])
                end
                # Check top neighbour
                if norm(von_neumann_abs[2]) ≤ spore_rad_lattice
                    if von_neumann_abs[2] in cw_index_map
                        diff_top = Db * dtdx2 * (c_old[von_neumann_nbrs[2]...] - c_old[idx...])
                    else
                        diff_top = 0
                    end
                else
                    diff_top = D * dtdx2 * (c_old[von_neumann_nbrs[2]...] - c_old[idx...])
                end
                # Check left neighbour
                if norm(von_neumann_abs[3]) ≤ spore_rad_lattice
                    if von_neumann_abs[3] in cw_index_map
                        diff_left = Db * dtdx2 * (c_old[von_neumann_nbrs[3]...] - c_old[idx...])
                    else
                        diff_left = 0
                    end
                else
                    diff_left = D * dtdx2 * (c_old[von_neumann_nbrs[3]...] - c_old[idx...])
                end
                # Check right neighbour
                if norm(von_neumann_abs[4]) ≤ spore_rad_lattice
                    if von_neumann_abs[4] in cw_index_map
                        diff_right = Db * dtdx2 * (c_old[von_neumann_nbrs[4]...] - c_old[idx...])
                    else
                        diff_right = 0
                    end
                else
                    diff_right = D * dtdx2 * (c_old[von_neumann_nbrs[4]...] - c_old[idx...])
                end
                # Check front neighbour
                if norm(von_neumann_abs[5]) ≤ spore_rad_lattice
                    if von_neumann_abs[5] in cw_index_map
                        diff_front = Db * dtdx2 * (c_old[von_neumann_nbrs[5]...] - c_old[idx...])
                    else
                        diff_front = 0
                    end
                else
                    diff_front = D * dtdx2 * (c_old[von_neumann_nbrs[5]...] - c_old[idx...])
                end
                # Check back neighbour
                if norm(von_neumann_abs[6]) ≤ spore_rad_lattice
                    if von_neumann_abs[6] in cw_index_map
                        diff_back = Db * dtdx2 * (c_old[von_neumann_nbrs[6]...] - c_old[idx...])
                    else
                        diff_back = 0
                    end
                else
                    diff_back = D * dtdx2 * (c_old[von_neumann_nbrs[6]...] - c_old[idx...])
                end
            end

            c_new[idx...] = c_old[idx...] + diff_bottom + diff_top + diff_left + diff_right + diff_front + diff_back
            
        elseif !in_spore
            # Exterior site

            # Check bottom neighbour
            if norm(von_neumann_abs[1]) ≤ spore_rad_lattice
                diff_bottom = Db * dtdx2 * (c_old[von_neumann_nbrs[1]...] - c_old[idx...])
            else
                diff_bottom = D * dtdx2 * (c_old[von_neumann_nbrs[1]...] - c_old[idx...])
            end
            # Check top neighbour
            if norm(von_neumann_abs[2]) ≤ spore_rad_lattice
                diff_top = Db * dtdx2 * (c_old[von_neumann_nbrs[2]...] - c_old[idx...])
            else
                diff_top = D * dtdx2 * (c_old[von_neumann_nbrs[2]...] - c_old[idx...])
            end
            # Check left neighbour
            if norm(von_neumann_abs[3]) ≤ spore_rad_lattice
                diff_left = Db * dtdx2 * (c_old[von_neumann_nbrs[3]...] - c_old[idx...])
            else
                diff_left = D * dtdx2 * (c_old[von_neumann_nbrs[3]...] - c_old[idx...])
            end
            # Check right neighbour
            if norm(von_neumann_abs[4]) ≤ spore_rad_lattice
                diff_right = Db * dtdx2 * (c_old[von_neumann_nbrs[4]...] - c_old[idx...])
            else
                diff_right = D * dtdx2 * (c_old[von_neumann_nbrs[4]...] - c_old[idx...])
            end
            # Check front neighbour
            if norm(von_neumann_abs[5]) ≤ spore_rad_lattice
                diff_front = Db * dtdx2 * (c_old[von_neumann_nbrs[5]...] - c_old[idx...])
            else
                diff_front = D * dtdx2 * (c_old[von_neumann_nbrs[5]...] - c_old[idx...])
            end
            # Check back neighbour
            if norm(von_neumann_abs[6]) ≤ spore_rad_lattice
                diff_back = Db * dtdx2 * (c_old[von_neumann_nbrs[6]...] - c_old[idx...])
            else
                diff_back = D * dtdx2 * (c_old[von_neumann_nbrs[6]...] - c_old[idx...])
            end

            c_new[idx...] = c_old[idx...] + diff_bottom + diff_top + diff_left + diff_right + diff_front + diff_back
        end
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
            if t % save_interval == 0 && save_ct ≤ n_save_frames
                c_evolution[save_ct, :, :, :] .= Array(c_A_gpu)
                times[save_ct] = t * dt
                save_ct += 1
                # println("Frame $save_ct saved.")
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
            if t % save_interval == 0 && save_ct ≤ n_save_frames
                c_med_evolution[save_ct, :, :, :] .= Array(c_A_gpu)
                c_spore_evolution[save_ct] = CUDA.reduce(max_reduce_kernel, c_spore_gpu, init=-Inf)
                times[save_ct] = t * dt
                save_ct += 1
                # println("Frame $save_ct saved.")
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

    function diffusion_time_dependent_GPU_hi_res(c_init, c₀, spore_center_indices, spore_rad, t_max; D=1.0, Pₛ=1.0, dt=0.005, dx=0.2, n_save_frames=100,
        c_thresholds=nothing, neumann_z=false)
        """
        Compute the evolution of a square lattice of concentration scalars
        based on the time-dependent diffusion equation.
        inputs:
            c_init (vector of float) - the initial state of the lattice
            c₀ (float) - the initial concentration at the spore
            spore_center_indices (array of tuples) - the indices of the spore locations
            spore_rad (float) - the radius of the spore
            t_max (int) - a maximum number of iterations
            D (float) - the diffusion constant; defaults to 1
            Db (float) - the diffusion constant through the spore; defaults to nothing
            Ps (float) - the permeation constant through the spore barrier; defaults to 1
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

        @assert length(spore_center_indices[1]) == 3 "spore_idx must be a 3D array"

        GC.gc()

        # Determine number of lattice rows/columns
        N = size(c_init)[1]
        H = size(c_init)[3]

        @assert spore_rad < N && spore_rad <H "spore_rad must be less than N and H"

        # Save update factor
        dtdx2 = dt / (dx^2)

        # Correction factor for permeation
        Db = Pₛ * dx
        println("Using D = $D, Db = $Db, Ps = $Pₛ")

        # Check stability
        if  D * dtdx2 ≥ 0.2
            println("Warning: inappropriate scaling of dx and dt due to D, may result in an unstable simulation; Ddt/dx2 = $(D*dtdx2).")
        end
        if  Db * dtdx2 ≥ 0.2
            println("Warning: inappropriate scaling of dx and dt due to Db, may result in an unstable simulation; Dbdt/dx2 = $(Db*dtdx2).")
        end

        # # Create spore mask
        # indices = [(i, j, k) for i in 1:N, j in 1:N, k in 1:H]
        # distances = [(i - spore_center_idx[1])^2 + (j - spore_center_idx[2])^2 + (k - spore_center_idx[3])^2 for (i, j, k) in indices]
        # spore_mask .= reshape([d ≤ spore_rad^2 for d in distances], N, N, H)

        # Construct cell wall index map
        steps = [0, -1, 1]
        moore_nbrs = vec([(di, dj, dk) for di in steps, dj in steps, dk in steps])
        # println("Moore neighbors: ", moore_nbrs)
        cw_index_map = []
        spore_rad_lattice = spore_rad / dx
        println("Spore radius in lattice units: ", spore_rad_lattice)
        spore_bnds = Int(ceil(spore_rad_lattice)+1)
        println("Spore bounds: ", spore_bnds)
        for i in 1:spore_bnds, j in 1:spore_bnds, k in 1:spore_bnds
            excluded = false
            excluded_nbrs = 0
            for (di, dj, dk) in moore_nbrs
                # println(sqrt((i + di)^2 + (j + dj)^2 + (k + dk)^2))
                if (i + di)^2 + (j + dj)^2 + (k + dk)^2 > spore_rad_lattice^2
                    if (di, dj, dk) == (0, 0, 0)
                        excluded = true
                    else
                        excluded_nbrs += 1
                        break
                    end
                end
            end
            if !excluded && excluded_nbrs > 0
                push!(cw_index_map, (i - 1, j - 1, k - 1))
            end
        end
        println(length(cw_index_map), " cell wall indices found.")

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

        # Initialise lattice states on GPU
        c_A_gpu = cu(c_init)
        c_B_gpu = CUDA.zeros(N, N, H)
        cw_index_map_gpu = cu(cw_index_map)
        spore_center_indices_gpu = cu(spore_center_indices)

        kernel_blocks, kernel_threads = invoke_smart_kernel_3D(size(c_init))

        # Run the simulation
        for t in 1:n_frames

            # Save frame
            if t % save_interval == 0 && save_ct ≤ n_save_frames
                c_evolution[save_ct, :, :] .= Array(c_A_gpu)[:, N ÷ 2, :]
                times[save_ct] = t * dt
                save_ct += 1
                # println("Frame $save_ct saved.")
            end

            # Update the lattice
            @cuda threads=kernel_threads blocks=kernel_blocks update_GPU_hi_res!(c_A_gpu, c_B_gpu, N, H, dtdx2, D, Db, spore_center_indices_gpu, cw_index_map_gpu, spore_rad_lattice, neumann_z)
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
        c_evolution[save_ct, :, :] .= Array(c_A_gpu)[:, N ÷ 2, :]

        c_test = zeros(N, H)
        cw_indices_2D_ne = [(i + spore_center_indices[1][1], k + spore_center_indices[1][3]) for (i, j, k) in cw_index_map if j == 1]
        cw_indices_2D_se = [(i + spore_center_indices[1][1], -k + spore_center_indices[1][3]) for (i, j, k) in cw_index_map if i > 0 && j == 1 && k > 0]
        cw_indices_2D_sw = [(-i + spore_center_indices[1][1], -k + spore_center_indices[1][3]) for (i, j, k) in cw_index_map if j == 1]
        cw_indices_2D_nw = [(-i + spore_center_indices[1][1], k + spore_center_indices[1][3]) for (i, j, k) in cw_index_map if i > 0 && j == 1 && k > 0]
        cw_indices_2D = vcat(cw_indices_2D_ne, cw_indices_2D_se, cw_indices_2D_sw, cw_indices_2D_nw)
        cw_indices_cartesian = CartesianIndex.(cw_indices_2D)
        println("Number of cell wall indices: ", length(cw_indices_cartesian))
        c_test[cw_indices_cartesian] .= c₀
        # count nonzero elements
        println("Nonzero elements: ", count(!iszero, c_test))
        c_evolution[1, :, :] .= c_test

        return c_evolution, times, t_thresholds
    end
end