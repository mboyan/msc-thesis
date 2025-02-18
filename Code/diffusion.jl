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


    function update_GPU_hi_res!(c_old, c_new, N, H, dtdx2, D, Db, Deff, sp_cen_indices, start_indices, end_indices, cw_idx_map_x, cw_idx_map_y, cw_idx_map_z, spore_rad_lattice, neumann_z)
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
            Deff (float) - the effective diffusion constant at the spore interface
            sp_cen_indices (flat array of int) - the indices of the spore locations
            start_indices (flat array of int) - the start indices of the cell wall locations per spore
            end_indices (flat array of int) - the end indices of the cell wall locations per spore
            cw_idx_map_x (flat array of int) - zero-based indices of the cell wall locations along x
            cw_idx_map_y (flat array of int) - zero-based indices of the cell wall locations along y
            cw_idx_map_z (flat array of int) - zero-based indices of the cell wall locations along z
            spore_rad_lattice (float) - the radius of the spore in lattice units
            neumann_z (bool) - whether to use Neumann boundary conditions in the z-direction
        """
        i, j, k = CUDA.blockIdx().x, CUDA.blockIdx().y, CUDA.blockIdx().z
        ti, tj, tk = CUDA.threadIdx().x, CUDA.threadIdx().y, CUDA.threadIdx().z
    
        # Determine the indices of the current cell
        idx = ((i - 1) * blockDim().x + ti, (j - 1) * blockDim().y + tj, (k - 1) * blockDim().z + tk)

        spore_rad_sq = spore_rad_lattice^2
        spore_half_rad_sq = (spore_rad_lattice*0.5)^2

        # Update the concentration value
        if 1 ≤ idx[1] ≤ N && 1 ≤ idx[2] ≤ N && 1 ≤ idx[3] ≤ H

            # Check if in spore / cell wall
            in_spore = false
            sp_cw_index = 0
            min_dist_sq = N * N + N * N + H * H
            closest_spore = 0
            for n in 1:(length(sp_cen_indices) ÷ 3)
                sp_cen_idx = (sp_cen_indices[3*n - 2], sp_cen_indices[3*n - 1], sp_cen_indices[3*n])
                dist_sq = (idx[1] - sp_cen_idx[1])^2 + (idx[2] - sp_cen_idx[2])^2 + (idx[3] - sp_cen_idx[3])^2
                # Record closest spore
                if dist_sq ≤ min_dist_sq
                    min_dist_sq = dist_sq
                    closest_spore = n
                end
                if dist_sq ≤ spore_rad_sq
                    in_spore = true
                    if dist_sq > spore_half_rad_sq
                        # Check if in cell wall
                        for m in start_indices[n]:end_indices[n]
                            if idx[1] == cw_idx_map_x[m] && idx[2] == cw_idx_map_y[m] && idx[3] == cw_idx_map_z[m]
                                # Save the index of the spore
                                sp_cw_index = n
                                break
                            end
                        end
                    end
                    break
                end
            end

            center = c_old[idx...]

            vneum_nbrs = ((idx[1], idx[2], mod1(idx[3] - 1, H)), (idx[1], idx[2], mod1(idx[3] + 1, H)),
                        (idx[1], mod1(idx[2] - 1, N), idx[3]), (idx[1], mod1(idx[2] + 1, N), idx[3]),
                        (mod1(idx[1] - 1, N), idx[2], idx[3]), (mod1(idx[1] + 1, N), idx[2], idx[3]))
            
            # Take absolute of relative coordinates
            vneum_abs = ((abs(vneum_nbrs[1][1] - sp_cen_indices[3*closest_spore - 2]),
            abs(vneum_nbrs[1][2] - sp_cen_indices[3*closest_spore - 1]),
            abs(vneum_nbrs[1][3] - sp_cen_indices[3*closest_spore])),
            (abs(vneum_nbrs[2][1] - sp_cen_indices[3*closest_spore - 2]),
            abs(vneum_nbrs[2][2] - sp_cen_indices[3*closest_spore - 1]),
            abs(vneum_nbrs[2][3] - sp_cen_indices[3*closest_spore])),
            (abs(vneum_nbrs[3][1] - sp_cen_indices[3*closest_spore - 2]),
            abs(vneum_nbrs[3][2] - sp_cen_indices[3*closest_spore - 1]),
            abs(vneum_nbrs[3][3] - sp_cen_indices[3*closest_spore])),
            (abs(vneum_nbrs[4][1] - sp_cen_indices[3*closest_spore - 2]),
            abs(vneum_nbrs[4][2] - sp_cen_indices[3*closest_spore - 1]),
            abs(vneum_nbrs[4][3] - sp_cen_indices[3*closest_spore])),
            (abs(vneum_nbrs[5][1] - sp_cen_indices[3*closest_spore - 2]),
            abs(vneum_nbrs[5][2] - sp_cen_indices[3*closest_spore - 1]),
            abs(vneum_nbrs[5][3] - sp_cen_indices[3*closest_spore])),
            (abs(vneum_nbrs[6][1] - sp_cen_indices[3*closest_spore - 2]),
            abs(vneum_nbrs[6][2] - sp_cen_indices[3*closest_spore - 1]),
            abs(vneum_nbrs[6][3] - sp_cen_indices[3*closest_spore])))

            # Cell wall site
            if sp_cw_index > 0 && min_dist_sq > spore_half_rad_sq * 1.1

                # Check bottom neighbour
                if vneum_abs[1][1]^2 + vneum_abs[1][2]^2 + vneum_abs[1][3]^2 ≤ spore_rad_sq
                    if vneum_nbrs[1][1] in cw_idx_map_x && vneum_nbrs[1][2] in cw_idx_map_y && vneum_nbrs[1][3] in cw_idx_map_z
                        diff_bottom = Db * dtdx2 * (c_old[vneum_nbrs[1]...] - center)
                    else
                        diff_bottom = 0.0
                    end
                else
                    diff_bottom = Deff * dtdx2 * (c_old[vneum_nbrs[1]...] - center)
                end
                # Check top neighbour
                if vneum_abs[2][1]^2 + vneum_abs[2][2]^2 + vneum_abs[2][3]^2 ≤ spore_rad_sq
                    if vneum_nbrs[2][1] in cw_idx_map_x && vneum_nbrs[2][2] in cw_idx_map_y && vneum_nbrs[2][3] in cw_idx_map_z
                        diff_top = Db * dtdx2 * (c_old[vneum_nbrs[2]...] - center)
                    else
                        diff_top = 0.0
                    end
                else
                    diff_top = Deff * dtdx2 * (c_old[vneum_nbrs[2]...] - center)
                end
                # Check left neighbour
                if vneum_abs[3][1]^2 + vneum_abs[3][2]^2 + vneum_abs[3][3]^2 ≤ spore_rad_sq
                    if vneum_nbrs[3][1] in cw_idx_map_x && vneum_nbrs[3][2] in cw_idx_map_y && vneum_nbrs[3][3] in cw_idx_map_z
                        diff_left = Db * dtdx2 * (c_old[vneum_nbrs[3]...] - center)
                    else
                        diff_left = 0.0
                    end
                else
                    diff_left = Deff * dtdx2 * (c_old[vneum_nbrs[3]...] - center)
                end
                # Check right neighbour
                if vneum_abs[4][1]^2 + vneum_abs[4][2]^2 + vneum_abs[4][3]^2 ≤ spore_rad_sq
                    if vneum_nbrs[4][1] in cw_idx_map_x && vneum_nbrs[4][2] in cw_idx_map_y && vneum_nbrs[4][3] in cw_idx_map_z
                        diff_right = Db * dtdx2 * (c_old[vneum_nbrs[4]...] - center)
                    else
                        diff_right = 0.0
                    end
                else
                    diff_right = Deff * dtdx2 * (c_old[vneum_nbrs[4]...] - center)
                end
                # Check front neighbour
                if vneum_abs[5][1]^2 + vneum_abs[5][2]^2 + vneum_abs[5][3]^2 ≤ spore_rad_sq
                    if vneum_nbrs[5][1] in cw_idx_map_x && vneum_nbrs[5][2] in cw_idx_map_y && vneum_nbrs[5][3] in cw_idx_map_z
                        diff_front = Db * dtdx2 * (c_old[vneum_nbrs[5]...] - center)
                    else
                        diff_front = 0.0
                    end
                else
                    diff_front = Deff * dtdx2 * (c_old[vneum_nbrs[5]...] - center)
                end
                # Check back neighbour
                if vneum_abs[6][1]^2 + vneum_abs[6][2]^2 + vneum_abs[6][3]^2 ≤ spore_rad_sq
                    if vneum_nbrs[6][1] in cw_idx_map_x && vneum_nbrs[6][2] in cw_idx_map_y && vneum_nbrs[6][3] in cw_idx_map_z
                        diff_back = Db * dtdx2 * (c_old[vneum_nbrs[6]...] - center)
                    else
                        diff_back = 0.0
                    end
                else
                    diff_back = Deff * dtdx2 * (c_old[vneum_nbrs[6]...] - center)
                end
                
                c_new[idx...] = center + diff_bottom + diff_top + diff_left + diff_right + diff_front + diff_back
                
            elseif !in_spore && min_dist_sq < spore_rad_sq * 1.25
                # Exterior site close to spore

                # Check bottom neighbour
                if vneum_abs[1][1]^2 + vneum_abs[1][2]^2 + vneum_abs[1][3]^2 ≤ spore_rad_sq
                    diff_bottom = Deff * dtdx2 * (c_old[vneum_nbrs[1]...] - center)
                else
                    diff_bottom = D * dtdx2 * (c_old[vneum_nbrs[1]...] - center)
                end
                # Check top neighbour
                if vneum_abs[2][1]^2 + vneum_abs[2][2]^2 + vneum_abs[2][3]^2 ≤ spore_rad_sq
                    diff_top = Deff * dtdx2 * (c_old[vneum_nbrs[2]...] - center)
                else
                    diff_top = D * dtdx2 * (c_old[vneum_nbrs[2]...] - center)
                end
                # Check left neighbour
                if vneum_abs[3][1]^2 + vneum_abs[3][2]^2 + vneum_abs[3][3]^2 ≤ spore_rad_sq
                    diff_left = Deff * dtdx2 * (c_old[vneum_nbrs[3]...] - center)
                else
                    diff_left = D * dtdx2 * (c_old[vneum_nbrs[3]...] - center)
                end
                # Check right neighbour
                if vneum_abs[4][1]^2 + vneum_abs[4][2]^2 + vneum_abs[4][3]^2 ≤ spore_rad_sq
                    diff_right = Deff * dtdx2 * (c_old[vneum_nbrs[4]...] - center)
                else
                    diff_right = D * dtdx2 * (c_old[vneum_nbrs[4]...] - center)
                end
                # Check front neighbour
                if vneum_abs[5][1]^2 + vneum_abs[5][2]^2 + vneum_abs[5][3]^2 ≤ spore_rad_sq
                    diff_front = Deff * dtdx2 * (c_old[vneum_nbrs[5]...] - center)
                else
                    diff_front = D * dtdx2 * (c_old[vneum_nbrs[5]...] - center)
                end
                # Check back neighbour
                if vneum_abs[6][1]^2 + vneum_abs[6][2]^2 + vneum_abs[6][3]^2 ≤ spore_rad_sq
                    diff_back = Deff * dtdx2 * (c_old[vneum_nbrs[6]...] - center)
                else
                    diff_back = D * dtdx2 * (c_old[vneum_nbrs[6]...] - center)
                end

                c_new[idx...] = center + diff_bottom + diff_top + diff_left + diff_right + diff_front + diff_back

            elseif !in_spore && min_dist_sq ≥ spore_rad_sq * 1.25
                # Exterior site far from spore
                c_new[idx...] = center + D * dtdx2 * (c_old[vneum_nbrs[1]...] + c_old[vneum_nbrs[2]...] + 
                                                    c_old[vneum_nbrs[3]...] + c_old[vneum_nbrs[4]...] +
                                                    c_old[vneum_nbrs[5]...] + c_old[vneum_nbrs[6]...] - 6 * center)
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

        # Construct cell wall index map
        steps = [0, -1, 1]
        moore_nbrs = vec([(di, dj, dk) for di in steps, dj in steps, dk in steps])
        # println("Moore neighbors: ", moore_nbrs)
        cw_idx_map = []
        spore_rad_lattice = spore_rad / dx
        println("Spore radius in lattice units: ", spore_rad_lattice)
        spore_bnds = Int(ceil(spore_rad_lattice)+1)
        println("Spore bounds: ", spore_bnds)
        cw_idx_lengths = zeros(Int, length(sp_cen_indices)+1)
        cw_idx_lengths[1] = 0
        for (n, sp_cen_idx) in enumerate(sp_cen_indices)
            cw_idx_length = 0
            for i in (-spore_bnds):spore_bnds, j in (-spore_bnds):spore_bnds, k in (-spore_bnds):spore_bnds
                included = false
                included_nbrs = 0
                for (di, dj, dk) in moore_nbrs
                    if (i + di - 1)^2 + (j + dj - 1)^2 + (k + dk - 1)^2 ≤ spore_rad_lattice^2
                        if (di, dj, dk) == (0, 0, 0)
                            included = true
                        else
                            included_nbrs += 1
                        end
                    end
                end
                if included && included_nbrs < 26
                    push!(cw_idx_map, (sp_cen_idx[1] + i - 1, sp_cen_idx[2] + j - 1, sp_cen_idx[3] + k - 1))
                    cw_idx_length += 1
                end
                cw_idx_lengths[n+1] = cw_idx_length
            end
        end
        println(length(cw_idx_map), " cell wall indices found.")
        println("Cell wall index lengths: ", cw_idx_lengths)

        # Precompute start and end indices for each spore
        start_indices = [sum(cw_idx_lengths[1:i])+1 for i in 1:length(cw_idx_lengths)-1]
        end_indices = [sum(cw_idx_lengths[1:i+1]) for i in 1:length(cw_idx_lengths)-1]
        println("Start indices: ", start_indices)
        println("End indices: ", end_indices)

        # Initialise concentrations in cell wall
        cw_indices_cartesian = CartesianIndex.(cw_idx_map)
        c_init[cw_indices_cartesian] .= c₀
        # for sp_cen_idx in sp_cen_indices
        #     steps = [-1, 1]
        #     full_quadrants = [1, 4, 6, 7]
        #     transformations = vec(collect(IterTools.product(steps, steps, steps)))
        #     cw_indices_2D = vcat([
        #         [(i * t[1] + sp_cen_idx[1], j * t[2] + sp_cen_idx[2], k * t[3] + sp_cen_idx[3])
        #             for (i, j, k) in cw_idx_map if (n ∉ full_quadrants && i > 0 && j > 0 && k > 0) || (n in full_quadrants)]
        #         for (n, t) in enumerate(transformations)
        #     ]...)
        #     cw_indices_cartesian = CartesianIndex.(cw_indices_2D)
        #     c_init[cw_indices_cartesian] .= c₀
        # end

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
        cw_idx_map_x = [x[1] for x in cw_idx_map]
        cw_idx_map_y = [x[2] for x in cw_idx_map]
        cw_idx_map_z = [x[3] for x in cw_idx_map]
        cw_idx_map_x_gpu = cu(cw_idx_map_x)
        cw_idx_map_y_gpu = cu(cw_idx_map_y)
        cw_idx_map_z_gpu = cu(cw_idx_map_z)
        flat_sp_cen_indices = vcat([collect(t) for t in sp_cen_indices]...)
        sp_cen_indices_gpu = cu(flat_sp_cen_indices)
        start_indices_gpu = cu(start_indices)
        end_indices_gpu = cu(end_indices)

        kernel_blocks, kernel_threads = invoke_smart_kernel_3D(size(c_init))
        println("Kernel blocks: $kernel_blocks, kernel threads: $kernel_threads")

        # Run the simulation
        for t in 1:n_frames

            # println("Frame $t")

            # Save frame
            if (t - 1) % save_interval == 0 && save_ct ≤ n_save_frames
                println(maximum(Array(c_A_gpu)))
                c_evolution[save_ct, :, :] .= Array(c_A_gpu)[:, N ÷ 2, :]
                times[save_ct] = t * dt
                # println("Frame $save_ct saved.")
                save_ct += 1
            end

            # Update the lattice
            @cuda threads=kernel_threads blocks=kernel_blocks update_GPU_hi_res!(c_A_gpu, c_B_gpu, N, H, dtdx2, D, Db, Deff,
                                                                                sp_cen_indices_gpu, start_indices_gpu, end_indices_gpu,
                                                                                cw_idx_map_x_gpu, cw_idx_map_y_gpu, cw_idx_map_z_gpu,
                                                                                spore_rad_lattice, neumann_z)
            c_A_gpu, c_B_gpu = c_B_gpu, c_A_gpu
            CUDA.synchronize()
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
        c_evolution[save_ct, :, :] .= Array(c_A_gpu)[:, N ÷ 2, :]

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