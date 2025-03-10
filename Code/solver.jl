module Solver
    __precompile__(false)
    """
    Functions used with different diffusion solvers.
    """

    using SparseArrays
    using LinearAlgebra
    using LinearMaps
    using CUDA
    
    export invoke_smart_kernel_3D
    export max_reduce_kernel
    export update_GPU!
    export update_GPU_low_res!
    export update_GPU_hi_res!
    export update_GPU_hi_res_coeffs!
    export initialise_lattice_and_operator_GPU!
    export initialise_lattice_and_operator_GPU_abs_bndry!
    # export initialise_lattice_and_build_operator!

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

    function update_GPU_hi_res!(lattice_old, lattice_new, N, H, dtdx2, D, Dcw, Db, region_ids, abs_bndry, neumann_z)
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
            Dcw (float) - the diffusion constant through the spore
            Db (float) - the effective diffusion constant at the spore interface
            region_ids (array) - the region IDs of the lattice
            abs_bndry (bool) - whether to use absorbing boundary conditions
            neumann_z (bool) - whether to use Neumann boundary conditions in the z-direction
        """
        i, j, k = CUDA.blockIdx().x, CUDA.blockIdx().y, CUDA.blockIdx().z
        ti, tj, tk = CUDA.threadIdx().x, CUDA.threadIdx().y, CUDA.threadIdx().z
    
        # Determine the indices of the current cell
        idx = ((i - 1) * blockDim().x + ti, (j - 1) * blockDim().y + tj, (k - 1) * blockDim().z + tk)

        # Apply absorbing boundary conditions
        if abs_bndry
            if idx[1] == 1 || idx[1] == N || idx[2] == 1 || idx[2] == N || idx[3] == 1 || idx[3] == H
                return nothing
            end
        end

        # Update the concentration value
        if 1 ≤ idx[1] ≤ N && 1 ≤ idx[2] ≤ N && 1 ≤ idx[3] ≤ H

            diff = 0.0

            @inbounds begin
                if region_ids[idx...] == 0 # Exterior site
                    # Check bottom neighbour
                    if region_ids[idx[1], idx[2], mod1(idx[3] - 1, H)] == 0 # Exterior - exterior
                        diff += D * (lattice_old[idx[1], idx[2], mod1(idx[3] - 1, H)] - lattice_old[idx...])
                    elseif region_ids[idx[1], idx[2], mod1(idx[3] - 1, H)] == 1 # Exterior - cell wall
                        diff += Db * (lattice_old[idx[1], idx[2], mod1(idx[3] - 1, H)] - lattice_old[idx...])
                    end
                    # Check top neighbour
                    if region_ids[idx[1], idx[2], mod1(idx[3] + 1, H)] == 0 # Exterior - exterior
                        diff += D * (lattice_old[idx[1], idx[2], mod1(idx[3] + 1, H)] - lattice_old[idx...])
                    elseif region_ids[idx[1], idx[2], mod1(idx[3] + 1, H)] == 1 # Exterior - cell wall
                        diff += Db * (lattice_old[idx[1], idx[2], mod1(idx[3] + 1, H)] - lattice_old[idx...])
                    end
                    # Check left neighbour
                    if region_ids[idx[1], mod1(idx[2] - 1, N), idx[3]] == 0 # Exterior - exterior
                        diff += D * (lattice_old[idx[1], mod1(idx[2] - 1, N), idx[3]] - lattice_old[idx...])
                    elseif region_ids[idx[1], mod1(idx[2] - 1, N), idx[3]] == 1 # Exterior - cell wall
                        diff += Db * (lattice_old[idx[1], mod1(idx[2] - 1, N), idx[3]] - lattice_old[idx...])
                    end
                    # Check right neighbour
                    if region_ids[idx[1], mod1(idx[2] + 1, N), idx[3]] == 0 # Exterior - exterior
                        diff += D * (lattice_old[idx[1], mod1(idx[2] + 1, N), idx[3]] - lattice_old[idx...])
                    elseif region_ids[idx[1], mod1(idx[2] + 1, N), idx[3]] == 1 # Exterior - cell wall
                        diff += Db * (lattice_old[idx[1], mod1(idx[2] + 1, N), idx[3]] - lattice_old[idx...])
                    end
                    # Check front neighbour
                    if region_ids[mod1(idx[1] - 1, N), idx[2], idx[3]] == 0 # Exterior - exterior
                        diff += D * (lattice_old[mod1(idx[1] - 1, N), idx[2], idx[3]] - lattice_old[idx...])
                    elseif region_ids[mod1(idx[1] - 1, N), idx[2], idx[3]] == 1 # Exterior - cell wall
                        diff += Db * (lattice_old[mod1(idx[1] - 1, N), idx[2], idx[3]] - lattice_old[idx...])
                    end
                    # Check back neighbour
                    if region_ids[mod1(idx[1] + 1, N), idx[2], idx[3]] == 0 # Exterior - exterior
                        diff += D * (lattice_old[mod1(idx[1] + 1, N), idx[2], idx[3]] - lattice_old[idx...])
                    elseif region_ids[mod1(idx[1] + 1, N), idx[2], idx[3]] == 1 # Exterior - cell wall
                        diff += Db * (lattice_old[mod1(idx[1] + 1, N), idx[2], idx[3]] - lattice_old[idx...])
                    end
                elseif region_ids[idx...] == 1 # Cell wall site
                    # Check bottom neighbour
                    if region_ids[idx[1], idx[2], mod1(idx[3] - 1, H)] == 0 # Cell wall - exterior
                        diff += Db * (lattice_old[idx[1], idx[2], mod1(idx[3] - 1, H)] - lattice_old[idx...])
                    elseif region_ids[idx[1], idx[2], mod1(idx[3] - 1, H)] == 1 # Cell wall - cell wall
                        diff += Dcw * (lattice_old[idx[1], idx[2], mod1(idx[3] - 1, H)] - lattice_old[idx...])
                    end
                    # Check top neighbour
                    if region_ids[idx[1], idx[2], mod1(idx[3] + 1, H)] == 0 # Cell wall - exterior
                        diff += Db * (lattice_old[idx[1], idx[2], mod1(idx[3] + 1, H)] - lattice_old[idx...])
                    elseif region_ids[idx[1], idx[2], mod1(idx[3] + 1, H)] == 1 # Cell wall - cell wall
                        diff += Dcw * (lattice_old[idx[1], idx[2], mod1(idx[3] + 1, H)] - lattice_old[idx...])
                    end
                    # Check left neighbour
                    if region_ids[idx[1], mod1(idx[2] - 1, N), idx[3]] == 0 # Cell wall - exterior
                        diff += Db * (lattice_old[idx[1], mod1(idx[2] - 1, N), idx[3]] - lattice_old[idx...])
                    elseif region_ids[idx[1], mod1(idx[2] - 1, N), idx[3]] == 1 # Cell wall - cell wall
                        diff += Dcw * (lattice_old[idx[1], mod1(idx[2] - 1, N), idx[3]] - lattice_old[idx...])
                    end
                    # Check right neighbour
                    if region_ids[idx[1], mod1(idx[2] + 1, N), idx[3]] == 0 # Cell wall - exterior
                        diff += Db * (lattice_old[idx[1], mod1(idx[2] + 1, N), idx[3]] - lattice_old[idx...])
                    elseif region_ids[idx[1], mod1(idx[2] + 1, N), idx[3]] == 1 # Cell wall - cell wall
                        diff += Dcw * (lattice_old[idx[1], mod1(idx[2] + 1, N), idx[3]] - lattice_old[idx...])
                    end
                    # Check front neighbour
                    if region_ids[mod1(idx[1] - 1, N), idx[2], idx[3]] == 0 # Cell wall - exterior
                        diff += Db * (lattice_old[mod1(idx[1] - 1, N), idx[2], idx[3]] - lattice_old[idx...])
                    elseif region_ids[mod1(idx[1] - 1, N), idx[2], idx[3]] == 1 # Cell wall - cell wall
                        diff += Dcw * (lattice_old[mod1(idx[1] - 1, N), idx[2], idx[3]] - lattice_old[idx...])
                    end
                    # Check back neighbour
                    if region_ids[mod1(idx[1] + 1, N), idx[2], idx[3]] == 0 # Cell wall - exterior
                        diff += Db * (lattice_old[mod1(idx[1] + 1, N), idx[2], idx[3]] - lattice_old[idx...])
                    elseif region_ids[mod1(idx[1] + 1, N), idx[2], idx[3]] == 1 # Cell wall - cell wall
                        diff += Dcw * (lattice_old[mod1(idx[1] + 1, N), idx[2], idx[3]] - lattice_old[idx...])
                    end
                end
            end
            
            @inbounds lattice_new[idx...] = lattice_old[idx...] + dtdx2 * diff
        end
        
        return nothing
    end


    function initialise_lattice_and_operator_GPU!(c_init, colidx, valsA, valsB, region_ids, c₀, sp_cen_i, sp_cen_j, sp_cen_k, spore_rad_lattice, cw_thickness, D, Dcw, Db, dtdx2, N, H, crank_nicolson=true)
        """
        GPU kernel for building the operator sparse matrix for implicitly solving the diffusion equation
        using the Crank-Nicolson method and initialising the lattice. Also assigns region IDs to the lattice.
        Implements periodic boundary conditions.
        inputs:
            c_init (CuArray) - the initial state of the lattice
            colidx (CuArray) - the column indices of the operator matrix
            valsA (CuArray) - the nonzero values of the operator matrix A
            valsB (CuArray) - the nonzero values of the operator matrix B
            region_ids (CuArray) - the region IDs of the lattice
            c₀ (float) - the initial concentration
            sp_cen_idx (array) - current spore center index
            spore_rad_lattice (float) - the radius of the spores in lattice units
            cw_thickness (int) - the thickness of the cell wall in lattice units
            D (float) - the diffusion constant
            Dcw (float) - the diffusion constant through the spore
            Db (float) - the effective diffusion constant at the spore interface
            dtdx2 (float) - the update factor
            N (int) - the number of lattice rows/columns
            H (int) - the number of lattice layers
            crank_nicolson (bool) - whether the operators are suited for Crank-Nicolson method
        """

        i, j, k = CUDA.blockIdx().x, CUDA.blockIdx().y, CUDA.blockIdx().z
        ti, tj, tk = CUDA.threadIdx().x, CUDA.threadIdx().y, CUDA.threadIdx().z
    
        # Determine the indices of the current cell
        idx = ((i - 1) * blockDim().x + ti, (j - 1) * blockDim().y + tj, (k - 1) * blockDim().z + tk)
        
        if idx[1] > N || idx[2] > N || idx[3] > H
            return nothing
        end
        
        # idx_lin = (idx[1] - 1) * N * H + (idx[2] - 1) * H + idx[3]
        idx_lin = (idx[3] - 1) * N * H + (idx[2] - 1) * H + idx[1]
        colidx[idx_lin*7-6] = idx_lin

        diag_val = 0f0
        
        dist = sqrt((idx[1] - sp_cen_i - 1)^2 + (idx[2] - sp_cen_j - 1)^2 + (idx[3] - sp_cen_k - 1)^2)
        # debugger[idx...] = sqdist
        
        region_id = region_ids[idx...]
        if (spore_rad_lattice - cw_thickness ≤ dist) && (dist ≤ spore_rad_lattice)
            # Cell wall site
            region_id = 1
            c_init[idx_lin] = c₀
        elseif dist < spore_rad_lattice - cw_thickness
            # Interior site
            region_id = 2
        end

        # Get neighbour coefficients
        nbr_ct = 1
        # ===== Bottom neighbour =====
        ni, nj, nk = idx[1], idx[2], mod1(idx[3] - 1, H)
        # n_idx = (ni - 1) * N * H + (nj - 1) * H + nk
        n_idx = (nk - 1) * N * H + (nj - 1) * H + ni
        colidx[idx_lin*7-5] = n_idx
        dist = sqrt((ni - sp_cen_i - 1)^2 + (nj - sp_cen_j - 1)^2 + (nk - sp_cen_k - 1)^2)
        coeff = 0f0
        if (spore_rad_lattice - cw_thickness ≤ dist) && (dist ≤ spore_rad_lattice)
            # Cell wall neighbour
            if region_id == 0 # Exterior - cell wall
                coeff = Float32(Db * dtdx2 * 0.5)
                # debugger[idx...] = 1
            elseif region_id == 1 # Cell wall - cell wall
                coeff = Float32(Dcw * dtdx2 * 0.5)
            end
        elseif dist ≤ spore_rad_lattice - cw_thickness
            # Exterior neighbour
            if region_id == 0 # Exterior - exterior
                coeff = Float32(D * dtdx2 * 0.5)
            elseif region_id == 1 # Cell wall - exterior
                coeff = Float32(Db * dtdx2 * 0.5)
            end
        end
        diag_val += coeff
        valsA[idx_lin*7-5] = - coeff
        valsB[idx_lin*7-5] = crank_nicolson ? coeff : 0f0

        # ===== Top neighbour =====
        ni, nj, nk = idx[1], idx[2], mod1(idx[3] + 1, H)
        # n_idx = (ni - 1) * N * H + (nj - 1) * H + nk
        n_idx = (nk - 1) * N * H + (nj - 1) * H + ni
        colidx[idx_lin*7-4] = n_idx
        dist = sqrt((ni - sp_cen_i - 1)^2 + (nj - sp_cen_j - 1)^2 + (nk - sp_cen_k - 1)^2)
        coeff = 0f0
        if (spore_rad_lattice - cw_thickness ≤ dist) && (dist ≤ spore_rad_lattice)
            # Cell wall neighbour
            if region_id == 0 # Exterior - cell wall
                coeff = Float32(Db * dtdx2 * 0.5)
            elseif region_id == 1 # Cell wall - cell wall
                coeff = Float32(Dcw * dtdx2 * 0.5)
            end
        elseif dist ≤ spore_rad_lattice - cw_thickness
            # Exterior neighbour
            if region_id == 0 # Exterior - exterior
                coeff = Float32(D * dtdx2 * 0.5)
            elseif region_id == 1 # Cell wall - exterior
                coeff = Float32(Db * dtdx2 * 0.5)
            end
        end
        diag_val += coeff
        valsA[idx_lin*7-4] = - coeff
        valsB[idx_lin*7-4] = crank_nicolson ? coeff : 0f0

        # ===== Left neighbour =====
        ni, nj, nk = idx[1], mod1(idx[2] - 1, H), idx[3]
        # n_idx = (ni - 1) * N * H + (nj - 1) * H + nk
        n_idx = (nk - 1) * N * H + (nj - 1) * H + ni
        colidx[idx_lin*7-3] = n_idx
        dist = sqrt((ni - sp_cen_i - 1)^2 + (nj - sp_cen_j - 1)^2 + (nk - sp_cen_k - 1)^2)
        coeff = 0f0
        if (spore_rad_lattice - cw_thickness ≤ dist) && (dist ≤ spore_rad_lattice)
            # Cell wall neighbour
            if region_id == 0 # Exterior - cell wall
                coeff = Float32(Db * dtdx2 * 0.5)
            elseif region_id == 1 # Cell wall - cell wall
                coeff = Float32(Dcw * dtdx2 * 0.5)
            end
        elseif dist ≤ spore_rad_lattice - cw_thickness
            # Exterior neighbour
            if region_id == 0 # Exterior - exterior
                coeff = Float32(D * dtdx2 * 0.5)
            elseif region_id == 1 # Cell wall - exterior
                coeff = Float32(Db * dtdx2 * 0.5)
            end
        end
        diag_val += coeff
        valsA[idx_lin*7-3] = - coeff
        valsB[idx_lin*7-3] = crank_nicolson ? coeff : 0f0

        # ===== Right neighbour =====
        ni, nj, nk = idx[1], mod1(idx[2] + 1, H), idx[3]
        # n_idx = (ni - 1) * N * H + (nj - 1) * H + nk
        n_idx = (nk - 1) * N * H + (nj - 1) * H + ni
        colidx[idx_lin*7-2] = n_idx
        dist = sqrt((ni - sp_cen_i - 1)^2 + (nj - sp_cen_j - 1)^2 + (nk - sp_cen_k - 1)^2)
        coeff = 0f0
        if (spore_rad_lattice - cw_thickness ≤ dist) && (dist ≤ spore_rad_lattice)
            # Cell wall neighbour
            if region_id == 0 # Exterior - cell wall
                coeff = Float32(Db * dtdx2 * 0.5)
            elseif region_id == 1 # Cell wall - cell wall
                coeff = Float32(Dcw * dtdx2 * 0.5)
            end
        elseif dist ≤ spore_rad_lattice - cw_thickness
            # Exterior neighbour
            if region_id == 0 # Exterior - exterior
                coeff = Float32(D * dtdx2 * 0.5)
            elseif region_id == 1 # Cell wall - exterior
                coeff = Float32(Db * dtdx2 * 0.5)
            end
        end
        diag_val += coeff
        valsA[idx_lin*7-2] = - coeff
        valsB[idx_lin*7-2] = crank_nicolson ? coeff : 0f0

        # ===== Front neighbour =====
        ni, nj, nk = mod1(idx[1] - 1, H), idx[2], idx[3]
        # n_idx = (ni - 1) * N * H + (nj - 1) * H + nk
        n_idx = (nk - 1) * N * H + (nj - 1) * H + ni
        colidx[idx_lin*7-1] = n_idx
        dist = sqrt((ni - sp_cen_i - 1)^2 + (nj - sp_cen_j - 1)^2 + (nk - sp_cen_k - 1)^2)
        coeff = 0f0
        if (spore_rad_lattice - cw_thickness ≤ dist) && (dist ≤ spore_rad_lattice)
            # Cell wall neighbour
            if region_id == 0 # Exterior - cell wall
                coeff = Float32(Db * dtdx2 * 0.5)
            elseif region_id == 1 # Cell wall - cell wall
                coeff = Float32(Dcw * dtdx2 * 0.5)
            end
        elseif dist ≤ spore_rad_lattice - cw_thickness
            # Exterior neighbour
            if region_id == 0 # Exterior - exterior
                coeff = Float32(D * dtdx2 * 0.5)
            elseif region_id == 1 # Cell wall - exterior
                coeff = Float32(Db * dtdx2 * 0.5)
            end
        end
        diag_val += coeff
        valsA[idx_lin*7-1] = - coeff
        valsB[idx_lin*7-1] = crank_nicolson ? coeff : 0f0

        # ===== Back neighbour =====
        ni, nj, nk = mod1(idx[1] + 1, H), idx[2], idx[3]
        # n_idx = (ni - 1) * N * H + (nj - 1) * H + nk
        n_idx = (nk - 1) * N * H + (nj - 1) * H + ni
        colidx[idx_lin*7] = n_idx
        dist = sqrt((ni - sp_cen_i - 1)^2 + (nj - sp_cen_j - 1)^2 + (nk - sp_cen_k - 1)^2)
        coeff = 0f0
        if (spore_rad_lattice - cw_thickness ≤ dist) && (dist ≤ spore_rad_lattice)
            # Cell wall neighbour
            if region_id == 0 # Exterior - cell wall
                coeff = Float32(Db * dtdx2 * 0.5)
            elseif region_id == 1 # Cell wall - cell wall
                coeff = Float32(Dcw * dtdx2 * 0.5)
            end
        elseif dist ≤ spore_rad_lattice - cw_thickness
            # Exterior neighbour
            if region_id == 0 # Exterior - exterior
                coeff = Float32(D * dtdx2 * 0.5)
            elseif region_id == 1 # Cell wall - exterior
                coeff = Float32(Db * dtdx2 * 0.5)
            end
        end
        diag_val += coeff
        valsA[idx_lin*7] = - coeff
        valsB[idx_lin*7] = crank_nicolson ? coeff : 0f0

        valsA[idx_lin*7-6] = 1 + diag_val
        valsB[idx_lin*7-6] = crank_nicolson ? 1 - diag_val : 1f0
        # pc_vals[idx_lin] = 1/(1 + diag_val)

        region_ids[idx...] = region_id

        return nothing
    end


    function initialise_lattice_and_operator_GPU_abs_bndry!(c_init, colidx, valsA, valsB, region_ids, c₀, sp_cen_i, sp_cen_j, sp_cen_k, spore_rad_lattice, cw_thickness, D, Dcw, Db, dtdx2, N, H, crank_nicolson=true)
        """
        GPU kernel for building the operator sparse matrix for implicitly solving the diffusion equation
        using the Crank-Nicolson method and initialising the lattice. Also assigns region IDs to the lattice.
        Implements absorbing boundary conditions.
        inputs:
            c_init (CuArray) - the initial state of the lattice
            colidx (CuArray) - the column indices of the operator matrix
            valsA (CuArray) - the nonzero values of the operator matrix A
            valsB (CuArray) - the nonzero values of the operator matrix B
            region_ids (CuArray) - the region IDs of the lattice
            c₀ (float) - the initial concentration
            sp_cen_idx (array) - current spore center index
            spore_rad_lattice (float) - the radius of the spores in lattice units
            cw_thickness (int) - the thickness of the cell wall in lattice units
            D (float) - the diffusion constant
            Dcw (float) - the diffusion constant through the spore
            Db (float) - the effective diffusion constant at the spore interface
            dtdx2 (float) - the update factor
            N (int) - the number of lattice rows/columns
            H (int) - the number of lattice layers
            crank_nicolson (bool) - whether the operators are suited for Crank-Nicolson method
        """

        i, j, k = CUDA.blockIdx().x, CUDA.blockIdx().y, CUDA.blockIdx().z
        ti, tj, tk = CUDA.threadIdx().x, CUDA.threadIdx().y, CUDA.threadIdx().z
    
        # Determine the indices of the current cell
        idx = ((i - 1) * blockDim().x + ti, (j - 1) * blockDim().y + tj, (k - 1) * blockDim().z + tk)

        if idx[1] > N || idx[2] > N || idx[3] > H
            return nothing
        end

        # idx_lin = (idx[1] - 1) * N * H + (idx[2] - 1) * H + idx[3]
        idx_lin = (idx[3] - 1) * N * H + (idx[2] - 1) * H + idx[1]
        colidx[idx_lin*7-6] = idx_lin

        diag_val = 0f0
        
        dist = sqrt((idx[1] - sp_cen_i - 1)^2 + (idx[2] - sp_cen_j - 1)^2 + (idx[3] - sp_cen_k - 1)^2)
        # debugger[idx...] = sqdist
        
        region_id = region_ids[idx...]
        if (spore_rad_lattice - cw_thickness ≤ dist) && (dist ≤ spore_rad_lattice)
            # Cell wall site
            region_id = 1
            c_init[idx_lin] = c₀
        elseif dist < spore_rad_lattice - cw_thickness
            # Interior site
            region_id = 2
        end

        # Get neighbour coefficients
        nbr_ct = 1
        # ===== Bottom neighbour =====
        if idx[3] > 1
            ni, nj, nk = idx[1], idx[2], idx[3] - 1
            # n_idx = (ni - 1) * N * H + (nj - 1) * H + nk
            n_idx = (nk - 1) * N * H + (nj - 1) * H + ni
            colidx[idx_lin*7-5] = n_idx
            dist = sqrt((ni - sp_cen_i - 1)^2 + (nj - sp_cen_j - 1)^2 + (nk - sp_cen_k - 1)^2)
            coeff = 0f0
            if (spore_rad_lattice - cw_thickness ≤ dist) && (dist ≤ spore_rad_lattice)
                # Cell wall neighbour
                if region_id == 0 # Exterior - cell wall
                    coeff = Float32(Db * dtdx2 * 0.5)
                    # debugger[idx...] = 1
                elseif region_id == 1 # Cell wall - cell wall
                    coeff = Float32(Dcw * dtdx2 * 0.5)
                end
            elseif dist ≤ spore_rad_lattice - cw_thickness
                # Exterior neighbour
                if region_id == 0 # Exterior - exterior
                    coeff = Float32(D * dtdx2 * 0.5)
                elseif region_id == 1 # Cell wall - exterior
                    coeff = Float32(Db * dtdx2 * 0.5)
                end
            end
            diag_val += coeff
            valsA[idx_lin*7-5] = - coeff
            valsB[idx_lin*7-5] = crank_nicolson ? coeff : 0f0
        end

        # ===== Top neighbour =====
        if idx[3] < H
            ni, nj, nk = idx[1], idx[2], idx[3] + 1
            # n_idx = (ni - 1) * N * H + (nj - 1) * H + nk
            n_idx = (nk - 1) * N * H + (nj - 1) * H + ni
            colidx[idx_lin*7-4] = n_idx
            dist = sqrt((ni - sp_cen_i - 1)^2 + (nj - sp_cen_j - 1)^2 + (nk - sp_cen_k - 1)^2)
            coeff = 0f0
            if (spore_rad_lattice - cw_thickness ≤ dist) && (dist ≤ spore_rad_lattice)
                # Cell wall neighbour
                if region_id == 0 # Exterior - cell wall
                    coeff = Float32(Db * dtdx2 * 0.5)
                elseif region_id == 1 # Cell wall - cell wall
                    coeff = Float32(Dcw * dtdx2 * 0.5)
                end
            elseif dist ≤ spore_rad_lattice - cw_thickness
                # Exterior neighbour
                if region_id == 0 # Exterior - exterior
                    coeff = Float32(D * dtdx2 * 0.5)
                elseif region_id == 1 # Cell wall - exterior
                    coeff = Float32(Db * dtdx2 * 0.5)
                end
            end
            diag_val += coeff
            valsA[idx_lin*7-4] = - coeff
            valsB[idx_lin*7-4] = crank_nicolson ? coeff : 0f0
        end

        # ===== Left neighbour =====
        if idx[2] > 1
            ni, nj, nk = idx[1], idx[2] - 1, idx[3]
            # n_idx = (ni - 1) * N * H + (nj - 1) * H + nk
            n_idx = (nk - 1) * N * H + (nj - 1) * H + ni
            colidx[idx_lin*7-3] = n_idx
            dist = sqrt((ni - sp_cen_i - 1)^2 + (nj - sp_cen_j - 1)^2 + (nk - sp_cen_k - 1)^2)
            coeff = 0f0
            if (spore_rad_lattice - cw_thickness ≤ dist) && (dist ≤ spore_rad_lattice)
                # Cell wall neighbour
                if region_id == 0 # Exterior - cell wall
                    coeff = Float32(Db * dtdx2 * 0.5)
                elseif region_id == 1 # Cell wall - cell wall
                    coeff = Float32(Dcw * dtdx2 * 0.5)
                end
            elseif dist ≤ spore_rad_lattice - cw_thickness
                # Exterior neighbour
                if region_id == 0 # Exterior - exterior
                    coeff = Float32(D * dtdx2 * 0.5)
                elseif region_id == 1 # Cell wall - exterior
                    coeff = Float32(Db * dtdx2 * 0.5)
                end
            end
            diag_val += coeff
            valsA[idx_lin*7-3] = - coeff
            valsB[idx_lin*7-3] = crank_nicolson ? coeff : 0f0
        end

        # ===== Right neighbour =====
        if idx[2] < H
            ni, nj, nk = idx[1], idx[2] + 1, idx[3]
            # n_idx = (ni - 1) * N * H + (nj - 1) * H + nk
            n_idx = (nk - 1) * N * H + (nj - 1) * H + ni
            colidx[idx_lin*7-2] = n_idx
            dist = sqrt((ni - sp_cen_i - 1)^2 + (nj - sp_cen_j - 1)^2 + (nk - sp_cen_k - 1)^2)
            coeff = 0f0
            if (spore_rad_lattice - cw_thickness ≤ dist) && (dist ≤ spore_rad_lattice)
                # Cell wall neighbour
                if region_id == 0 # Exterior - cell wall
                    coeff = Float32(Db * dtdx2 * 0.5)
                elseif region_id == 1 # Cell wall - cell wall
                    coeff = Float32(Dcw * dtdx2 * 0.5)
                end
            elseif dist ≤ spore_rad_lattice - cw_thickness
                # Exterior neighbour
                if region_id == 0 # Exterior - exterior
                    coeff = Float32(D * dtdx2 * 0.5)
                elseif region_id == 1 # Cell wall - exterior
                    coeff = Float32(Db * dtdx2 * 0.5)
                end
            end
            diag_val += coeff
            valsA[idx_lin*7-2] = - coeff
            valsB[idx_lin*7-2] = crank_nicolson ? coeff : 0f0
        end

        # ===== Front neighbour =====
        if idx[1] > 1
            ni, nj, nk = idx[1] - 1, idx[2], idx[3]
            # n_idx = (ni - 1) * N * H + (nj - 1) * H + nk
            n_idx = (nk - 1) * N * H + (nj - 1) * H + ni
            colidx[idx_lin*7-1] = n_idx
            dist = sqrt((ni - sp_cen_i - 1)^2 + (nj - sp_cen_j - 1)^2 + (nk - sp_cen_k - 1)^2)
            coeff = 0f0
            if (spore_rad_lattice - cw_thickness ≤ dist) && (dist ≤ spore_rad_lattice)
                # Cell wall neighbour
                if region_id == 0 # Exterior - cell wall
                    coeff = Float32(Db * dtdx2 * 0.5)
                elseif region_id == 1 # Cell wall - cell wall
                    coeff = Float32(Dcw * dtdx2 * 0.5)
                end
            elseif dist ≤ spore_rad_lattice - cw_thickness
                # Exterior neighbour
                if region_id == 0 # Exterior - exterior
                    coeff = Float32(D * dtdx2 * 0.5)
                elseif region_id == 1 # Cell wall - exterior
                    coeff = Float32(Db * dtdx2 * 0.5)
                end
            end
            diag_val += coeff
            valsA[idx_lin*7-1] = - coeff
            valsB[idx_lin*7-1] = crank_nicolson ? coeff : 0f0
        end

        # ===== Back neighbour =====
        if idx[1] < H
            ni, nj, nk = idx[1] + 1, idx[2], idx[3]
            # n_idx = (ni - 1) * N * H + (nj - 1) * H + nk
            n_idx = (nk - 1) * N * H + (nj - 1) * H + ni
            colidx[idx_lin*7] = n_idx
            dist = sqrt((ni - sp_cen_i - 1)^2 + (nj - sp_cen_j - 1)^2 + (nk - sp_cen_k - 1)^2)
            coeff = 0f0
            if (spore_rad_lattice - cw_thickness ≤ dist) && (dist ≤ spore_rad_lattice)
                # Cell wall neighbour
                if region_id == 0 # Exterior - cell wall
                    coeff = Float32(Db * dtdx2 * 0.5)
                elseif region_id == 1 # Cell wall - cell wall
                    coeff = Float32(Dcw * dtdx2 * 0.5)
                end
            elseif dist ≤ spore_rad_lattice - cw_thickness
                # Exterior neighbour
                if region_id == 0 # Exterior - exterior
                    coeff = Float32(D * dtdx2 * 0.5)
                elseif region_id == 1 # Cell wall - exterior
                    coeff = Float32(Db * dtdx2 * 0.5)
                end
            end
            diag_val += coeff
            valsA[idx_lin*7] = - coeff
            valsB[idx_lin*7] = crank_nicolson ? coeff : 0f0
        end

        valsA[idx_lin*7-6] = 1 + diag_val
        valsB[idx_lin*7-6] = crank_nicolson ? 1 - diag_val : 1f0
        # pc_vals[idx_lin] = 1/(1 + diag_val)

        region_ids[idx...] = region_id

        return nothing
    end


    # function initialise_lattice_and_build_operator!(c_init, c₀, sp_cen_indices, spore_rad_lattice, D, Dcw, Db, dtdx2, crank_nicolson)
    #     """
    #     Build operator sparse matrix for implicitly solving the diffusion equation
    #     using the Crank-Nicolson method.
    #     inputs:
    #         c_init (array) - the initial state of the lattice
    #         c₀ (float) - the initial concentration
    #         sp_cen_indices (array) - the indices of the spore centers
    #         spore_rad_lattice (float) - the radius of the spores in lattice units
    #         D (float) - the diffusion constant
    #         Dcw (float) - the diffusion constant through the spore
    #         Db (float) - the effective diffusion constant at the spore interface
    #         dtdx2 (float) - the update factor
    #         crank_nicolson (bool) - whether the operators are suited for Crank-Nicolson method
    #     outputs:
    #         op_A (sparse matrix) - the operator matrix A
    #         op_B (sparse matrix) - the operator matrix B
    #         region_ids (array) - the region IDs of the lattice
    #     """

    #     N = size(c_init)[1]
    #     H = size(c_init)[3]
    #     Nt = N * N * H

    #     op_A = spzeros(Float32, Nt, Nt)
    #     op_B = spzeros(Float32, Nt, Nt)
    #     region_ids = zeros(Int, N, N, H)

    #     ddia = sqrt(2)
    #     ddia_triple = 3 * ddia
    #     lattice_dia = 2*N^2 + H^2

    #     # Initialise concentrations in cell wall
    #     for i in 1:N, j in 1:N, k in 1:H

    #         idx_lin = lin_idx(i, j, k, N, H)
    #         diag_val = 0f0

    #         vneum_nbrs = [(i, j, mod1(k - 1, H)), (i, j, mod1(k + 1, H)),
    #                     (i, mod1(j - 1, N), k), (i, mod1(j + 1, N), k),
    #                     (mod1(i - 1, N), j, k), (mod1(i + 1, N), j, k)]
            
    #         region_id = 0
    #         min_dist = lattice_dia
    #         for sp_cen_idx in sp_cen_indices

    #             # Compute distance to spore center
    #             dist = norm([i - sp_cen_idx[1] - 1, j - sp_cen_idx[2] - 1, k - sp_cen_idx[3] - 1])
    #             if dist < min_dist
    #                 min_dist = dist
    #             end
                
    #             if spore_rad_lattice - ddia ≤ dist ≤ spore_rad_lattice
    #                 # Cell wall site
    #                 region_id = 1
    #                 c_init[i, j, k] = c₀
    #                 break
    #             elseif dist < spore_rad_lattice - ddia
    #                 # Interior site
    #                 region_id = 2
    #                 break
    #             end
    #         end

    #         region_ids[i, j, k] = region_id
            
    #         # Get neighbour coefficients
    #         if min_dist ≤ spore_rad_lattice + ddia_triple

    #             # Iterate over von Neumann neighbours
    #             for (ni, nj, nk) in vneum_nbrs
    #                 for sp_cen_idx in sp_cen_indices
                        
    #                     n_idx = lin_idx(ni, nj, nk, N, H)
    #                     # println("Index: $idx_lin, Neighbour: $n_idx")
    #                     dist = norm([ni - sp_cen_idx[1] - 1, nj- sp_cen_idx[2] - 1, nk - sp_cen_idx[3] - 1])
    #                     coeff = 0f0

    #                     if spore_rad_lattice - ddia ≤ dist ≤ spore_rad_lattice
    #                         # Cell wall neighbour
    #                         if region_id == 0 # Exterior - cell wall
    #                             coeff = Float32(Db * dtdx2 * 0.5)
    #                         elseif region_id == 1 # Cell wall - cell wall
    #                             coeff = Float32(Dcw * dtdx2 * 0.5)
    #                         end

    #                     elseif dist ≤ spore_rad_lattice - ddia
    #                         # Exterior neighbour
    #                         if region_id == 0 # Exterior - exterior
    #                             coeff = Float32(D * dtdx2 * 0.5)
    #                         elseif region_id == 1 # Cell wall - exterior
    #                             coeff = Float32(Db * dtdx2 * 0.5)
    #                         end
    #                     end

    #                     diag_val += coeff
    #                     op_A[idx_lin, idx_lin] = - coeff
    #                     op_B[idx_lin, idx_lin] = crank_nicolson ? coeff : 0f0
    #                 end
    #             end
    #         end

    #         op_A[idx_lin, idx_lin] = 1 + diag_val 
    #         op_B[idx_lin, idx_lin] = crank_nicolson ? 1 - diag_val : 1f0
    #     end
    #     println("Concentrations initialised.")

    #     # Check if matrices are singular
    #     if iszero(det(op_A))
    #         println("Matrix A is singular.")
    #     end
    #     if iszero(det(op_B))
    #         println("Matrix B is singular.")
    #     end

    #     return op_A, op_B, region_ids
    # end
end