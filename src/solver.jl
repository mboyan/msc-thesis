module Solver
    __precompile__(false)
    """
    Functions used with different diffusion solvers.
    """

    using SparseArrays
    using LinearAlgebra
    using CUDA
    
    export invoke_smart_kernel_3D
    export max_reduce_kernel
    export update_GPU!
    export update_GPU_spore_cluster!
    export update_GPU_low_res!
    export update_GPU_low_res_spore_cluster!
    export update_GPU_hi_res!
    export update_GPU_hi_res_coeffs!
    export initialise_lattice_and_operator_GPU!
    export initialise_lattice_and_operator_GPU_abs_bndry!

    function invoke_smart_kernel_3D(size, threads_per_block=(8, 8, 8))
        """
        Invoke a kernel with the appropriate number of blocks and threads per block.
        """
        blocks_per_grid = (Int(ceil(size[1] / threads_per_block[1])),
                           Int(ceil(size[2] / threads_per_block[2])),
                           Int(ceil(size[3] / threads_per_block[3])))
        return blocks_per_grid, threads_per_block
    end

    function max_reduce_kernel(a, b)
        return max(a, b)
    end


    function update_GPU!(c_old, c_new, N, H, dtdx2, D, Db, spore_idx, abs_bndry, neumann_z)
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
            abs_bndry (bool) - whether to use absorbing boundary conditions
            neumann_z (bool) - whether to use Neumann boundary conditions in the z-direction
        """
        i, j, k = CUDA.blockIdx().x, CUDA.blockIdx().y, CUDA.blockIdx().z
        ti, tj, tk = CUDA.threadIdx().x, CUDA.threadIdx().y, CUDA.threadIdx().z
    
        # Determine the indices of the current cell
        idx = ((i - 1) * blockDim().x + ti, (j - 1) * blockDim().y + tj, (k - 1) * blockDim().z + tk)
    
        # Update the concentration value
        if 1 ≤ idx[1] ≤ N && 1 ≤ idx[2] ≤ N && 1 ≤ idx[3] ≤ H

            # Apply absorbing boundary conditions
            if abs_bndry
                if idx[1] == 1 || idx[1] == N || idx[2] == 1 || idx[2] == N || idx[3] == 1 || idx[3] == H
                    return nothing
                end
            end
            
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

    
    function update_GPU_spore_cluster!(c_old, c_new, N, H, dtdx2, D, Db, spore_idx, cluster_spacing, cluster_size, abs_bndry, neumann_z)
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
            cluster_spacing (int) - the spacing between spores in the cluster in lattice units
            cluster_size (int) - the number of spores in the cluster
            abs_bndry (bool) - whether to use absorbing boundary conditions
            neumann_z (bool) - whether to use Neumann boundary conditions in the z-direction
        """
        i, j, k = CUDA.blockIdx().x, CUDA.blockIdx().y, CUDA.blockIdx().z
        ti, tj, tk = CUDA.threadIdx().x, CUDA.threadIdx().y, CUDA.threadIdx().z
    
        # Determine the indices of the current cell
        idx = ((i - 1) * blockDim().x + ti, (j - 1) * blockDim().y + tj, (k - 1) * blockDim().z + tk)
    
        # Update the concentration value
        if 1 ≤ idx[1] ≤ N && 1 ≤ idx[2] ≤ N && 1 ≤ idx[3] ≤ H

            # Apply absorbing boundary conditions
            if abs_bndry
                if idx[1] == 1 || idx[1] == N || idx[2] == 1 || idx[2] == N || idx[3] == 1 || idx[3] == H
                    return nothing
                end
            end
            
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

            # First neighbour
            if cluster_size > 1
                spore_dist_vec = (idx[1] - spore_idx[1], idx[2] - spore_idx[2], idx[3] - spore_idx[3] + cluster_spacing)
                at_spore = spore_dist_vec == (0, 0, 0)
                Ddtdx2bottom = spore_dist_vec == (0, 0, 1) || at_spore ? Db * dtdx2 : Ddtdx2bottom
                Ddtdx2top = spore_dist_vec == (0, 0, -1) || at_spore ? Db * dtdx2 : Ddtdx2top
                Ddtdx2left = spore_dist_vec == (0, 1, 0) || at_spore ? Db * dtdx2 : Ddtdx2left
                Ddtdx2right = spore_dist_vec == (0, -1, 0) || at_spore ? Db * dtdx2 : Ddtdx2right
                Ddtdx2front = spore_dist_vec == (1, 0, 0) || at_spore ? Db * dtdx2 : Ddtdx2front
                Ddtdx2back = spore_dist_vec == (-1, 0, 0) || at_spore ? Db * dtdx2 : Ddtdx2back
            end

            # Second neighbour
            if cluster_size > 2
                spore_dist_vec = (idx[1] - spore_idx[1], idx[2] - spore_idx[2], idx[3] - spore_idx[3] - cluster_spacing)
                at_spore = spore_dist_vec == (0, 0, 0)
                Ddtdx2bottom = spore_dist_vec == (0, 0, 1) || at_spore ? Db * dtdx2 : Ddtdx2bottom
                Ddtdx2top = spore_dist_vec == (0, 0, -1) || at_spore ? Db * dtdx2 : Ddtdx2top
                Ddtdx2left = spore_dist_vec == (0, 1, 0) || at_spore ? Db * dtdx2 : Ddtdx2left
                Ddtdx2right = spore_dist_vec == (0, -1, 0) || at_spore ? Db * dtdx2 : Ddtdx2right
                Ddtdx2front = spore_dist_vec == (1, 0, 0) || at_spore ? Db * dtdx2 : Ddtdx2front
                Ddtdx2back = spore_dist_vec == (-1, 0, 0) || at_spore ? Db * dtdx2 : Ddtdx2back
            end

            # Third neighbour
            if cluster_size > 3
                spore_dist_vec = (idx[1] - spore_idx[1], idx[2] - spore_idx[2] + cluster_spacing, idx[3] - spore_idx[3])
                at_spore = spore_dist_vec == (0, 0, 0)
                Ddtdx2bottom = spore_dist_vec == (0, 0, 1) || at_spore ? Db * dtdx2 : Ddtdx2bottom
                Ddtdx2top = spore_dist_vec == (0, 0, -1) || at_spore ? Db * dtdx2 : Ddtdx2top
                Ddtdx2left = spore_dist_vec == (0, 1, 0) || at_spore ? Db * dtdx2 : Ddtdx2left
                Ddtdx2right = spore_dist_vec == (0, -1, 0) || at_spore ? Db * dtdx2 : Ddtdx2right
                Ddtdx2front = spore_dist_vec == (1, 0, 0) || at_spore ? Db * dtdx2 : Ddtdx2front
                Ddtdx2back = spore_dist_vec == (-1, 0, 0) || at_spore ? Db * dtdx2 : Ddtdx2back
            end

            # Fourth neighbour
            if cluster_size > 4
                spore_dist_vec = (idx[1] - spore_idx[1], idx[2] - spore_idx[2] - cluster_spacing, idx[3] - spore_idx[3])
                at_spore = spore_dist_vec == (0, 0, 0)
                Ddtdx2bottom = spore_dist_vec == (0, 0, 1) || at_spore ? Db * dtdx2 : Ddtdx2bottom
                Ddtdx2top = spore_dist_vec == (0, 0, -1) || at_spore ? Db * dtdx2 : Ddtdx2top
                Ddtdx2left = spore_dist_vec == (0, 1, 0) || at_spore ? Db * dtdx2 : Ddtdx2left
                Ddtdx2right = spore_dist_vec == (0, -1, 0) || at_spore ? Db * dtdx2 : Ddtdx2right
                Ddtdx2front = spore_dist_vec == (1, 0, 0) || at_spore ? Db * dtdx2 : Ddtdx2front
                Ddtdx2back = spore_dist_vec == (-1, 0, 0) || at_spore ? Db * dtdx2 : Ddtdx2back
            end

            # Fifth neighbour
            if cluster_size > 5
                spore_dist_vec = (idx[1] - spore_idx[1] + cluster_spacing, idx[2] - spore_idx[2], idx[3] - spore_idx[3])
                at_spore = spore_dist_vec == (0, 0, 0)
                Ddtdx2bottom = spore_dist_vec == (0, 0, 1) || at_spore ? Db * dtdx2 : Ddtdx2bottom
                Ddtdx2top = spore_dist_vec == (0, 0, -1) || at_spore ? Db * dtdx2 : Ddtdx2top
                Ddtdx2left = spore_dist_vec == (0, 1, 0) || at_spore ? Db * dtdx2 : Ddtdx2left
                Ddtdx2right = spore_dist_vec == (0, -1, 0) || at_spore ? Db * dtdx2 : Ddtdx2right
                Ddtdx2front = spore_dist_vec == (1, 0, 0) || at_spore ? Db * dtdx2 : Ddtdx2front
                Ddtdx2back = spore_dist_vec == (-1, 0, 0) || at_spore ? Db * dtdx2 : Ddtdx2back
            end

            # Sixth neighbour
            if cluster_size > 6
                spore_dist_vec = (idx[1] - spore_idx[1] - cluster_spacing, idx[2] - spore_idx[2], idx[3] - spore_idx[3])
                at_spore = spore_dist_vec == (0, 0, 0)
                Ddtdx2bottom = spore_dist_vec == (0, 0, 1) || at_spore ? Db * dtdx2 : Ddtdx2bottom
                Ddtdx2top = spore_dist_vec == (0, 0, -1) || at_spore ? Db * dtdx2 : Ddtdx2top
                Ddtdx2left = spore_dist_vec == (0, 1, 0) || at_spore ? Db * dtdx2 : Ddtdx2left
                Ddtdx2right = spore_dist_vec == (0, -1, 0) || at_spore ? Db * dtdx2 : Ddtdx2right
                Ddtdx2front = spore_dist_vec == (1, 0, 0) || at_spore ? Db * dtdx2 : Ddtdx2front
                Ddtdx2back = spore_dist_vec == (-1, 0, 0) || at_spore ? Db * dtdx2 : Ddtdx2back
            end

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

    function update_GPU_low_res_spore_cluster!(c_old, c_new, N, H, inv_dx2, D, spore_vol_idx, c_spore, inv_tau, dt, cluster_spacing, cluster_size, neumann_z)
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
            cluster_spacing (int) - the spacing between spores in the cluster in lattice units
            cluster_size (int) - the number of neighbours in the cluster
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
            cond_spore = idx[1] == spore_vol_idx[1] && idx[2] == spore_vol_idx[2] && idx[3] == spore_vol_idx[3]
            if cluster_size > 1 # Single neighbour
                cond_spore = cond_spore || idx[1] == spore_vol_idx[1] && idx[2] == spore_vol_idx[2] && idx[3] == spore_vol_idx[3] - cluster_spacing
            end
            if cluster_size > 2 # Two neighbours
                cond_spore = cond_spore || idx[1] == spore_vol_idx[1] && idx[2] == spore_vol_idx[2] && idx[3] == spore_vol_idx[3] + cluster_spacing
            end
            if cluster_size > 3 # Three neighbours
                cond_spore = cond_spore || idx[1] == spore_vol_idx[1] && idx[2] == spore_vol_idx[2] - cluster_spacing && idx[3] == spore_vol_idx[3]
            end
            if cluster_size > 4 # Four neighbours
                cond_spore = cond_spore || idx[1] == spore_vol_idx[1] && idx[2] == spore_vol_idx[2] + cluster_spacing && idx[3] == spore_vol_idx[3]
            end
            if cluster_size > 5 # Five neighbours
                cond_spore = cond_spore || idx[1] == spore_vol_idx[1] - cluster_spacing && idx[2] == spore_vol_idx[2] && idx[3] == spore_vol_idx[3]
            end
            if cluster_size > 6 # Six neighbours
                cond_spore = cond_spore || idx[1] == spore_vol_idx[1] + cluster_spacing && idx[2] == spore_vol_idx[2] && idx[3] == spore_vol_idx[3]
            end
            if cond_spore
                delta_c_half = (c_spore[idx...] - center) * (1 - 0.5 * dttau)
                c_new[idx...] = center + D * dtdx2 * (bottom + top + left + right + front + back - 6 * center) + dttau * delta_c_half
                c_spore[idx...] = c_new[idx...] + delta_c_half * (1 - 0.5 * dttau)
            else
                c_new[idx...] = center + D * dtdx2 * (bottom + top + left + right + front + back - 6 * center)
            end
        end

        return nothing
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


    function initialise_lattice_and_operator_GPU!(c_init, colidx, valsA, valsB, region_ids, pc_vals, c₀, sp_cen_i, sp_cen_j, sp_cen_k, spore_rad_lattice, cw_thickness, D, Dcw, Db, dtdx2, N, H, crank_nicolson=true, empty_interior=true, abs_bndry=false)
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
            empty_interior (bool) - whether to make interior beyond the cell wall inaccessible for diffusion
            abs_bndry (bool) - whether to use absorbing boundary conditions
        """

        i, j, k = CUDA.blockIdx().x, CUDA.blockIdx().y, CUDA.blockIdx().z
        ti, tj, tk = CUDA.threadIdx().x, CUDA.threadIdx().y, CUDA.threadIdx().z
    
        # Determine the indices of the current cell
        idx = ((i - 1) * blockDim().x + ti, (j - 1) * blockDim().y + tj, (k - 1) * blockDim().z + tk)
        
        if idx[1] > N || idx[2] > N || idx[3] > H
            return nothing
        end

        if crank_nicolson
            factor = 0.5
        else
            factor = 1.0
        end
        
        idx_lin = (idx[3] - 1) * N^2 + (idx[2] - 1) * N + idx[1]
        colidx[idx_lin*7-6] = idx_lin

        diag_val = 0f0
        
        dist = sqrt((idx[1] - sp_cen_i)^2 + (idx[2] - sp_cen_j)^2 + (idx[3] - sp_cen_k)^2)
        # debugger[idx...] = sqdist
        
        region_id = region_ids[idx...]
        if (spore_rad_lattice - cw_thickness ≤ dist) && (dist ≤ spore_rad_lattice)
            # Cell wall site
            region_id = 1
            c_init[idx_lin] = 1f0#c₀
        elseif (dist < spore_rad_lattice - cw_thickness) && empty_interior
            # Interior site
            region_id = 2
        elseif (dist < spore_rad_lattice - cw_thickness) && !empty_interior
            # Spore interior site
            region_id = 1
            c_init[idx_lin] = 1f0
        end

        # Get neighbour coefficients
        # ===== Bottom neighbour =====
        if (abs_bndry && idx[3] > 1) || !abs_bndry
            ni, nj, nk = idx[1], idx[2], mod1(idx[3] - 1, H)
            n_idx = (nk - 1) * N^2 + (nj - 1) * N + ni
            colidx[idx_lin*7-5] = n_idx
            dist = sqrt((ni - sp_cen_i)^2 + (nj - sp_cen_j)^2 + (nk - sp_cen_k)^2)
            coeff = 0f0
            if (spore_rad_lattice - cw_thickness ≤ dist) && (dist ≤ spore_rad_lattice)
                # Cell wall neighbour
                if region_id == 0 # Exterior - cell wall
                    coeff = Float32(Db * dtdx2 * factor)
                    # debugger[idx...] = 1
                elseif region_id == 1 # Cell wall - cell wall
                    coeff = Float32(Dcw * dtdx2 * factor)
                end
            elseif dist > spore_rad_lattice
                # Exterior neighbour
                if region_id == 0 # Exterior - exterior
                    coeff = Float32(D * dtdx2 * factor)
                elseif region_id == 1 # Cell wall - exterior
                    coeff = Float32(Db * dtdx2 * factor)
                end
            elseif !empty_interior
                # Interior neighbour
                coeff = Float32(Dcw * dtdx2 * factor)
            end
            diag_val += coeff
            valsA[idx_lin*7-5] = valsA[idx_lin*7-5] != 0 ? - min(abs(valsA[idx_lin*7-5]), coeff) : - coeff
            valsB[idx_lin*7-5] = crank_nicolson ? (valsB[idx_lin*7-5] != 0 ? min(valsB[idx_lin*7-5], coeff) : coeff) : 0f0
        end

        # ===== Top neighbour =====
        if (abs_bndry && idx[3] < H) || !abs_bndry
            ni, nj, nk = idx[1], idx[2], mod1(idx[3] + 1, H)
            n_idx = (nk - 1) * N^2 + (nj - 1) * N + ni
            colidx[idx_lin*7-4] = n_idx
            dist = sqrt((ni - sp_cen_i)^2 + (nj - sp_cen_j)^2 + (nk - sp_cen_k)^2)
            coeff = 0f0
            if (spore_rad_lattice - cw_thickness ≤ dist) && (dist ≤ spore_rad_lattice)
                # Cell wall neighbour
                if region_id == 0 # Exterior - cell wall
                    coeff = Float32(Db * dtdx2 * factor)
                elseif region_id == 1 # Cell wall - cell wall
                    coeff = Float32(Dcw * dtdx2 * factor)
                end
            elseif dist > spore_rad_lattice
                # Exterior neighbour
                if region_id == 0 # Exterior - exterior
                    coeff = Float32(D * dtdx2 * factor)
                elseif region_id == 1 # Cell wall - exterior
                    coeff = Float32(Db * dtdx2 * factor)
                end
            elseif !empty_interior
                # Interior neighbour
                coeff = Float32(Dcw * dtdx2 * factor)
            end
            diag_val += coeff
            # valsA[idx_lin*7-4] = - coeff
            # valsB[idx_lin*7-4] = crank_nicolson ? coeff : 0f0
            valsA[idx_lin*7-4] = valsA[idx_lin*7-4] != 0 ? - min(abs(valsA[idx_lin*7-4]), coeff) : - coeff
            valsB[idx_lin*7-4] = crank_nicolson ? (valsB[idx_lin*7-4] != 0 ? min(valsB[idx_lin*7-4], coeff) : coeff) : 0f0
        end

        # ===== Left neighbour =====
        if (abs_bndry && idx[2] > 1) || !abs_bndry
            ni, nj, nk = idx[1], mod1(idx[2] - 1, H), idx[3]
            n_idx = (nk - 1) * N^2 + (nj - 1) * N + ni
            colidx[idx_lin*7-3] = n_idx
            dist = sqrt((ni - sp_cen_i)^2 + (nj - sp_cen_j)^2 + (nk - sp_cen_k)^2)
            coeff = 0f0
            if (spore_rad_lattice - cw_thickness ≤ dist) && (dist ≤ spore_rad_lattice)
                # Cell wall neighbour
                if region_id == 0 # Exterior - cell wall
                    coeff = Float32(Db * dtdx2 * factor)
                elseif region_id == 1 # Cell wall - cell wall
                    coeff = Float32(Dcw * dtdx2 * factor)
                end
            elseif dist > spore_rad_lattice
                # Exterior neighbour
                if region_id == 0 # Exterior - exterior
                    coeff = Float32(D * dtdx2 * factor)
                elseif region_id == 1 # Cell wall - exterior
                    coeff = Float32(Db * dtdx2 * factor)
                end
            elseif !empty_interior
                # Interior neighbour
                coeff = Float32(Dcw * dtdx2 * factor)
            end
            diag_val += coeff
            # valsA[idx_lin*7-3] = - coeff
            # valsB[idx_lin*7-3] = crank_nicolson ? coeff : 0f0
            valsA[idx_lin*7-3] = valsA[idx_lin*7-3] != 0 ? - min(abs(valsA[idx_lin*7-3]), coeff) : - coeff
            valsB[idx_lin*7-3] = crank_nicolson ? (valsB[idx_lin*7-3] != 0 ? min(valsB[idx_lin*7-3], coeff) : coeff) : 0f0
        end

        # ===== Right neighbour =====
        if (abs_bndry && idx[2] < N) || !abs_bndry
            ni, nj, nk = idx[1], mod1(idx[2] + 1, H), idx[3]
            n_idx = (nk - 1) * N^2 + (nj - 1) * N + ni
            colidx[idx_lin*7-2] = n_idx
            dist = sqrt((ni - sp_cen_i)^2 + (nj - sp_cen_j)^2 + (nk - sp_cen_k)^2)
            coeff = 0f0
            if (spore_rad_lattice - cw_thickness ≤ dist) && (dist ≤ spore_rad_lattice)
                # Cell wall neighbour
                if region_id == 0 # Exterior - cell wall
                    coeff = Float32(Db * dtdx2 * factor)
                elseif region_id == 1 # Cell wall - cell wall
                    coeff = Float32(Dcw * dtdx2 * factor)
                end
            elseif dist > spore_rad_lattice
                # Exterior neighbour
                if region_id == 0 # Exterior - exterior
                    coeff = Float32(D * dtdx2 * factor)
                elseif region_id == 1 # Cell wall - exterior
                    coeff = Float32(Db * dtdx2 * factor)
                end
            elseif !empty_interior
                # Interior neighbour
                coeff = Float32(Dcw * dtdx2 * factor)
            end
            diag_val += coeff
            # valsA[idx_lin*7-2] = - coeff
            # valsB[idx_lin*7-2] = crank_nicolson ? coeff : 0f0
            valsA[idx_lin*7-2] = valsA[idx_lin*7-2] != 0 ? - min(abs(valsA[idx_lin*7-2]), coeff) : - coeff
            valsB[idx_lin*7-2] = crank_nicolson ? (valsB[idx_lin*7-2] != 0 ? min(valsB[idx_lin*7-2], coeff) : coeff) : 0f0
        end

        # ===== Front neighbour =====
        if (abs_bndry && idx[1] > 1) || !abs_bndry
            ni, nj, nk = mod1(idx[1] - 1, H), idx[2], idx[3]
            n_idx = (nk - 1) * N^2 + (nj - 1) * N + ni
            colidx[idx_lin*7-1] = n_idx
            dist = sqrt((ni - sp_cen_i)^2 + (nj - sp_cen_j)^2 + (nk - sp_cen_k)^2)
            coeff = 0f0
            if (spore_rad_lattice - cw_thickness ≤ dist) && (dist ≤ spore_rad_lattice)
                # Cell wall neighbour
                if region_id == 0 # Exterior - cell wall
                    coeff = Float32(Db * dtdx2 * factor)
                elseif region_id == 1 # Cell wall - cell wall
                    coeff = Float32(Dcw * dtdx2 * factor)
                end
            elseif dist > spore_rad_lattice
                # Exterior neighbour
                if region_id == 0 # Exterior - exterior
                    coeff = Float32(D * dtdx2 * factor)
                elseif region_id == 1 # Cell wall - exterior
                    coeff = Float32(Db * dtdx2 * factor)
                end
            elseif !empty_interior
                # Interior neighbour
                coeff = Float32(Dcw * dtdx2 * factor)
            end
            diag_val += coeff
            # valsA[idx_lin*7-1] = - coeff
            # valsB[idx_lin*7-1] = crank_nicolson ? coeff : 0f0
            valsA[idx_lin*7-1] = valsA[idx_lin*7-1] != 0 ? - min(abs(valsA[idx_lin*7-1]), coeff) : - coeff
            valsB[idx_lin*7-1] = crank_nicolson ? (valsB[idx_lin*7-1] != 0 ? min(valsB[idx_lin*7-1], coeff) : coeff) : 0f0
        end

        # ===== Back neighbour =====
        if (abs_bndry && idx[1] < N) || !abs_bndry
            ni, nj, nk = mod1(idx[1] + 1, H), idx[2], idx[3]
            n_idx = (nk - 1) * N^2 + (nj - 1) * N + ni
            colidx[idx_lin*7] = n_idx
            dist = sqrt((ni - sp_cen_i)^2 + (nj - sp_cen_j)^2 + (nk - sp_cen_k)^2)
            coeff = 0f0
            if (spore_rad_lattice - cw_thickness ≤ dist) && (dist ≤ spore_rad_lattice)
                # Cell wall neighbour
                if region_id == 0 # Exterior - cell wall
                    coeff = Float32(Db * dtdx2 * factor)
                elseif region_id == 1 # Cell wall - cell wall
                    coeff = Float32(Dcw * dtdx2 * factor)
                end
            elseif dist > spore_rad_lattice
                # Exterior neighbour
                if region_id == 0 # Exterior - exterior
                    coeff = Float32(D * dtdx2 * factor)
                elseif region_id == 1 # Cell wall - exterior
                    coeff = Float32(Db * dtdx2 * factor)
                end
            elseif !empty_interior
                # Interior neighbour
                coeff = Float32(Dcw * dtdx2 * factor)
            end
            diag_val += coeff
            # valsA[idx_lin*7] = - coeff
            # valsB[idx_lin*7] = crank_nicolson ? coeff : 0f0
            valsA[idx_lin*7] = valsA[idx_lin*7] != 0 ? - min(abs(valsA[idx_lin*7]), coeff) : - coeff
            valsB[idx_lin*7] = crank_nicolson ? (valsB[idx_lin*7] != 0 ? min(valsB[idx_lin*7]) : coeff) : 0f0
        end

        valsA[idx_lin*7-6] = valsA[idx_lin*7-6] == 0 ? 1 + diag_val : min(valsA[idx_lin*7-6], 1 + diag_val)
        valsB[idx_lin*7-6] = crank_nicolson ? max(valsB[idx_lin*7-6], 1 - diag_val) : 1f0
        pc_vals[idx_lin] = 1/(1 + diag_val)

        region_ids[idx...] = max(region_ids[idx...], region_id)

        return nothing
    end


    function initialise_lattice_and_operator_GPU_abs_bndry!(c_init, colidx, valsA, valsB, region_ids, c₀, sp_cen_i, sp_cen_j, sp_cen_k, spore_rad_lattice, cw_thickness, D, Dcw, Db, dtdx2, N, H, crank_nicolson=true, empty_interior=true)
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
            empty_interior (bool) - whether to make interior beyond the cell wall inaccessible for diffusion
        """

        i, j, k = CUDA.blockIdx().x, CUDA.blockIdx().y, CUDA.blockIdx().z
        ti, tj, tk = CUDA.threadIdx().x, CUDA.threadIdx().y, CUDA.threadIdx().z
    
        # Determine the indices of the current cell
        idx = ((i - 1) * blockDim().x + ti, (j - 1) * blockDim().y + tj, (k - 1) * blockDim().z + tk)

        if idx[1] > N || idx[2] > N || idx[3] > H
            return nothing
        end

        if crank_nicolson
            factor = 0.5
        else
            factor = 1.0
        end

        idx_lin = (idx[3] - 1) * N^2 + (idx[2] - 1) * N + idx[1]
        colidx[idx_lin*7-6] = idx_lin

        diag_val = 0f0
        
        dist = sqrt((idx[1] - sp_cen_i)^2 + (idx[2] - sp_cen_j)^2 + (idx[3] - sp_cen_k)^2)
        # debugger[idx...] = sqdist
        
        region_id = region_ids[idx...]
        if (spore_rad_lattice - cw_thickness ≤ dist) && (dist ≤ spore_rad_lattice)
            # Cell wall site
            region_id = 1
            c_init[idx_lin] = 1f0#c₀
        elseif (dist < spore_rad_lattice - cw_thickness) && empty_interior
            # Interior site
            region_id = 2
        elseif (dist < spore_rad_lattice - cw_thickness) && !empty_interior
            # Spore interior site
            region_id = 1
            c_init[idx_lin] = 1f0
        end

        # Get neighbour coefficients
        # ===== Bottom neighbour =====
        if idx[3] > 1
            ni, nj, nk = idx[1], idx[2], idx[3] - 1
            n_idx = (nk - 1) * N^2 + (nj - 1) * N + ni
            colidx[idx_lin*7-5] = n_idx
            dist = sqrt((ni - sp_cen_i)^2 + (nj - sp_cen_j)^2 + (nk - sp_cen_k)^2)
            coeff = 0f0
            if (spore_rad_lattice - cw_thickness ≤ dist) && (dist ≤ spore_rad_lattice)
                # Cell wall neighbour
                if region_id == 0 # Exterior - cell wall
                    coeff = Float32(Db * dtdx2 * factor)
                    # debugger[idx...] = 1
                elseif region_id == 1 # Cell wall - cell wall
                    coeff = Float32(Dcw * dtdx2 * factor)
                end
            elseif dist > spore_rad_lattice
                # Exterior neighbour
                if region_id == 0 # Exterior - exterior
                    coeff = Float32(D * dtdx2 * factor)
                elseif region_id == 1 # Cell wall - exterior
                    coeff = Float32(Db * dtdx2 * factor)
                end
            elseif !empty_interior
                # Interior neighbour
                coeff = Float32(Dcw * dtdx2 * factor)
            end
            diag_val += coeff
            valsA[idx_lin*7-5] = - coeff
            valsB[idx_lin*7-5] = crank_nicolson ? coeff : 0f0
        end

        # ===== Top neighbour =====
        if idx[3] < H
            ni, nj, nk = idx[1], idx[2], idx[3] + 1
            n_idx = (nk - 1) * N^2 + (nj - 1) * N + ni
            colidx[idx_lin*7-4] = n_idx
            dist = sqrt((ni - sp_cen_i)^2 + (nj - sp_cen_j)^2 + (nk - sp_cen_k)^2)
            coeff = 0f0
            if (spore_rad_lattice - cw_thickness ≤ dist) && (dist ≤ spore_rad_lattice)
                # Cell wall neighbour
                if region_id == 0 # Exterior - cell wall
                    coeff = Float32(Db * dtdx2 * factor)
                elseif region_id == 1 # Cell wall - cell wall
                    coeff = Float32(Dcw * dtdx2 * factor)
                end
            elseif dist > spore_rad_lattice
                # Exterior neighbour
                if region_id == 0 # Exterior - exterior
                    coeff = Float32(D * dtdx2 * factor)
                elseif region_id == 1 # Cell wall - exterior
                    coeff = Float32(Db * dtdx2 * factor)
                end
            elseif !empty_interior
                # Interior neighbour
                coeff = Float32(Dcw * dtdx2 * factor)
            end
            diag_val += coeff
            valsA[idx_lin*7-4] = - coeff
            valsB[idx_lin*7-4] = crank_nicolson ? coeff : 0f0
        end

        # ===== Left neighbour =====
        if idx[2] > 1
            ni, nj, nk = idx[1], idx[2] - 1, idx[3]
            n_idx = (nk - 1) * N^2 + (nj - 1) * N + ni
            colidx[idx_lin*7-3] = n_idx
            dist = sqrt((ni - sp_cen_i)^2 + (nj - sp_cen_j)^2 + (nk - sp_cen_k)^2)
            coeff = 0f0
            if (spore_rad_lattice - cw_thickness ≤ dist) && (dist ≤ spore_rad_lattice)
                # Cell wall neighbour
                if region_id == 0 # Exterior - cell wall
                    coeff = Float32(Db * dtdx2 * factor)
                elseif region_id == 1 # Cell wall - cell wall
                    coeff = Float32(Dcw * dtdx2 * factor)
                end
            elseif dist > spore_rad_lattice
                # Exterior neighbour
                if region_id == 0 # Exterior - exterior
                    coeff = Float32(D * dtdx2 * factor)
                elseif region_id == 1 # Cell wall - exterior
                    coeff = Float32(Db * dtdx2 * factor)
                end
            elseif !empty_interior
                # Interior neighbour
                coeff = Float32(Dcw * dtdx2 * factor)
            end
            diag_val += coeff
            valsA[idx_lin*7-3] = - coeff
            valsB[idx_lin*7-3] = crank_nicolson ? coeff : 0f0
        end

        # ===== Right neighbour =====
        if idx[2] < N
            ni, nj, nk = idx[1], idx[2] + 1, idx[3]
            n_idx = (nk - 1) * N^2 + (nj - 1) * N + ni
            colidx[idx_lin*7-2] = n_idx
            dist = sqrt((ni - sp_cen_i)^2 + (nj - sp_cen_j)^2 + (nk - sp_cen_k)^2)
            coeff = 0f0
            if (spore_rad_lattice - cw_thickness ≤ dist) && (dist ≤ spore_rad_lattice)
                # Cell wall neighbour
                if region_id == 0 # Exterior - cell wall
                    coeff = Float32(Db * dtdx2 * factor)
                elseif region_id == 1 # Cell wall - cell wall
                    coeff = Float32(Dcw * dtdx2 * factor)
                end
            elseif dist > spore_rad_lattice
                # Exterior neighbour
                if region_id == 0 # Exterior - exterior
                    coeff = Float32(D * dtdx2 * factor)
                elseif region_id == 1 # Cell wall - exterior
                    coeff = Float32(Db * dtdx2 * factor)
                end
            elseif !empty_interior
                # Interior neighbour
                coeff = Float32(Dcw * dtdx2 * factor)
            end
            diag_val += coeff
            valsA[idx_lin*7-2] = - coeff
            valsB[idx_lin*7-2] = crank_nicolson ? coeff : 0f0
        end

        # ===== Front neighbour =====
        if idx[1] > 1
            ni, nj, nk = idx[1] - 1, idx[2], idx[3]
            n_idx = (nk - 1) * N^2 + (nj - 1) * N + ni
            colidx[idx_lin*7-1] = n_idx
            dist = sqrt((ni - sp_cen_i)^2 + (nj - sp_cen_j)^2 + (nk - sp_cen_k)^2)
            coeff = 0f0
            if (spore_rad_lattice - cw_thickness ≤ dist) && (dist ≤ spore_rad_lattice)
                # Cell wall neighbour
                if region_id == 0 # Exterior - cell wall
                    coeff = Float32(Db * dtdx2 * factor)
                elseif region_id == 1 # Cell wall - cell wall
                    coeff = Float32(Dcw * dtdx2 * factor)
                end
            elseif dist > spore_rad_lattice
                # Exterior neighbour
                if region_id == 0 # Exterior - exterior
                    coeff = Float32(D * dtdx2 * factor)
                elseif region_id == 1 # Cell wall - exterior
                    coeff = Float32(Db * dtdx2 * factor)
                end
            elseif !empty_interior
                # Interior neighbour
                coeff = Float32(Dcw * dtdx2 * factor)
            end
            diag_val += coeff
            valsA[idx_lin*7-1] = - coeff
            valsB[idx_lin*7-1] = crank_nicolson ? coeff : 0f0
        end

        # ===== Back neighbour =====
        if idx[1] < N
            ni, nj, nk = idx[1] + 1, idx[2], idx[3]
            n_idx = (nk - 1) * N^2 + (nj - 1) * N + ni
            colidx[idx_lin*7] = n_idx
            dist = sqrt((ni - sp_cen_i)^2 + (nj - sp_cen_j)^2 + (nk - sp_cen_k)^2)
            coeff = 0f0
            if (spore_rad_lattice - cw_thickness ≤ dist) && (dist ≤ spore_rad_lattice)
                # Cell wall neighbour
                if region_id == 0 # Exterior - cell wall
                    coeff = Float32(Db * dtdx2 * factor)
                elseif region_id == 1 # Cell wall - cell wall
                    coeff = Float32(Dcw * dtdx2 * factor)
                end
            elseif dist > spore_rad_lattice
                # Exterior neighbour
                if region_id == 0 # Exterior - exterior
                    coeff = Float32(D * dtdx2 * factor)
                elseif region_id == 1 # Cell wall - exterior
                    coeff = Float32(Db * dtdx2 * factor)
                end
            elseif !empty_interior
                # Interior neighbour
                coeff = Float32(Dcw * dtdx2 * factor)
            end
            diag_val += coeff
            valsA[idx_lin*7] = - coeff
            valsB[idx_lin*7] = crank_nicolson ? coeff : 0f0
        end

        valsA[idx_lin*7-6] = 1 + diag_val
        valsB[idx_lin*7-6] = crank_nicolson ? 1 - diag_val : 1f0

        region_ids[idx...] = region_id

        return nothing
    end


end