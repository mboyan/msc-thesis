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
    export update_GPU_low_res!
    export update_GPU_hi_res!
    export initialise_lattice_and_build_operator!

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
                lattice_new[idx...] = lattice_old[idx...]
                return nothing
            end
            
            bottom = lattice_old[idx[1], idx[2], mod1(idx[3] - 1, H)]
            top = lattice_old[idx[1], idx[2], mod1(idx[3] + 1, H)]
            left = lattice_old[idx[1], mod1(idx[2] - 1, N), idx[3]]
            right = lattice_old[idx[1], mod1(idx[2] + 1, N), idx[3]]
            front = lattice_old[mod1(idx[1] - 1, N), idx[2], idx[3]]
            back = lattice_old[mod1(idx[1] + 1, N), idx[2], idx[3]]

            diff_bottom, diff_top, diff_left, diff_right, diff_front, diff_back = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            # self_contributions = 0.0
            # nbr_contributions = 0.0

            if region_id == 0 # Exterior site
                # Check bottom neighbour
                if bottom < 10 # Exterior - exterior
                    # diff_bottom = D * bottom
                    diff_bottom = D * (bottom - center)
                    # self_contributions += D * center
                    # nbr_contributions += D * bottom
                elseif bottom < 100 # Exterior - cell wall
                    # diff_bottom = Deff * rem(bottom, 10)
                    diff_bottom = Deff * (rem(bottom, 10.0) - center)
                    # self_contributions += Deff * center
                    # nbr_contributions += Deff * rem(bottom, 10.0)
                end
                # Check top neighbour
                if top < 10 # Exterior - exterior
                    # diff_top = D * top
                    diff_top = D * (top - center)
                    # self_contributions += D * center
                    # nbr_contributions += D * top
                elseif top < 100 # Exterior - cell wall
                    # diff_top = Deff * rem(top, 10)
                    diff_top = Deff * (rem(top, 10.0) - center)
                    # self_contributions += Deff * center
                    # nbr_contributions += Deff * rem(top, 10.0)
                end
                # Check left neighbour
                if left < 10 # Exterior - exterior
                    # diff_left = D * left
                    diff_left = D * (left - center)
                    # self_contributions += D * center
                    # nbr_contributions += D * left
                elseif left < 100 # Exterior - cell wall
                    # diff_left = Deff * rem(left, 10)
                    diff_left = Deff * (rem(left, 10.0) - center)
                    # self_contributions += Deff * center
                    # nbr_contributions += Deff * rem(left, 10.0)
                end
                # Check right neighbour
                if right < 10 # Exterior - exterior
                    # diff_right = D * right
                    diff_right = D * (right - center)
                    # self_contributions += D * center
                    # nbr_contributions += D * right
                elseif right < 100 # Exterior - cell wall
                    # diff_right = Deff * rem(right, 10)
                    diff_right = Deff * (rem(right, 10.0) - center)
                    # self_contributions += Deff * center
                    # nbr_contributions += Deff * rem(right, 10.0)
                end
                # Check front neighbour
                if front < 10 # Exterior - exterior
                    # diff_front = D * front
                    diff_front = D * (front - center)
                    # self_contributions += D * center
                    # nbr_contributions += D * front
                elseif front < 100 # Exterior - cell wall
                    # diff_front = Deff * rem(front, 10)
                    diff_front = Deff * (rem(front, 10.0) - center)
                    # self_contributions += Deff * center
                    # nbr_contributions += Deff * rem(front, 10.0)
                end
                # Check back neighbour
                if back < 10 # Exterior - exterior
                    # diff_back = D * back
                    diff_back = D * (back - center)
                    # self_contributions += D * center
                    # nbr_contributions += D * back
                elseif back < 100 # Exterior - cell wall
                    # diff_back = Deff * rem(back, 10)
                    diff_back = Deff * (rem(back, 10.0) - center)
                    # self_contributions += Deff * center
                    # nbr_contributions += Deff * rem(back, 10.0)
                end
            elseif region_id == 1 # Cell wall site
                # Check bottom neighbour
                if bottom < 10 # Cell wall - exterior
                    # diff_bottom = Deff * bottom
                    diff_bottom = Deff * (bottom - center)
                    # self_contributions += Deff * center
                    # nbr_contributions += Deff * bottom
                elseif bottom < 100 # Cell wall - cell wall
                    # diff_bottom = Db * rem(bottom, 10)
                    diff_bottom = Db * (rem(bottom, 10.0) - center)
                    # self_contributions += Db * center
                    # nbr_contributions += Db * rem(bottom, 10.0)
                end
                # Check top neighbour
                if top < 10 # Cell wall - exterior
                    # diff_top = Deff * top
                    diff_top = Deff * (top - center)
                    # self_contributions += Deff * center
                    # nbr_contributions += Deff * top
                elseif top < 100 # Cell wall - cell wall
                    # diff_top = Db * rem(top, 10)
                    diff_top = Db * (rem(top, 10.0) - center)
                    # self_contributions += Db * center
                    # nbr_contributions += Db * rem(top, 10.0)
                end
                # Check left neighbour
                if left < 10 # Cell wall - exterior
                    # diff_left = Deff * left
                    diff_left = Deff * (left - center)
                    # self_contributions += Deff * center
                    # nbr_contributions += Deff * left
                elseif left < 100 # Cell wall - cell wall
                    # diff_left = Db * rem(left, 10)
                    diff_left = Db * (rem(left, 10.0) - center)
                    # self_contributions += Db * center
                    # nbr_contributions += Db * rem(left, 10.0)
                end
                # Check right neighbour
                if right < 10 # Cell wall - exterior
                    # diff_right = Deff * right
                    diff_right = Deff * (right - center)
                    # self_contributions += Deff * center
                    # nbr_contributions += Deff * right
                elseif right < 100 # Cell wall - cell wall
                    # diff_right = Db * rem(right, 10)
                    diff_right = Db * (rem(right, 10.0) - center)
                    # self_contributions += Db * center
                    # nbr_contributions += Db * rem(right, 10.0)
                end
                # Check front neighbour
                if front < 10 # Cell wall - exterior
                    # diff_front = Deff * front
                    diff_front = Deff * (front - center)
                    # self_contributions += Deff * center
                    # nbr_contributions += Deff * front
                elseif front < 100 # Cell wall - cell wall
                    # diff_front = Db * rem(front, 10)
                    diff_front = Db * (rem(front, 10.0) - center)
                    # self_contributions += Db * center
                    # nbr_contributions += Db * rem(front, 10.0)
                end
                # Check back neighbour
                if back < 10 # Cell wall - exterior
                    # diff_back = Deff * back
                    diff_back = Deff * (back - center)
                    # self_contributions += Deff * center
                    # nbr_contributions += Deff * back
                elseif back < 100 # Cell wall - cell wall
                    # diff_back = Db * rem(back, 10)
                    diff_back = Db * (rem(back, 10.0) - center)
                    # self_contributions += Db * center
                    # nbr_contributions += Db * rem(back, 10.0)
                end
            # else # Interior site
            #     return nothing
            end

            c_new = center + dtdx2 * (diff_bottom + diff_top + diff_left + diff_right + diff_front + diff_back)
            # c_new = center + dtdx2 * (diff_bottom + diff_top + diff_left + diff_right + diff_front + diff_back - self_contributions)
            # c_new = center + dtdx2 * (nbr_contributions - 6 * self_contributions)

            if region_id == 0
                lattice_new[idx...] = c_new
            elseif region_id == 1
                lattice_new[idx...] = 10.0 + c_new
            end
        end
        
        return nothing
    end

    function lin_idx(i, j, k, N, H)
        """
        Convert 3D indices to linear index.
        inputs:
            i (int) - the row index
            j (int) - the column index
            k (int) - the layer index
            N (int) - the number of lattice rows/columns
            H (int) - the number of lattice layers
        outputs:
            (int) - the linear index
        """
        return (i - 1) * N * H + (j - 1) * H + k
    end


    function initialise_lattice_and_build_operator!(c_init, c₀, sp_cen_indices, spore_rad_lattice, D, Db, Deff, dtdx2, crank_nicolson=true)
        """
        Build operator sparse matrix for implicitly solving the diffusion equation
        using the Crank-Nicolson method.
        inputs:
            c_init (array) - the initial state of the lattice
            c₀ (float) - the initial concentration
            sp_cen_indices (array) - the indices of the spore centers
            spore_rad_lattice (float) - the radius of the spores in lattice units
            D (float) - the diffusion constant
            Db (float) - the diffusion constant through the spore
            Deff (float) - the effective diffusion constant at the spore interface
            dtdx2 (float) - the update factor
            crank_nicolson (bool) - whether the operators are suited for Crank-Nicolson method
        outputs:
            op_A (sparse matrix) - the operator matrix A
            op_B (sparse matrix) - the operator matrix B
            region_ids (array) - the region IDs of the lattice
        """

        N = size(c_init)[1]
        H = size(c_init)[3]
        Nt = N * N * H

        op_A = spzeros(Float32, Nt, Nt)
        op_B = spzeros(Float32, Nt, Nt)
        region_ids = zeros(Int, N, N, H)

        ddia = sqrt(2)
        ddia_triple = 3 * ddia
        lattice_dia = 2*N^2 + H^2

        # Initialise concentrations in cell wall
        for i in 1:N, j in 1:N, k in 1:H

            idx = lin_idx(i, j, k, N, H)
            diag_val = 0f0

            vneum_nbrs = [(i, j, mod1(k - 1, H)), (i, j, mod1(k + 1, H)),
                        (i, mod1(j - 1, N), k), (i, mod1(j + 1, N), k),
                        (mod1(i - 1, N), j, k), (mod1(i + 1, N), j, k)]
            
            region_id = 0
            min_dist = lattice_dia
            for sp_cen_idx in sp_cen_indices

                # Compute distance to spore center
                dist = norm([i - sp_cen_idx[1] - 1, j - sp_cen_idx[2] - 1, k - sp_cen_idx[3] - 1])
                if dist < min_dist
                    min_dist = dist
                end
                
                if spore_rad_lattice - ddia ≤ dist ≤ spore_rad_lattice
                    # Cell wall site
                    region_id = 1
                    c_init[i, j, k] = c₀
                    break
                elseif dist < spore_rad_lattice - ddia
                    # Interior site
                    region_id = 2
                    break
                end
            end

            region_ids[i, j, k] = region_id
            
            # Get neighbour coefficients
            if min_dist ≤ spore_rad_lattice + ddia_triple

                # Iterate over von Neumann neighbours
                for (ni, nj, nk) in vneum_nbrs
                    for sp_cen_idx in sp_cen_indices
                        
                        n_idx = lin_idx(ni, nj, nk, N, H)
                        # println("Index: $idx, Neighbour: $n_idx")
                        dist = norm([ni - sp_cen_idx[1] - 1, nj- sp_cen_idx[2] - 1, nk - sp_cen_idx[3] - 1])
                        coeff = 0f0

                        if spore_rad_lattice - ddia ≤ dist ≤ spore_rad_lattice
                            # Cell wall neighbour
                            if region_id == 0 # Exterior - cell wall
                                coeff = Float32(Deff * dtdx2 * 0.5)
                            elseif region_id == 1 # Cell wall - cell wall
                                coeff = Float32(Db * dtdx2 * 0.5)
                            end

                        elseif dist ≤ spore_rad_lattice - ddia
                            # Exterior neighbour
                            if region_id == 0 # Exterior - exterior
                                coeff = Float32(D * dtdx2 * 0.5)
                            elseif region_id == 1 # Cell wall - exterior
                                coeff = Float32(Deff * dtdx2 * 0.5)
                            end
                        end

                        diag_val += coeff
                        op_A[idx, n_idx] = - coeff
                        op_B[idx, n_idx] = crank_nicolson ? coeff : 0f0
                        # if crank_nicolson
                        #     op_B[idx, n_idx] = coeff
                        # else
                        #     op_B[idx, n_idx] = 0f0
                        # end
                    end
                end
            end

            op_A[idx, idx] = 1 + diag_val 
            op_B[idx, idx] = 1 - diag_val
        end
        println("Concentrations initialised.")

        # Check if matrices are singular
        if iszero(det(op_A))
            println("Matrix A is singular.")
        end
        if iszero(det(op_B))
            println("Matrix B is singular.")
        end

        # Check if non-diagonal elements are non-zero
        if iszero(sum(op_A) - sum(diag(op_A)))
            println("Matrix A has all zero non-diagonal elements.")
        end
        if iszero(sum(op_B) - sum(diag(op_B)))
            println("Matrix B has all zero non-diagonal elements.")
        end

        return op_A, op_B, region_ids
    end
end