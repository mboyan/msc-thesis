module Conversions
    """
    Contains conversion utilites
    """
    
    using QuadGK
    using LinearAlgebra

    export mL_to_cubic_um
    export inverse_mL_to_cubic_um
    export cubic_um_to_mL
    export inverse_cubic_um_to_mL
    export inverse_uL_to_mL
    export convert_D_to_Ps
    export convert_Ps_to_D
    export measure_coverage
    export extract_mean_cw_concentration
    export compute_spore_concentration

    function mL_to_cubic_um(mL)
        """
        Convert milliliters to micrometers cubed.
        inputs:
            mL (float): volume in milliliters
        outputs:
            (float) volume in micrometers cubed
        """
        return mL * 1e12
    end

    function inverse_mL_to_cubic_um(mL_inv)
        """
        Convert inverse milliliters to inverse micrometers cubed.
        inputs:
            mL_inv (float): volume in inverse milliliters
        outputs:
            (float) volume in inverse micrometers cubed
        """
        return mL_inv * 1e-12
    end

    function cubic_um_to_mL(cubic_um)
        """
        Convert micrometers cubed to milliliters.
        inputs:
            cubic_um (float): volume in micrometers cubed
        outputs:
            (float) volume in milliliters
        """
        return cubic_um * 1e-12
    end

    function inverse_cubic_um_to_mL(cubic_um_inv)
        """
        Convert inverse micrometers cubed to inverse milliliters.
        inputs:
            microns_cubed_inv (float): number density in inverse micrometers cubed
        outputs:
            (float) number density in inverse milliliters
        """
        return cubic_um_inv * 1e12
    end

    function inverse_uL_to_mL(uL_inv)
        """
        Convert inverse milliliters to inverse micrometers cubed.
        inputs:
            uL_inv (float): number density in inverse microliters
        outputs:
            (float) number density in inverse milliliters
        """
        return uL_inv * 1000
    end

    function convert_D_to_Ps(D, K, d)
        """
        Convert diffusion coefficient to permeability.
        inputs:
            D (float): diffusion coefficient in micrometers squared per second
            K (float): partition coefficient
            d (float): thickness of the membrane in micrometers
        outputs:
            (float) permeation constant in micrometers per second
        """
        return D * K / d
    end

    function convert_Ps_to_D(Ps, K, d)
        """
        Convert permeability to diffusion coefficient.
        inputs:
            Ps (float): permeability in micrometers per second
            K (float): partition coefficient
            d (float): thickness of the membrane in micrometers
        outputs:
            (float) diffusion constant in micrometers squared per second
        """
        return Ps * d / K
    end

    function coverage_integral(ϕ, R, d)
        """
        The coverage function for a sphere.
        inputs:
            phi (float): vertical angle in radians
            R (float): radius of the sphere
            d (float): distance between the centers of the spheres
        outputs:
            (float) coverage function value
        """
        Δ = d * cos(ϕ) - sqrt(R^2 - (d * sin(ϕ))^2) - R
        return exp(-Δ) * sin(ϕ)
    end

    function measure_coverage(sample_shere_center::Tuple, nbr_sphere_centers, rad=1)
        """
        Measure the cumulative shadow intensity of neighboring spheres on a sample sphere.
        inputs:
            sample_shere_center (Tuple{Float64, 1}): center of the sample sphere
            nbr_sphere_centers (Array{Tuple{Float64}, 1}): centers of the neighboring spheres
            rad (float): radius of the spheres
        outputs:
            (float) cumulative shadow intensity
        """
        intsum = 0.0
        for center in nbr_sphere_centers
            d = norm(center .- sample_shere_center)
            ϕ₀ = asin(rad / d)
            integral, err = quadgk(ϕ -> coverage_integral(ϕ, rad, d), 0, ϕ₀)
            intsum += integral
        end

        return 0.5 * intsum
    end

    function extract_mean_cw_concentration(c_frames, region_ids)
        """
        Extract the mean concentration from the cell wall region.
        inputs:
            c_lattice (Array{Float64, 3}): concentration lattice
            region_ids (Array{Int, 1}): region ids
        outputs:
            c_avg (Array{Float32, 1}): average concentrations
        """

        # Add new axis to region_ids
        region_ids = reshape(region_ids, 1, size(region_ids)[1], size(region_ids)[2])

        # Mask the cell wall region and take the average concentration
        c_cell_wall = c_frames .* (region_ids .== 1)
        c_avg = sum(c_cell_wall, dims=(2, 3)) ./ sum(region_ids .== 1)

        return c_avg
    end

    function compute_spore_concentration(c_frames, region_ids, spore_rad, dx, cw_thickness=nothing)
        """
        Compute the inhibitor concentration relative to the spore volume
        from the cell wall region.
        inputs:
            c_frames (Array{Float64, 3}): concentration lattice
            region_ids (Array{Int, 1}): region ids
            spore_rad (float): spore radius
            cw_thickness (float): cell wall thickness
            dx (float): lattice spacing
        """

        if isnothing(cw_thickness)
            cw_thickness = dx
        end

        # Add new axis to region_ids
        region_ids = reshape(region_ids, 1, size(region_ids)...)
        # println("Region ids: ", size(region_ids))

        # Isolate only central spore region
        if ndims(c_frames) == 3
            # Extrapolate 3D volume from 2D section
            center = size(c_frames[1, :, :]) .÷ 2 .* dx
            indices = CartesianIndices(c_frames[1, :, :])
            X = [idx[1] * dx for idx in indices]  # Row indices
            Y = [idx[2] * dx for idx in indices]  # Column indices
            dist = sqrt.((X .- center[1]).^2 + (Y .- center[2]).^2)
            central_spore_mask = dist .<= spore_rad
            central_spore_mask = reshape(central_spore_mask, 1, size(central_spore_mask)...)
            # println("Central spore nodes: ", sum(central_spore_mask))

            region_ids = region_ids .* central_spore_mask

            # Compute the cell wall moles
            moles_cw_voxels_sec = c_frames .* (region_ids .== 1) .* dx^3
            moles_cw_sec = sum(moles_cw_voxels_sec, dims=(2, 3))
            moles_cw = 2 * moles_cw_sec * spore_rad / cw_thickness
            # println("Moles CW: ", moles_cw)

            # Compute the spore volume
            spore_vol = 4/3 * π * spore_rad^3
            # println("Spore volume: ", spore_vol)

        elseif ndims(c_frames) == 4
            # Compute the spore volume accurately
            center = size(c_frames[1, :, :, :]) .÷ 2 .* dx
            indices = CartesianIndices(c_frames[1, :, :, :])
            X = [idx[1] * dx for idx in indices]  # Row indices
            Y = [idx[2] * dx for idx in indices]  # Column indices
            Z = [idx[3] * dx for idx in indices]  # Depth indices
            dist = sqrt.((X .- center[1]).^2 + (Y .- center[2]).^2 + (Z .- center[3]).^2)
            central_spore_mask = dist .<= spore_rad
            central_spore_mask = reshape(central_spore_mask, 1, size(central_spore_mask)...)
            # println("Central spore nodes: ", sum(central_spore_mask))

            region_ids = region_ids .* central_spore_mask
            # println("Cell wall nodes: ", sum(region_ids .== 1))

            # Compute the cell wall moles
            moles_cw_voxels = c_frames .* (region_ids .== 1) .* dx^3
            moles_cw = sum(moles_cw_voxels, dims=(2, 3, 4))
            # println("Moles CW: ", moles_cw)

            # Compute the spore volume
            spore_vol = sum(central_spore_mask) * dx^3
            # println("Spore volume: ", spore_vol)
        else
            error("Invalid number of dimensions of c_frames")
        end
        
        # Compute the inhibitor concentration relative to the spore volume
        c_spore = moles_cw / spore_vol

        return c_spore[:]
    end

end