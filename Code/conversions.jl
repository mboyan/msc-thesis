module Conversions
    """
    Contains conversion utilites
    """
    
    using QuadGK
    using LinearAlgebra

    export mL_to_cubic_um
    export inverse_mL_to_cubic_um
    export inverse_cubic_um_to_mL
    export inverse_uL_to_mL
    export convert_D_to_Ps
    export convert_Ps_to_D
    export measure_coverage
    export extract_mean_cw_concentration

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

    function measure_coverage(sample_shere_center, nbr_sphere_centers, rad=1)
        """
        Measure the cumulative shadow intensity of neighboring spheres on a sample sphere.
        inputs:
            sample_shere_center (Array{Float64, 1}): center of the sample sphere
            nbr_sphere_centers (Array{Array{Float64, 1}, 1}): centers of the neighboring spheres
            rad (float): radius of the spheres
        outputs:
            (float) cumulative shadow intensity
        """
        intsum = 0.0
        for center in eachrow(nbr_sphere_centers)
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

end