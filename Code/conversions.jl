module Conversions
    """
    Contains conversion utilites
    """
    
    using QuadGK
    using LinearAlgebra
    using MeshGrid

    export cm_to_um
    export um_to_cm
    export nm_to_um
    export um_to_nm
    export cm2_to_um2
    export um2_to_cm2
    export mL_to_cubic_um
    export inverse_mL_to_cubic_um
    export cubic_um_to_mL
    export inverse_cubic_um_to_mL
    export inverse_uL_to_mL
    export convert_D_to_Ps
    export convert_Ps_to_D
    export compute_stokes_radius
    export compute_spore_area_and_volume_from_dia
    export compute_D_from_radius_and_viscosity
    export measure_coverage
    export extract_mean_cw_concentration
    export compute_spore_concentration
    export generate_spore_positions

    function cm_to_um(cm)
        """
        Convert centimeters to micrometers.
        inputs:
            cm (float): length in centimeters
        outputs:
            (float) length in micrometers
        """
        return cm * 1e4
    end

    function um_to_cm(um)
        """
        Convert micrometers to centimeters.
        inputs:
            um (float): length in micrometers
        outputs:
            (float) length in centimeters
        """
        return um * 1e-4
    end

    function nm_to_um(nm)
        """
        Convert nanometers to micrometers.
        inputs:
            nm (float): length in nanometers
        outputs:
            (float) length in micrometers
        """
        return nm * 1e-3
    end

    function um_to_nm(um)
        """
        Convert micrometers to nanometers.
        inputs:
            um (float): length in micrometers
        outputs:
            (float) length in nanometers
        """
        return um * 1e3
    end

    function cm2_to_um2(cm2)
        """
        Convert square centimeters to square micrometers.
        inputs:
            cm2 (float): area in square centimeters
        outputs:
            (float) area in square micrometers
        """
        return cm2 * 1e8
    end

    function um2_to_cm2(um2)
        """
        Convert square micrometers to square centimeters.
        inputs:
            um2 (float): area in square micrometers
        outputs:
            (float) area in square centimeters
        """
        return um2 * 1e-8
    end

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

    function compute_spore_area_and_volume_from_dia(diameter)
        """
        Compute the area and volume of a spherical spore
        given its diameter.
        inputs:
            diameter (float): diameter of the spore
        outputs:
            A (float): the area of the spore
            V (float): the volume of the spore
        """
        rad = diameter / 2.0
        A = 4 * π * rad^2
        V = 4/3 * π * rad^3
        return A, V
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

    function measure_coverage(sample_shere_center::Tuple, nbr_sphere_centers; rad=1, dx=1)
        """
        Measure the cumulative shadow intensity of neighboring spheres on a sample sphere.
        inputs:
            sample_shere_center (Tuple{Float64, 1}): center of the sample sphere
            nbr_sphere_centers (Array{Tuple{Float64}, 1}): centers of the neighboring spheres
            rad (float): radius of the spheres
            dx (float): lattice spacing, if 1, the absolute distance is used
        outputs:
            (float) cumulative shadow intensity
        """
        sample_shere_center = sample_shere_center .* dx
        nbr_sphere_centers = [center .* dx for center in nbr_sphere_centers]
        intsum = 0.0
        for center in nbr_sphere_centers
            d = norm(center .- sample_shere_center)
            ϕ₀ = asin(rad / d)
            integral, err = quadgk(ϕ -> coverage_integral(ϕ, rad, d), 0, ϕ₀)
            intsum += integral
        end

        return 0.5 * intsum
    end

    function compute_stokes_radius(mass, density)
        """
        Compute the stokes radius of a molecule
        based on the molecular mass and density.
        inputs:
            mass (float): molecular mass of the substance in grams per mole
            density (float): density of the substance in grams per milliliters
        outputs:
            (float) Stokes radius in micrometers
        """
        NA = 6.022e23  # Avogadro's number
        vol = mass / (density * NA * 1e-12)
        println("Molecular volume: ", vol)
        return (3 * vol / 4π)^(1/3)# * 1e6
    end

    function compute_D_from_radius_and_viscosity(a, eta)
        """
        Compute the diffusion coefficient from the Stokes radius and viscosity.
        inputs:
            a (float): Stokes radius in micrometers
            eta (float): viscosity in centipoise / millipascal seconds
        outputs:
            (float) diffusion coefficient in micrometers squared per second
        """
        kT = 4.1e-21  # Boltzmann constant in Joules
        eta = eta * 1e-3  # Convert centipoise to pascal seconds
        a = a * 1e-6  # Convert micrometers to meters
        return kT / (6 * π * eta * a) * 1e12  # Convert to micrometers squared per second
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

    function generate_spore_positions(spore_density, Lx, Lz; base_height=nothing)
        """
        Generate positions of spores in a 3D grid.
        inputs:
            spore_density (float): density of spores in spores/mL
            Lx (float): length of the grid in micrometers
            Lz (float): height of the grid in micrometers
            base_height (float): height of the base of the grid in micrometers, if specified, a 2D grid is generated
        """

        # Convert spore density to spores/micrometer^3
        spore_density = inverse_mL_to_cubic_um(spore_density)
        println("Spore density: $(spore_density) spores per micrometer^3")

        # Calculate the number of spores to place
        V_grid = Lx^2 * Lz
        # n_spores = spore_density * V_grid

        if isnothing(base_height)
            n_spores_1D = cbrt(spore_density)
        else
            n_spores_1D = sqrt(spore_density)
        end

        spore_spacing = 1 / n_spores_1D

        println("Populating volume of $(V_grid) micrometers^3 with $(spore_density) spores per um^3, $(n_spores_1D) spores per dimension")
        println("Spore spacing: $(spore_spacing) micrometers")

        spores_x = collect(0:spore_spacing:Lx)
        if isnothing(base_height)
            spores_z = collect(0:spore_spacing:Lz)
            spores_x, spores_y, spores_z = meshgrid(spores_x, spores_x, spores_z)
        else
            spores_x, spores_y = meshgrid(spores_x, spores_x)
            spores_z = zeros(length(spores_x)) .+ base_height
        end

        spore_coords = zeros(Float64, length(spores_x), 3)
        spore_coords .= hcat(vec(spores_x), vec(spores_y), vec(spores_z))  # Concatenate the coordinates

        return spore_coords, spore_spacing
    end

end