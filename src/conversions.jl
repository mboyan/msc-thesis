module Conversions
    """
    Contains conversion utilites
    """
    
    using QuadGK
    using Cubature
    using LinearAlgebra
    using MeshGrid
    using Distributions
    using SpecialFunctions

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
    export inverse_um_to_mL
    export convert_D_to_Ps
    export convert_Ps_to_D
    export compute_stokes_radius
    export composite_Ps
    export compute_spore_area_and_volume_from_dia
    export compute_c_eq
    export compute_D_from_radius_and_viscosity
    export measure_coverage
    export measure_shielding_index
    export extract_mean_cw_concentration
    export compute_spore_concentration
    export generate_spore_positions
    export compute_ps_layer_volume
    export flatten_recursive


    flatten_recursive(x) = (x isa AbstractVector) ? vcat(map(flatten_recursive, x)...) : [x]

    """
    Convert centimeters to micrometers.
    inputs:
        cm (Float): length in centimeters
    outputs:
        (Float) length in micrometers
    """
    function cm_to_um(cm)
        return cm * 1e4
    end

    """
    Convert micrometers to centimeters.
    inputs:
        um (Float): length in micrometers
    outputs:
        (Float) length in centimeters
    """
    function um_to_cm(um)
        return um * 1e-4
    end

    """
    Convert nanometers to micrometers.
    inputs:
        nm (Float): length in nanometers
    outputs:
        (Float) length in micrometers
    """
    function nm_to_um(nm)
        return nm * 1e-3
    end

    """
    Convert micrometers to nanometers.
    inputs:
        um (Float): length in micrometers
    outputs:
        (Float) length in nanometers
    """
    function um_to_nm(um)
        return um * 1e3
    end

    """
    Convert square centimeters to square micrometers.
    inputs:
        cm2 (Float): area in square centimeters
    outputs:
        (Float) area in square micrometers
    """
    function cm2_to_um2(cm2)
        return cm2 * 1e8
    end

    """
    Convert square micrometers to square centimeters.
    inputs:
        um2 (Float): area in square micrometers
    outputs:
        (Float) area in square centimeters
    """
    function um2_to_cm2(um2)
        return um2 * 1e-8
    end

    """
    Convert milliliters to micrometers cubed.
    inputs:
        mL (Float): volume in milliliters
    outputs:
        (Float) volume in micrometers cubed
    """
    function mL_to_cubic_um(mL)
        return mL * 1e12
    end

    """
    Convert inverse milliliters to inverse micrometers cubed.
    inputs:
        mL_inv (Float): volume in inverse milliliters
    outputs:
        (Float) volume in inverse micrometers cubed
    """
    function inverse_mL_to_cubic_um(mL_inv)
        return mL_inv * 1e-12
    end

    """
    Convert micrometers cubed to milliliters.
    inputs:
        cubic_um (Float): volume in micrometers cubed
    outputs:
        (Float) volume in milliliters
    """
    function cubic_um_to_mL(cubic_um)
        return cubic_um * 1e-12
    end

    """
    Convert inverse micrometers cubed to inverse milliliters.
    inputs:
        microns_cubed_inv (Float): number density in inverse micrometers cubed
    outputs:
        (Float) number density in inverse milliliters
    """
    function inverse_cubic_um_to_mL(cubic_um_inv)
        return cubic_um_inv * 1e12
    end

    """
    Convert inverse milliliters to inverse micrometers cubed.
    inputs:
        uL_inv (Float): number density in inverse microliters
    outputs:
        (Float) number density in inverse milliliters
    """
    function inverse_um_to_mL(uL_inv)
        return uL_inv * 1000
    end

    """
    Convert diffusion coefficient to permeability.
    inputs:
        D (Float): diffusion coefficient in micrometers squared per second
        K (Float): partition coefficient
        d (Float): thickness of the membrane in micrometers
    outputs:
        (Float) permeation constant in micrometers per second
    """
    function convert_D_to_Ps(D, K, d)
        return D * K / d
    end

    """
    Convert permeability to diffusion coefficient.
    inputs:
        Ps (Float): permeability in micrometers per second
        K (Float): partition coefficient
        d (Float): thickness of the membrane in micrometers
    outputs:
        (Float) diffusion constant in micrometers squared per second
    """
    function convert_Ps_to_D(Ps, K, d)
        return Ps * d / K
    end

    """
    Compute the composite permeability of a series of membranes in parallel.
    inputs:
        permeabilities (Array{Float64, 1}): array of permeabilities in micrometers per second
    outputs:
        (Float) composite permeability in micrometers per second
    """
    function composite_Ps(permeabilities)
        return 1 / sum(1 ./ permeabilities)
    end

    """
    Compute the area and volume of a spherical spore
    given its diameter.
    inputs:
        diameter (Float): diameter of the spore
    outputs:
        A (Float): the area of the spore
        V (Float): the volume of the spore
    """
    function compute_spore_area_and_volume_from_dia(diameter)
        rad = diameter / 2.0
        A = 4 * π * rad^2
        V = 4/3 * π * rad^3
        return A, V
    end

    """
    Compute the equilibrium concentration of a spore in a solution.
    inputs:
        ρₛ (Float): spore density in spores/mL
        V (Float): volume of the solution in micrometers cubed
        c₀ (Float): initial concentration of the solution in M
        c_ex (Float): exogenous concentration in M
    outputs:
        c_eq (Float): equilibrium concentration in M
    """
    function compute_c_eq(ρₛ, V, c₀, c_ex)
        ρₛ = inverse_mL_to_cubic_um(ρₛ)  # Convert from spores/mL to spores/m^3
        ϕ = ρₛ * V
        return ϕ * c₀ + (1 - ϕ) * c_ex
    end

    """
    The coverage function for a sphere.
    inputs:
        phi (Float): vertical angle in radians
        R (Float): radius of the sphere
        d (Float): distance between the centers of the spheres
    outputs:
        (Float) coverage function value
    """
    function coverage_integral(ϕ, R, d)
        Δ = d * cos(ϕ) - sqrt(R^2 - (d * sin(ϕ))^2) - R
        return exp(-Δ) * sin(ϕ)
    end

    """
    Measure the cumulative shadow intensity of neighboring spheres on a sample sphere.
    inputs:
        sample_shere_center (Tuple{Float64, 1}): center of the sample sphere
        nbr_sphere_centers (Array{Tuple{Float64}, 1}): centers of the neighboring spheres
        rad (Float): radius of the spheres
        dx (Float): lattice spacing, if 1, the absolute distance is used
    outputs:
        (Float) cumulative shadow intensity
    """
    function measure_coverage(sample_shere_center::Tuple, nbr_sphere_centers; rad=1, dx=1)
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

    """
    Measure the shielding index of neighboring spheres on a sample sphere
    using the non-dominant eigenvalues of the unnormalised orientation tensor.
    inputs:
        sample_shere_center (Tuple{Float64, 1}): center of the sample sphere
        nbr_sphere_centers (Array{Tuple{Float64}, 1}): centers of the neighboring spheres
    outputs:
        (Float) cumulative shadow intensity
    """
    function measure_shielding_index(sample_shere_center::Tuple, nbr_sphere_centers)

        # Create matrix
        M = zeros(Float64, 3, 3)
        for nbr in nbr_sphere_centers
            nbr = collect(nbr)
            u = nbr .- sample_shere_center
            u = u ./ norm(u)
            M += u * u'
        end

        # Eigenvalues
        λ = eigvals(M)
        λ = sort(λ, rev=true)
        
        return λ[2] + λ[3]
    end

    """
    Compute the stokes radius of a molecule
    based on the molecular mass and density.
    inputs:
        mass (Float): molecular mass of the substance in grams per mole
        density (Float): density of the substance in grams per milliliters
    outputs:
        (Float) Stokes radius in micrometers
    """
    function compute_stokes_radius(mass, density)
        NA = 6.022e23  # Avogadro's number
        vol = mass / (density * NA * 1e-12)
        println("Molecular volume: ", vol)
        return (3 * vol / 4π)^(1/3)# * 1e6
    end

    """
    Compute the diffusion coefficient from the Stokes radius and viscosity.
    inputs:
        a (Float): Stokes radius in micrometers
        eta (Float): viscosity in centipoise / millipascal seconds
    outputs:
        (Float) diffusion coefficient in micrometers squared per second
    """
    function compute_D_from_radius_and_viscosity(a, eta)
        kT = 4.1e-21  # Boltzmann constant in Joules
        eta = eta * 1e-3  # Convert centipoise to pascal seconds
        a = a * 1e-6  # Convert micrometers to meters
        return kT / (6 * π * eta * a) * 1e12  # Convert to micrometers squared per second
    end

    """
    Extract the mean concentration from the cell wall region.
    inputs:
        c_lattice (Array{Float64, 3}): concentration lattice
        region_ids (Array{Int, 1}): region ids
    outputs:
        c_avg (Array{Float32, 1}): average concentrations
    """
    function extract_mean_cw_concentration(c_frames, region_ids)

        # Add new axis to region_ids
        region_ids = reshape(region_ids, 1, size(region_ids)[1], size(region_ids)[2])

        # Mask the cell wall region and take the average concentration
        c_cell_wall = c_frames .* (region_ids .== 1)
        c_avg = sum(c_cell_wall, dims=(2, 3)) ./ sum(region_ids .== 1)

        return c_avg
    end

    """
    Compute the inhibitor concentration relative to the spore volume
    from the cell wall region.
    inputs:
        c_frames (Array{Float64, 3}): concentration lattice
        region_ids (Array{Int, 1}): region ids
        spore_rad (Float): spore radius
        cw_thickness (Float): cell wall thickness
        dx (Float): lattice spacing
    """
    function compute_spore_concentration(c_frames, region_ids, spore_rad, dx, cw_thickness=nothing)

        if isnothing(cw_thickness)
            cw_thickness = dx
        end

        # Add new axis to region_ids
        region_ids = reshape(region_ids, 1, size(region_ids)...)
        # println("Region ids: ", size(region_ids))

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
        
        # Compute the inhibitor concentration relative to the spore volume
        c_spore = moles_cw / spore_vol

        return c_spore[:]
    end

    """
    Generate positions of spores in a 3D grid.
    inputs:
        spore_density (Float): density of spores in spores/mL
        Lx (Float): length of the grid in micrometers
        Lz (Float): height of the grid in micrometers
        base_height (Float): height of the base of the grid in micrometers, if specified, a 2D grid is generated
    """
    function generate_spore_positions(spore_density, Lx, Lz; base_height=nothing)

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

    """
    Compute polysaccharide layer volume.
    inputs:
        R - spore radius in μm
        d_hp - hydrophobin layer thickness in μm
        d_ps - polysaccharide layer thickness in μm
        porosity - non-vacant fraction of the polysaccharide layer
    output:
        polysaccharide layer volume in μm^3
    """
    function compute_ps_layer_volume(R, d_hp, d_ps; porosity=0.32)
        return porosity .* π .* ((R .- d_hp).^3 .- (R .- d_hp .- d_ps).^3)
    end
end