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
    using Symbolics, SymbolicUtils

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
    export build_coupling_library
    export closed_form_inverse
    export build_reparam
    export auto_reparameterise
    export extract_coupled_pairs
    export invert_reparams
    export reference_geometry

    export HDMRResult
    export ModelConfig
    export Reparameterisation


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

    # ====================================
    # ===== MODEL REPARAMETERISATION =====
    # ====================================
    """
    Symbolic representation of a parameter coupling.
    expr:     the composite expression (e.g. c_cs / (K_cC + c_cs))
    original: the original parameters appearing in expr
    composite_name: the name to give the composite
    fixed_vars: any constants needed
    """
    struct SymbolicCoupling
        composite_name :: Symbol
        expr           :: Num                    # Symbolics expression
        original       :: Vector{Symbol}         # parameters this replaces/involves
        fixed_vals     :: Dict{Symbol, Float64}  # constants substituted at build time
    end

    struct Reparameterisation
        composite_names :: Vector{Symbol}   # Names of the composite parameters exposed to the sampler
        original_names  :: Vector{Symbol}   # Names of the original parameters they replace
        forward         :: Function         # composite → original (what the model receives)
        inverse         :: Function         # original → composite (for initialising from example_params)
    end

    """
    Parameter configuration for a single model.
    Holds names, anchor values, and prior bounds.
    """
    struct ModelConfig
        name        :: String
        param_names :: Vector{Symbol}
        anchor      :: Vector{Float64}      # example_params — known plausible point
        lower       :: Vector{Float64}      # prior lower bounds (log scale for LogNormal params)
        upper       :: Vector{Float64}      # prior upper bounds (log scale for LogNormal params)
        log_scaled  :: Vector{Bool}         # true if parameter is LogNormal (sweep in log space)
        target_pairs:: Vector{Tuple{Int,Int}} # pairs to test for interactions (from algebraic analysis)
        reparams    :: Vector{Reparameterisation}
    end

    struct HDMRResult
        config          :: ModelConfig
        S_first         :: Matrix{Float64}   # d × n_outputs
        S_pair          :: Matrix{Float64}   # n_pairs × n_outputs
        S_total         :: Matrix{Float64}   # d × n_outputs
        S_mean          :: Vector{Float64}   # d — averaged over outputs
        d_eff           :: Int
        log_ml          :: Float64           # approximate log marginal likelihood
        z_scores_all    :: Vector{Float64}   # from all sweep samples combined
    end

    @variables K_cC Pₛ Pₛ_cs neg_δ_γ neg_δ_ω μ_γ μ_ω
    @variables c₀_cs_ref t_ref A_ref V_cw_ref V_ref ρₛ_ref

    # ─────────────────────────────────────────────────────────────────────────────
    # Intermediate expressions derived directly from the call chain
    # ─────────────────────────────────────────────────────────────────────────────

    # Inhibition: geometric time constant and filling fraction (geometry fixed at anchor)
    τ_β_expr  = V_ref / (Pₛ * A_ref)
    ϕ_expr    = ρₛ_ref * V_ref
    β_ref_expr = ϕ_expr + (1 - ϕ_expr) * exp(-t_ref / (τ_β_expr * (1 - ϕ_expr)))
    
    # Induction: carbon source concentration and saturation signal
    τ_cs_expr  = V_cw_ref / (A_ref * Pₛ_cs)
    c_cs_expr  = c₀_cs_ref * (1 - exp(-t_ref / τ_cs_expr))
    s_ref_expr = c_cs_expr / (K_cC + c_cs_expr)

    # Variance composites (from neg_δ parameterisation)
    σ_γ_expr = μ_γ * exp(-neg_δ_γ)
    σ_ω_expr = μ_ω * exp(-neg_δ_ω)

    # Normalised CDF arguments at reference conditions
    A_γ_ref_expr = (β_ref_expr - μ_γ) / σ_γ_expr
    A_ω_ref_expr = (s_ref_expr - μ_ω) / σ_ω_expr

    # ─────────────────────────────────────────────────────────────────────────────
    # Reference geometry computed from anchor example_params
    # Caller must supply these from the anchor's ξ_anchor, κ_anchor, d_hp
    # ─────────────────────────────────────────────────────────────────────────────
    """
    Compute the fixed geometric and physical reference constants used to
    substitute into symbolic coupling expressions. All symbolic coupling
    expressions are evaluated at these reference values, converting
    fully symbolic expressions into expressions in the model parameters
    only. Should be called once per experimental condition using the
    anchor parameter values (example_params).

    Inputs:
        ξ_anchor  (Float64) - spore radius at the anchor point in μm.
                            Taken from the ξ distribution mean in example_params.
        κ_anchor  (Float64) - cell wall thickness at the anchor point in μm.
                            Taken from the κ distribution mean in example_params.
        d_hp      (Float64) - hydrophobin layer thickness in μm. Treated as
                            fixed (not a free parameter); taken from def_params.
        ρₛ        (Float64) - spore density in spores/μm³. Sets the volume
                            fraction ϕ = ρₛ·V, which controls how much
                            inhibitor is depleted at equilibrium.
        c₀_cs     (Float64) - initial external carbon source concentration in M.
                            Sets the maximum achievable c_cs as t → ∞.
        t_ref     (Float64) - reference time in seconds at which the coupling
                            expressions are evaluated. Should be chosen as a
                            representative experimental time, e.g. the median
                            observation time across all assay conditions.

    Outputs:
        Dict with keys:
            :V_ref     (Float64) - spore volume 4π/3·ξ³ in μm³
            :A_ref     (Float64) - spore surface area 4π·ξ² in μm²
            :V_cw_ref  (Float64) - vacant cell wall (polysaccharide layer pore)
                                volume in μm³, computed via
                                compute_ps_layer_volume at the anchor geometry
            :ρₛ_ref    (Float64) - spore density, passed through unchanged
            :c₀_cs_ref (Float64) - initial carbon source concentration, passed
                                through unchanged
            :t_ref     (Float64) - reference time, passed through unchanged
            :ϕ_ref     (Float64) - spore volume fraction ρₛ·V, dimensionless.
                                Represents the fraction of total volume
                                occupied by spores; used in β_ref inversion.
    """
    function reference_geometry(ξ_anchor, κ_anchor, d_hp, ρₛ, c₀_cs, t_ref)
        V   = 4/3 * π * ξ_anchor^3
        A   = 4π  * ξ_anchor^2
        V_cw = compute_ps_layer_volume(ξ_anchor, d_hp, κ_anchor)
        ϕ   = ρₛ * V
        return Dict(
            :V_ref     => V,
            :A_ref     => A,
            :V_cw_ref  => V_cw,
            :ρₛ_ref    => ρₛ,
            :c₀_cs_ref => c₀_cs,
            :t_ref     => t_ref,
            :ϕ_ref     => ϕ
        )
    end

    # ─────────────────────────────────────────────────────────────────────────────
    # Coupling library
    # Each entry: composite expression, original parameters, fixed constants
    # ─────────────────────────────────────────────────────────────────────────────
    """
    Construct the symbolic coupling library for the "independent" germination
    model by substituting reference geometry constants into fully symbolic
    coupling expressions. Each entry in the returned Dict describes one
    natural composite quantity — a combination of model parameters that
    appears as a unit in the germination probability formula and whose
    variation is more identifiable than the individual parameters comprising it.

    The library is hierarchically ordered: composites σ_γ and σ_ω (entries 1
    and 2) should be applied before A_γ_ref and A_ω_ref (entries 5 and 6),
    since the latter reduce to lower-dimensional couplings once the width
    composites are substituted.

    Inputs:
        geom (Dict{Symbol,Float64}) - reference geometry constants as returned
                                    by reference_geometry(). Required keys:
            :V_ref     - spore volume at anchor in μm³
            :A_ref     - spore surface area at anchor in μm²
            :V_cw_ref  - vacant cell wall volume at anchor in μm³
            :ρₛ_ref    - spore density in spores/μm³
            :c₀_cs_ref - initial carbon source concentration in M
            :t_ref     - reference time in seconds

    Outputs:
        Dict{Symbol, SymbolicCoupling} with entries:

            :σ_γ     - Inhibition threshold width.
                    Expression: μ_γ · exp(−neg_δ_γ)
                    Original parameters: [:μ_γ, :neg_δ_γ]
                    Composite σ_γ is the standard deviation of the inhibition
                    threshold distribution. Invertible for neg_δ_γ via
                    neg_δ_γ = log(μ_γ / σ_γ), given μ_γ.

            :σ_ω     - Induction threshold width.
                    Expression: μ_ω · exp(−neg_δ_ω)
                    Original parameters: [:μ_ω, :neg_δ_ω]
                    Composite σ_ω is the standard deviation of the induction
                    threshold distribution. Invertible for neg_δ_ω via
                    neg_δ_ω = log(μ_ω / σ_ω), given μ_ω.

            :s_ref   - Induction signal at reference conditions.
                    Expression: c_cs_ref / (K_cC + c_cs_ref), where
                    c_cs_ref = c₀_cs · (1 − exp(−t_ref / τ_cs)) and
                    τ_cs = V_cw / (A · Pₛ_cs).
                    Original parameters: [:K_cC, :Pₛ_cs]
                    s_ref ∈ (0,1) is the Hill-saturation of the carbon source
                    signal at the reference time and geometry. Invertible for
                    K_cC via K_cC = c_cs_ref·(1−s_ref)/s_ref, or for Pₛ_cs
                    via Pₛ_cs = V_cw / (A·τ_cs) where τ_cs is solved from
                    the exponential accumulation equation.

            :β_ref   - Inhibitor depletion at reference conditions.
                    Expression: ϕ + (1−ϕ)·exp(−t_ref / (τ_β·(1−ϕ))), where
                    τ_β = V / (Pₛ·A) and ϕ = ρₛ·V.
                    Original parameters: [:Pₛ]
                    β_ref ∈ (ϕ, 1): β_ref = 1 means no inhibitor has been
                    lost (t=0 or Pₛ=0); β_ref = ϕ means inhibitor has fully
                    equilibrated with the external volume. Invertible for
                    Pₛ via τ_β = −t_ref/((1−ϕ)·log((β_ref−ϕ)/(1−ϕ))),
                    then Pₛ = V/(τ_β·A).

            :A_γ_ref - Normalised inhibition CDF argument at reference conditions.
                    Expression: (β_ref − μ_γ) / (μ_γ · exp(−neg_δ_γ))
                    Original parameters: [:Pₛ, :μ_γ, :neg_δ_γ]
                    A_γ_ref is the argument to Φ in the inhibition factor
                    1 − Φ(A_γ_ref). After applying :β_ref and :σ_γ first,
                    this reduces to (β_ref − μ_γ)/σ_γ, a 2-parameter coupling
                    between β_ref and μ_γ only.

            :A_ω_ref - Normalised induction CDF argument at reference conditions.
                    Expression: (s_ref − μ_ω) / (μ_ω · exp(−neg_δ_ω))
                    Original parameters: [:K_cC, :Pₛ_cs, :μ_ω, :neg_δ_ω]
                    A_ω_ref is the argument to Φ in the induction factor
                    Φ(A_ω_ref). After applying :s_ref and :σ_ω first, this
                    reduces to (s_ref − μ_ω)/σ_ω, a 2-parameter translational
                    coupling between s_ref and μ_ω only.
    """
    function build_coupling_library(geom::Dict)
        # Substitute reference geometry constants into symbolic expressions
        geom_subs = Dict(
            V_ref     => geom[:V_ref],
            A_ref     => geom[:A_ref],
            V_cw_ref  => geom[:V_cw_ref],
            ρₛ_ref    => geom[:ρₛ_ref],
            c₀_cs_ref => geom[:c₀_cs_ref],
            t_ref     => geom[:t_ref]
        )
        sub(expr) = substitute(expr, geom_subs)

        Dict(

            # ── 1. σ_γ = μ_γ · exp(-neg_δ_γ) ──────────────────────────────────
            # Captures (μ_γ, neg_δ_γ) coupling: both determine the width of
            # the inhibition threshold distribution.
            # Composite σ_γ is directly interpretable as the inhibition std.
            # Invert for neg_δ_γ: neg_δ_γ = -log(σ_γ / μ_γ) = log(μ_γ / σ_γ)
            :σ_γ => SymbolicCoupling(
                :σ_γ,
                sub(σ_γ_expr),
                [:μ_γ, :neg_δ_γ],
                Dict()   # no additional constants after geometry substitution
            ),

            # ── 2. σ_ω = μ_ω · exp(-neg_δ_ω) ──────────────────────────────────
            # Captures (μ_ω, neg_δ_ω) coupling: both determine the width of
            # the induction threshold distribution.
            # Invert for neg_δ_ω: neg_δ_ω = log(μ_ω / σ_ω)
            :σ_ω => SymbolicCoupling(
                :σ_ω,
                sub(σ_ω_expr),
                [:μ_ω, :neg_δ_ω],
                Dict()
            ),

            # ── 3. s_ref = c_cs_ref / (K_cC + c_cs_ref) ───────────────────────
            # Captures (K_cC, Pₛ_cs) coupling: K_cC sets the saturation threshold,
            # Pₛ_cs sets how fast c_cs_ref builds up. Both appear inside s_ref.
            # At fixed reference geometry and time, s_ref is a scalar in (0,1).
            # Invert for K_cC: K_cC = c_cs_ref * (1 - s_ref) / s_ref
            # Invert for Pₛ_cs: solve τ_cs = V_cw/(A*Pₛ_cs) from
            #   c_cs_ref = c₀_cs*(1 - exp(-t_ref/τ_cs))
            #   → τ_cs = -t_ref / log(1 - c_cs_ref/c₀_cs)
            #   → Pₛ_cs = V_cw / (A * τ_cs)
            :s_ref => SymbolicCoupling(
                :s_ref,
                sub(s_ref_expr),
                [:K_cC, :Pₛ_cs],
                Dict()
            ),

            # ── 4. β_ref: inhibitor depletion at reference conditions ───────────
            # Captures Pₛ in isolation (geometry fixed at anchor):
            # β_ref = ϕ + (1-ϕ)*exp(-t_ref / (τ_β*(1-ϕ)))
            # where τ_β = V/(Pₛ*A).
            # β_ref ∈ (ϕ, 1): 1 = no depletion, ϕ = full depletion.
            # Invert for Pₛ: τ_β = -t_ref/((1-ϕ)*log((β_ref-ϕ)/(1-ϕ)))
            #                 Pₛ  = V / (τ_β * A)
            :β_ref => SymbolicCoupling(
                :β_ref,
                sub(β_ref_expr),
                [:Pₛ],
                Dict()
            ),

            # ── 5. A_γ_ref: normalised inhibition argument ──────────────────────
            # Captures (Pₛ, μ_γ, neg_δ_γ) jointly:
            # A_γ_ref = (β_ref - μ_γ) / (μ_γ * exp(-neg_δ_γ))
            # This is the actual argument to Φ in the germination probability.
            # Fixing A_γ_ref + one of {μ_γ, neg_δ_γ, Pₛ} determines the others.
            # Most useful when Pₛ is already reparameterised via β_ref (coupling 4),
            # reducing this to (μ_γ, neg_δ_γ) only.
            :A_γ_ref => SymbolicCoupling(
                :A_γ_ref,
                sub(A_γ_ref_expr),
                [:Pₛ, :μ_γ, :neg_δ_γ],
                Dict()
            ),

            # ── 6. A_ω_ref: normalised induction argument ───────────────────────
            # Captures (K_cC, Pₛ_cs, μ_ω, neg_δ_ω) jointly:
            # A_ω_ref = (s_ref - μ_ω) / (μ_ω * exp(-neg_δ_ω))
            # After reparameterising via s_ref (coupling 3) and σ_ω (coupling 2),
            # this reduces to (s_ref - μ_ω) / σ_ω and captures only
            # the residual (s_ref, μ_ω) translational coupling.
            :A_ω_ref => SymbolicCoupling(
                :A_ω_ref,
                sub(A_ω_ref_expr),
                [:K_cC, :Pₛ_cs, :μ_ω, :neg_δ_ω],
                Dict()
            ),
        )
    end

    """
    closed_form_inverse(composite_name, eliminate, all_symbolic, expr)

    Return the symbolic closed-form inverse for known coupling expressions
    where symbolic_solve cannot handle the nonlinearity. All five composites
    in the independent germination model have analytical inverses derived
    from the model equations directly.

    Inputs:
        composite_name (Symbol)           - name of the composite (:σ_γ, :σ_ω,
                                            :s_ref, :β_ref)
        eliminate      (Symbol)           - original parameter to solve for
        all_symbolic   (Dict{Symbol,Num}) - symbolic variable map
        expr           (Num)              - the composite expression (with
                                            geometry already substituted), used
                                            to extract any embedded constants

    Outputs:
        Num or nothing - symbolic expression for the eliminated parameter,
                        or nothing if no closed form is known for this
                        composite/eliminate combination
    """
    function closed_form_inverse(composite_name::Symbol, eliminate::Symbol,
                                all_symbolic::Dict{Symbol,Num},
                                geom::Dict)

        @variables composite_var

        μ_γ = all_symbolic[:μ_γ]
        μ_ω = all_symbolic[:μ_ω]
        K_cC_sym = all_symbolic[:K_cC]

        # Geometry constants read directly from geom
        c₀_cs_val = geom[:c₀_cs_ref]
        t_val     = geom[:t_ref]
        V_cw_val  = geom[:V_cw_ref]
        A_val     = geom[:A_ref]
        V_val     = geom[:V_ref]
        ϕ_val     = geom[:ϕ_ref]

        if composite_name == :σ_γ && eliminate == :neg_δ_γ
            # σ_γ = μ_γ · exp(-neg_δ_γ)  →  neg_δ_γ = log(μ_γ / σ_γ)
            return log(μ_γ / composite_var)

        elseif composite_name == :σ_ω && eliminate == :neg_δ_ω
            # σ_ω = μ_ω · exp(-neg_δ_ω)  →  neg_δ_ω = log(μ_ω / σ_ω)
            return log(μ_ω / composite_var)

        elseif composite_name == :s_ref && eliminate == :K_cC
            # s = c_cs / (K_cC + c_cs)  →  K_cC = c_cs · (1 - s) / s
            c_cs_val = c₀_cs_val * (1 - exp(-t_val / (V_cw_val / (A_val * 1.0))))
            # c_cs depends on Pₛ_cs which is free — use the symbolic form
            # s · (K_cC + c_cs) = c_cs  →  K_cC = c_cs · (1/s - 1)
            # Here c_cs is still symbolic in Pₛ_cs; express K_cC in terms of
            # composite_var and the remaining free parameter Pₛ_cs
            Pₛ_cs_sym = all_symbolic[:Pₛ_cs]
            τ_cs_sym  = V_cw_val / (A_val * Pₛ_cs_sym)
            c_cs_sym  = c₀_cs_val * (1 - exp(-t_val / τ_cs_sym))
            return c_cs_sym * (1 - composite_var) / composite_var

        elseif composite_name == :s_ref && eliminate == :Pₛ_cs
            # From s and K_cC: c_cs = s · K_cC / (1 - s)
            # From c_cs: τ_cs = -t / log(1 - c_cs/c₀_cs)
            #            Pₛ_cs = V_cw / (A · τ_cs)
            c_cs_sym = composite_var * K_cC_sym / (1 - composite_var)
            τ_cs_sym = -t_val / log(1 - c_cs_sym / c₀_cs_val)
            return V_cw_val / (A_val * τ_cs_sym)

        elseif composite_name == :β_ref && eliminate == :Pₛ
            # β = ϕ + (1-ϕ)·exp(-t / ((1-ϕ)·τ_β))
            # τ_β = -t / ((1-ϕ) · log((β-ϕ)/(1-ϕ)))
            # Pₛ  = V / (τ_β · A)
            τ_β_sym = -t_val / ((1 - ϕ_val) * log((composite_var - ϕ_val) / (1 - ϕ_val)))
            return V_val / (τ_β_sym * A_val)

        else
            return nothing
        end
    end

    """
    Symbolically derive the forward and inverse transforms for a single
    reparameterisation, given a SymbolicCoupling and the original parameter
    to eliminate. Uses Symbolics.jl to solve the composite expression for
    the eliminated parameter and compiles both directions to native Julia
    functions via build_function.

    Inputs:
        coupling      (SymbolicCoupling) - one entry from the coupling library,
                                        describing the composite expression,
                                        the original parameters it involves,
                                        and any fixed constants to substitute
                                        before inversion.
        eliminate     (Symbol)           - the original parameter to solve for
                                        in terms of the composite. Must be
                                        a member of coupling.original. This
                                        should be the parameter with the
                                        lower first-order Sobol index, i.e.
                                        the less individually identifiable
                                        of the coupled pair.
        all_symbolic  (Dict{Symbol,Num}) - mapping from parameter name symbols
                                        to their Symbolics.jl symbolic
                                        variables. Must contain entries for
                                        all names appearing in coupling.original
                                        and coupling.fixed_vals.

    Outputs:
        forward_fn    (Function) - compiled function (composite_val, remaining...)
                                → Float64. Given the composite value and the
                                values of all original parameters except the
                                eliminated one, returns the value of the
                                eliminated original parameter. This is the
                                transform applied inside to_model_params before
                                calling the mechanistic model.
        inverse_fn    (Function) - compiled function (original_vals...) → Float64.
                                Given the values of all original parameters in
                                coupling.original (in order), returns the
                                composite value. Used to initialise the sampler
                                anchor and prior bounds from example_params via
                                invert_reparams.
        original_expr (Num)      - the symbolic expression for the eliminated
                                parameter in terms of the composite and the
                                remaining originals. Printed for verification
                                and stored for documentation.
    """
    function build_reparam(coupling::SymbolicCoupling, eliminate::Symbol,
                            all_symbolic::Dict{Symbol, Num}, geom::Dict)

        expr = coupling.expr
        # Substitute fixed constants
        for (k, v) in coupling.fixed_vals
            expr = substitute(expr, Dict(all_symbolic[k] => v))
        end

        # The composite variable
        @variables composite_var

        # Solve: composite_var = expr, for the eliminated parameter
        target_sym = all_symbolic[eliminate]
        # solutions = Symbolics.solve_for(composite_var ~ expr, target_sym)

        # ── Tier 1: try symbolic_solve (handles nonlinear) ───────────────────────
        original_expr = nothing
        try
            solutions = Symbolics.symbolic_solve(composite_var ~ expr, target_sym)
            if !isempty(solutions)
                original_expr = first(solutions)
            end
        catch e
            @warn "symbolic_solve failed for $(coupling.composite_name) → $eliminate: $e"
        end

        # ── Tier 2: fall back to known closed-form inversions ────────────────────
        if isnothing(original_expr)
            original_expr = closed_form_inverse(
                coupling.composite_name, eliminate, all_symbolic, geom)
        end

        if isnothing(original_expr)
            error("Cannot invert $(coupling.composite_name) for $eliminate. " *
                "Add a manual Reparameterisation instead.")
        end

        println("  Inverse expression: $eliminate = $original_expr")

        remaining_syms = [all_symbolic[s] for s in coupling.original if s != eliminate]

        # original_expr = first(solutions)

        # Compile both directions to fast Julia functions
        # Forward: (composite_var, remaining originals) → eliminated original
        forward_fn = build_function(original_expr,
                                    [composite_var; remaining_syms],
                                    expression=Val{false})

        # Inverse: original params → composite
        inverse_fn = build_function(expr,
                                    [all_symbolic[s] for s in coupling.original],
                                    expression=Val{false})

        return forward_fn, inverse_fn, original_expr
    end

    """
    Automatically construct a vector of Reparameterisation objects from
    HDMR-identified coupled parameter pairs and the symbolic coupling library.
    For each pair, finds the matching library entry, derives the forward and
    inverse transforms via build_reparam, and wraps them in the Reparameterisation
    struct expected by to_model_params and invert_reparams.

    Parameters for which no library entry exists produce a warning and are
    skipped — these must be handled by manually constructed Reparameterisation
    objects appended to the returned vector.

    Inputs:
        coupled_pairs     (Vector{Tuple{Symbol,Symbol}}) - list of parameter
                        pairs identified as strongly coupled by HDMR, e.g.
                        from extract_coupled_pairs(). The first element of
                        each tuple is taken as the parameter to eliminate
                        (replace with the composite); it should be the
                        parameter with the lower first-order Sobol index.
        coupling_library  (Dict{Symbol,SymbolicCoupling}) - the library returned
                        by build_coupling_library(). Each entry covers one
                        composite quantity and the original parameters it
                        involves.
        all_symbolic      (Dict{Symbol,Num}) - mapping from parameter name symbols
                        to Symbolics.jl symbolic variables. Must cover all
                        parameters appearing in any coupling in the library.

    Outputs:
        reparams  (Vector{Reparameterisation}) - one entry per successfully
                processed coupled pair, in the same order as coupled_pairs
                (skipped pairs omitted). These are passed directly to
                ModelConfig and applied in sequence by to_model_params and
                invert_reparams. Order matters: composites that appear as
                inputs to later couplings (e.g. σ_γ before A_γ_ref) must
                appear earlier in the vector.
    """
    function auto_reparameterise(coupled_pairs::Vector{Tuple{Symbol,Symbol}},
                                coupling_library::Dict{Symbol, SymbolicCoupling},
                                all_symbolic::Dict{Symbol, Num}, geom::Dict)

        reparams = Reparameterisation[]

        for (p1, p2) in coupled_pairs
            # Find which library entry covers this pair
            matching = [name => c for (name, c) in coupling_library
                        if p1 ∈ c.original || p2 ∈ c.original]

            if isempty(matching)
                @warn "No symbolic coupling found for ($p1, $p2). Add to library manually."
                continue
            end

            name, coupling = first(matching)
            # Eliminate whichever original parameter is less individually identifiable
            # (lower S_first) — this is the one we replace with the composite
            eliminate = p1   # caller decides based on S_first values

            forward_fn, inverse_fn, inv_expr = build_reparam(
                coupling, eliminate, all_symbolic, geom)

            println("Generated reparam for ($p1, $p2):")
            println("  Composite: $(coupling.composite_name) = $(coupling.expr)")
            println("  Solving for $eliminate: $inv_expr")

            push!(reparams, Reparameterisation(
                [coupling.composite_name, p2],
                [p1, p2],
                (comp, def) -> begin
                    cv = comp[coupling.composite_name]
                    remaining = [comp[s] for s in coupling.original if s != eliminate]
                    orig_val = forward_fn(cv, remaining...)
                    Dict(eliminate => orig_val, p2 => comp[p2])
                end,
                (orig) -> begin
                    vals = [orig[s] for s in coupling.original]
                    Dict(coupling.composite_name => inverse_fn(vals...),
                        p2 => orig[p2])
                end
            ))
        end

        return reparams
    end

    """
    Extract parameter pairs whose pairwise HDMR interaction index exceeds a
    given threshold in at least one output dimension. These are the pairs for
    which reparameterisation is warranted before SMC, as their joint influence
    on the output cannot be decomposed into independent contributions from each
    parameter alone.

    Inputs:
        result     (HDMRResult)  - the result struct returned by run_hdmr(),
                                containing S_pair (n_pairs × n_outputs matrix
                                of pairwise Sobol indices) and the ModelConfig
                                which maps pair indices to parameter name pairs.
        threshold  (Float64)     - minimum S_pair value in any output dimension
                                for a pair to be considered strongly coupled.
                                Default 0.05, consistent with the effective
                                dimensionality threshold in
                                effective_dimensionality(). Pairs below this
                                threshold in all outputs contribute less than
                                5% of total variance through their interaction
                                and can be treated as approximately independent.

    Outputs:
        pairs  (Vector{Tuple{Symbol,Symbol}}) - parameter name pairs whose
            interaction index exceeds the threshold in at least one output.
            Ordered as they appear in config.target_pairs. The first element
            of each tuple corresponds to the lower-indexed parameter in the
            ModelConfig, not necessarily the less identifiable one — the
            caller should reorder based on S_first before passing to
            auto_reparameterise.
    """
    function extract_coupled_pairs(result::HDMRResult; threshold::Float64=0.05)
        pairs = Tuple{Symbol,Symbol}[]
        for (p, (j, k)) in enumerate(result.config.target_pairs)
            if any(result.S_pair[p, :] .> threshold)
                push!(pairs, (result.config.param_names[j],
                            result.config.param_names[k]))
            end
        end
        return pairs
    end

    """
    Inverse: convert example_params (original space) to composite space.
    Used to initialise anchor and prior bounds.
    """
    function invert_reparams(original_params::Dict{Symbol,Float64},
                            reparams::Vector{Reparameterisation})
        result = copy(original_params)
        for r in reparams
            orig = Dict(name => result[name] for name in r.original_names)
            comp = r.inverse(orig)
            for name in r.original_names
                delete!(result, name)
            end
            merge!(result, comp)
        end
        return result
    end
end