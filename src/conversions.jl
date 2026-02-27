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
    export build_reparam
    export auto_reparameterise


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
    Given a composite expression and one original parameter to eliminate,
    symbolically solve for that parameter and return compiled forward/inverse functions.
    """
    function build_reparam(coupling::SymbolicCoupling, eliminate::Symbol,
                            all_symbolic::Dict{Symbol, Num})

        expr = coupling.expr
        # Substitute fixed constants
        for (k, v) in coupling.fixed_vals
            expr = substitute(expr, Dict(all_symbolic[k] => v))
        end

        # The composite variable
        @variables composite_var

        # Solve: composite_var = expr, for the eliminated parameter
        target_sym = all_symbolic[eliminate]
        solutions = Symbolics.solve_for(composite_var ~ expr, target_sym)

        if isempty(solutions)
            error("Cannot invert $(coupling.composite_name) for $eliminate symbolically. Write manually.")
        end

        original_expr = first(solutions)

        # Compile both directions to fast Julia functions
        # Forward: (composite_var, remaining originals) → eliminated original
        forward_fn = build_function(original_expr,
                                    [composite_var;
                                    [all_symbolic[s] for s in coupling.original if s != eliminate]],
                                    expression=Val{false})

        # Inverse: original params → composite
        inverse_fn = build_function(expr,
                                    [all_symbolic[s] for s in coupling.original],
                                    expression=Val{false})

        return forward_fn, inverse_fn, original_expr
    end

    """
    Given HDMR results identifying which pairs are strongly coupled,
    automatically construct Reparameterisation objects.
    """
    function auto_reparameterise(coupled_pairs::Vector{Tuple{Symbol,Symbol}},
                                coupling_library::Dict{Symbol, SymbolicCoupling},
                                all_symbolic::Dict{Symbol, Num})

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
                coupling, eliminate, all_symbolic)

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
end