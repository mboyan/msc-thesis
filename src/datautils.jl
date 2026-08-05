module DataUtils
__precompile__(false)
    """
    Contains utility functions.
    """

    using DataFrames
    using CSV
    using FastGaussQuadrature
    using BlackBoxOptim
    using Optim
    using NLopt
    using LsqFit
    using MeshGrid
    using Distributions
    using ArgCheck
    using QuasiMonteCarlo
    using LinearAlgebra
    using Random
    using JLD2
    using CUDA
    using Flux
    using Flux: gradient
    using Statistics
    using StatsBase
    using ProgressMeter
    using Base.Threads
    # using LogExpFunctions
    
    include("./conversions.jl")
    include("./germstats.jl")
    include("./germstats_gpu.jl")
    using .Conversions
    using .GermStats
    using .GermStatsGPU

    export calibrate_marginals
    export calibrate_copula
    export sample_parameters
    export parse_ijadpanahsaravi_data
    export unpack_ijadpanahsaravi_data
    export calibrate_priors
    export dantigny
    export dantigny_time_shifted
    export infer_dantigny_parameters
    export generate_dantigny_dataset
    export train_multioutput_nn_mixed_precision
    export predict_with_uncertainty_mixed_precision
    export fit_dantigny_to_germination_curve
    export fit_dantigny_time_shifted_to_germination_curve
    export fit_model_to_data
    export get_params_for_idx
    export fit_model_to_data_equilibrium
    export sensitivity_analysis
    export sequential_monte_carlo

    CUDA.allowscalar(false)  # Prevent slow scalar operations

    # ======================
    # ===== Statistics =====
    # ======================
    """
    Alternative calculation of mean
    through logarithms (avoids overflow)
    """
    function mean_log(values; dims=nothing)
        log_mean = mean(log.(values); dims=dims)
        return exp.(log_mean)
    end

    """
    Compute a weighted median from a set of values and weights.
    """
    function weighted_median(values::Vector{T}, weights::Vector{T}) where T

        # Check if the length of values and weights match
        if length(values) != length(weights)
            throw(ArgumentError("Values and weights must have the same length."))
        end
        
        # Combine values and weights into a sorted array based on values
        sorted_indices = sortperm(values)
        sorted_values = values[sorted_indices]
        sorted_weights = weights[sorted_indices]
        
        # Calculate cumulative weights
        cum_weights = cumsum(sorted_weights)
        total_weight = sum(sorted_weights)
        # println(total_weight)
        
        # Find the median index
        median_idx = findfirst(cum -> cum >= total_weight / 2, cum_weights)
        
        return sorted_values[median_idx]
    end

    """
    Compute a quantile of a set of weighted values.
    inputs:
        x (Vector) - values
        w (Vector) - weights
        p (Real) - quantile
    output:
        the linearly interpolated value at the given quantile
    """
    function weighted_quantile(x::AbstractVector, w::AbstractVector, p::Real)

        @assert length(x) == length(w)
        @assert 0 ≤ p ≤ 1

        # Sort by x
        idx = sortperm(x)
        xs = x[idx]
        ws = w[idx] ./ sum(w)

        cdf = cumsum(ws)

        i = searchsortedfirst(cdf, p)

        if i == 1
            return xs[1]
        elseif i > length(xs)
            return xs[end]
        else
            # Linear interpolation
            x0, x1 = xs[i-1], xs[i]
            F0, F1 = cdf[i-1], cdf[i]
            return x0 + (p - F0) / (F1 - F0) * (x1 - x0)
        end
    end

    """
    Refit a prior distribution through a set of weighted samples
    inputs:
        dist_current (Distribution) - current distribution
        θ (Vector) - samples
        w_norm (Vector) - normalized weights
        bounds (Tuple) - bounds for the 95% interval of the distribution
        α (Float) - learning rate
    output:
        the fitted prior distribution
    """
    function fit_dist(dist_current, θ, w_norm, bounds; α=0.5)

        μ_current = dist_current.μ
        σ_current = dist_current.σ
        dist_type = typeof(dist_current)

        ess = 1 / sum(w_norm .^ 2) # effective sample size
        ess_ratio = ess / length(w_norm)
        exploration_factor = 1.0 + ess_ratio * 0.5
        println("ESS: $ess, exploration factor: $exploration_factor")

        if dist_type == LogNormal{Float64}

            x = log.(θ)
            m = weighted_median(vec(x), w_norm)
            iqr = weighted_quantile(x, w_norm, 0.75) - weighted_quantile(x, w_norm, 0.25)
            σ = iqr / 1.349

            μ_interp = μ_current + α * (m - μ_current)
            σ_interp = σ_current + α * (σ * exploration_factor - σ_current)

            # Bound distribution
            z_hi = μ_interp + 1.96σ_interp
            z_lo = μ_interp - 1.96σ_interp

            z_hi = min(z_hi, log(bounds[2]))
            z_lo = max(z_lo, log(bounds[1]))

            σ_interp = (z_hi - z_lo) / (2 * 1.96)
            μ_interp = (z_hi + z_lo) / 2

        elseif dist_type == Normal{Float64}
            
            μ = sum(w_norm .* θ)
            σ = sqrt(sum(w_norm .* (θ .- μ).^2))

            μ_interp = μ_current + α * (μ - μ_current)
            σ_interp = σ_current + α * (σ * exploration_factor - σ_current)

            # Bound distribution
            x_hi = μ_interp + 1.96σ_interp
            x_lo = μ_interp - 1.96σ_interp

            x_hi = min(x_hi, bounds[2])
            x_lo = max(x_lo, bounds[1])

            σ_interp = max((x_hi - x_lo) / (2 * 1.96), 1e-12)
            μ_interp = (x_hi + x_lo) / 2

        else
            error("Invalid distribution type")
        end

        return dist_type(μ_interp, σ_interp)
    end

    """
    Perform near-PD projection (e.g. for when
    a matrix is not Cholesky factorizable)
    inputs:
        A (Matrix) - matrix to be corrected
    outputs:
        A3 (Matrix) - SPD matrix
    """
    function nearestSPD(A)
        
        B = (A + A') / 2
        _, S, V = svd(B)
        H = V * Diagonal(S) * V'
        A2 = (B + H) / 2
        A3 = (A2 + A2') / 2

        if isposdef(A3)
            return A3
        end

        spacing = eps(Float64) * norm(A)
        I = diagm(ones(size(A, 1)))
        k = 1
        while !isposdef(A3)
            A3 += I * spacing * k
            k += 1
        end
        return A3
    end

    """
    Calibrate a set of distributions given a set of weighted samples.
    Optionally, condition-specific weights can also be used.
    inputs:
        marg_current (Vector{Distribution}) - the current distributions
        θ (Matrix) - parameter samples
        weights (Vector) - sample weights
        bounds (Vector{Tuple}) - bounds for the 95% IQR of the current distributions
    output:
        marginals (Vector{Distribution}) - the calibrated distributions
    """
    function calibrate_marginals(marg_current, θ, weights, bounds)
        Np = size(θ, 1)

        weights ./= sum(weights)

        marginals = Vector{Distribution}(undef, Np)
        for i in 1:Np
            marginals[i] = fit_dist(marg_current[i], θ[i, :], weights, bounds[i])
        end
        return marginals
    end

    """
    Transform global and condition-specific parameter
    values from given marginal distributions
    to Gaussian space (for computing a Gaussian copula).
    inputs: 
        Θg (Matrix) - global parameter values
        Θs (Matrix) - condition-specific parameter values
        marg_g (Vector{Distribution}) - global parameter distributions
        marg_s (Vector{Distribution}) - condition-specific parameter distributions
        glob_tag (BitVector) - boolean weaving pattern for global vs specific parameters
        eps_u (Float) - tolerance avoiding the sampling of Inf
    output:
        Z (Matrix) - transformed parameter values
    """
    function to_gaussian_space(Θg, Θs, marg_g, marg_s, glob_tag; eps_u = nothing)
        
        Ng = size(Θg, 1)
        Nc = size(Θs, 1)
        Ns = size(Θg, 2)
        Z = zeros(Ns, Ng + Nc)

        # Avoid U of 0 and 1
        if eps_u === nothing
            eps_u = 1 / (Ns + 1)
        end
    
        g_st = 1
        s_ct = 1
        for i in 1:(Ng + Nc)
            if glob_tag[i]
                u = cdf.(marg_g[g_st], Θg[g_st, :])
                u = clamp.(u, eps_u, 1 - eps_u)
                Z[:, i] = quantile.(Normal(), u)
                g_st += 1
            else
                u = cdf.(marg_s[s_ct, :], Θs[s_ct, :])
                u = clamp.(u, eps_u, 1 - eps_u)
                Z[:, i] = quantile.(Normal(), u)
                s_ct += 1
            end
        end
        return Z
    end

    """
    Compute the correlation of weighted
    Gaussian-transformed parameter values.
    inputs:
        Z (Matrix) - transformed parameter values
        w (Vector) - weights
    output:
        correlation matrix
    """
    function weighted_correlation(Z, w)

        n_samples, n_params = size(Z)
        w_norm = w ./ sum(w)
        
        # Weighted mean
        μ = sum(Z .* w_norm, dims=1)  # Shape: (1, n_params)
        
        # Covariance
        Σ = zeros(n_params, n_params)
        for i in 1:n_samples
            δ = Z[i, :] .- vec(μ)  # Deviation for sample i
            Σ .+= w_norm[i] .* (δ * δ')
        end
        
        # To correlation
        σ = sqrt.(diag(Σ))
        σ = max.(σ, 1e-10)  # Prevent division by zero
        R = Σ ./ (σ * σ')
        
        # Clean up
        for i in 1:n_params
            R[i, i] = 1.0
        end
        R = (R + R') / 2
        R = clamp.(R, -1.0, 1.0)
        for i in 1:n_params
            R[i, i] = 1.0
        end
        
        return R
    end

    """
    Construct a Gaussian copula from weighted 
    parameter values and their marginal distributions.
    inputs:
        Θg (Matrix) - global parameter values
        Θs (Matrix) - condition-specific parameter values
        marg_g (Vector{Distribution}) - global parameter distributions
        marg_s (Vector{Distribution}) - condition-specific parameter distributions
        w_glob (Vector) - global weights
        w_spec (Matrix) - condition-specific weights
        glob_tag (BitVector) - boolean weaving pattern for global vs specific parameters
    output:
        multivariate distribution of weighted samples
    """
    function calibrate_copula(Θg, Θs, marg_g, marg_s, w_glob, w_spec, glob_tag)

        w_eff = w_glob .* w_spec
        w_eff ./= sum(w_eff)

        n_eff = 1 / sum(w_eff.^2)
        # println("  Effective sample size for copula: $(round(n_eff, digits=1))")
    
        # if n_eff < 100
        #     println("  ⚠️ Low n_eff - correlations unreliable!")
        # end

        Z = to_gaussian_space(Θg, Θs, marg_g, marg_s, glob_tag)
        R_raw  = weighted_correlation(Z, w_eff)

        # Enforce symmetry
        R_raw = (R_raw + R_raw') / 2

        # === CRITICAL: CAP CORRELATIONS ===
        n = size(R_raw, 1)
        max_allowed_corr = 0.95  # HARD LIMIT
        
        R_capped = copy(R_raw)
        n_capped = 0
        
        @inbounds for i in 1:n
            @inbounds for j in (i+1):n
                if abs(R_raw[i,j]) > max_allowed_corr
                    R_capped[i,j] = sign(R_raw[i,j]) * max_allowed_corr
                    R_capped[j,i] = R_capped[i,j]
                    n_capped += 1
                    
                    if abs(R_raw[i,j]) > 0.999
                        println("    ⚠️ Extreme correlation capped: R[$i,$j] = $(round(R_raw[i,j], digits=5)) → $max_allowed_corr")
                    end
                end
            end
        end
        
        if n_capped > 0
            println("  Capped $n_capped correlations")
        end
        
        # Shrinkage regularization
        λ = max(0.05, min(0.2, 1.0 - n_eff/200))
        R_reg = (1 - λ) * R_capped + λ * I(n)
        println("  Applied $(round(100*λ, digits=1))% shrinkage")
        
        # Ensure PD
        min_eig = eigmin(Symmetric(R_reg))
        if min_eig < 1e-6
            R_reg = R_reg + (1e-6 - min_eig + 1e-8) * I(n)
        end
        
        # Final check
        if eigmin(Symmetric(R_reg)) <= 0
            R_reg = nearestSPD(R_reg)
        end
        
        # Verify condition number
        eigenvalues = eigvals(Symmetric(R_reg))
        κ = maximum(eigenvalues) / minimum(eigenvalues)
        println("  Condition number: $(round(κ, digits=1))")
        
        if κ > 1000
            println("  ⚠️ WARNING: Parameters may be non-identifiable")
        end
        
        return MvNormal(zeros(size(R_reg,1)), R_reg)
    end

    """
    Sample a multivariate distribution (Gaussian copula).
    inputs:
        p (Matrix) - normalized locations for sampling the copula
        copula (MvNormal) - Gaussian copula
        marg_g (Vector{Distribution}) - global parameter marginals
        marg_s (Vector{Distribution}) - carbon-specific parameter marginals
        glob_tag (Vector{Bool}) - tags whether a parameter is global or not
    outputs:
        Θ (Matrix) - new parameter values
    """
    function sample_parameters(p, copula, marg_g, marg_s, glob_tag)

        d, N = size(p)

        # Sobol → iid Gaussians
        Z0 = quantile.(Normal(), p)

        # Partition Gaussians and copula
        Zg = Z0[glob_tag, :]
        Zs = Z0[.!glob_tag, :]
        Σ = copula.Σ
        # display(Σ)
        Σgg = Σ[glob_tag, glob_tag]
        Σss = Σ[.!glob_tag, .!glob_tag]
        Σsg = Σ[.!glob_tag, glob_tag]

        # Conditional mean and covariance (for spec. params)
        μs = Σsg * (Σgg \ Zg) 
        Σs = Σss - Σsg * (Σgg \ Σsg')

        # Correlate (copula)
        L = cholesky(Symmetric(Σs + 1e-10I)).L
        Zs = μs .+ L * Zs

        # Back to uniforms
        Ug = cdf.(Normal(), Zg)
        Us = cdf.(Normal(), Zs)

        # Apply marginals
        θg = similar(Zg)
        for i in eachindex(marg_g)
            θg[i,:] = quantile.(marg_g[i], Ug[i,:])
        end
        θs = similar(Zs)
        for i in eachindex(marg_s)
            θs[i,:] = quantile.(marg_s[i], Us[i,:])
        end

        return θg, θs
    end
    
    # ===========================
    # ===== Data processing =====
    # ===========================
    """
    Parses the dataset from Ijadpanahsaravi et al. (2023)
    with multiple inoculum densities and returns a DataFrame.
    """
    function parse_ijadpanahsaravi_data()

        df_germination = DataFrame(CSV.File("../src/Data/swelling_germination_results.csv"; header=true))

        # Filter the data to only include swelling
        df_germination_swelling = filter(row -> row[1] == "Swelling", df_germination)

        # Expression for parsing
        regex_triplet = r"(-?\d+\.\d+)\[(-?\d+\.\d+);(-?\d+\.\d+)\]*.*"

        # Function to parse a string and return three numbers
        function parse_numbers(s)
            m = match(regex_triplet, s)
            if m !== nothing
                num1 = parse(Float64, m.captures[1])
                num2 = parse(Float64, m.captures[2])
                num3 = parse(Float64, m.captures[3])
                return num1, num2, num3
            else
                return missing, missing, missing
            end
        end

        # Parse Pmax and its confidence intervals
        Pmax_parsed = [parse_numbers(row[1]) for row in eachrow(df_germination_swelling[!, 4])]
        Pmax_vals = [x[1] for x in Pmax_parsed]
        PmaxCIlow_vals = [x[2] for x in Pmax_parsed]
        PmaxCIhigh_vals = [x[3] for x in Pmax_parsed]

        # Parse tau and its confidence intervals
        tau_parsed = [parse_numbers(row[1]) for row in eachrow(df_germination_swelling[!, 5])]
        tau_vals = [x[1] for x in tau_parsed]
        tauCIlow_vals = [x[2] for x in tau_parsed]
        tauCIhigh_vals = [x[3] for x in tau_parsed]

        # Parse d and its confidence intervals
        d_parsed = [parse_numbers(row[1]) for row in eachrow(df_germination_swelling[!, 6])]
        d_vals = [x[1] for x in d_parsed]
        dCIlow_vals = [x[2] for x in d_parsed]
        dCIhigh_vals = [x[3] for x in d_parsed]

        # Reconstruct the DataFrame with the parsed values
        df_germination_rebuilt = DataFrame(
            :CarbonSource => df_germination_swelling[!, 2],
            :Density => inverse_um_to_mL(df_germination_swelling[!, 3] / 150),
            :Pmax => Pmax_vals,
            :Pmax_CI_Lower => PmaxCIlow_vals,
            :Pmax_CI_Upper => PmaxCIhigh_vals,
            :tau => tau_vals,
            :tau_CI_Lower => tauCIlow_vals,
            :tau_CI_Upper => tauCIhigh_vals,
            :d => d_vals,
            :d_CI_Lower => dCIlow_vals,
            :d_CI_Upper => dCIhigh_vals,
            :RMSE => df_germination_swelling[!, 7],
            :N => df_germination_swelling[!, 8],
            :M => df_germination_swelling[!, 9]
        )

        df_germination_rebuilt
    end

    """
    Extracts relevant metrics from the dataset
    from from Ijadpanahsaravi et al. (2023)
    """
    function unpack_ijadpanahsaravi_data(t_max=48)

        # Load data
        df_germination_rebuilt = parse_ijadpanahsaravi_data()
        df_germination_rebuilt = filter(row -> row[1] != "Arg", df_germination_rebuilt) # Remove "Arg" from the dataset
        _, times, sources, densities, _, _, _, _ = generate_dantigny_dataset(df_germination_rebuilt, t_max)

        n_src = length(sources)
        n_dens = length(densities)

        # Precompute lab data
        data_lookup = Dict((row[1], row[2]) => row for row in eachrow(df_germination_rebuilt))
        lab_means = zeros(Float64, n_dens, n_src, 3)
        CIs = zeros(Float64, n_dens, n_src, 3, 2)
        CI_widths = zeros(Float64, n_dens, n_src, 3)
        uncert_lab = zeros(Float64, n_dens, n_src, 3) # Lab uncertainty (based on RMSE)

        for i in eachindex(densities)
            for j in eachindex(sources)
                data_row = data_lookup[(sources[j], densities[i])]
                lab_means[i, j, :] = [data_row["Pmax"] * 0.01, data_row["tau"], data_row["d"]]
                CIs[i, j, :, :] = [
                    data_row["Pmax_CI_Lower"] * 0.01 data_row["Pmax_CI_Upper"] * 0.01; # normalise from %
                    data_row["tau_CI_Lower"] data_row["tau_CI_Upper"];
                    data_row["d_CI_Lower"] data_row["d_CI_Upper"]
                    ]
                CI_widths[i, j, :] = CIs[i, j, :, 2] .- CIs[i, j, :, 1]

                uncert_base = CI_widths[i, j, :] ./ 3.92

                uncert_lab[i, j, :] = sqrt.(uncert_base.^2 .+ (data_row["RMSE"] * 0.01)^2) # normalise from % and square
            end
        end

        return times, sources, densities, lab_means, CIs, CI_widths, uncert_lab
    end

    # ==========================
    # ===== Dantigny model =====
    # ==========================
    """
    Dantigny model for the germination of a fungal culture.
    inputs:
        t: time (in hours)
        p_max: maximum germination rate
        τ: time constant (in hours)
        ν: design parameter (dimensionless)
    outputs:
        p: germination rate (dimensionless)
    """
    function dantigny(t, p_max, τ, ν)
        p = p_max * (1 - 1 / (1 + (t / τ)^ν))
        return p
    end

    """
    Dantigny model for the germination of a fungal culture
    incorporating a time shift.
    inputs:
        t: time (in hours)
        p_max: maximum germination rate
        τ: time constant (in hours)
        ν: design parameter (dimensionless)
        δ: time shift (in hours)
    outputs:
        p: germination rate (dimensionless)
    """
    function dantigny_time_shifted(t, p_max, τ, ν, δ)
        p = p_max * (1 - 1 / (1 + ((t - δ) / τ)^ν))
        return p
    end

    """
    Infer the effective Dantigny parameters
    p_max, τ_g and ν from a germination time series
    inputs:
        germ_response (Vector) - germination fractions over time
        times (Vector) - time points for germination fractions (in hours)
        smooth_rad (Int) - radius of smoothing around τ_g
    output:
        p_max (Float36) - saturation germination fraction
        τ_g (Float36) - germination half-saturation time
        ν (Float36) - onset steepness parameter
    """
    function infer_dantigny_parameters(germ_response, times)

        @assert length(germ_response) == length(times)

        dt = times[2] - times[1]

        # Saturation fraction as end value if saturation is reached
        close_to_unity = 1.0 - germ_response[end] < 0.05
        if germ_response[end] - germ_response[end - 1] < dt / 36 || close_to_unity
            
            p_max = close_to_unity ? 1.0 : germ_response[end]

            min_val, half_sat_idx = findmin(abs.(germ_response .- 0.5 * p_max))
            τ_g = times[half_sat_idx]

            # Smooth middle of curve
            diffs = diff(germ_response)
            smooth_rad = round(Int, sum(diffs .> mean(diffs)) * 0.1)
            println("Smoothing radius: $smooth_rad")
            gresp_smooth = copy(germ_response)
            for n in 1:100
                mix_idx = max(2, half_sat_idx - smooth_rad)
                max_idx = min(length(times) - 1, half_sat_idx + smooth_rad)
                for i in mix_idx:max_idx
                    gresp_smooth[i] = 0.5 * (gresp_smooth[i - 1] + gresp_smooth[i + 1])
                end
            end

            # Approximate derivative at half-saturation time
            deriv = (gresp_smooth[half_sat_idx + 1] - gresp_smooth[half_sat_idx]) / dt

            ν = 4 * τ_g * deriv / p_max
        else
            pmax0 = min(0.9, germ_response[end]*2)
            τ0 = times[end]
            ν0 = 2.0
            d, rmse, uncert = fit_dantigny_to_germination_curve(germ_response, times; p0=[pmax0, τ0, ν0])
            p_max, τ_g, ν = d
            println("Dantigny summaries fitted with RMSE $rmse and uncertainties $uncert")
        end

        return p_max, τ_g, ν
    end

    """
    Generate time-dependent germination data using the Dantigny model
    and parameters from a dictionary.
    inputs:
        df_germination (DataFrame): Dantigny model parameters
        t_max (Float): maximum time point in hours
        n_pts (Int): number of time points to generate
    outputs:
        dantigny_data (Matrix): matrix of time-dependent germination data
        times (Vector): vector of time points (in hours)
        sources (Vector): unique string identifiers of carbon sources
        densities (Vector): unique spore densities used in the data
        errs (Matrix): negative and positive offsets of p_max CIs from the mean
        p_maxs (Matrix): maximum germination percentages
        taus (Matrix): characteristic germination times (in hours)
        nus (Matrix): design parameters
    """
    function generate_dantigny_dataset(df_germination, t_max, n_pts=1000)
        
        sources = unique(df_germination[!, :CarbonSource])
        densities = unique(df_germination[!, :Density])
        
        times = collect(LinRange(0, t_max, n_pts))
        dantigny_data = zeros(length(sources), length(densities), n_pts)
        errs = zeros(length(sources), length(densities), 2)
        p_maxs = zeros(length(sources), length(densities))
        taus = zeros(length(sources), length(densities))
        nus = zeros(length(sources), length(densities))
        for (i, source) in enumerate(sources)
            for (j, density) in enumerate(densities)
                # Get the parameters for the current source and density
                params = df_germination[(df_germination[!, :CarbonSource] .== source) .& (df_germination[!, :Density] .== density), :]
                
                if nrow(params) == 0
                    continue
                end
                
                p_max = params[1, :Pmax]
                τ = params[1, :tau]
                ν = params[1, :d]
                
                # Generate the time course using the Dantigny model
                dantigny_data[i, j, :] = dantigny.(times, p_max, τ, ν)

                # Confidence_intervals
                errs[i, j, 1] = max(p_max - params[1, :Pmax_CI_Lower], 0)
                errs[i, j, 2] = params[1, :Pmax_CI_Upper] - p_max

                # Dantigny parameters
                p_maxs[i, j] = p_max
                taus[i, j] = τ
                nus[i, j] = ν
            end
        end

        return dantigny_data * 0.01, times, sources, densities, errs, p_maxs, taus, nus
    end

    # =============================
    # ===== Prior calibration =====
    # =============================
    """
    Exponential cooling schedule.
    T_init: Initial temperature (high = flat weights)
    T_final: Final temperature (low = sharp weights)
    """
    function get_temp_param(iteration, n_iter; T_init=100.0, T_final=4.0)
        
        # Exponential decay
        α = (T_final / T_init)^(1 / (n_iter - 1))
        T = T_init * α^(iteration - 1)
        
        return 1.0 / T  # Return inverse temperature (your temp_param)
    end

    """
        Iteratively calibrates parameter priors
        based on lab-derived Dantigny summaries.
        inputs:
            n_iter (Int) - maximum number of calibration iterations
            n_samples (Int) - size of Sobol sample
            def_params (Dict) - defined parameters (not optimised)
            bounds_abs (Dict) - initial-guess distributions for the case of absolute γ-thresholds
            bounds_rel (Dict) - initial-guess distributions for the case of relative γ-thresholds
            use_surrogates (Bool) - whether to use a surrogate NN for mapping successive parameter values to Dantigny summaries
        """
    function calibrate_priors(n_iter, n_samples, def_params, bounds_abs, bounds_rel; use_surrogates=true)
        
        Random.seed!(1236)

        # Load data
        aliases, combination_IDs, descriptions, param_key_sets = load_model_collection()
        times, sources, densities, lab_means, CIs, CI_widths, uncert_lab = unpack_ijadpanahsaravi_data()
        times_sec = times * 3600 # for matching physics variables in mechanistic model

        n_src = length(sources)
        n_dens = length(densities)

        # Determine relative vs absolute threshold models
        ids_rel = ["0", "B", "Bi"]

        # Determine global vs inducer-specific parameters
        params_glob_keys = [:b, :Pₛ, :μ_γ, :neg_δ_γ, :μ_ψ, :neg_δ_ψ]

        # Precompute distributions
        param_dists_abs = Dict()
        for (key, val) in bounds_abs
            if startswith(string(key), "neg_δ_")
                mean = 0.5 * (val[1] + val[2])
                sd = (val[2] - val[1]) / (2 * 1.96)
                param_dists_abs[key] = Normal(mean, sd)
            else
                mean = (log(val[1]) + log(val[2])) * 0.5
                sd = (log(val[2]) - log(val[1])) / (2 * 1.96)
                param_dists_abs[key] = LogNormal(mean, sd)
            end
        end

        param_dists_rel = Dict()
        for (key, val) in bounds_rel
            if startswith(string(key), "neg_δ_")
                mean = 0.5 * (val[1] + val[2])
                sd = (val[2] - val[1]) / (2 * 1.96)
                param_dists_rel[key] = Normal(mean, sd)
            else
                mean = (log(val[1]) + log(val[2])) * 0.5
                sd = (log(val[2]) - log(val[1])) / (2 * 1.96)
                param_dists_rel[key] = LogNormal(mean, sd)
            end
        end

        # Dictionaries for saving priors and surrogate models
        priors_all = Dict()
        surrogates_all = Dict()

        # χ^2 threshold
        χ_sq = 7.815

        # Acceptable ranges for Dantigny data
        output_ranges = [0 1; 1e-6 1e5; 0 20]

        # Record data points (input parameters + Dantigny summaries)
        data_pts_dict = Dict()

        # Diagnostics
        π_d = zeros(Float64, n_dens, n_src, 3)
        π_diffs = zeros(Float64, n_dens, n_src, 3)
        p_d = zeros(Float64, n_dens, n_src)
        q_d = zeros(Float64, n_dens, n_src, 3)
        m_d = zeros(Float64, n_dens, n_src)
        iqr = zeros(Float64, n_dens, n_src, 3)
        iqr_check = zeros(Bool, n_dens, n_src, 3)
        iqr_diff = zeros(Float64, n_dens, n_src, 3)
        p_z = zeros(Float64, n_dens, n_src)
        p_z_diffs = zeros(Float64, n_dens, n_src)

        π_d_crit = zeros(Bool, n_dens, n_src, 3)
        q_d_crit = zeros(Bool, n_dens, n_src, 3)
        m_d_crit = zeros(Bool, n_dens, n_src, 3)

        # Placeholders
        input_params_all = Vector{Dict}(undef, n_src)
        p_out = Vector{Float64}(undef, length(times))
        diffs_dantigny = zeros(3, n_samples)
        p_maxs = zeros(Float64, n_samples)
        taus = zeros(Float64, n_samples)
        nus = zeros(Float64, n_samples)
        d = zeros(Float64, 3, n_samples)
        z_dist = zeros(Float64, n_samples)
        rmses = zeros(Float64, n_dens, n_src, n_samples)

        # Data collection
        priors_running_all = Dict()
        priors_running = Vector{Dict}(undef, n_iter)
        θ_final_all = Dict()
        w_glob_final_all = Dict()
        w_spec_final_all = Dict()
        w_spec_record_all = Dict()

        debug_vals = Dict()

        # Iterate over models
        for m in 1:1#59:59#eachindex(aliases)

            println("Running $(aliases[m])")

            # Parameter keys
            param_keys = param_key_sets[m]
            key_src_strings = [Symbol(string(key) * " " * src) for key in param_keys, src in sources]

            # Dimensions and bitvectors
            n_dims = length(param_keys)
            glob_tag = [key in params_glob_keys for key in param_keys]
            n_glob = sum(glob_tag)
            n_spec = n_dims - n_glob

            debug_vals[:d] = zeros(Float64, n_iter, n_dens, n_src, n_samples, 3)
            debug_vals[:lab_means] = zeros(Float64, n_iter, n_dens, n_src, n_samples, 3)
            debug_vals[:diffs] = zeros(Float64, n_iter, n_dens, n_src, n_samples, 3)
            debug_vals[:z_dist] = zeros(Float64, n_iter, n_dens, n_src, n_samples)
            debug_vals[:w_spec] = zeros(Float64, n_iter, n_src, n_samples)
            debug_vals[:params] = zeros(Float64, n_iter, n_src, n_dims, n_samples)
            debug_vals[:penalties] = zeros(Float64, n_iter, n_dens, n_src, n_samples)

            # Data collection
            priors = Dict()
            surrogates = Dict()

            # Check if model uses absolute or relative bounds
            if combination_IDs[m] in ids_rel
                sample_dists = filter(p -> p[1] in param_keys, param_dists_rel)
                abs_thresh = false
            else
                sample_dists = filter(p -> p[1] in param_keys, param_dists_abs)
                abs_thresh = true
            end
            
            # Initiate criteria stability counters
            π_d_crit_ct = 0
            q_d_crit_ct = 0
            m_d_crit_ct = 0

            # Generate (normalized) parameter samples
            sobol_pts = QuasiMonteCarlo.sample(n_samples, n_dims, SobolSample())

            # Shrink samples to 95%
            sobol_pts = 0.025 .+ 0.95 .* sobol_pts

            # Sobol point indices
            idx_shuffle = collect(1:n_dims)

            # Parameter and weights placeholders
            θ = similar(sobol_pts)
            θ_glob = zeros(Float64, n_glob, n_samples)
            θ_spec = zeros(Float64, n_src, n_spec, n_samples)
            w_spec = zeros(Float64, n_src, n_samples)
            w_glob = zeros(Float64, n_samples)
            w_mean = zeros(Float64, n_src)
            w_std = similar(w_mean)

            # Distribution placeholders
            marginals_temp = Vector{Distribution}(undef, n_dims)
            marg_glob = Vector{Distribution}(undef, n_glob)
            marg_spec = Matrix{Distribution}(undef, (n_src, n_spec))

            # Assign global/specific parameter bounds
            bounds_glob = Vector{Tuple}(undef, n_glob)
            bounds_spec = Vector{Tuple}(undef, n_spec)

            # Record data points (input parameters + Dantigny summaries)
            data_pts = zeros(Float64, n_src, n_iter, n_dims + 3, n_samples)
            w_spec_record = zeros(Float64, n_src, n_iter, n_samples)

            # --- GENERATE INPUT PARAMETER SAMPLE ---
            sample_params = Dict()
            for (i, src) in enumerate(sources)
                g_ct = 1
                s_ct = 1
                for (k, key) in enumerate(param_keys)

                    sample_param = quantile.(sample_dists[key], sobol_pts[k, :]) # limit to sampling to 95% within bounds
                    sample_params[key] = sample_param

                    # Split in general and inducer-specific parameters
                    if key in params_glob_keys
                        θ_glob[g_ct, :] .= sample_param
                        marg_glob[g_ct] = sample_dists[key]
                        bounds_glob[g_ct] = abs_thresh ? bounds_abs[key] : bounds_rel[key]
                        g_ct += 1
                    else
                        θ_spec[i, s_ct, :] .= sample_param
                        marg_spec[i, s_ct] = sample_dists[key]
                        bounds_spec[s_ct] = abs_thresh ? bounds_abs[key] : bounds_rel[key]
                        s_ct += 1
                    end

                    data_pts[i, 1, k, :] .= sample_params[key]

                    # Initial guess priors
                    priors[key_src_strings[k, i]] = sample_dists[key]
                end
            end

            # Transform sigmas
            for key in param_keys
                if startswith(string(key), "neg_δ_")
                    suffix = string(key)[end]
                    sample_params[Symbol("σ_" * suffix)] = abs.(sample_params[Symbol("μ_" * suffix)]) .* clamp.(exp.(-sample_params[key]), 1e-12, 1e6)
                end
            end

            # Merge with default parameters
            input_params_all .= Ref(merge(sample_params, def_params))

            # --- NN TRAINING SET ---
            X_train = zeros(n_dims, n_samples, n_src)
            X_train[glob_tag, :, :] .= repeat(θ_glob, outer=[1, 1, 2])
            X_train[.!glob_tag, :, :] .= permutedims(θ_spec, [2, 3, 1])
            Y_train = zeros(3, n_samples, n_dens, n_src)

            X_train_accumulated = Dict{Tuple{Int,Int}, Matrix{Float64}}()
            Y_train_accumulated = Dict{Tuple{Int,Int}, Matrix{Float64}}()

            # Initialise data accumulators
            for i in eachindex(densities)
                for j in eachindex(sources)
                    X_train_accumulated[(i,j)] = zeros(Float64, n_dims, 0)
                    Y_train_accumulated[(i,j)] = zeros(Float64, 3, 0)
                end
            end

            # --- CALIBRATION LOOP ---
            @inbounds for s in 1:n_iter

                # print("\rIteration $s")
                println("\n===== Iteration $s =====")
                
                # Shuffle Sobol points
                # shuffle!(idx_shuffle)
                # sobol_pts = sobol_pts[idx_shuffle, :]

                if s > 1
                    marg_glob .= calibrate_marginals(marg_glob, θ_glob, w_glob, bounds_glob)
                    println("\nGlobal marginals: $marg_glob")
                    for (j, src) in enumerate(sources)

                        sample_params = Dict()

                        marg_spec[j, :] = calibrate_marginals(marg_spec[j, :], θ_spec[j, :, :], w_spec[j, :], bounds_spec)
                        println("Specific marginals: $marg_spec")
                        copula = calibrate_copula(θ_glob, θ_spec[j, :, :], marg_glob, marg_spec[j, :], w_glob, w_spec[j, :], glob_tag)

                        marginals_temp[glob_tag] .= marg_glob
                        marginals_temp[.!glob_tag] .= marg_spec[j, :]
                        
                        θ_glob, θ_spec[j, :, :] = sample_parameters(sobol_pts, copula, marg_glob, marg_spec[j, :], glob_tag) # USES COPULA
                        # for (k, marg) in enumerate(marg_glob)
                        #     θ_glob[k, :] .= quantile.(marg, sobol_pts[glob_tag, :][k, :])
                        # end
                        # for (k, marg) in enumerate(marg_spec[j, :])
                        #     θ_spec[j, k, :] .= quantile.(marg, sobol_pts[.!glob_tag, :][k, :])
                        # end

                        # Assign new samples
                        g_ct = 1
                        s_ct = 1
                        for (k, key) in enumerate(param_keys)
                            if key in params_glob_keys
                                sample_params[key] = θ_glob[g_ct, :]
                                g_ct += 1
                            else
                                sample_params[key] = θ_spec[j, s_ct, :]
                                s_ct += 1
                            end
                            priors[key_src_strings[k, j]] = marginals_temp[k]
                        end

                        # Transform sigmas
                        for key in param_keys
                            if startswith(string(key), "neg_δ_")
                                suffix = string(key)[end]
                                sample_params[Symbol("σ_" * suffix)] = abs.(sample_params[Symbol("μ_" * suffix)]) .* clamp.(exp.(-sample_params[key]), 1e-12, 1e6)
                            end
                        end

                        input_params_all[j] = merge(sample_params, def_params) # Merge with default parameters
                    end
                end

                println("Global parameter means: $(mean(θ_glob; dims=2))")
                println("Specific parameter means: $(mean(θ_spec; dims=3))")

                # --- ADAPTIVE SAMPLING STRATEGY ---
                if (use_surrogates == true && s == 1) || use_surrogates == false
                    # First iteration: run ALL samples on mechanistic model
                    n_mech_samples = n_samples
                    mech_indices = 1:n_samples
                    use_surrogate_this_iter = false
                else
                    # Later iterations: strategic subsampling
                    n_mech_samples = max(200, round(Int, 0.15 * n_samples))  # At least 200 or 15%
                    
                    # Sample strategically:
                    # 1. High-weight samples (50%)
                    # 2. Random samples (50%) for coverage
                    n_high_weight = div(n_mech_samples, 2)
                    n_random = n_mech_samples - n_high_weight
                    
                    # Get high-weight indices
                    weight_order = sortperm(w_glob, rev=true)
                    high_weight_idx = weight_order[1:n_high_weight]
                    
                    # Random samples from rest
                    remaining_idx = setdiff(1:n_samples, high_weight_idx)
                    random_idx = sample(remaining_idx, n_random, replace=false)
                    
                    mech_indices = sort([high_weight_idx; random_idx])
                    use_surrogate_this_iter = true
                end

                println("Running mechanistic model on $(n_mech_samples) samples ($(round(100*n_mech_samples/n_samples, digits=1))%)")

                max_uncertainties = zeros(n_dens, n_src)
                z_acc_specific = zeros(n_src, n_samples)

                # --- DANTIGNY SUMMARIES AND DIAGNOSTICS OVER EXPERIMENTAL CONDITIONS ---
                @inbounds for (i, density) in enumerate(densities) # iterate over spore densities (exp. data)
                    density_scaled = inverse_mL_to_cubic_um(density)

                    @inbounds for j in eachindex(sources) # Iterate over sources

                        # Mechanistic uncertainty combines fit quality + prediction uncertainty
                        uncert_mech = zeros(Float64, 3, n_samples)

                        println("\n---- Running model $(aliases[m]) with density $density and source $(sources[j]) -----")

                        # println("Input parameters: $(input_params_all[j])")

                        # Weave global and local parameters
                        θ[glob_tag, :] .= θ_glob
                        θ[.!glob_tag, :] .= θ_spec[j, :, :]
                        data_pts[j, s, 1:n_dims, :] .= θ
                        debug_vals[:params][s, j, :, :] .= θ

                        if !use_surrogate_this_iter

                            # --- RUN ALL SAMPLES THROUGH MECHANISTIC MODEL ---
                            for n in 1:n_samples
                                p_out .= compute_germination_response(aliases[m], times_sec, density_scaled, Dict(k => v[mod1(n, length(v))] for (k, v) in input_params_all[j]))
                                d[:, n], rmses[i, j, n], uncert_mech[:, n] = fit_dantigny_to_germination_curve(p_out, times)
                                Y_train[:, n, i, j] .= d[:, n] # Save to training data
                            end

                        else
                            # --- HYBRID APPROACH ---

                            # 1. Run mechanistic model on subsample
                            d_mech = zeros(Float64, 3, n_mech_samples)
                            rmse_mech = zeros(Float64, n_mech_samples)
                            uncert_params = zeros(Float64, 3, n_mech_samples)
                            for (idx_local, n_global) in enumerate(mech_indices)
                                try
                                    p_out .= compute_germination_response(
                                        aliases[m], times_sec, density_scaled,
                                        Dict(k => v[mod1(n_global, length(v))] for (k, v) in input_params_all[j])
                                    )
                                    d_mech[:, idx_local], rmse_mech[idx_local], uncert_params[:, idx_local] = fit_dantigny_to_germination_curve(p_out, times)
                                catch
                                    println(Dict(k => v[mod1(n_global, length(v))] for (k, v) in input_params_all[j]))
                                end
                            end
                            
                            # 2. Predict all samples with surrogate
                            d_surr, σ_pred = predict_with_uncertainty_mixed_precision(
                                surrogates[(i, j)], 
                                θ,
                                n_dropout_samples=50,
                                output_ranges=[(r[1], r[2]) for r in eachrow(output_ranges)]
                            )

                            # DIAGNOSTICS
                            # if any(isnan.(d_surr))
                            #     println("🔍 NaN Diagnostics for density=$i, source=$j")
                                
                            #     # Check which outputs are NaN
                            #     nan_samples = any(isnan.(d_surr), dims=1) |> vec
                            #     println("  $(sum(nan_samples)) samples have NaN predictions")
                                
                            #     # Check input parameter ranges
                            #     θ_min = minimum(θ, dims=2) |> vec
                            #     θ_max = maximum(θ, dims=2) |> vec
                            #     println("  θ ranges this iteration:")
                            #     for (idx, (mn, mx)) in enumerate(zip(θ_min, θ_max))
                            #         println("    Param $idx: $mn to $mx")
                            #     end
                                
                            #     # Compare to training data ranges
                            #     X_train_this = X_train_accumulated[(i,j)]
                            #     if size(X_train_this, 2) > 0
                            #         X_train_min = minimum(X_train_this, dims=2) |> vec
                            #         X_train_max = maximum(X_train_this, dims=2) |> vec
                            #         println("  Training data ranges:")
                            #         for (idx, (mn, mx)) in enumerate(zip(X_train_min, X_train_max))
                            #             println("    Param $idx: $mn to $mx")
                            #         end
                                    
                            #         # Check extrapolation
                            #         extrapolating = (θ_min .< X_train_min) .| (θ_max .> X_train_max)
                            #         if any(extrapolating)
                            #             println("  ⚠️ Extrapolating on parameters: $(findall(extrapolating))")
                            #         end
                            #     end
                                
                            #     # Check which outputs are problematic
                            #     for out_idx in 1:3
                            #         if any(isnan.(d_surr[out_idx, :]))
                            #             println("  Output $out_idx has $(sum(isnan.(d_surr[out_idx, :]))) NaNs")
                            #             # Check non-NaN range
                            #             valid_vals = d_surr[out_idx, .!isnan.(d_surr[out_idx, :])]
                            #             if length(valid_vals) > 0
                            #                 println("    Valid range: $(minimum(valid_vals)) to $(maximum(valid_vals))")
                            #             end
                            #         end
                            #     end
                            # end
                            
                            # 3. Validate surrogate on mechanistic subsample
                            d_surr_subsample = d_surr[:, mech_indices]
                            valid_mask = .!any(isnan.(d_mech), dims=1) |> vec
                            
                            d_mech = d_mech[:, valid_mask]
                            d_surr_subsample = d_surr_subsample[:, valid_mask]
                            uncert_params = uncert_params[:, valid_mask]
                            validation_errors = abs.(d_mech .- d_surr_subsample)

                            println(sum(valid_mask), " valid mechanistic model subsamples")
                            
                            # Compute validation metrics
                            mae_validation = mean(validation_errors, dims=2) |> vec
                            max_error = maximum(validation_errors, dims=2) |> vec
                            
                            println("  Validation MAE: P_max=$(round(mae_validation[1], digits=4)), τ=$(round(mae_validation[2], digits=2)), ν=$(round(mae_validation[3], digits=3))")
                            println("  Validation max error: P_max=$(round(max_error[1], digits=4)), τ=$(round(max_error[2], digits=2)), ν=$(round(max_error[3], digits=3))")
                            
                            # 4. Decide: trust surrogate or retrain?
                            needs_retraining = (
                                mae_validation[1] > 0.05 ||  # P_max error > 5%
                                mae_validation[2] > 2.0 ||   # τ error > 2 hours
                                mae_validation[3] > 0.5 ||   # ν error > 0.5
                                max_error[1] > 0.15 ||
                                max_error[2] > 10.0 ||
                                max_error[3] > 1.5
                            )

                            if needs_retraining
                                println("  ⚠️  Surrogate validation failed - accumulating data for retraining")
                                
                                # Accumulate NEW mechanistic data
                                X_new = θ[:, mech_indices]
                                Y_new = d_mech
                                
                                # Filter valid samples
                                X_new = X_new[:, valid_mask]
                                
                                # Clamp to reasonable bounds
                                Y_new = clamp.(Y_new, output_ranges[:, 1], output_ranges[:, 2])
                                
                                # Append to accumulated data
                                X_train_accumulated[(i,j)] = hcat(X_train_accumulated[(i,j)], X_new)
                                Y_train_accumulated[(i,j)] = hcat(Y_train_accumulated[(i,j)], Y_new)
                                
                                # Keep only recent N samples to avoid memory bloat
                                max_accumulated = 8192
                                n_accumulated = size(X_train_accumulated[(i,j)], 2)
                                if n_accumulated > max_accumulated
                                    # Keep most recent samples
                                    keep_idx = (n_accumulated - max_accumulated + 1):n_accumulated
                                    X_train_accumulated[(i,j)] = X_train_accumulated[(i,j)][:, keep_idx]
                                    Y_train_accumulated[(i,j)] = Y_train_accumulated[(i,j)][:, keep_idx]
                                end
                                
                                # Retrain with accumulated data
                                println("  Retraining surrogate with $(size(X_train_accumulated[(i,j)], 2)) accumulated samples...")
                                surrogates[(i,j)] = train_multioutput_nn_mixed_precision(
                                    X_train_accumulated[(i,j)], 
                                    Y_train_accumulated[(i,j)];
                                    batch_size=find_optimal_batch_size(n_dims, size(X_train_accumulated[(i,j)], 2)),
                                    epochs=500
                                )
                                
                                # Re-predict with updated surrogate
                                d_surr, σ_pred = predict_with_uncertainty_mixed_precision(
                                    surrogates[(i, j)], 
                                    θ,
                                    n_dropout_samples=50,
                                    output_ranges=[(r[1], r[2]) for r in eachrow(output_ranges)]
                                )

                                # DIAGNOSTICS AGAIN
                                # if any(isnan.(d_surr))
                                #     println("🔍 NaN Diagnostics (retraining) for density=$i, source=$j")
                                    
                                #     # Check which outputs are NaN
                                #     nan_samples = any(isnan.(d_surr), dims=1) |> vec
                                #     println("  $(sum(nan_samples)) samples have NaN predictions")
                                    
                                #     # Check input parameter ranges
                                #     θ_min = minimum(θ, dims=2) |> vec
                                #     θ_max = maximum(θ, dims=2) |> vec
                                #     println("  θ ranges this iteration:")
                                #     for (idx, (mn, mx)) in enumerate(zip(θ_min, θ_max))
                                #         println("    Param $idx: $mn to $mx")
                                #     end
                                    
                                #     # Compare to training data ranges
                                #     X_train_this = X_train_accumulated[(i,j)]
                                #     if size(X_train_this, 2) > 0
                                #         X_train_min = minimum(X_train_this, dims=2) |> vec
                                #         X_train_max = maximum(X_train_this, dims=2) |> vec
                                #         println("  Training data ranges:")
                                #         for (idx, (mn, mx)) in enumerate(zip(X_train_min, X_train_max))
                                #             println("    Param $idx: $mn to $mx")
                                #         end
                                        
                                #         # Check extrapolation
                                #         extrapolating = (θ_min .< X_train_min) .| (θ_max .> X_train_max)
                                #         if any(extrapolating)
                                #             println("  ⚠️ Extrapolating on parameters: $(findall(extrapolating))")
                                #         end
                                #     end
                                    
                                #     # Check which outputs are problematic
                                #     for out_idx in 1:3
                                #         if any(isnan.(d_surr[out_idx, :]))
                                #             println("  Output $out_idx has $(sum(isnan.(d_surr[out_idx, :]))) NaNs")
                                #             # Check non-NaN range
                                #             valid_vals = d_surr[out_idx, .!isnan.(d_surr[out_idx, :])]
                                #             if length(valid_vals) > 0
                                #                 println("    Valid range: $(minimum(valid_vals)) to $(maximum(valid_vals))")
                                #             end
                                #         end
                                #     end
                                # end
                            else
                                println("  ✓ Surrogate validated successfully")
                            end
                            
                            # 5. Use surrogate predictions for ALL samples
                            d .= d_surr

                            # Use uncertainty for RMSE estimate
                            # rmses[i, j, :] = mean_log(σ_pred, dims=1) |> vec

                            uncert_fit = mean(uncert_params; dims=2)
                            uncert_mech = sqrt.(uncert_fit.^2 .+ σ_pred)

                            # Track maximum uncertainty
                            max_uncertainties[i, j] = maximum(σ_pred)
                            
                            # Warn if extrapolating heavily
                            rel_uncertainty = σ_pred ./ (d .+ 1e-10)
                            max_rel_unc = maximum(rel_uncertainty)
                            
                            if max_rel_unc > 0.5  # >50% relative uncertainty
                                println("⚠️  High uncertainty for density=$i, source=$j")
                                println("    Max relative uncertainty: $(round(max_rel_unc*100, digits=1))%")
                                println("    Consider retraining surrogate or using mechanistic model")
                            end
                        end
                        
                        nan_mask = all(.!isnan.(d); dims=1) |> vec
                        println("$(sum(nan_mask)) valid Dantigny summaries")
                        println("Mean p_max: $(mean(d[1, nan_mask])) ($(minimum(d[1, nan_mask])) - $(maximum(d[1, nan_mask])), $(sum(isnan.(d[1, :]))) NaNs)")
                        println("Mean tau_g: $(mean(d[2, nan_mask])) ($(minimum(d[2, nan_mask])) - $(maximum(d[2, nan_mask])), $(sum(isnan.(d[2, :]))) NaNs)")
                        println("Mean nu: $(mean(d[3, nan_mask])) ($(minimum(d[3, nan_mask])) - $(maximum(d[3, nan_mask])), $(sum(isnan.(d[3, :]))) NaNs)")

                        # println("Maximum RMSE: $(maximum(rmses))")

                        # Total uncertainty per sample
                        uncert_total = sqrt.(uncert_lab[i, j, :].^2 .+ uncert_mech.^2)
                        valid_uncert = all(.!isnan.(uncert_total); dims=1) |>vec
                        println("Mechanistic model uncertainties range from $(minimum(uncert_mech[:, valid_uncert]; dims=2)) to $(maximum(uncert_mech[:, valid_uncert]; dims=2))")
                        println("Invalid uncertainties: $(sum(.!valid_uncert))")

                        # Find valid Dantigny results
                        in_bounds_and_full = ((all(d .> output_ranges[:, 1] .&& d .< output_ranges[:, 2]; dims=1) .&& all(.!isnan.(d[2:3, :]); dims=1)) |> vec)
                        println("$(sum(in_bounds_and_full)) summaries in bounds & full")
                        valid_summary = in_bounds_and_full .&& valid_uncert
                        d_valid = d[:, valid_summary]

                        # Included fraction (Criterion 1)
                        π_d_new = dropdims(mean(d_valid .> CIs[i, j, :, 1] .&& d_valid .< CIs[i, j, :, 2]; dims=2); dims=2)
                        if s > 1 # Criterion 5
                            π_diffs[i, j, :] = π_d_new ./ max.(π_d[i, j, :], 1e-6)
                        end
                        π_d[i, j, :] .= π_d_new

                        # Lab mean quantiles (Criterion 2)
                        q_d[i, j, :] = mean(d_valid .< lab_means[i, j, :]; dims=2) |> vec

                        # Mahalanobis distance (Criterion 3)
                        μ_d = mean(d_valid, dims=2) |> vec
                        Σ_d = cov(d_valid', corrected=false)
                        diff_d = lab_means[i, j, :] .- μ_d

                        # Debug covariance matrix
                        if any(isnan.(Σ_d) .|| isinf.(Σ_d))
                            display(Σ_d)
                        end
                        m_d[i, j] = dot(diff_d, Σ_d \ diff_d)

                        # IQR comparison (Criterion 4)
                        d_vecs = [vec(di) for di in eachrow(d_valid)]
                        iqr_new = quantile.(d_vecs, 0.75) .- quantile.(d_vecs, 0.25)
                        if s > 1 # Criterion 5
                            iqr_diff[i, j, :] = iqr_new ./ max.(iqr[i, j, :], 1e-6)
                        end
                        iqr[i, j, :] .= iqr_new
                        iqr_check[i, j, :] .= iqr[i, j, :] .> CI_widths[i, j, :]

                        # Covariance matrix
                        Σ_lab = diagm(uncert_lab[i, j, :].^2)

                        # Compute running deviations
                        @inbounds for n in 1:n_samples
                            if valid_summary[n]
                                
                                diffs = d[:, n] .- lab_means[i, j, :]
                                debug_vals[:lab_means][s, i, j, n, :] .= lab_means[i, j, :]
                                debug_vals[:diffs][s, i, j, n, :] .= diffs
                                
                                if !use_surrogate_this_iter
                                    
                                    z_dev = dot(diffs, Σ_lab \ diffs)
                                    
                                    # Unreliability penalty from SE
                                    se_penalty = sum(max.(0.0, uncert_mech[:, n] ./ uncert_lab[i, j, :] .- 1.0).^2)
                                    # se_penalty = sum(log.(uncert_mech[:, n] ./ uncert_lab[i, j, :]).^2)
                                    
                                    # RMSE penalty
                                    # rmse_penalty = (rmses[i, j, n] / 0.02)^2
                                    
                                    # z_dist[n] = min(z_dev + se_penalty + rmse_penalty, 1e6) # ADD RMSE LATER!!!!
                                    z_dist[n] = min(z_dev + se_penalty, 1e8)

                                    # println("$n: z_dist for $(d[:, n]) is $(z_dist[n]), reliability penalty is $(se_penalty).")
                                    debug_vals[:penalties][s, i, j, n] = se_penalty

                                else
                                    # Uncertainty-weighted Mahalanobis distance
                                    Σ_n = diagm(uncert_total[:, n].^2)
                                    z_dist[n] = dot(diffs, Σ_n \ diffs)
                                end

                                z_acc_specific[j, n] += z_dist[n]

                            else

                                z_dist[n] = 1e8  # Large penalty
                                z_acc_specific[j, n] += z_dist[n]

                            end
                        end

                        debug_vals[:d][s, i, j, :, :] .= d'
                        debug_vals[:z_dist][s, i, j, :] .= z_dist
                        if i == 1 data_pts[j, s, (n_dims + 1):end, :] .= d end

                        # Fraction of lab means inside joint regions (Criterion 5)
                        p_z_new = mean(z_dist .< χ_sq)
                        if s > 1
                            p_z_diffs[i, j] = p_z_new / max(p_z[i, j], 1e-6)
                        end
                        p_z[i, j] = p_z_new

                        # Total predictive probability (Criterion 6)
                        p_d[i, j] = mean(all(d_valid .> CIs[i, j, :, 1] .&& d_valid .< CIs[i, j, :, 2], dims=1))
                    end
                end

                z_acc_specific ./= n_dens
                println("Mean z_acc_specific: $(mean(z_acc_specific; dims=2))")

                # Penalize weights due to RMSE
                # reliabilities = exp.(-0.5 .* dropdims(mean(rmses.^2, dims=1); dims=1) ./ 0.004) # using an error scale of 2%

                # Compute condition-specific weights
                temp_param = get_temp_param(s, n_iter, T_init=10000.0, T_final=10000.0)
                println("Tempering parameter: $temp_param")
                log_w_spec = -temp_param .* z_acc_specific
                w_spec .= exp.(log_w_spec)
                
                w_spec_record[:, s, :] .= w_spec
                debug_vals[:w_spec][s, :, :] .= w_spec

                # Compute global weights as product of specific ones
                log_w_glob = sum(log_w_spec, dims=1) |> vec  # Sum of logs = log of product
                w_glob .= exp.(log_w_glob)

                # Weight statistics (Criterion 5)
                w_mean_new = mean(w_spec, dims=2) |> vec
                w_std_new = std(w_spec, dims=2) |> vec
                if s > 1
                    w_mean_diff = w_mean_new / w_mean
                    w_std_diff = w_std_new / w_std
                end
                w_mean = w_mean_new
                w_std = w_std_new

                # --- INITIAL SURROGATE TRAINING ---
                if s == 1 && use_surrogates
                    println("Building GPU-accelerated surrogate...")
                    
                    for i in eachindex(densities)
                        for j in eachindex(sources)

                            valid_mask = .!any(isnan.(Y_train[:, :, i, j]), dims=1) |> vec
                            println("$(sum(valid_mask)) valid samples")

                            X_valid = X_train[:, valid_mask, j]
                            Y_valid = Y_train[:, valid_mask, i, j]

                            # Clamp within plausible ranges
                            Y_valid .= clamp.(Y_valid, output_ranges[:, 1], output_ranges[:, 2])

                            # Initialize accumulated data
                            X_train_accumulated[(i,j)] = copy(X_valid)
                            Y_train_accumulated[(i,j)] = copy(Y_valid)
                            
                            # Train on GPU
                            surrogates[(i, j)] = train_multioutput_nn_mixed_precision(
                                X_valid, Y_valid; 
                                batch_size=find_optimal_batch_size(n_dims, sum(valid_mask))
                            )
                        end
                    end
                end

                println("w_mean: $w_mean")
                println("w_std: $w_std")
                println("$(sum(w_spec .> w_mean; dims=2)) weights above mean")

                # Termination criteria
                π_d_crit_new = all(π_d .> 0.2 .&& π_d .< 0.8) # Criterion 1
                q_d_crit_new = all(q_d .> 0.1 .&& q_d .< 0.9) # Criterion 2
                m_d_crit_new = all(m_d .< χ_sq) # Criterion 3

                if (π_d_crit_new != π_d_crit)
                    π_d_crit_ct = 0
                else
                    π_d_crit_ct += 1
                end
                if (q_d_crit_new != q_d_crit)
                    q_d_crit_ct = 0
                else
                    q_d_crit_ct += 1
                end
                if (m_d_crit_new != m_d_crit)
                    m_d_crit_ct = 0
                else
                    m_d_crit_ct += 1
                end

                π_d_crit = π_d_crit_new
                q_d_crit = q_d_crit_new
                m_d_crit = m_d_crit_new

                if s == 1
                    stability_crit = true
                else
                    stability_crit = all(abs.(1.0 .- π_diffs) .< 0.1) && all(abs.(1.0 .- iqr_diff) .< 0.1) && all(abs.(1.0 .- p_z_diffs) .< 0.1) && all(abs.(1.0 .- w_mean_diff) .< 0.1) && all(abs.(1.0 .- w_std_diff) .< 0.1)
                end
                println("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
                println("π_d_crit_ct: $π_d_crit_ct")
                println("q_d_crit_ct: $q_d_crit_ct")
                println("m_d_crit_ct: $m_d_crit_ct")
                # println("π_d: $π_d")
                println("π_d: $(minimum(π_d)) - $(maximum(π_d))")
                # println("q_d: $q_d")
                println("q_d: $(minimum(q_d)) - $(maximum(q_d))")
                # println("m_d: $m_d")
                println("m_d: $(minimum(m_d)) - $(maximum(m_d))")
                # println("iqr: $iqr")
                println("iqr: $(minimum(iqr)) - $(maximum(iqr))")
                if s > 1
                    println("Max π_diffs: $(maximum(π_diffs))")
                    println("Max iqr_diff: $(maximum(iqr_diff))")
                    println("Max p_z_diffs: $(maximum(p_z_diffs))")
                    println("Max w_mean_diff: $(maximum(w_mean_diff))")
                    println("Max w_std_diff: $(maximum(w_std_diff))")
                end
                println("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
                println("Criterion 1 (Fraction of means in CIs): $π_d_crit_new ($(π_d .> 0.2 .&& π_d .< 0.8))")
                println("Criterion 2 (Quantiles of lab means): $q_d_crit_new ($(q_d .> 0.1 .&& q_d .< 0.9))")
                println("Criterion 3 (Mahalanobis distances): $m_d_crit_new ($(m_d .< χ_sq))")
                println("Criterion 4 (IQR larger than CI): $(all(iqr_check))")
                if s > 1
                    println("Criterion 5 (Stability of diagnostics): $stability_crit")
                end
                println("Criterion 6 (Minimal fraction complete inclusion): $(all(p_d .> 1e-2)) ($p_d)")
                println("Criterion 1, 2, 3 (soft): $((π_d_crit && q_d_crit && m_d_crit) || (π_d_crit_ct > 5 && q_d_crit_ct > 5 && m_d_crit_ct > 5))")
                if (all(iqr_check) # Criterion 4
                    && stability_crit # Criterion 5
                    && all(p_d .> 1e-2) # Criterion 6
                    && ((π_d_crit && q_d_crit && m_d_crit) || (π_d_crit_ct > 5 && q_d_crit_ct > 5 && m_d_crit_ct > 5))) # Soft criteria 1, 2 and 3
                    println("Termination criteria met at iteration $s")
                    break
                end
                println("===========================")
                
                @show priors
                priors_running[s] = copy(priors)
            end

            priors_all[aliases[m]] = priors
            priors_running_all[aliases[m]] = priors_running
            surrogates_all[aliases[m]] = surrogates
            θ_final_all[aliases[m]] = θ
            w_glob_final_all[aliases[m]] = w_glob
            w_spec_final_all[aliases[m]] = w_spec
            data_pts_dict[aliases[m]] = data_pts
            w_spec_record_all[aliases[m]] = w_spec_record
        end

        jldsave("../src/Data/priors.jld2"; priors_all, priors_running_all, surrogates_all, θ_final_all, w_glob_final_all, w_spec_final_all, data_pts_dict, w_spec_record_all, debug_vals)

        return data_pts_dict
    end


    # ===== Surrogate model =====
    struct MultiOutputSurrogate
        model::Chain
        X_mean::Vector{Float64}
        X_std::Vector{Float64}
        Y_min::Vector{Float64}
        Y_range::Vector{Float64}
        n_inputs::Int
        n_outputs::Int
    end

    """
        Automatic Mixed Precision training optimized for RTX A4000
        
        Uses:
        - FP16 for forward/backward passes (Tensor Cores)
        - FP32 for loss computation and weight updates
        - Dynamic loss scaling to prevent underflow
        """
    function train_multioutput_nn_mixed_precision(
        X_train, Y_train;
        hidden_dims=[128, 64, 32],
        epochs=1000,
        batch_size=64,
        learning_rate=0.001,
        validation_split=0.15,
        early_stopping_patience=50,
        loss_scale=1024.0,  # Initial loss scale
        #output_ranges=[(0, 1), (1e-6, 1e4), (0, 10)],
        verbose=false
    )
        
        n_inputs = size(X_train, 1)
        n_outputs = size(Y_train, 1)
        n_samples = size(X_train, 2)
        
        if verbose
            println("Using Automatic Mixed Precision on RTX A4000")
            println("This will use Tensor Cores for ~2-4× speedup")
        end
        
        # ==================== Preprocessing ====================
        
        # Normalize (in Float32 for accuracy)
        X_mean = mean(X_train, dims=2) |> vec
        X_std = std(X_train, dims=2) |> vec
        X_std[X_std .< 1e-8] .= 1.0
        X_normalized = (X_train .- X_mean) ./ X_std
        
        # Log-transform outputs
        Y_log = copy(Y_train)
        Y_log[2, :] = log.(max.(Y_train[2, :], 1e-10))
        Y_log[3, :] = log.(max.(Y_train[3, :], 1e-10))

        # Normalize outputs to [0, 1] range for each parameter
        Y_min = [minimum(Y_log[i, :]) for i in 1:n_outputs]
        Y_max = [maximum(Y_log[i, :]) for i in 1:n_outputs]
        Y_range = Y_max .- Y_min
        Y_range[Y_range .< 1e-8] .= 1.0

        Y_normalized = (Y_log .- Y_min) ./ Y_range
        
        # Train/val split
        n_val = floor(Int, validation_split * n_samples)
        n_train = n_samples - n_val
        
        indices = shuffle(1:n_samples)
        train_idx = indices[1:n_train]
        val_idx = indices[n_train+1:end]
        
        X_train_split = X_normalized[:, train_idx]
        Y_train_split = Y_normalized[:, train_idx]
        X_val = X_normalized[:, val_idx]
        Y_val = Y_normalized[:, val_idx]
        
        # ==================== Build Model (Float32) ====================
        
        layers = []
        push!(layers, Dense(n_inputs, hidden_dims[1], swish))
        push!(layers, Dropout(0.1))
        
        for i in 2:length(hidden_dims)
            push!(layers, Dense(hidden_dims[i-1], hidden_dims[i], swish))
            push!(layers, Dropout(0.1))
        end
        
        push!(layers, Dense(hidden_dims[end], n_outputs), sigmoid)
        
        model = Chain(layers...)
        
        # ==================== Move to GPU ====================
        
        # Model stays in Float32
        model_gpu = model |> gpu
        
        # Convert data to Float16 and move to GPU
        X_train_gpu = Float16.(X_train_split) |> gpu
        Y_train_gpu = Float16.(Y_train_split) |> gpu
        X_val_gpu = Float16.(X_val) |> gpu
        Y_val_gpu = Float16.(Y_val) |> gpu
        
        if verbose
            println("\nModel parameters: ", sum(length, Flux.params(model_gpu)))
            println("Data type: Float16 (activations), Float32 (weights)")
        end
        
        # ==================== Training Setup ====================
        
        # Optimizer (operates on Float32 weights)
        opt_state = Flux.setup(Flux.Adam(learning_rate), model_gpu)
        
        # Dynamic loss scaling
        current_scale = loss_scale
        scale_factor = 2.0
        scale_window = 2000  # Steps before increasing scale
        steps_since_overflow = 0
        
        # ==================== Helper Functions ====================
        
        function create_batches(X, Y, batch_size)
            n = size(X, 2)
            indices = shuffle(1:n)
            batches = []
            
            for i in 1:batch_size:n
                batch_end = min(i + batch_size - 1, n)
                batch_idx = indices[i:batch_end]
                push!(batches, (X[:, batch_idx], Y[:, batch_idx]))
            end
            
            return batches
        end

        function check_overflow(x)
            if x === nothing
                return false
            elseif x isa AbstractArray
                return any(isnan.(x)) || any(isinf.(x))
            elseif x isa NamedTuple
                return any(check_overflow(v) for v in values(x))
            elseif x isa Tuple
                return any(check_overflow(v) for v in x)
            else
                return false
            end
        end

        function unscale_grads!(g, scale)
            if g isa AbstractArray
                g ./= scale
            elseif g isa NamedTuple
                foreach(v -> unscale_grads!(v, scale), values(g))
            elseif g isa Tuple
                foreach(v -> unscale_grads!(v, scale), g)
            end
            return g
        end
        
        """
        Single training step with mixed precision
        Returns: (loss, had_overflow)
        """
        function mixed_precision_step!(model, x_fp16, y_fp16, opt_state, scale)

            # Forward pass in FP16
            loss_val, grads = Flux.withgradient(model) do m
                # Activations in FP16 (uses Tensor Cores!)
                ŷ_fp16 = m(x_fp16)
                
                # Compute loss in FP32 for stability
                ŷ_fp32 = Float32.(ŷ_fp16)
                y_fp32 = Float32.(y_fp16)
                
                loss_fp32 = Flux.mse(ŷ_fp32, y_fp32)
                
                # Scale loss to prevent gradient underflow
                loss_fp32 * scale
            end
            
            # Check for overflow/NaN in gradients
            grad_model = grads[1]
            has_overflow = check_overflow(grad_model)
            
            if !has_overflow
                unscale_grads!(grad_model, scale)
                Flux.update!(opt_state, model, grad_model)
            end
            
            # Return unscaled loss
            return loss_val / scale, has_overflow
        end
        
        # ==================== Training Loop ====================
        
        best_val_loss = Inf
        patience_counter = 0
        best_model_state = Flux.state(model_gpu)
        
        train_losses = Float64[]
        val_losses = Float64[]
        
        if verbose
            println("\nStarting mixed precision training...")
            progress = Progress(epochs, desc="Training: ")
        end
        
        global_step = 0
        
        for epoch in 1:epochs
            batches = create_batches(X_train_gpu, Y_train_gpu, batch_size)
            
            epoch_train_loss = 0.0
            n_batches = length(batches)
            overflow_count = 0
            
            for (x_batch, y_batch) in batches
                global_step += 1
                
                # Mixed precision training step
                loss_val, had_overflow = mixed_precision_step!(
                    model_gpu, x_batch, y_batch, opt_state, current_scale
                )
                
                if had_overflow
                    # Reduce loss scale on overflow
                    current_scale = max(current_scale / scale_factor, 1.0)
                    overflow_count += 1
                    steps_since_overflow = 0
                else
                    # Increase loss scale if stable
                    steps_since_overflow += 1
                    if steps_since_overflow >= scale_window
                        current_scale = min(current_scale * scale_factor, 65536.0)
                        steps_since_overflow = 0
                    end
                    
                    epoch_train_loss += Float64(loss_val)
                end
            end
            
            epoch_train_loss /= n_batches
            push!(train_losses, epoch_train_loss)
            
            # Validation (in FP16 for speed, convert to FP32 for loss)
            ŷ_val_fp16 = model_gpu(X_val_gpu)
            ŷ_val_fp32 = Float32.(ŷ_val_fp16)
            Y_val_fp32 = Float32.(Y_val_gpu)
            val_loss = Float64(Flux.mse(ŷ_val_fp32, Y_val_fp32))
            push!(val_losses, val_loss)
            
            # Early stopping
            if val_loss < best_val_loss
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = Flux.state(model_gpu)
            else
                patience_counter += 1
            end
            
            if patience_counter >= early_stopping_patience
                if verbose
                    println("\nEarly stopping at epoch $epoch")
                end
                break
            end
            
            # Progress
            if verbose && (epoch % 10 == 0 || epoch == 1)
                ProgressMeter.next!(progress;
                    showvalues = [
                        (:epoch, epoch),
                        (:train_loss, round(epoch_train_loss, digits=6)),
                        (:val_loss, round(val_loss, digits=6)),
                        (:best_val, round(best_val_loss, digits=6)),
                        (:loss_scale, round(current_scale, digits=1)),
                        (:overflows, overflow_count)
                    ])
            end
        end
        
        # Restore best model and move to CPU
        Flux.loadmodel!(model_gpu, best_model_state)
        model_cpu = model_gpu |> cpu
        
        CUDA.reclaim()
        
        if verbose
            println("\nTraining complete!")
            println("Best validation loss: $(round(best_val_loss, digits=6))")
            println("Final loss scale: $(round(current_scale, digits=1))")
        end
        
        return MultiOutputSurrogate(
            model_cpu,
            X_mean,
            X_std,
            Y_min,
            Y_range,
            n_inputs,
            n_outputs
        )
    end
    
    """
    Mixed precision uncertainty estimation
    """
    function predict_with_uncertainty_mixed_precision(
        surrogate::MultiOutputSurrogate,
        X_new;
        n_dropout_samples=50,
        output_ranges=[(0, 1), (1e-6, 1e4), (0, 10)]
    )
        
        single_sample = ndims(X_new) == 1
        if single_sample
            X_new = reshape(X_new, :, 1)
        end
        
        n_samples = size(X_new, 2)
        
        # Normalize
        X_normalized = (X_new .- surrogate.X_mean) ./ surrogate.X_std
        
        # Move to GPU and convert to FP16
        model_gpu = surrogate.model |> gpu
        X_gpu = Float16.(X_normalized) |> gpu
        
        # Enable dropout
        Flux.testmode!(model_gpu, false)
        
        # Replicate for parallel dropout
        X_replicated = repeat(X_gpu, outer=(1, n_dropout_samples))
        
        # Forward pass in FP16 (uses Tensor Cores!)
        Y_fp16 = model_gpu(X_replicated)
        
        # Convert back to FP32 for statistics
        Y_fp32 = Float32.(Y_fp16)
        
        # Disable dropout
        Flux.testmode!(model_gpu, true)
        
        # Move to CPU
        Y_normalized_all = Y_fp32 |> cpu
        CUDA.reclaim()
        
        # Reshape
        Y_normalized_all = reshape(Y_normalized_all, surrogate.n_outputs, n_samples, n_dropout_samples)

        # Denormalize and back-transform
        Y_log_all = Y_normalized_all .* reshape(surrogate.Y_range, :, 1, 1) .+ 
                reshape(surrogate.Y_min, :, 1, 1)

        Y_all = copy(Y_log_all)
        Y_all[2, :, :] = exp.(Y_log_all[2, :, :])
        Y_all[3, :, :] = exp.(Y_log_all[3, :, :])

        # Clamp with output ranges
        Y_all[1, :, :] = clamp.(Y_all[1, :, :], output_ranges[1]...)
        Y_all[2, :, :] = clamp.(Y_all[2, :, :], output_ranges[2]...)
        Y_all[3, :, :] = clamp.(Y_all[3, :, :], output_ranges[3]...)
        
        # Compute statistics
        mean_pred = mean(Y_all, dims=3)[:, :, 1]
        std_pred = std(Y_all, dims=3)[:, :, 1]
        
        if single_sample
            return mean_pred[:, 1], std_pred[:, 1]
        else
            return mean_pred, std_pred
        end
    end
    
    """
    Find optimal batch size based on GPU memory
    
    Rule of thumb:
    - Larger batches = better GPU utilization but more memory
    - Sweet spot: 32-256 for most GPUs
    """
    function find_optimal_batch_size(n_inputs, n_samples; max_memory_gb=8)

        if !CUDA.functional()
            return 32  # CPU default
        end
        
        available_memory = CUDA.available_memory() / 1e9  # GB
        
        # Estimate memory per sample (rough heuristic)
        # memory_per_sample ≈ n_params × hidden_size × 4 bytes × 2 (forward + backward)
        estimated_memory_per_sample = n_inputs * 128 * 4 * 2 / 1e9  # GB
        
        # Use 50% of available memory for batch
        safe_batch_size = floor(Int, available_memory * 0.5 / estimated_memory_per_sample)
        
        # Clamp to reasonable range
        batch_size = clamp(safe_batch_size, 32, min(256, n_samples ÷ 10))
        
        println("Recommended batch size: $batch_size")
        return batch_size
    end

    # ========================
    # ===== Data fitting =====
    # ========================
    """
    Fit Dantigny model to simulated germination curve
    from a mechanistic model.
    inputs:
        germ_response (Vector): germination fractions per time
        times (Vector): time frames for germination fractions (in hours)
        check_stderr (Bool): whether to perform a standard error check
        p0 (Vector): initial parameter guesses
    outputs:
        p_max (Float): maximum germination percentage
        τ (Float): half-saturation time for germination
        ν (Float): design parameter
        rmse (Float): root mean-squared error
    """
    function fit_dantigny_to_germination_curve(germ_response, times; p0=nothing)

        dantigny_wrapper(t, p) = dantigny.(t, 1 / (1 + exp(-p[1])), exp(p[2]), exp(p[3])) # [p_max, τ, ν] after transformations to become strictly positive

        # Initial guesses
        if isnothing(p0)
            p0 = [0.5, times[round(Int, 0.5 * length(times))], 2.0]
        end

        # Transform to strictly positive scales
        p0[1] = log(p0[1]) - log1p(-p0[1])
        p0[2] = log(p0[2])
        p0[3] = log(p0[3])

        fit = LsqFit.curve_fit(dantigny_wrapper, times, germ_response, p0)
        params = coef(fit)
        rmse = sqrt(mean(residuals(fit) .^ 2))
        # param_uncertainties = try
        #     stderror(fit)
        # catch e
        #     if isa(e, LinearAlgebra.LAPACKException)
        #         # Matrix is singular - return NaN or zeros
        #         fill(Inf, length(params))
        #     else
        #         rethrow(e)
        #     end
        # end

        # Transform parameters
        params[1] = 1 / (1 + exp(-params[1]))
        params[2] = exp(params[2])
        params[3] = exp(params[3])

        return params, rmse #, param_uncertainties
    end

    function fit_dantigny_time_shifted_to_germination_curve(germ_response, times; p0=nothing)

        dantigny_wrapper(t, p) = dantigny_time_shifted.(t, 1 / (1 + exp(-p[1])), exp(p[2]), exp(p[3]), p[4]) # [p_max, τ, ν, delta] after transformations to become strictly positive

        # Initial guesses
        if isnothing(p0)
            p0 = [0.5, times[round(Int, 0.5 * length(times))], 2.0, 0.0]
        end

        # Transform to strictly positive scales
        p0[1] = log(p0[1]) - log1p(-p0[1])
        p0[2] = log(p0[2])
        p0[3] = log(p0[3])
        # p0[4] = log(p0[4])

        fit = LsqFit.curve_fit(dantigny_wrapper, times, germ_response, p0)
        params = coef(fit)
        rmse = sqrt(mean(residuals(fit) .^ 2))
        # param_uncertainties = try
        #     stderror(fit)
        # catch e
        #     if isa(e, LinearAlgebra.LAPACKException)
        #         # Matrix is singular - return NaN or zeros
        #         fill(Inf, length(params))
        #     else
        #         rethrow(e)
        #     end
        # end

        # Transform parameters
        params[1] = 1 / (1 + exp(-params[1]))
        params[2] = exp(params[2])
        params[3] = exp(params[3])
        # params[4] = exp(params[4])

        return params, rmse #, param_uncertainties
    end

    # function get_smart_initial_guess(germ_response, times)
    #     p_max_guess = maximum(germ_response) * 0.98
    #     p_norm = germ_response ./ p_max_guess
    #     idx_half = searchsortedfirst(p_norm, 0.5)
        
    #     τ_guess = if idx_half > 1 && idx_half <= length(times)
    #         times[idx_half]
    #     else
    #         times[Int(0.5 * length(times))]
    #     end
        
    #     ν_guess = if idx_half > 3 && idx_half < length(times) - 3
    #         Δp = p_norm[idx_half + 3] - p_norm[idx_half - 3]
    #         Δt = times[idx_half + 3] - times[idx_half - 3]
    #         max(0.3, min(4.0, Δp / Δt))
    #     else
    #         2.0
    #     end
        
    #     return [p_max_guess, τ_guess, ν_guess]
    # end

    # function fit_dantigny_to_germination_curve(
    #     germ_response, times; 
    #     p0=nothing,
    #     compute_uncertainties=false
    # )
        
    #     dantigny_wrapper(t, p) = @. (1 / (1 + exp(-p[1]))) * (1 - 1 / (1 + (t / exp(p[2]))^exp(p[3])))

    #     # Better initial guess
    #     if isnothing(p0)
    #         p_max_phys = maximum(germ_response) * 0.98
    #         p_norm = germ_response ./ p_max_phys
    #         idx_half = searchsortedfirst(p_norm, 0.5)
    #         τ_phys = idx_half > 1 ? times[idx_half] : times[Int(0.5 * length(times))]
            
    #         # Transform to optimization space
    #         p0 = [
    #             log(p_max_phys) - log1p(-p_max_phys),
    #             log(τ_phys),
    #             log(2.0)
    #         ]
    #     end

    #     fit = LsqFit.curve_fit(
    #         dantigny_wrapper, times, germ_response, p0
    #     )
        
    #     params_opt = coef(fit)
        
    #     # Transform back to physical space
    #     params = [
    #         1 / (1 + exp(-params_opt[1])),
    #         exp(params_opt[2]),
    #         exp(params_opt[3])
    #     ]
        
    #     rmse = sqrt(mean(residuals(fit) .^ 2))

    #     return params, rmse
    # end


    """
    Fit a selected germination model to the data.
    inputs:
        model_type (String): model type to fit
        def_params (Dict): default parameter values
        dantigny_data (Matrix): time-dependent data from varying inducers and spore densities to fit
        times (Vector): time points
        sources (Vector): carbon sources
        densities (Vector): spore densities
        bounds_dict (Dict): bounds for the free parameters
        max_steps (Int): maximum number of steps for the optimization
        debug (Bool) - whether to print additional debugging messages
    outputs:
        params_out (Dict): optimized parameters
    """
    function fit_model_to_data(model_type, def_params, dantigny_data, times, sources, densities, bounds_dict; max_steps=10000, debug=false)

        models = load_model_collection()

        @argcheck model_type in models[1]

        # Reshape input
        densities_tile = repeat(densities, outer=[1, length(sources), length(times)])
        densities_tile = permutedims(densities_tile, (2, 1, 3))
        times_tile = repeat(times, outer=[1, length(sources), length(densities)])
        times_tile = permutedims(times_tile, (2, 3, 1))

        # Determine number of nodes depending on the integral dimension
        if model_type in ["independent", "inhibitor", "inhibitor_thresh", "inhibitor_perm", 
                            "combined_inhibitor", "combined_inhibitor_thresh", "combined_inhibitor_perm"]
            n_nodes = 36 # 2D integral
        elseif model_type in ["inducer", "inducer_thresh", "inducer_signal",
                            "combined_inducer", "combined_inducer_thresh", "combined_inducer_signal", "special_independent",
                            "inhibitor_thresh_inducer_signal", "combined_inhibitor_thresh_inducer_signal", "combined_inhibitor_thresh_inducer_thresh", "combined_inhibitor_thresh_signal_inducer_thresh"]
            n_nodes = 10 # 3D integral
        elseif model_type in ["special_inducer", "special_combined", "special_thresh", "special_signal"]
            n_nodes = 6 # 4D integral
        elseif startswith(model_type, "feedback")
            n_nodes = 512#1024
        end
        println("Number of nodes/samples: ", n_nodes)

        model_type_split = split(model_type, "_")

        gh_integral = false
        if model_type_split[1] == "feedback"
            if (haskey(bounds_dict, :μ_γ) && haskey(bounds_dict, :μ_ω))
                sample_dim = 5
            else
                sample_dim = 4
            end
            sobol_pts = QuasiMonteCarlo.sample(n_nodes, sample_dim, SobolSample())
        else
            gh_integral = true

            # Gauss-Hermite nodes
            ghnodes, ghweights = gausshermite(n_nodes)
            u = √2 .* ghnodes
            hw = ghweights ./ √π
        end

        # Unpack means and stds and weight samples
        μ_ξ = def_params[:μ_ξ]
        σ_ξ = def_params[:σ_ξ]
        μ_ξ_log = log(μ_ξ^2 / sqrt(σ_ξ^2 + μ_ξ^2))
        σ_ξ_log = sqrt(log(σ_ξ^2 / μ_ξ^2 + 1))
        if gh_integral ξ = exp.(μ_ξ_log .+ σ_ξ_log .* u) end

        if haskey(def_params, :μ_κ)
            μ_κ = def_params[:μ_κ]
            σ_κ = def_params[:σ_κ]
            μ_κ_log = log(μ_κ^2 / sqrt(σ_κ^2 + μ_κ^2))
            σ_κ_log = sqrt(log(σ_κ^2 / μ_κ^2 + 1))
            if gh_integral
                κ = exp.(μ_κ_log .+ σ_κ_log .* u)

                ξ2, κ2 = meshgrid(ξ, κ)
            end
        end

        # Multi-dimensional Gauss-Hermite weights
        if gh_integral
            W = hw * hw'
            W3 = reshape(hw, n_nodes,1,1) .* reshape(hw, 1,n_nodes,1) .* reshape(hw, 1,1,n_nodes)
            W4 = reshape(hw, n_nodes,1,1,1) .* reshape(hw, 1,n_nodes,1,1) .* reshape(hw, 1,1,n_nodes,1) .* reshape(hw, 1,1,1,n_nodes)
        end

        # Construct distributions and geometric samples
        if !gh_integral
            dist_ξ = LogNormal(μ_ξ_log, σ_ξ_log)
            dist_κ = LogNormal(μ_κ_log, σ_κ_log)

            samples_ξ = clamp_inplace!(quantile(dist_ξ, sobol_pts[1,:]))
            samples_κ = clamp_inplace!(quantile(dist_κ, sobol_pts[2,:]))

            samples_AV = compute_spore_area_and_volume_from_dia.(2 .* samples_ξ)
            samples_A, samples_Vₛ = (getindex.(samples_AV, 1), getindex.(samples_AV, 2))
            # samples_V_out = 1.0/ρₛ .- samples_Vₛ
            samples_V_ps = compute_ps_layer_volume.(samples_ξ, def_params[:d_hp], samples_κ)
        end

        # Define number of specific parameter occurrences (general or per carbon source)
        n_src = length(sources)
        param_occurrences_dict = Dict(
            :s_max => n_src,
            :b_max => 1,
            :Pₛ => 1,
            :Pₛ_cs => n_src,
            :k_C => n_src,
            :k_I => n_src,
            :K_I => n_src,
            :K_cC => n_src,
            :K_cI => n_src,
            :n => n_src,
            :μ_γ => 1,
            :δ_γ => 1,
            :μ_ω => n_src,
            :δ_ω => n_src,
            :μ_ψ => 1,	
            :δ_ψ => 1,
            :μ_α => 1,
            :δ_α => 1
        )

        # Find model index and parameter keys
        model_index = findfirst(models[1])
        param_keys = models[2][model_index]
        
        if model_type == "independent"
            # Independent inducer/inhibitor
            println("Model: independent factors")
            wrapper = (inputs, params) -> gresp_independent_factors_gh(
                u, W,
                inputs[1], #t
                inputs[2], #ρₛ
                def_params[:c₀_cs],
                def_params[:d_hp],
                ξ2,
                κ2,
                params[1], #Pₛ
                params[2], #Pₛ_cs
                params[3], #K_cC
                params[4], #μ_γ
                params[4] * exp(params[5]), # σ_γ = μ_γ * exp(δ_γ)
                params[6], #μ_ω
                params[6] * exp(params[7]) # σ_ω = μ_ω * exp(δ_ω)
            )

        elseif model_type == "inhibitor_thresh" # Inducer shifts inhibition threshold
            println("Model: inducer-modulated inhibitor (threshold)")
            wrapper = (inputs, params) -> Main.gresp_inducer_dep_inhibitor_thresh_gh(
                u, W,
                inputs[1], #t
                inputs[2], #ρₛ
                def_params[:c₀_cs],
                def_params[:d_hp],
                ξ2,
                κ2,
                params[1], #Pₛ
                params[2], #Pₛ_cs
                exp(params[3]), #k_C
                params[4], #K_cC
                params[5], #μ_γ
                params[5] * exp(params[6]) # σ_γ = μ_γ * exp(δ_γ)
            )

        elseif model_type == "inducer" # Inhibitor shifts induction threshold and modulates inducer signal strength
            println("Model: inhibitor-modulated inducer (combined)")
            wrapper = (inputs, params) -> Main.gresp_inhibitor_dep_inducer_gh(
                u, W3,
                inputs[1], #t
                inputs[2], #ρₛ
                def_params[:c₀_cs],
                def_params[:d_hp],
                ξ2,
                κ2,
                params[1], #Pₛ
                params[2], #Pₛ_cs
                exp(params[3]), #k_I
                params[4], #K_cI
                params[5], #K_cC
                params[6], #K_I
                params[7], #n
                params[8], #μ_ω
                params[8] * exp(params[9]), # σ_ω = μ_ω * exp(δ_ω)
                params[10], #μ_ψ
                params[10] * exp(params[11]) # σ_ψ = μ_ψ * exp(δ_ψ)
            )
                
        elseif model_type == "inducer_thresh" # Inhibitor shifts induction threshold
            println("Model: inhibitor-modulated inducer (threshold)")
            wrapper = (inputs, params) -> Main.gresp_inhibitor_dep_inducer_thresh_gh(
                u, W3,
                inputs[1], #t
                inputs[2], #ρₛ
                def_params[:c₀_cs],
                def_params[:d_hp],
                ξ2,
                κ2,
                params[1], #Pₛ
                params[2], #Pₛ_cs
                params[3], #K_cI
                params[4], #K_cC
                exp(params[5]), #k_I
                params[6], #μ_ω
                params[6] * exp(params[7]), # σ_ω = μ_ω * exp(δ_ω)
                params[8], #μ_ψ
                params[8] * exp(params[9]) # σ_ψ = μ_ψ * exp(δ_ψ)
            )
            
        elseif model_type == "inducer_signal" # Inhibitor shifts induction threshold
            println("Model: inhibitor-modulated inducer (signal)")
            wrapper = (inputs, params) -> Main.gresp_inhibitor_dep_inducer_signal_gh(
                u, W3,
                inputs[1], #t
                inputs[2], #ρₛ
                def_params[:c₀_cs],
                def_params[:d_hp],
                ξ2,
                κ2,
                params[1], #Pₛ
                params[2], #Pₛ_cs
                params[3], #K_cC
                params[4], #K_I
                params[5], #n
                params[6], #μ_ω
                params[6] * exp(params[7]), # σ_ω = μ_ω * exp(δ_ω)
                params[8], #μ_ψ
                params[8] * exp(params[9]) # σ_ψ = μ_ψ * exp(δ_ψ)
            )

        elseif model_type == "combined_inhibitor"
            println("Model: inducer-modulated inhibitor (combined)")
            wrapper = (inputs, params) -> Main.gresp_inducer_dep_inhibitor_2_factor_gh(
                u, W,
                inputs[1], #t
                inputs[2], #ρₛ
                def_params[:c₀_cs],
                def_params[:d_hp],
                ξ2,
                κ2,
                params[1], #Pₛ,
                params[2], #Pₛ_cs,
                params[3], #K_cC,
                exp(params[4]), #k_C,
                params[5], #μ_γ,
                params[5] * exp(params[6]), # σ_γ = μ_γ * exp(δ_γ)
                params[7], #μ_ω,
                params[7] * exp(params[8]) # σ_ω = μ_ω * exp(δ_ω)
            )

        elseif model_type == "combined_inhibitor_thresh"
            println("Model: inducer-modulated inhibitor (combined)")
            wrapper = (inputs, params) -> Main.gresp_inducer_dep_inhibitor_thresh_2_factor_gh(
                u, W,
                inputs[1], #t
                inputs[2], #ρₛ
                def_params[:c₀_cs],
                def_params[:d_hp],
                ξ2,
                κ2,
                params[1], #Pₛ
                params[2], #Pₛ_cs
                exp(params[3]), #k_C
                params[4], #K_cC
                params[5], #μ_γ
                params[5] * exp(params[6]), # σ_γ = μ_γ * exp(δ_γ)
                params[7], #μ_ω
                params[7] * exp(params[8]) # σ_ω = μ_ω * exp(δ_ω)
            )

        elseif model_type == "combined_inducer"
            println("Model: inhibitor-modulated inducer (combined)")
            wrapper = (inputs, params) -> Main.gresp_inhibitor_dep_inducer_2_factor_gh(
                u, W3,
                inputs[1], #t
                inputs[2], #ρₛ
                def_params[:c₀_cs],
                def_params[:d_hp],
                ξ2,
                κ2,
                params[1], #Pₛ
                params[2], #Pₛ_cs
                params[3], #K_cI
                params[4], #K_cC
                params[5], #K_I
                params[6], #n
                exp(params[7]), #k_I
                params[8], #μ_γ
                params[8] * exp(params[9]), # σ_γ = μ_γ * exp(δ_γ)
                params[10], #μ_ω
                params[10] * exp(params[11]), # σ_ω = μ_ω * exp(δ_ω)
                params[12], #μ_ψ
                params[12] * exp(params[13]) # σ_ψ = μ_ψ * exp(δ_ψ)
            )

        elseif model_type == "combined_inducer_thresh"
            println("Model: inhibitor-modulated inducer (threshold)")
            wrapper = (inputs, params) -> Main.gresp_inhibitor_dep_inducer_thresh_2_factor_gh(
                u, W3,
                inputs[1], #t
                inputs[2], #ρₛ
                def_params[:c₀_cs],
                def_params[:d_hp],
                ξ2,
                κ2,
                params[1], #Pₛ
                params[2], #Pₛ_cs
                params[3], #K_cI
                params[4], #K_cC
                exp(params[5]), #k_I
                params[6], #μ_γ
                params[6] * exp(params[7]), # σ_γ = μ_γ * exp(δ_γ)
                params[8], #μ_ω
                params[8] * exp(params[9]), # σ_ω = μ_ω * exp(δ_ω)
                params[10], #μ_ψ
                params[10] * exp(params[11]) # σ_ψ = μ_ψ * exp(δ_ψ)
            )

        elseif model_type == "combined_inducer_signal"
            println("Model: inhibitor-modulated inducer (signal)")
            wrapper = (inputs, params) -> Main.gresp_inhibitor_dep_inducer_signal_2_factor_gh(
                u, W3,
                inputs[1], #t
                inputs[2], #ρₛ
                def_params[:c₀_cs],
                def_params[:d_hp],
                ξ2,
                κ2,
                params[1], #Pₛ
                params[2], #Pₛ_cs
                params[3], #K_cC
                params[4], #K_I
                params[5], #n
                params[6], #μ_γ
                params[6] * exp(params[7]), # σ_γ = μ_γ * exp(δ_γ)
                params[8], #μ_ω
                params[8] * exp(params[9]), # σ_ω = μ_ω * exp(δ_ω)
                params[10], #μ_ψ
                params[10] * exp(params[11]) # σ_ψ = μ_ψ * exp(δ_ψ)
            )
            # param_keys = [:Pₛ, :Pₛ_cs, :K_cC, :K_I, :n, :μ_γ, :δ_γ, :μ_ω, :δ_ω, :μ_ψ, :δ_ψ]
            # param_occurrences = [1, n_src, n_src, n_src, n_src, 1, 1, n_src, n_src, 1, 1]

        elseif model_type == "special_inducer"
            println("Model: inhibitor-modulated inducer (combined) with varying permeability")
            wrapper = (inputs, params) -> Main.gresp_inducer_var_perm_gh(
                u, W4,
                inputs[1], #t
                inputs[2], #ρₛ
                def_params[:c₀_cs],
                def_params[:d_hp],
                ξ2,
                κ2,
                params[1], #Pₛ
                params[2], #Pₛ_cs
                params[3], #K_cC
                params[4], #K_I
                exp(params[5]), #k_I
                params[6], #n
                params[7], #μ_ω
                params[7] * exp(params[8]), # σ_ω = μ_ω * exp(δ_ω)
                params[9], #μ_ψ
                params[9] * exp(params[10]), # σ_ψ = μ_ψ * exp(δ_ψ)
                params[11], #μ_α
                params[11] * exp(params[12]) # σ_α = μ_α * exp(δ_α)
            )

        elseif model_type == "special_independent"
            println("Model: independent factors with varying permeability")
            wrapper = (inputs, params) -> Main.gresp_independent_factors_var_perm_gh(
                u, W3,
                inputs[1], #t
                inputs[2], #ρₛ
                def_params[:c₀_cs],
                def_params[:d_hp],
                ξ2,
                κ2,
                params[1], #Pₛ
                params[2], #Pₛ_cs
                params[3], #K_cC
                params[4], #μ_γ
                params[4] * exp(params[5]), # σ_γ = μ_γ * exp(δ_γ)
                params[6], #μ_ω
                params[6] * exp(params[7]), # σ_ω = μ_ω * exp(δ_ω)
                params[8], #μ_α
                params[8] * exp(params[9]) # σ_α = μ_α * exp(δ_α)
            )

        elseif model_type == "special_combined"
            println("Model: 2-factor germination with inhibitor-modulated inducer (combined) and varying permeability")
            wrapper = (inputs, params) -> Main.gresp_inducer_2_factors_var_perm_gh(
                u, W4,
                inputs[1], #t
                inputs[2], #ρₛ
                def_params[:c₀_cs],
                def_params[:d_hp],
                ξ2,
                κ2,
                params[1], #Pₛ
                params[2], #Pₛ_cs
                params[3], #K_cC
                params[4], #K_I
                exp(params[5]), #k_I
                params[6], #n
                params[7], #μ_γ
                params[7] * exp(params[8]), # σ_γ = μ_γ * exp(δ_γ)
                params[9], #μ_ω
                params[9] * exp(params[10]), # σ_ω = μ_ω * exp(δ_ω)
                params[11], #μ_ψ
                params[11] * exp(params[12]), # σ_ψ = μ_ψ * exp(δ_ψ)
                params[13], #μ_α
                params[13] * exp(params[14]) # σ_α = μ_α * exp(δ_α)
            )

        elseif model_type == "special_combined_thresh"
            println("Model: 2-factor germination with inhibitor-modulated inducer (threshold) and varying permeability")
            wrapper = (inputs, params) -> Main.gresp_inducer_thresh_2_factors_var_perm_gh(
                u, W4,
                inputs[1], #t
                inputs[2], #ρₛ
                def_params[:c₀_cs],
                def_params[:d_hp],
                ξ2,
                κ2,
                params[1], #Pₛ
                params[2], #Pₛ_cs
                params[3], #K_cC
                exp(params[4]), #k_I
                params[5], #μ_γ
                params[5] * exp(params[6]), # σ_γ = μ_γ * exp(δ_γ)
                params[7], #μ_ω
                params[7] * exp(params[8]), # σ_ω = μ_ω * exp(δ_ω)
                params[9], #μ_ψ
                params[9] * exp(params[10]), # σ_ψ = μ_ψ * exp(δ_ψ)
                params[11], #μ_α
                params[11] * exp(params[12]) # σ_α = μ_α * exp(δ_α)
            )

        elseif model_type == "special_combined_signal"
            println("Model: 2-factor germination with inhibitor-modulated inducer (signal) and varying permeability")
            wrapper = (inputs, params) -> Main.gresp_inducer_signal_2_factors_var_perm_gh(
                u, W4,
                inputs[1], #t
                inputs[2], #ρₛ
                def_params[:c₀_cs],
                def_params[:d_hp],
                ξ2,
                κ2,
                params[1], #Pₛ
                params[2], #Pₛ_cs
                params[3], #K_cC
                params[4], #K_I
                params[5], #n
                params[6], #μ_γ
                params[6] * exp(params[7]), # σ_γ = μ_γ * exp(δ_γ)
                params[8], #μ_ω
                params[8] * exp(params[9]), # σ_ω = μ_ω * exp(δ_ω)
                params[10], #μ_ψ
                params[10] * exp(params[11]), # σ_ψ = μ_ψ * exp(δ_ψ)
                params[12], #μ_α
                params[12] * exp(params[13]) # σ_α = μ_α * exp(δ_α)
            )
        
            # ==================== FEEDBACK MODELS ========================
        elseif model_type == "feedback_inhibitor_inducer_perm" # A
            println("Model: inhibitor-dependent germination with inducer-dependent inhibitor/inducer permeability")
            wrapper = (V_out, params) -> Main.gresp_feedback(
                Main.ode_inducer_dependent_perm!,
                Main.thresh_criterion_inhibitor,
                sobol_pts,
                times,
                [samples_A, samples_Vₛ, V_out, samples_V_ps],
                def_params[:c₀_cs],
                [params[1]], # s_max
                params[2], # Pₛ_I
                params[3], # Pₛ_C
                [nothing, params[4]], # K_cC
                params[5], # μ_ψ
                params[5] * exp(params[6]), # σ_ψ = μ_ψ * exp(δ_ψ)
                [params[7]], # μ_γ
                [params[7] * exp(params[8])] # σ_γ = μ_γ * exp(δ_γ)
            )

        elseif model_type == "feedback_combined_inducer_perm" # A
            println("Model: 2-factor germination with inducer-dependent inhibitor/inducer permeability")
            wrapper = (V_out, params) -> Main.gresp_feedback(
                Main.ode_inducer_dependent_perm!,
                Main.thresh_criterion_combined,
                sobol_pts,
                times,
                [samples_A, samples_Vₛ, V_out, samples_V_ps],
                def_params[:c₀_cs],
                [params[1]], # s_max
                params[2], # Pₛ_I
                params[3], # Pₛ_C
                [nothing, params[4]], # K_cC
                params[5], # μ_ψ
                params[5] * exp(params[6]), # σ_ψ = μ_ψ * exp(δ_ψ)
                [params[7], params[9]], # [μ_γ, μ_ω]
                [params[7] * exp(params[8]), params[9] * exp(params[10])] # [σ_γ = μ_γ * exp(δ_γ), σ_ω = μ_ω * exp(δ_ω)]
            )
        
        elseif model_type == "feedback_inducer_inhibitor_perm" # D
            println("Model: Inducer-dependent germination with inhibitor-dependent inhibitor/inducer permeability")
            wrapper = (V_out, params) -> Main.gresp_feedback(
                Main.ode_inhibitor_dependent_perm!,
                Main.thresh_criterion_inducer,
                sobol_pts,
                times,
                [samples_A, samples_Vₛ, V_out, samples_V_ps],
                def_params[:c₀_cs],
                [params[1]], # b_max
                params[2], # Pₛ_I
                params[3], # Pₛ_C
                [params[4], params[5]], # K_cI
                params[6], # μ_ψ
                params[6] * exp(params[7]), # σ_ψ = μ_ψ * exp(δ_ψ)
                [params[8]], # [μ_ω]
                [params[8] * exp(params[9])] # [σ_ω = μ_ω * exp(δ_ω)]
            )
        
        elseif model_type == "feedback_combined_inhibitor_perm" # D
            println("Model: 2-factor germination with inhibitor-dependent inhibitor/inducer permeability")
            wrapper = (V_out, params) -> Main.gresp_feedback(
                Main.ode_inhibitor_dependent_perm!,
                Main.thresh_criterion_combined,
                sobol_pts,
                times,
                [samples_A, samples_Vₛ, V_out, samples_V_ps],
                def_params[:c₀_cs],
                [params[1]], # b_max
                params[2], # Pₛ_I
                params[3], # Pₛ_C
                [params[4], params[5]], # K_cI
                params[6], # μ_ψ
                params[6] * exp(params[7]), # σ_ψ = μ_ψ * exp(δ_ψ)
                [params[8], params[10]], # [μ_γ, μ_ω]
                [params[8] * exp(params[9]), params[10] * exp(params[11])] # [σ_ω = μ_ω * exp(δ_ω)]
            )
        
        elseif model_type == "feedback_inhibitor_inducer_perm_thresh" # AB
            println("Model: Inhibitor-dependent germination with inducer-dependent permeability and inhibition threshold")
            wrapper = (V_out, params) -> Main.gresp_feedback(
                Main.ode_inducer_dependent_perm!,
                Main.thresh_criterion_inhibitor_shift,
                sobol_pts,
                times,
                [samples_A, samples_Vₛ, V_out, samples_V_ps],
                def_params[:c₀_cs],
                [params[1]], # s_max
                params[2], # Pₛ_I
                params[3], # Pₛ_C
                [nothing, params[4]], # K_cC
                params[5], # μ_ψ
                params[5] * exp(params[6]), # σ_ψ = μ_ψ * exp(δ_ψ)
                [params[7]], # μ_γ
                [params[7] * exp(params[8])]; # σ_γ = μ_γ * exp(δ_γ)
                ks=[exp(params[9])] # k_C
            )

        elseif model_type == "feedback_combined_inducer_perm_thresh" # AB
            println("Model: 2-factor germination with inducer-dependent permeability and inhibition threshold")
            wrapper = (V_out, params) -> Main.gresp_feedback(
                Main.ode_inducer_dependent_perm!,
                Main.thresh_criterion_combined_inhibitor_shift,
                sobol_pts,
                times,
                [samples_A, samples_Vₛ, V_out, samples_V_ps],
                def_params[:c₀_cs],
                [params[1]], # s_max
                params[2], # Pₛ_I
                params[3], # Pₛ_C
                [nothing, params[4]], # K_cC
                params[5], # μ_ψ
                params[5] * exp(params[6]), # σ_ψ = μ_ψ * exp(δ_ψ)
                [params[7], params[9]], # [μ_γ, μ_ω]
                [params[7] * exp(params[8]), params[9] * exp(params[10])]; # [σ_γ = μ_γ * exp(δ_γ), σ_ω = μ_ω * exp(δ_ω)]
                ks=[exp(params[11])] # k_C
            )
        
        elseif model_type == "feedback_inhibitor_inducer_perm_inhibitor_signal" # AC
            println("Model: Inhibitor-dependent germination with inducer-dependent permeability and inhibitor-dependent induction signal")
            wrapper = (V_out, params) -> Main.gresp_feedback(
                Main.ode_inducer_dependent_perm_inhibitor_dependent_signal!,
                Main.thresh_criterion_inhibitor,
                sobol_pts,
                times,
                [samples_A, samples_Vₛ, V_out, samples_V_ps],
                def_params[:c₀_cs],
                [params[1]], # s_max
                params[2], # Pₛ_I
                params[3], # Pₛ_C
                [nothing, params[4], params[5]], # K_cC, K_I
                params[6], # μ_ψ
                params[6] * exp(params[7]), # σ_ψ = μ_ψ * exp(δ_ψ)
                [params[8]], # μ_γ
                [params[8] * exp(params[9])]; # σ_γ = μ_γ * exp(δ_γ)
                n=params[10] # n
            )

        elseif model_type == "feedback_inducer_inducer_perm_inhibitor_signal" # AC
            println("Model: Inducer-dependent germination with inducer-dependent permeability and inhibitor-dependent induction signal")
            wrapper = (V_out, params) -> Main.gresp_feedback(
                Main.ode_inducer_dependent_perm_inhibitor_dependent_signal!,
                Main.thresh_criterion_inducer,
                sobol_pts,
                times,
                [samples_A, samples_Vₛ, V_out, samples_V_ps],
                def_params[:c₀_cs],
                [params[1]], # s_max
                params[2], # Pₛ_I
                params[3], # Pₛ_C
                [nothing, params[4], params[5]], # K_cC, K_I
                params[6], # μ_ψ
                params[6] * exp(params[7]), # σ_ψ = μ_ψ * exp(δ_ψ)
                [params[8]], # μ_ω
                [params[8] * exp(params[9])]; # σ_ω = μ_ω * exp(δ_ω)
                n=params[10] # n
            )
            
        elseif model_type == "feedback_combined_inducer_perm_inhibitor_signal" # AC
            println("Model: 2-factor germination with inducer-dependent permeability and inhibitor-dependent induction signal")
            wrapper = (V_out, params) -> Main.gresp_feedback(
                Main.ode_inducer_dependent_perm_inhibitor_dependent_signal!,
                Main.thresh_criterion_combined,
                sobol_pts,
                times,
                [samples_A, samples_Vₛ, V_out, samples_V_ps],
                def_params[:c₀_cs],
                [params[1]], # s_max
                params[2], # Pₛ_I
                params[3], # Pₛ_C
                [nothing, params[4], params[5]], # K_cC, K_I
                params[6], # μ_ψ
                params[6] * exp(params[7]), # σ_ψ = μ_ψ * exp(δ_ψ)
                [params[8], params[10]], # [μ_γ, μ_ω]
                [params[8] * exp(params[9]), params[10] * exp(params[11])]; # [σ_γ = μ_γ * exp(δ_γ), σ_ω = μ_ω * exp(δ_ω)]
                n=params[12] # n
            )
            
        elseif model_type == "feedback_inhibitor_inhibitor_inducer_perm" # AD
            println("Model: Inhibitor-dependent germination with inhibitor- and inducer-dependent inhibitor/inducer permeability")
            wrapper = (V_out, params) -> Main.gresp_feedback(
                Main.ode_inducer_and_inhibitor_dependent_perm!,
                Main.thresh_criterion_inhibitor,
                sobol_pts,
                times,
                [samples_A, samples_Vₛ, V_out, samples_V_ps],
                def_params[:c₀_cs],
                [params[1], params[2]], # b_max, s_max
                params[3], # Pₛ_I
                params[4], # Pₛ_C
                [params[5], params[6]], # K_cI, K_cC
                params[7], # μ_ψ
                params[7] * exp(params[8]), # σ_ψ = μ_ψ * exp(δ_ψ)
                [params[9]], # μ_γ
                [params[9] * exp(params[10])] # σ_γ = μ_γ * exp(δ_γ)
            )
        
        elseif model_type == "feedback_inducer_inhibitor_inducer_perm" # AD
            println("Model: Inducer-dependent germination with inhibitor- and inducer-dependent inhibitor/inducer permeability")
            wrapper = (V_out, params) -> Main.gresp_feedback(
                Main.ode_inducer_and_inhibitor_dependent_perm!,
                Main.thresh_criterion_inducer,
                sobol_pts,
                times,
                [samples_A, samples_Vₛ, V_out, samples_V_ps],
                def_params[:c₀_cs],
                [params[1], params[2]], # b_max, s_max
                params[3], # Pₛ_I
                params[4], # Pₛ_C
                [params[5], params[6]], # K_cI, K_cC
                params[7], # μ_ψ
                params[7] * exp(params[8]), # σ_ψ = μ_ψ * exp(δ_ψ)
                [params[9]], # μ_ω
                [params[9] * exp(params[10])] # σ_ω = μ_ω * exp(δ_ω)
            )
        
        elseif model_type == "feedback_combined_inhibitor_inducer_perm" # AD
            println("Model: 2-factor germination with inhibitor- and inducer-dependent inhibitor/inducer permeability")
            wrapper = (V_out, params) -> Main.gresp_feedback(
                Main.ode_inducer_and_inhibitor_dependent_perm!,
                Main.thresh_criterion_combined,
                sobol_pts,
                times,
                [samples_A, samples_Vₛ, V_out, samples_V_ps],
                def_params[:c₀_cs],
                [params[1], params[2]], # b_max, s_max
                params[3], # Pₛ_I
                params[4], # Pₛ_C
                [params[5], params[6]], # K_cI, K_cC
                params[7], # μ_ψ
                params[7] * exp(params[8]), # σ_ψ = μ_ψ * exp(δ_ψ)
                [params[9], params[11]], # μ_γ, μ_ω
                [params[9] * exp(params[10]), params[11] * exp(params[12])] # σ_γ = μ_γ * exp(δ_γ), σ_ω = μ_ω * exp(δ_ω)
            )
        
        elseif model_type == "feedback_inducer_inhibitor_thresh_inducer_perm" # AE
            println("Model: Inducer-dependent germination with inhibitor-dependent induction threshold and inducer-dependent inhibitor/inducer permeability")
            wrapper = (V_out, params) -> Main.gresp_feedback(
                Main.ode_inducer_dependent_perm!,
                Main.thresh_criterion_inducer_shift,
                sobol_pts,
                times,
                [samples_A, samples_Vₛ, V_out, samples_V_ps],
                def_params[:c₀_cs],
                [params[1]], # s_max
                params[2], # Pₛ_I
                params[3], # Pₛ_C
                [params[4], params[5]], # K_cI, K_cC
                params[6], # μ_ψ
                params[6] * exp(params[7]), # σ_ψ = μ_ψ * exp(δ_ψ)
                [params[8]], # μ_ω
                [params[8] * exp(params[9])]; # σ_ω = μ_ω * exp(δ_ω)
                ks=[exp(params[10])] # k_I
            )

        elseif model_type == "feedback_combined_inhibitor_thresh_inducer_perm" # AE
            println("Model: 2-factor germination with inhibitor-dependent induction threshold and inducer-dependent inhibitor/inducer permeability")
            wrapper = (V_out, params) -> Main.gresp_feedback(
                Main.ode_inducer_dependent_perm!,
                Main.thresh_criterion_combined_inducer_shift,
                sobol_pts,
                times,
                [samples_A, samples_Vₛ, V_out, samples_V_ps],
                def_params[:c₀_cs],
                [params[1]], # s_max
                params[2], # Pₛ_I
                params[3], # Pₛ_C
                [params[4], params[5]], # K_cI, K_cC
                params[6], # μ_ψ
                params[6] * exp(params[7]), # σ_ψ = μ_ψ * exp(δ_ψ)
                [params[8], params[10]], # μ_ω
                [params[8] * exp(params[9]), params[10] * exp(params[11])]; # σ_ω = μ_ω * exp(δ_ω)
                ks=[exp(params[12])] # k_I
            )
            
        elseif model_type == "inhibitor_thresh_inducer_signal" # BC
            println("Model: Inhibitor-dependent germination with inducer-dependent inhibition threshold and inhibitor-dependent induction signal")
            wrapper = (inputs, params) -> Main.gresp_inh_dep_ind_signal_ind_dep_inh_thresh_gh(
                u, W3,
                inputs[1], #t
                inputs[2], #ρₛ
                def_params[:c₀_cs],
                def_params[:d_hp],
                ξ2,
                κ2,
                params[1], #Pₛ
                params[2], #Pₛ_cs
                exp(params[3]), #k_C
                params[4], #K_cC
                params[5], #K_I
                params[6], #n
                params[7], #μ_γ
                params[7] * exp(params[8]), #σ_γ
                params[9], #μ_ψ
                params[9] * exp(params[10]) #σ_ψ
            )

        elseif model_type == "combined_inhibitor_thresh_inducer_signal" # BC
            println("Model: 2-factor germination with inducer-dependent inhibition threshold and inhibitor-dependent induction signal")
            wrapper = (inputs, params) -> Main.gresp_inh_dep_ind_signal_ind_dep_inh_thresh_2_factor_gh(
                u, W3,
                inputs[1], #t
                inputs[2], #ρₛ
                def_params[:c₀_cs],
                def_params[:d_hp],
                ξ2,
                κ2,
                params[1], #Pₛ
                params[2], #Pₛ_cs
                exp(params[3]), #k_C
                params[4], #K_cC
                params[5], #K_I
                params[6], #n
                params[7], #μ_γ
                params[7] * exp(params[8]), #σ_γ
                params[9], #μ_ω
                params[9] * exp(params[10]), #σ_ω
                params[11], #μ_ψ
                params[11] * exp(params[12]) #σ_ψ
            )
            
        elseif model_type == "feedback_inhibitor_inducer_thresh_inhibitor_perm" # BD
            println("Model: Inhibitor-dependent germination with inducer-dependent inhibition threshold and inhibitor-dependent inhibitor/inducer permeability")
            wrapper = (V_out, params) -> Main.gresp_feedback(
                Main.ode_inhibitor_dependent_perm!,
                Main.thresh_criterion_inhibitor_shift,
                sobol_pts,
                times,
                [samples_A, samples_Vₛ, V_out, samples_V_ps],
                def_params[:c₀_cs],
                [params[1]], # b_max
                params[2], # Pₛ_I
                params[3], # Pₛ_C
                [params[4], params[5]], # K_cI, K_cC
                params[6], # μ_ψ
                params[6] * exp(params[7]), # σ_ψ = μ_ψ * exp(δ_ψ)
                [params[8]], # [μ_γ]
                [params[8] * exp(params[9])], # [σ_γ = μ_γ * exp(δ_γ)]
                ks=[exp(params[10])] # k_C
            )
        
        elseif model_type == "feedback_combined_inducer_thresh_inhibitor_perm" # BD
            println("Model: Inhibitor-dependent germination with inducer-dependent inhibition threshold and inhibitor-dependent inhibitor/inducer permeability")
            wrapper = (V_out, params) -> Main.gresp_feedback(
                Main.ode_inhibitor_dependent_perm!,
                Main.thresh_criterion_combined_inhibitor_shift,
                sobol_pts,
                times,
                [samples_A, samples_Vₛ, V_out, samples_V_ps],
                def_params[:c₀_cs],
                [params[1]], # b_max
                params[2], # Pₛ_I
                params[3], # Pₛ_C
                [params[4], params[5]], # K_cI, K_cC
                params[6], # μ_ψ
                params[6] * exp(params[7]), # σ_ψ = μ_ψ * exp(δ_ψ)
                [params[8], params[10]], # [μ_γ, μ_ω]
                [params[8] * exp(params[9]), params[10] * exp(params[11])], # [σ_γ = μ_γ * exp(δ_γ), σ_ω = μ_ω * exp(δ_ω)]
                ks=[exp(params[12])] # k_C
            )
        
        elseif model_type == "combined_inhibitor_thresh_inducer_thresh" # BE
            println("Model: inhibitor-modulated induction threshold / inducer-modulated inhibition threshold")
            wrapper = (inputs, params) -> Main.gresp_inh_dep_ind_thresh_ind_dep_inh_thresh_2_factor_gh(
                u, W3,
                inputs[1], #t
                inputs[2], #ρₛ
                def_params[:c₀_cs],
                def_params[:d_hp],
                ξ2,
                κ2,
                params[1], #Pₛ
                params[2], #Pₛ_cs
                params[3], #K_cI
                params[4], #K_cC
                exp(params[5]), #k_I
                exp(params[6]), #k_C
                params[7], #μ_γ
                params[7] * exp(params[8]), # σ_γ = μ_γ * exp(δ_γ)
                params[9], #μ_ω
                params[9] * exp(params[10]), # σ_ω = μ_ω * exp(δ_ω)
                params[11], #μ_ψ
                params[11] * exp(params[12]) # σ_ψ = μ_ψ * exp(δ_ψ)
            )

        elseif model_type == "feedback_inducer_inhibitor_perm_signal" # CD
            println("Model: Inhibitor-dependent germination with inducer-dependent inhibition threshold and inhibitor-dependent inhibitor/inducer permeability")
            wrapper = (V_out, params) -> Main.gresp_feedback(
                Main.ode_inhibitor_dependent_perm!,
                Main.thresh_criterion_inducer_signal,
                sobol_pts,
                times,
                [samples_A, samples_Vₛ, V_out, samples_V_ps],
                def_params[:c₀_cs],
                [params[1]], # b_max
                params[2], # Pₛ_I
                params[3], # Pₛ_C
                [params[4], params[5], params[6]], # K_cI, K_cC, K_I
                params[7], # μ_ψ
                params[7] * exp(params[8]), # σ_ψ = μ_ψ * exp(δ_ψ)
                [params[9]], # [μ_ω]
                [params[9] * exp(params[10])] # [σ_ω = μ_ω * exp(δ_ω)]
            )
        
        elseif model_type == "feedback_combined_inhibitor_perm_signal" # CD
            println("Model: 2-factor germination with inducer-dependent inhibition threshold and inhibitor-dependent inhibitor/inducer permeability")
            wrapper = (V_out, params) -> Main.gresp_feedback(
                Main.ode_inhibitor_dependent_perm!,
                Main.thresh_criterion_combined_inducer_signal,
                sobol_pts,
                times,
                [samples_A, samples_Vₛ, V_out, samples_V_ps],
                def_params[:c₀_cs],
                [params[1]], # b_max
                params[2], # Pₛ_I
                params[3], # Pₛ_C
                [params[4], params[5], params[6]], # K_cI, K_cC, K_I
                params[7], # μ_ψ
                params[7] * exp(params[8]), # σ_ψ = μ_ψ * exp(δ_ψ)
                [params[9], params[11]], # [μ_γ, μ_ω]
                [params[9] * exp(params[10]), params[11] * exp(params[12])], # [σ_γ = μ_γ * exp(δ_γ), σ_ω = μ_ω * exp(δ_ω)]
                n=params[13] # n
            )
        
        elseif model_type == "feedback_inducer_inhibitor_perm_thresh" # DE
            println("Model: Inhibitor-dependent germination with inhibitor-dependent inhibitor/inducer permeability and induction threshold")
            wrapper = (V_out, params) -> Main.gresp_feedback(
                Main.ode_inhibitor_dependent_perm!,
                Main.thresh_criterion_inducer_shift,
                sobol_pts,
                times,
                [samples_A, samples_Vₛ, V_out, samples_V_ps],
                def_params[:c₀_cs],
                [params[1]], # b_max
                params[2], # Pₛ_I
                params[3], # Pₛ_C
                [params[4], params[5]], # K_cI, K_cC
                params[6], # μ_ψ
                params[6] * exp(params[7]), # σ_ψ = μ_ψ * exp(δ_ψ)
                [params[8]], # [μ_ω]
                [params[8] * exp(params[9])], # [σ_ω = μ_ω * exp(δ_ω)]
                ks=[exp(params[10])] # k_I
            )
        
        elseif model_type == "feedback_combined_inhibitor_perm_thresh" # DE
            println("Model: Inhibitor-dependent germination with inhibitor-dependent inhibitor/inducer permeability and induction threshold")
            wrapper = (V_out, params) -> Main.gresp_feedback(
                Main.ode_inhibitor_dependent_perm!,
                Main.thresh_criterion_combined_inducer_shift,
                sobol_pts,
                times,
                [samples_A, samples_Vₛ, V_out, samples_V_ps],
                def_params[:c₀_cs],
                [params[1]], # b_max
                params[2], # Pₛ_I
                params[3], # Pₛ_C
                [params[4], params[5]], # K_cI, K_cC
                params[6], # μ_ψ
                params[6] * exp(params[7]), # σ_ψ = μ_ψ * exp(δ_ψ)
                [params[8], params[10]], # [μ_γ, μ_ω]
                [params[8] * exp(params[9]), params[10] * exp(params[11])], # [σ_γ = μ_γ * exp(δ_γ), σ_ω = μ_ω * exp(δ_ω)]
                ks=[exp(params[12])] # k_I
            )
        
        elseif model_type == "feedback_inhibitor_inducer_perm_thresh_inhibitor_signal" # ABC
            println("Model: Inducer-dependent germination with inducer-dependent permeability and inhibitor-dependent induction signal")
            wrapper = (V_out, params) -> Main.gresp_feedback(
                Main.ode_inducer_dependent_perm_inhibitor_dependent_signal!,
                Main.thresh_criterion_inhibitor_shift,
                sobol_pts,
                times,
                [samples_A, samples_Vₛ, V_out, samples_V_ps],
                def_params[:c₀_cs],
                [params[1]], # s_max
                params[2], # Pₛ_I
                params[3], # Pₛ_C
                [nothing, params[4], params[5]], # K_cC, K_I
                params[6], # μ_ψ
                params[6] * exp(params[7]), # σ_ψ = μ_ψ * exp(δ_ψ)
                [params[8]], # μ_γ
                [params[8] * exp(params[9])]; # σ_γ = μ_γ * exp(δ_γ)
                ks=[exp(params[10])], # k_C
                n=params[11] # n
            )
            
        elseif model_type == "feedback_combined_inducer_perm_thresh_inhibitor_signal" # ABC
            println("Model: 2-factor germination with inducer-dependent permeability and inhibitor-dependent induction signal")
            wrapper = (V_out, params) -> Main.gresp_feedback(
                Main.ode_inducer_dependent_perm_inhibitor_dependent_signal!,
                Main.thresh_criterion_combined_inhibitor_shift,
                sobol_pts,
                times,
                [samples_A, samples_Vₛ, V_out, samples_V_ps],
                def_params[:c₀_cs],
                [params[1]], # s_max
                params[2], # Pₛ_I
                params[3], # Pₛ_C
                [nothing, params[4], params[5]], # K_cC, K_I
                params[6], # μ_ψ
                params[6] * exp(params[7]), # σ_ψ = μ_ψ * exp(δ_ψ)
                [params[8], params[11]], # [μ_γ, μ_ω]
                [params[8] * exp(params[9]), params[10] * exp(params[11])]; # [σ_γ = μ_γ * exp(δ_γ), σ_ω = μ_ω * exp(δ_ω)]
                ks=[exp(params[12])], # k_C
                n=params[13] # n
            )
            
        elseif model_type == "feedback_inhibitor_inducer_perm_thresh_inhibitor_perm" # ABD
            println("Model: Inhibitor-dependent germination with inducer-dependent inhibition threshold and inhibitor- and inducer-dependent inhibitor/inducer permeability")
            wrapper = (V_out, params) -> Main.gresp_feedback(
                Main.ode_inducer_and_inhibitor_dependent_perm!,
                Main.thresh_criterion_inhibitor_shift,
                sobol_pts,
                times,
                [samples_A, samples_Vₛ, V_out, samples_V_ps],
                def_params[:c₀_cs],
                [params[1], params[2]], # b_max, s_max
                params[3], # Pₛ_I
                params[4], # Pₛ_C
                [params[5], params[6]], # K_cI, K_cC
                params[7], # μ_ψ
                params[7] * exp(params[8]), # σ_ψ = μ_ψ * exp(δ_ψ)
                [params[9]], # μ_γ
                [params[9] * exp(params[10])]; # σ_γ = μ_γ * exp(δ_γ)
                ks=[exp(params[11])] # k_C
            )
        
        elseif model_type == "feedback_combined_inducer_perm_thresh_inhibitor_perm" # ABD
            println("Model: 2-factor germination with inducer-dependent inhibition threshold and inhibitor- and inducer-dependent inhibitor/inducer permeability")
            wrapper = (V_out, params) -> Main.gresp_feedback(
                Main.ode_inducer_and_inhibitor_dependent_perm!,
                Main.thresh_criterion_combined_inhibitor_shift,
                sobol_pts,
                times,
                [samples_A, samples_Vₛ, V_out, samples_V_ps],
                def_params[:c₀_cs],
                [params[1], params[2]], # b_max, s_max
                params[3], # Pₛ_I
                params[4], # Pₛ_C
                [params[5], params[6]], # K_cI, K_cC
                params[7], # μ_ψ
                params[7] * exp(params[8]), # σ_ψ = μ_ψ * exp(δ_ψ)
                [params[9], params[11]], # μ_γ, μ_ω
                [params[9] * exp(params[10]), params[11] * exp(params[12])]; # σ_γ = μ_γ * exp(δ_γ), σ_ω = μ_ω * exp(δ_ω)
                ks=[exp(params[13])] # k_C
            )
        
        elseif model_type == "feedback_combined_inducer_perm_thresh_inhibitor_thresh" # ABE
            println("Model: 2-factor germination with inducer-dependent inhibitor/inducer permeability and inhibition threshold and inhibitor-dependent induction threshold")
            wrapper = (V_out, params) -> Main.gresp_feedback(
                Main.ode_inducer_dependent_perm!,
                Main.thresh_criterion_combined_shift,
                sobol_pts,
                times,
                [samples_A, samples_Vₛ, V_out, samples_V_ps],
                def_params[:c₀_cs],
                [params[1]], # s_max
                params[2], # Pₛ_I
                params[3], # Pₛ_C
                [params[4], params[5]], # :K_cI, K_cC
                params[6], # μ_ψ
                params[6] * exp(params[7]), # σ_ψ = μ_ψ * exp(δ_ψ)
                [params[8], params[10]], # [μ_γ, μ_ω]
                [params[8] * exp(params[9]), params[10] * exp(params[11])]; # [σ_γ = μ_γ * exp(δ_γ), σ_ω = μ_ω * exp(δ_ω)]
                ks=[exp(params[12]), exp(params[13])] # k_I, k_C
            )
        
        elseif model_type == "feedback_inhibitor_inhibitor_inducer_perm_inhibitor_signal" # ACD
            println("Model: Inhibitor-dependent germination with inducer/inhibitor-dependent inhibitor/inducer permeability and inhibitor-dependent induction signal")
            wrapper = (V_out, params) -> Main.gresp_feedback(
                Main.ode_inducer_and_inhibitor_dependent_perm_inhibitor_dependent_signal!,
                Main.thresh_criterion_inhibitor,
                sobol_pts,
                times,
                [samples_A, samples_Vₛ, V_out, samples_V_ps],
                def_params[:c₀_cs],
                [params[1], params[2]], # b_max, s_max
                params[3], # Pₛ_I
                params[4], # Pₛ_C
                [params[5], params[6], params[7]], # K_cI, K_cC, K_I
                params[8], # μ_ψ
                params[8] * exp(params[9]), # σ_ψ = μ_ψ * exp(δ_ψ)
                [params[10]], # μ_γ
                [params[10] * exp(params[11])]; # σ_γ = μ_γ * exp(δ_γ)
                n=params[12] # n
            )
        
        elseif model_type == "feedback_inducer_inhibitor_inducer_perm_inhibitor_signal" # ACD
            println("Model: Inhibitor-dependent germination with inducer/inhibitor-dependent inhibitor/inducer permeability and inhibitor-dependent induction signal")
            wrapper = (V_out, params) -> Main.gresp_feedback(
                Main.ode_inducer_and_inhibitor_dependent_perm_inhibitor_dependent_signal!,
                Main.thresh_criterion_inducer,
                sobol_pts,
                times,
                [samples_A, samples_Vₛ, V_out, samples_V_ps],
                def_params[:c₀_cs],
                [params[1], params[2]], # b_max, s_max
                params[3], # Pₛ_I
                params[4], # Pₛ_C
                [params[5], params[6], params[7]], # K_cI, K_cC, K_I
                params[8], # μ_ψ
                params[8] * exp(params[9]), # σ_ψ = μ_ψ * exp(δ_ψ)
                [params[10]], # μ_ω
                [params[10] * exp(params[11])]; # σ_ω = μ_ω * exp(δ_ω)
                n=params[12] # n
            )
        
        elseif model_type == "feedback_combined_inhibitor_inducer_perm_inhibitor_signal" # ACD
            println("Model: 2-factor germination with inducer/inhibitor-dependent inhibitor/inducer permeability and inhibitor-dependent induction signal")
            wrapper = (V_out, params) -> Main.gresp_feedback(
                Main.ode_inducer_and_inhibitor_dependent_perm_inhibitor_dependent_signal!,
                Main.thresh_criterion_combined,
                sobol_pts,
                times,
                [samples_A, samples_Vₛ, V_out, samples_V_ps],
                def_params[:c₀_cs],
                [params[1], params[2]], # b_max, s_max
                params[3], # Pₛ_I
                params[4], # Pₛ_C
                [params[5], params[6], params[7]], # K_cI, K_cC, K_I
                params[8], # μ_ψ
                params[8] * exp(params[9]), # σ_ψ = μ_ψ * exp(δ_ψ)
                [params[10], params[12]], # μ_γ, μ_ω
                [params[10] * exp(params[11]), params[12] * exp(params[13])]; # σ_γ = μ_γ * exp(δ_γ), σ_ω = μ_ω * exp(δ_ω)
                n=params[14] # n
            )
        
        elseif model_type == "feedback_inducer_inducer_perm_inhibitor_thresh_signal" # ACE
            println("Model: Inducer-dependent germination with inducer-dependent permeability and inhibitor-dependent induction threshold and signal")
            wrapper = (V_out, params) -> Main.gresp_feedback(
                Main.ode_inducer_dependent_perm_inhibitor_dependent_signal!,
                Main.thresh_criterion_inducer_signal_shift,
                sobol_pts,
                times,
                [samples_A, samples_Vₛ, V_out, samples_V_ps],
                def_params[:c₀_cs],
                [params[1]], # s_max
                params[2], # Pₛ_I
                params[3], # Pₛ_C
                [params[4], params[5], params[6]], # K_cI, K_cC, K_I
                params[7], # μ_ψ
                params[7] * exp(params[8]), # σ_ψ = μ_ψ * exp(δ_ψ)
                [params[9]], # μ_ω
                [params[9] * exp(params[10])]; # σ_ω = μ_ω * exp(δ_ω)
                ks=[exp(params[11])], # k_I
                n=params[12] # n
            )
            
        elseif model_type == "feedback_combined_inducer_perm_inhibitor_thresh_signal" # ACE
            println("Model: 2-factor germination with inducer-dependent permeability and inhibitor-dependent induction threshold and signal")
            wrapper = (V_out, params) -> Main.gresp_feedback(
                Main.ode_inducer_dependent_perm_inhibitor_dependent_signal!,
                Main.thresh_criterion_combined_inducer_signal_shift,
                sobol_pts,
                times,
                [samples_A, samples_Vₛ, V_out, samples_V_ps],
                def_params[:c₀_cs],
                [params[1]], # s_max
                params[2], # Pₛ_I
                params[3], # Pₛ_C
                [params[4], params[5], params[6]], # K_cI, K_cC, K_I
                params[7], # μ_ψ
                params[7] * exp(params[8]), # σ_ψ = μ_ψ * exp(δ_ψ)
                [params[9], params[11]], # [μ_γ, μ_ω]
                [params[9] * exp(params[10]), params[11] * exp(params[12])]; # [σ_γ = μ_γ * exp(δ_γ), σ_ω = μ_ω * exp(δ_ω)]
                ks=[exp(params[13])], # k_I
                n=params[14] # n
            )
            
        elseif model_type == "feedback_inducer_inhibitor_inducer_perm_inhibitor_thresh" # ADE
            println("Model: Inhibitor-dependent germination with inhibitor- and inducer-dependent inhibitor/inducer permeability, inhibitor-dependent induction threshold")
            wrapper = (V_out, params) -> Main.gresp_feedback(
                Main.ode_inducer_and_inhibitor_dependent_perm!,
                Main.thresh_criterion_inducer_shift,
                sobol_pts,
                times,
                [samples_A, samples_Vₛ, V_out, samples_V_ps],
                def_params[:c₀_cs],
                [params[1], params[2]], # b_max, s_max
                params[3], # Pₛ_I
                params[4], # Pₛ_C
                [params[5], params[6]], # K_cI, K_cC
                params[7], # μ_ψ
                params[7] * exp(params[8]), # σ_ψ = μ_ψ * exp(δ_ψ)
                [params[9]], # μ_γ
                [params[9] * exp(params[10])]; # σ_γ = μ_γ * exp(δ_γ)
                ks=[exp(params[11])] # k_I
            )
        
        elseif model_type == "feedback_combined_inhibitor_inducer_perm_inhibitor_thresh" # ADE
            println("Model: 2-factor germination with inhibitor- and inducer-dependent inhibitor/inducer permeability, inhibitor-dependent induction threshold")
            wrapper = (V_out, params) -> Main.gresp_feedback(
                Main.ode_inducer_and_inhibitor_dependent_perm!,
                Main.thresh_criterion_combined_inducer_shift,
                sobol_pts,
                times,
                [samples_A, samples_Vₛ, V_out, samples_V_ps],
                def_params[:c₀_cs],
                [params[1], params[2]], # b_max, s_max
                params[3], # Pₛ_I
                params[4], # Pₛ_C
                [params[5], params[6]], # K_cI, K_cC
                params[7], # μ_ψ
                params[7] * exp(params[8]), # σ_ψ = μ_ψ * exp(δ_ψ)
                [params[9], params[11]], # μ_γ, μ_ω
                [params[9] * exp(params[10]), params[11] * exp(params[12])]; # σ_γ = μ_γ * exp(δ_γ), σ_ω = μ_ω * exp(δ_ω)
                ks=[exp(params[13])] # k_I
            )
        
        elseif model_type == "feedback_inhibitor_inducer_thresh_inhibitor_perm_signal" # BCD
            println("Model: Inhibitor-dependent germination with inhibitor-dependent inhibitor/inducer permeability and inducer-dependent inhibition threshold")
            wrapper = (V_out, params) -> Main.gresp_feedback(
                Main.ode_inhibitor_dependent_perm!,
                Main.thresh_criterion_inhibitor_signal_shift,
                sobol_pts,
                times,
                [samples_A, samples_Vₛ, V_out, samples_V_ps],
                def_params[:c₀_cs],
                [params[1]], # b_max
                params[2], # Pₛ_I
                params[3], # Pₛ_C
                [params[4], params[5], params[6]], # K_cI, K_cC, K_I
                params[7], # μ_ψ
                params[7] * exp(params[8]), # σ_ψ = μ_ψ * exp(δ_ψ)
                [params[9]], # [μ_γ]
                [params[9] * exp(params[10])]; # [σ_γ = μ_γ * exp(δ_γ)]
                ks=params[11], # k_C
                n=params[12] # n
            )
        
        elseif model_type == "feedback_combined_inducer_thresh_inhibitor_perm_signal" # BCD
            println("Model: 2-factor germination with inhibitor-dependent inhibitor/inducer permeability and inducer-dependent inhibition threshold")
            wrapper = (V_out, params) -> Main.gresp_feedback(
                Main.ode_inhibitor_dependent_perm!,
                Main.thresh_criterion_combined_inhibitor_signal_shift,
                sobol_pts,
                times,
                [samples_A, samples_Vₛ, V_out, samples_V_ps],
                def_params[:c₀_cs],
                [params[1]], # b_max
                params[2], # Pₛ_I
                params[3], # Pₛ_C
                [params[4], params[5], params[6]], # K_cI, K_cC, K_I
                params[7], # μ_ψ
                params[7] * exp(params[8]), # σ_ψ = μ_ψ * exp(δ_ψ)
                [params[9], params[11]], # [μ_γ, μ_ω]
                [params[9] * exp(params[10]), params[11] * exp(params[12])]; # [σ_ω = μ_ω * exp(δ_ω)]
                ks=params[13], # k_C
                n=params[14] # n
            )
        
        elseif model_type == "combined_inhibitor_thresh_signal_inducer_thresh" # BCE
            println("Model: inhibitor-modulated induction threshold and signal / inducer-modulated inhibition threshold")
            wrapper = (inputs, params) -> Main.gresp_inh_dep_ind_thresh_signal_ind_dep_inh_thresh_2_factor_gh(
                u, W3,
                inputs[1], #t
                inputs[2], #ρₛ
                def_params[:c₀_cs],
                def_params[:d_hp],
                ξ2,
                κ2,
                params[1], #Pₛ
                params[2], #Pₛ_cs
                params[3], #K_cI
                params[4], #K_cC
                params[5], #K_I
                params[6], #n
                exp(params[7]), #k_I
                exp(params[8]), #k_C
                params[9], #μ_γ
                params[9] * exp(params[10]), # σ_γ = μ_γ * exp(δ_γ)
                params[11], #μ_ω
                params[11] * exp(params[12]), # σ_ω = μ_ω * exp(δ_ω)
                params[13], #μ_ψ
                params[13] * exp(params[14]) # σ_ψ = μ_ψ * exp(δ_ψ)
            )

        elseif model_type == "feedback_combined_inhibitor_perm_thresh_inducer_thresh" # BDE
            println("Model: 2-factor germination with inhibitor-dependent inhibitor/inducer permeability and inducer threshold, inducer-dependent inhibition threshold")
            wrapper = (V_out, params) -> Main.gresp_feedback(
                Main.ode_inhibitor_dependent_perm!,
                Main.thresh_criterion_combined_shift,
                sobol_pts,
                times,
                [samples_A, samples_Vₛ, V_out, samples_V_ps],
                def_params[:c₀_cs],
                [params[1]], # b_max
                params[2], # Pₛ_I
                params[3], # Pₛ_C
                [params[4], params[5]], # K_cI, K_cC
                params[6], # μ_ψ
                params[6] * exp(params[7]), # σ_ψ = μ_ψ * exp(δ_ψ)
                [params[8], params[10]], # [μ_γ, μ_ω]
                [params[8] * exp(params[9]), params[10] * exp(params[11])]; # [σ_ω = μ_ω * exp(δ_ω)]
                ks=[params[12], params[13]] # k_I, k_C
            )
        
        elseif model_type == "feedback_inducer_inhibitor_perm_thresh_signal" # CDE
            println("Model: Inducer-dependent germination with inhibitor-dependent inhibitor/inducer permeability, induction threshold and signal")
            wrapper = (V_out, params) -> Main.gresp_feedback(
                Main.ode_inhibitor_dependent_perm!,
                Main.thresh_criterion_inducer_signal_shift,
                sobol_pts,
                times,
                [samples_A, samples_Vₛ, V_out, samples_V_ps],
                def_params[:c₀_cs],
                [params[1]], # b_max
                params[2], # Pₛ_I
                params[3], # Pₛ_C
                [params[4], params[5], params[6]], # K_cI, K_cC, K_I
                params[7], # μ_ψ
                params[7] * exp(params[8]), # σ_ψ = μ_ψ * exp(δ_ψ)
                [params[9]], # [μ_ω]
                [params[9] * exp(params[10])]; # [σ_ω = μ_ω * exp(δ_ω)]
                ks=[params[11]], # k_I
                n=params[12] # n
            )
        
        elseif model_type == "feedback_combined_inhibitor_perm_thresh_signal" # CDE
            println("Model: 2-factor germination with inhibitor-dependent inhibitor/inducer permeability, induction threshold and signal")
            wrapper = (V_out, params) -> Main.gresp_feedback(
                Main.ode_inhibitor_dependent_perm!,
                Main.thresh_criterion_combined_inducer_signal_shift,
                sobol_pts,
                times,
                [samples_A, samples_Vₛ, V_out, samples_V_ps],
                def_params[:c₀_cs],
                [params[1]], # b_max
                params[2], # Pₛ_I
                params[3], # Pₛ_C
                [params[4], params[5], params[6]], # K_cI, K_cC, K_I
                params[7], # μ_ψ
                params[7] * exp(params[8]), # σ_ψ = μ_ψ * exp(δ_ψ)
                [params[9], params[11]], # [μ_γ, μ_ω]
                [params[9] * exp(params[10]), params[11] * exp(params[12])]; # [σ_ω = μ_ω * exp(δ_ω)]
                ks=[params[13]], # k_I
                n=params[14] # n
            )
        
        elseif model_type == "feedback_inhibitor_inducer_perm_thresh_inhibitor_perm_signal" # ABCD
            println("Model: Inhibitor-dependent germination with inducer/inhibitor-dependent inhibitor/inducer permeability, inducer_dependent inhibition threshold and inhibitor-dependent induction signal")
            wrapper = (V_out, params) -> Main.gresp_feedback(
                Main.ode_inducer_and_inhibitor_dependent_perm_inhibitor_dependent_signal!,
                Main.thresh_criterion_inhibitor_shift,
                sobol_pts,
                times,
                [samples_A, samples_Vₛ, V_out, samples_V_ps],
                def_params[:c₀_cs],
                [params[1], params[2]], # b_max, s_max
                params[3], # Pₛ_I
                params[4], # Pₛ_C
                [params[5], params[6], params[7]], # K_cI, K_cC, K_I
                params[8], # μ_ψ
                params[8] * exp(params[9]), # σ_ψ = μ_ψ * exp(δ_ψ)
                [params[10]], # μ_γ
                [params[10] * exp(params[11])]; # σ_γ = μ_γ * exp(δ_γ)
                ks=[params[12]], # k_C
                n=params[13] # n
            )
        
        elseif model_type == "feedback_combined_inducer_perm_thresh_inhibitor_perm_signal" # ABCD
            println("Model: 2-factor germination with inducer/inhibitor-dependent inhibitor/inducer permeability, inducer_dependent inhibition threshold and inhibitor-dependent induction signal")
            wrapper = (V_out, params) -> Main.gresp_feedback(
                Main.ode_inducer_and_inhibitor_dependent_perm_inhibitor_dependent_signal!,
                Main.thresh_criterion_combined_inhibitor_shift,
                sobol_pts,
                times,
                [samples_A, samples_Vₛ, V_out, samples_V_ps],
                def_params[:c₀_cs],
                [params[1], params[2]], # b_max, s_max
                params[3], # Pₛ_I
                params[4], # Pₛ_C
                [params[5], params[6], params[7]], # K_cI, K_cC, K_I
                params[8], # μ_ψ
                params[8] * exp(params[9]), # σ_ψ = μ_ψ * exp(δ_ψ)
                [params[10], params[12]], # μ_γ, μ_ω
                [params[10] * exp(params[11]), params[12] * exp(params[13])]; # σ_γ = μ_γ * exp(δ_γ), σ_ω = μ_ω * exp(δ_ω)
                ks=[params[14]], # k_C
                n=params[15] # n
            )
        
        elseif model_type == "feedback_combined_inhibitor_thresh_signal_inducer_perm_thresh" # ABCE
            println("Model: 2-factor germination with inducer-dependent permeability/inhibition threshold and inhibitor-dependent induction signal and threshold")
            wrapper = (V_out, params) -> Main.gresp_feedback(
                Main.ode_inducer_dependent_perm_inhibitor_dependent_signal!,
                Main.thresh_criterion_combined_inhibitor_shift_inducer_signal_shift,
                sobol_pts,
                times,
                [samples_A, samples_Vₛ, V_out, samples_V_ps],
                def_params[:c₀_cs],
                [params[1]], # s_max
                params[2], # Pₛ_I
                params[3], # Pₛ_C
                [params[4], params[5], params[6]], # K_cI, K_cC, K_I
                params[7], # μ_ψ
                params[7] * exp(params[8]), # σ_ψ = μ_ψ * exp(δ_ψ)
                [params[9], params[11]], # [μ_γ, μ_ω]
                [params[9] * exp(params[10]), params[11] * exp(params[12])]; # [σ_γ = μ_γ * exp(δ_γ), σ_ω = μ_ω * exp(δ_ω)]
                ks=[params[13], params[14]],
                n=params[15] # n
            )
            
        elseif model_type == "feedback_combined_inhibitor_perm_thresh_inducer_perm_thresh" # ABDE
            println("Model: 2-factor germination with inhibitor- and inducer-dependent inhibitor/inducer permeability, inhibitor- and inducer-dependent thresholds")
            wrapper = (V_out, params) -> Main.gresp_feedback(
                Main.ode_inducer_and_inhibitor_dependent_perm!,
                Main.thresh_criterion_combined_shift,
                sobol_pts,
                times,
                [samples_A, samples_Vₛ, V_out, samples_V_ps],
                def_params[:c₀_cs],
                [params[1], params[2]], # b_max, s_max
                params[3], # Pₛ_I
                params[4], # Pₛ_C
                [params[5], params[6]], # K_cI, K_cC
                params[7], # μ_ψ
                params[7] * exp(params[8]), # σ_ψ = μ_ψ * exp(δ_ψ)
                [params[9], params[11]], # μ_γ, μ_ω
                [params[9] * exp(params[10]), params[11] * exp(params[12])]; # σ_γ = μ_γ * exp(δ_γ), σ_ω = μ_ω * exp(δ_ω)
                ks=[params[13], params[14]] # k_I, k_C
            )
        
        elseif model_type == "feedback_inducer_inhibitor_perm_thresh_signal_inducer_perm" # ACDE
            println("Model: Inhibitor-dependent germination with inducer/inhibitor-dependent inhibitor/inducer permeability and inhibitor-dependent induction signal and threshold")
            wrapper = (V_out, params) -> Main.gresp_feedback(
                Main.ode_inducer_and_inhibitor_dependent_perm_inhibitor_dependent_signal!,
                Main.thresh_criterion_inducer_signal_shift,
                sobol_pts,
                times,
                [samples_A, samples_Vₛ, V_out, samples_V_ps],
                def_params[:c₀_cs],
                [params[1], params[2]], # b_max, s_max
                params[3], # Pₛ_I
                params[4], # Pₛ_C
                [params[5], params[6], params[7]], # K_cI, K_cC, K_I
                params[8], # μ_ψ
                params[8] * exp(params[9]), # σ_ψ = μ_ψ * exp(δ_ψ)
                [params[10]], # μ_ω
                [params[10] * exp(params[11])]; # σ_ω = μ_ω * exp(δ_ω)
                ks=[params[12]], # k_I
                n=params[13] # n
            )
        
        elseif model_type == "feedback_combined_inhibitor_perm_thresh_signal_inducer_perm" # ACDE
            println("Model: 2-factor germination with inducer/inhibitor-dependent inhibitor/inducer permeability and inhibitor-dependent induction signal and threshold")
            wrapper = (V_out, params) -> Main.gresp_feedback(
                Main.ode_inducer_and_inhibitor_dependent_perm_inhibitor_dependent_signal!,
                Main.thresh_criterion_combined_inducer_signal_shift,
                sobol_pts,
                times,
                [samples_A, samples_Vₛ, V_out, samples_V_ps],
                def_params[:c₀_cs],
                [params[1], params[2]], # b_max, s_max
                params[3], # Pₛ_I
                params[4], # Pₛ_C
                [params[5], params[6], params[7]], # K_cI, K_cC, K_I
                params[8], # μ_ψ
                params[8] * exp(params[9]), # σ_ψ = μ_ψ * exp(δ_ψ)
                [params[10], params[12]], # μ_γ, μ_ω
                [params[10] * exp(params[11]), params[12] * exp(params[13])]; # σ_γ = μ_γ * exp(δ_γ), σ_ω = μ_ω * exp(δ_ω)
                ks=[params[14]], # k_I
                n=params[15] # n
            )
        
        elseif model_type == "feedback_combined_inhibitor_perm_thresh_signal_inducer_thresh" # BCDE
            println("Model: 2-factor germination with inhibitor-dependent inhibitor/inducer permeability, induction threshold and signal, and inducer-dependent inhibition threshold")
            wrapper = (V_out, params) -> Main.gresp_feedback(
                Main.ode_inhibitor_dependent_perm!,
                Main.thresh_criterion_combined_inhibitor_shift_inducer_signal_shift,
                sobol_pts,
                times,
                [samples_A, samples_Vₛ, V_out, samples_V_ps],
                def_params[:c₀_cs],
                [params[1]], # b_max
                params[2], # Pₛ_I
                params[3], # Pₛ_C
                [params[4], params[5], params[6]], # K_cI, K_cC, K_I
                params[7], # μ_ψ
                params[7] * exp(params[8]), # σ_ψ = μ_ψ * exp(δ_ψ)
                [params[9], params[11]], # [μ_γ, μ_ω]
                [params[9] * exp(params[10]), params[11] * exp(params[12])]; # [σ_ω = μ_ω * exp(δ_ω)]
                ks=[params[13], params[14]],
                n=params[15]
            )
        
        elseif model_type == "feedback_combined_inhibitor_perm_thresh_signal_inducer_perm_thresh" # ABCDE
            println("Model: 2-factor germination with inducer/inhibitor-dependent inhibitor/inducer permeability, inducer-dependent inhibition threshold and inhibitor-dependent induction threshold/signal")
            wrapper = (V_out, params) -> Main.gresp_feedback(
                Main.ode_inducer_and_inhibitor_dependent_perm_inhibitor_dependent_signal!,
                Main.thresh_criterion_combined_inhibitor_shift_inducer_signal_shift,
                sobol_pts,
                times,
                [samples_A, samples_Vₛ, V_out, samples_V_ps],
                def_params[:c₀_cs],
                [params[1], params[2]], # b_max, s_max
                params[3], # Pₛ_I
                params[4], # Pₛ_C
                [params[5], params[6], params[7]], # K_cI, K_cC, K_I
                params[8], # μ_ψ
                params[8] * exp(params[9]), # σ_ψ = μ_ψ * exp(δ_ψ)
                [params[10], params[12]], # μ_γ, μ_ω
                [params[10] * exp(params[11]), params[12] * exp(params[13])]; # σ_γ = μ_γ * exp(δ_γ), σ_ω = μ_ω * exp(δ_ω)
                ks=[params[14], params[15]], # k_I, k_C
                n=params[16] # n
            )
        
        else
            error("Model type not recognized.")
        end

        if debug
            # Simply print wrapper function and parameters
            
            # println("Parameters: ", sort(vcat(param_keys, collect(keys(def_params)))))
            println("Parameters: ", sort(param_keys))
            params_out = nothing
            rmse = nothing
        
        else
            # Model fitting

            param_occurrences = [param_occurrences_dict[pkey] for pkey in param_keys]

            # Duplicate bounds/parameters for each source
            param_starts = cumsum(param_occurrences) .- param_occurrences .+ 1
            param_keys_dup = vcat([key for key in param_keys for _ in 1:param_occurrences[param_keys .== key][1]]...)
            param_starts = cumsum(param_occurrences) .- param_occurrences .+ 1
            bounds = [bounds_dict[key] for key in param_keys_dup]

            param_indices_per_src = [param_starts .+ ((i - 1) .% param_occurrences) for i in 1:length(sources)]
            
            if gh_integral

                # Objective function
                input_tuples =  [tuple.(times_tile[i, :, :], inverse_mL_to_cubic_um.(densities_tile[i, :, :])) for i in 1:length(sources)]
                dantigny_data_flat = [collect(dantigny_data[i, :, :]) for i in 1:length(sources)]

                obj = params -> begin
                    err = 0.0
                    @inbounds for i in eachindex(sources)
                        params_select = view(params, param_indices_per_src[i])
                        ŷ = [wrapper(inputs, params_select) for inputs in input_tuples[i]]
                        err += sum(abs2, ŷ .- dantigny_data_flat[i])
                    end
                    return err
                end
                objgrad = (params,_) -> begin
                    err = 0
                    @inbounds for i in eachindex(sources)
                        params_select = view(params, param_indices_per_src[i])
                        ŷ = [wrapper(inputs, params_select) for inputs in input_tuples[i]]
                        err += sum(abs2, ŷ .- dantigny_data_flat[i])
                    end
                    return err
                end
            else

                # Objective function (feedback model)
                input_densities = inverse_mL_to_cubic_um.(densities)
                samples_V_out = 1 ./ input_densities .- samples_Vₛ'

                params = zeros(length(keys(bounds_dict)))
                for (i, key) in enumerate(keys(bounds_dict))
                    params[i] = mean(bounds_dict[key])
                end

                obj = params -> begin
                    err = 0.0
                    @inbounds for i in eachindex(sources)
                        params_select = view(params, param_indices_per_src[i])

                        # run a single simulation at all times
                        ŷ = reduce(vcat, [wrapper(samples_V_out[j, :], params_select) for j in eachindex(input_densities)]')

                        err += sum(abs2, ŷ .- dantigny_data[i, :, :])
                    end
                    return err
                end
                objgrad = (params,_) -> begin
                    err = 0.0
                    @inbounds for i in eachindex(sources)
                        params_select = view(params, param_indices_per_src[i])
                        
                        # run a single simulation at all times
                        ŷ = reduce(vcat, [wrapper(samples_V_out[j, :], params_select) for j in eachindex(input_densities)]')

                        err += sum(abs2, ŷ .- dantigny_data[i, :, :])
                    end
                    return err
                end
            end
            
            # Fit model
            println("Running first optimisation stage")
            res = bboptimize(params -> obj(params);
                        SearchRange = bounds,
                        MaxSteps = max_steps,
                        Method = :adaptive_de_rand_1_bin_radiuslimited)
                        # Method = :adaptive_de_rand_1_bin)
            p_opt = best_candidate(res)
            best_fit = best_fitness(res)

            println("Running second optimisation stage")
            opt = Opt(:LN_COBYLA, length(bounds))
            lower_bounds!(opt, [bnd[1] for bnd in bounds])
            upper_bounds!(opt, [bnd[2] for bnd in bounds])
            xtol_rel!(opt, 1e-4)
            maxeval!(opt, 2000)
            min_objective!(opt, objgrad)

            (best_fit, res, code) = NLopt.optimize(opt, p_opt)
            p_opt = res
            println("Final fitness: ", best_fit)
            

            # Compute rmse
            rmse = sqrt(best_fit / length(dantigny_data))

            # Create a dictionary for the optimized parameters
            params_out = Dict()
            for (i, key) in enumerate(param_keys)
                for j in 1:param_occurrences[i]
                    # Transform parameters back to original scale
                    key_split = split(string(key), "_")
                    if key_split[1] == "δ"
                        key_new = Symbol(:σ_, key_split[2])
                        val = p_opt[param_starts[i - 1] + j - 1] * exp(p_opt[param_starts[i] + j - 1])
                    elseif (key == :k_I || key == :k_C)
                        println("Converting k to original scale")
                        key_new = key
                        val = exp(p_opt[param_starts[i] + j - 1])
                    else
                        key_new = key
                        val = p_opt[param_starts[i] + j - 1]
                    end
                    if haskey(params_out, key_new)
                        push!(params_out[key_new], val)
                    else
                        params_out[key_new] = [val]
                    end
                end
            end
        end

        return params_out, rmse
    end


    function get_params_for_idx(params, idx)
        """
        Get the parameters for a specific index.
        inputs:
            params (Dict): dictionary of parameters
            idx (Int): index to get the parameters for
        outputs:
            params_out (Dict): dictionary of parameters for the specified index
        """

        params_out = Dict()

        for (key, value) in params
            params_out[key] = value[mod1(idx, length(value))]
        end

        return params_out
    end


    function fit_model_to_data_equilibrium(model_type, def_params, germ_data, densities, bounds_dict; c_ex_vals=nothing, ref_density=nothing, max_steps=10000)
        """
        Fit a selected equilibrium germination model to the data.
        inputs:
            model_type (String): model type to fit
            def_params (Dict): default parameter values
            germ_data (Array): germination data (concatenated datesets if both exogenous and endogenous models are fitted)
            densities (Vector): spore densities in spores/mL
            bounds_dict (Dict): bounds for the free parameters
            c_ex_vals (Vector): exogenous inhibitor concentrations (optional)
            ref_density (Float): reference density for exogenous inhibitor (optional, only used when fitting both exogenous and endogenous models)
            max_steps (Int): maximum number of steps for the optimization
        outputs:
            params_out (Dict): optimized parameters
        """

        @argcheck model_type in ["inhibitor", "combined_inducer", "combined_inducer_thresh", "combined_inducer_signal", "independent",
                                "inhibitor_ex", "combined_inducer_ex", "combined_inducer_thresh_ex", "combined_inducer_signal_ex", "independent_ex"]

        if isnothing(ref_density)
            ref_density = inverse_mL_to_cubic_um(densities[1]) # Use first density as reference if not provided
        elseif !isnothing(c_ex_vals)
            ref_density = inverse_mL_to_cubic_um(ref_density) # Convert to cubic micrometers if provided
        end

        # Unpack radius distribution
        μ_ξ = def_params[:μ_ξ]
        σ_ξ = def_params[:σ_ξ]
        μ_ξ_log = log(μ_ξ^2 / sqrt(σ_ξ^2 + μ_ξ^2))
        σ_ξ_log = sqrt(log(σ_ξ^2 / μ_ξ^2 + 1))
        dist_ξ = LogNormal(μ_ξ_log, σ_ξ_log)

        if model_type in ["inhibitor", "inhibitor_ex"]
            println("Model: Inducer-dependent inhibitor threshold and release")
            wrapper = (ρₛ, params) -> Main.gresp_inducer_dep_inhibitor_eq(
                ρₛ,
                dist_ξ,
                params[1], #μ_γ
                params[1] * exp(params[2]) # σ_γ = μ_γ * exp(δ_γ)
            )
            param_keys = [:μ_γ, :δ_γ]
        elseif model_type in ["combined_inducer", "combined_inducer_ex"]
            println("Model: Two-factor germination with inhibitor-dependent induction threshold and signal")
            wrapper = (ρₛ, params) -> Main.gresp_inhibitor_dep_inducer_2_factors_eq(
                ρₛ,
                dist_ξ,
                def_params[:c₀_cs],
                params[1], #K_cI
                params[2], #K_cC
                params[3], #K_I
                exp(params[4]), #k_I
                params[5], #n
                params[6], #μ_γ
                params[6] * exp(params[7]), # σ_γ = μ_γ * exp(δ_γ)
                params[8], #μ_ω
                params[8] * exp(params[9]), # σ_ω = μ_ω * exp(δ_ω)
                params[10], #μ_ψ
                params[10] * exp(params[11]) # σ_ψ = μ_ψ * exp(δ_ψ)
            )
            param_keys = [:K_cI, :K_cC, :K_I, :k_I, :n, :μ_γ, :δ_γ, :μ_ω, :δ_ω, :μ_ψ, :δ_ψ]
        elseif model_type in ["combined_inducer_thresh", "combined_inducer_thresh_ex"]
            println("Model: Two-factor germination with inhibitor-dependent induction threshold")
            wrapper = (ρₛ, params) -> Main.gresp_inhibitor_dep_inducer_thresh_2_factors_eq(
                ρₛ,
                dist_ξ,
                def_params[:c₀_cs],
                params[1], #K_cC
                exp(params[2]), #k_I
                params[3], #μ_γ
                params[3] * exp(params[4]), # σ_γ = μ_γ * exp(δ_γ)
                params[5], #μ_ω
                params[5] * exp(params[6]), # σ_ω = μ_ω * exp(δ_ω)
                params[7], #μ_ψ
                params[7] * exp(params[8]) # σ_ψ = μ_ψ * exp(δ_ψ)
            )
            param_keys = [:K_cC, :k_I, :μ_γ, :δ_γ, :μ_ω, :δ_ω, :μ_ψ, :δ_ψ]
        elseif model_type in ["combined_inducer_signal", "combined_inducer_signal_ex"]
            println("Model: Two-factor germination with inhibitor-dependent induction signal")
            wrapper = (ρₛ, params) -> Main.gresp_inhibitor_dep_inducer_signal_2_factors_eq(
                ρₛ,
                dist_ξ,
                def_params[:c₀_cs],
                params[1], #K_cC
                params[2], #K_I
                params[3], #n
                params[4], #μ_γ
                params[4] * exp(params[5]), # σ_γ = μ_γ * exp(δ_γ)
                params[6], #μ_ω
                params[6] * exp(params[7]), # σ_ω = μ_ω * exp(δ_ω)
                params[8], #μ_ψ
                params[8] * exp(params[9]) # σ_ψ = μ_ψ * exp(δ_ψ)
            )
            param_keys = [:K_cC, :K_I, :n, :μ_γ, :δ_γ, :μ_ω, :δ_ω, :μ_ψ, :δ_ψ]
        elseif model_type in ["independent", "independent_ex"]
            println("Model: Independent factors")
            wrapper = (ρₛ, params) -> Main.gresp_independent_eq(
                ρₛ,
                dist_ξ,
                def_params[:c₀_cs],
                params[1], #K_cC
                params[2], #μ_γ
                params[2] * exp(params[3]), # σ_γ = μ_γ * exp(δ_γ)
                params[4], #μ_ω
                params[4] * exp(params[5]) # σ_ω = μ_ω * exp(δ_ω)
            )
            param_keys = [:K_cC, :μ_γ, :δ_γ, :μ_ω, :δ_ω]
        end
        if model_type == "inhibitor_ex"
            println("Model: Inducer-dependent inhibitor threshold and release (exogenous inhibitor)")
            wrapper_ex = (c_ex, params) -> Main.gresp_inducer_dep_inhibitor_eq_c_ex(
                ref_density, # Assuming single density for exogenous inhibitor
                dist_ξ,
                c_ex,
                params[1], #μ_γ
                params[1] * exp(params[2]), # σ_γ = μ_γ * exp(δ_γ)
                params[3], #μ_ψ
                params[3] * exp(params[4]) # σ_ψ = μ_ψ * exp(δ_ψ)
            )
            param_keys = [:μ_γ, :δ_γ, :μ_ψ, :δ_ψ]
        elseif model_type == "combined_inducer_ex"
            println("Model: Two-factor germination with inhibitor-dependent induction threshold and signal (exogenous inhibitor)")
            wrapper_ex = (c_ex, params) -> Main.gresp_inhibitor_dep_inducer_2_factors_eq_c_ex(
                ref_density, # Assuming single density for exogenous inhibitor
                dist_ξ,
                c_ex,
                def_params[:c₀_cs],
                params[1], #K_cI
                params[2], #K_cC
                params[3], #K_I
                exp(params[4]), #k_I
                params[5], #n
                params[6], #μ_γ
                params[6] * exp(params[7]), # σ_γ = μ_γ * exp(δ_γ)
                params[8], #μ_ω
                params[8] * exp(params[9]), # σ_ω = μ_ω * exp(δ_ω)
                params[10], #μ_ψ
                params[10] * exp(params[11]) # σ_ψ = μ_ψ * exp(δ_ψ)
            )
            param_keys = [:K_cI, :K_cC, :K_I, :k_I, :n, :μ_γ, :δ_γ, :μ_ω, :δ_ω, :μ_ψ, :δ_ψ]
        elseif model_type == "combined_inducer_thresh_ex"
            println("Model: Two-factor germination with inhibitor-dependent induction threshold (exogenous inhibitor)")
            wrapper_ex = (c_ex, params) -> Main.gresp_inhibitor_dep_inducer_thresh_2_factors_eq_c_ex(
                ref_density, # Assuming single density for exogenous inhibitor
                dist_ξ,
                c_ex,
                def_params[:c₀_cs],
                params[1], #K_cI
                params[2], #K_cC
                exp(params[3]), #k_I
                params[4], #μ_γ
                params[4] * exp(params[5]), # σ_γ = μ_γ * exp(δ_γ)
                params[6], #μ_ω
                params[6] * exp(params[7]), # σ_ω = μ_ω * exp(δ_ω)
                params[8], #μ_ψ
                params[8] * exp(params[9]) # σ_ψ = μ_ψ * exp(δ_ψ)
            )
            param_keys = [:K_cI, :K_cC, :k_I, :μ_γ, :δ_γ, :μ_ω, :δ_ω, :μ_ψ, :δ_ψ]
        elseif model_type == "combined_inducer_signal_ex"
            println("Model: Two-factor germination with inhibitor-dependent induction signal (exogenous inhibitor)")
            wrapper_ex = (c_ex, params) -> Main.gresp_inhibitor_dep_inducer_signal_2_factors_eq_c_ex(
                ref_density, # Assuming single density for exogenous inhibitor
                dist_ξ,
                c_ex,
                def_params[:c₀_cs],
                params[1], #K_cC
                params[2], #K_I
                params[3], #n
                params[4], #μ_γ
                params[4] * exp(params[5]), # σ_γ = μ_γ * exp(δ_γ)
                params[6], #μ_ω
                params[6] * exp(params[7]), # σ_ω = μ_ω * exp(δ_ω)
                params[8], #μ_ψ
                params[8] * exp(params[9]) # σ_ψ = μ_ψ * exp(δ_ψ)
            )
            param_keys = [:K_cC, :K_I, :n, :μ_γ, :δ_γ, :μ_ω, :δ_ω, :μ_ψ, :δ_ψ]
        elseif model_type == "independent_ex"
            println("Model: Independent factors (exogenous inhibitor)")
            wrapper_ex = (c_ex, params) -> Main.gresp_independent_eq_c_ex(
                ref_density, # Assuming single density for exogenous inhibitor
                dist_ξ,
                c_ex,
                def_params[:c₀_cs],
                params[1], #K_cC
                params[2], #μ_γ
                params[2] * exp(params[3]), # σ_γ = μ_γ * exp(δ_γ)
                params[4], #μ_ω
                params[4] * exp(params[5]), # σ_ω = μ_ω * exp(δ_ω)
                params[6], #μ_ψ
                params[6] * exp(params[7]) # σ_ψ = μ_ψ * exp(δ_ψ)
            )
            param_keys = [:K_cC, :μ_γ, :δ_γ, :μ_ω, :δ_ω, :μ_ψ, :δ_ψ]
        end

        bounds = [bounds_dict[key] for key in param_keys]
        densities_input = inverse_mL_to_cubic_um.(densities)

        if isnothing(c_ex_vals) # No exogenous inhibitor concentrations provided
            n_inputs = length(densities_input)
            # Objective function
            obj = params -> begin
                ŷ = [wrapper(ρₛ, params) for ρₛ in densities_input]
                err = sum(abs2, ŷ .- germ_data) 
                return err
            end
            objgrad = (params, _) -> begin
                ŷ = [wrapper(ρₛ, params) for ρₛ in densities_input]
                err = sum(abs2, ŷ .- germ_data) 
                return err
            end
        elseif length(densities) == 1 # Exogenous inhibitor concentrations provided with single density
            n_inputs = length(c_ex_vals)
            # Objective function
            obj = params -> begin
                ŷ = [wrapper_ex(c_ex, params) for c_ex in c_ex_vals]
                err = sum(abs2, ŷ .- germ_data) 
                return err
            end
            objgrad = (params, _) -> begin
                ŷ = [wrapper_ex(c_ex, params) for c_ex in c_ex_vals]
                err = sum(abs2, ŷ .- germ_data) 
                return err
            end
        elseif !isnothing(ref_density) && length(densities) > 1 # Exogenous inhibitor concentrations provided with multiple densities
            n_inputs = length(c_ex_vals) + length(densities)
            # Objective function
            obj = params -> begin
                ŷ = vcat([wrapper(ρₛ, params) for ρₛ in densities_input], [wrapper_ex(c_ex, params) for c_ex in c_ex_vals])
                err = sum(abs2, ŷ .- germ_data) 
                return err
            end
            objgrad = (params, _) -> begin
                ŷ = vcat([wrapper(ρₛ, params) for ρₛ in densities_input], [wrapper_ex(c_ex, params) for c_ex in c_ex_vals])
                err = sum(abs2, ŷ .- germ_data) 
                return err
            end
        else
            error("Invalid input: either provide c_ex_vals or a non-singular density array.")
        end
            

        # Fit model
        println("Running first optimisation stage")
        res = bboptimize(params -> obj(params);
                    SearchRange = bounds,
                    MaxSteps = max_steps,
                    Method = :adaptive_de_rand_1_bin_radiuslimited)
        p_opt = best_candidate(res)
        best_fit = best_fitness(res)

        println("Running second optimisation stage")
        opt = Opt(:LN_COBYLA, length(bounds))
        lower_bounds!(opt, [bnd[1] for bnd in bounds])
        upper_bounds!(opt, [bnd[2] for bnd in bounds])
        xtol_rel!(opt, 1e-4)
        maxeval!(opt, 2000)
        min_objective!(opt, objgrad)

        (best_fit, res, code) = NLopt.optimize(opt, p_opt)
        p_opt = res
        println("Final fitness: ", best_fit)

        # Compute rmse
        rmse = sqrt(best_fit / n_inputs)

        # Create a dictionary for the optimized parameters
        params_out = Dict()
        for (i, key) in enumerate(param_keys)
            # Transform parameters back to original scale
            key_split = split(string(key), "_")
            if key_split[1] == "δ"
                println("Converting δ = $(p_opt[i]) to σ scale")
                key_new = Symbol(:σ_, key_split[2])
                val = p_opt[i - 1] * exp(p_opt[i])
            elseif (key == :k_I || key == :k_C)
                println("Converting k to original scale")
                key_new = key
                val = exp(p_opt[i])
            else
                key_new = key
                val = p_opt[i]
            end
            if haskey(params_out, key_new)
                push!(params_out[key_new], val)
            else
                params_out[key_new] = [val]
            end
        end

        return params_out, rmse
    end

    # ===============
    # ===== BMA =====
    # ===============
    # """
    # Parameter configuration for a single model.
    # Holds names, anchor values, and prior bounds.
    # """
    # struct ModelConfig
    #     name        :: String
    #     param_names :: Vector{Symbol}
    #     anchor      :: Vector{Float64}      # example_params — known plausible point
    #     lower       :: Vector{Float64}      # prior lower bounds (log scale for LogNormal params)
    #     upper       :: Vector{Float64}      # prior upper bounds (log scale for LogNormal params)
    #     log_scaled  :: Vector{Bool}         # true if parameter is LogNormal (sweep in log space)
    #     target_pairs:: Vector{Tuple{Int,Int}} # pairs to test for interactions (from algebraic analysis)
    # end

    # """
    # Generate a 1D sweep for parameter j.
    # All other parameters held at anchor.
    # Returns (sweep_values, parameter_vectors).
    # """
    # function make_1d_sweep(config::ModelConfig, j::Int, n_points::Int=100)
    #     d = length(config.anchor)
    #     sweep_vals = range(config.lower[j], config.upper[j], length=n_points)

    #     # Each column is a full parameter vector
    #     param_matrix = repeat(config.anchor, 1, n_points)  # d × n_points
    #     param_matrix[j, :] .= sweep_vals

    #     return collect(sweep_vals), param_matrix
    # end

    # """
    # Generate a 2D grid for parameters j and k.
    # All other parameters held at anchor.
    # Returns (sweep_j, sweep_k, parameter_matrix) where
    # parameter_matrix is d × (n_j * n_k).
    # """
    # function make_2d_sweep(config::ModelConfig, j::Int, k::Int,
    #                     n_j::Int=50, n_k::Int=50)
    #     d = length(config.anchor)
    #     sweep_j = range(config.lower[j], config.upper[j], length=n_j)
    #     sweep_k = range(config.lower[k], config.upper[k], length=n_k)

    #     n_total = n_j * n_k
    #     param_matrix = repeat(config.anchor, 1, n_total)

    #     idx = 1
    #     for vj in sweep_j, vk in sweep_k
    #         param_matrix[j, idx] = vj
    #         param_matrix[k, idx] = vk
    #         idx += 1
    #     end

    #     return collect(sweep_j), collect(sweep_k), param_matrix
    # end

    # """
    # Experimental setting for a model (density, measurement times)
    # """
    # struct ExperimentSetting
    #     times       ::Vector{Float64}
    #     density     ::Float64
    #     def_params  ::Dict
    # end

    # """
    # Convert parameter vector from sweep space to model input space.
    # Log-scaled parameters are exponentiated before passing to model.
    # """
    # function to_model_params(config::ModelConfig, param_vec::Vector{Float64},)
    #     d = length(param_vec)
    #     model_params = Dict{Symbol, Float64}()
    #     for i in 1:d
    #         val = config.log_scaled[i] ? exp(param_vec[i]) : param_vec[i]
    #         model_params[config.param_names[i]] = val
    #     end
    #     return model_params
    # end

    # """
    # Compute weight penalising half-saturation times
    # beyond a plausible experimental time.
    # """
    # function experimental_relevance_weight(τ_g::Float64, T_max::Float64; margin::Float64=3.0)
    #     excess_logs = max(0.0, log(τ_g / T_max) / log(margin))
    #     return exp(-excess_logs^2)
    # end

    # """
    # Thin wrapper: given a parameter matrix (d × N), run the mechanistic
    # model on each column and return an output matrix (n_outputs × N).
    # """
    # function run_model_batch(config::ModelConfig, param_matrix::Matrix{Float64}, exp_setting::ExperimentSetting) # REMEMBER TO SCALE DENSITY AND TIMES!!!!!!!!!!!!!!!!!!!!!!!
    #     n_cols = size(param_matrix, 2)
    #     outputs = Vector{Vector{Float64}}(undef, n_cols)
    #     stderr  = Vector{Vector{Float64}}(undef, n_cols)

    #     model_alias = config.name
    #     density = exp_setting.density
    #     times = exp_setting.times
    #     times_sec = times .* 3600
    #     p_out = Vector{Float64}(undef, length(times))

    #     Threads.@threads for n in 1:n_cols
    #         params = to_model_params(config, param_matrix[:, n])
    #         p_out .= compute_gresp_xform_params(model_alias, times_sec, density, params, exp_setting.def_params)
    #         d, rmse, se = fit_dantigny_to_germination_curve(p_out, times)
    #         # if any(d[2:3] .> 1e6) && any(se[2:3] .< 100)
    #         #     # d = [d[1], NaN, NaN]
    #         #     println("Large d=$d with (weirdly) SE $se")
    #         # end
    #         outputs[n] = d
    #         stderr[n]  = se
    #     end

    #     # Stack into n_outputs × N matrix; replace failed runs with NaN
    #     n_out = length(outputs[1])
    #     result = fill(NaN, n_out, n_cols)
    #     se_mat = fill(NaN, n_out, n_cols)
    #     for n in 1:n_cols
    #         if !any(isnan.(outputs[n])) && !any(isinf.(outputs[n]))
    #             result[:, n] .= outputs[n]
    #             se_mat[:, n] .= stderr[n]
    #         end
    #     end
    #     return result, se_mat
    # end

    # """
    # Compute the anchor output y0 (scalar per output dimension).
    # """
    # function compute_anchor_output(config::ModelConfig, exp_setting::ExperimentSetting)
    #     anchor_params = to_model_params(config, config.anchor)
    #     # println("anchor_params: $anchor_params")
    #     p_out = compute_gresp_xform_params(config.name, exp_setting.times .* 3600, exp_setting.density, anchor_params, exp_setting.def_params)
    #     d_vals, rmse, uncert = fit_dantigny_to_germination_curve(p_out, exp_setting.times)
    #     return d_vals  # [p_max, tau_g, nu]
    # end

    # """
    # Estimate first-order HDMR term for parameter j:
    #     f_j(θ_j) = y(θ_j; θ*_rest) - y0

    # Returns:
    #     sweep_vals  : the values of θ_j swept
    #     f_j_vals    : n_outputs × n_points matrix of f_j values
    #     y_vals      : n_outputs × n_points raw outputs (for score computation)
    # """
    # function compute_first_order_term(config::ModelConfig, j::Int,
    #                                 y0::Vector{Float64},
    #                                 exp_setting::ExperimentSetting;
    #                                 n_points::Int=100)
    #     sweep_vals, param_matrix = make_1d_sweep(config, j, n_points)
    #     y_vals, se_mat = run_model_batch(config, param_matrix, exp_setting)

    #     # f_j = y - y0, broadcast over columns
    #     f_j_vals = y_vals .- y0

    #     return sweep_vals, f_j_vals, y_vals, se_mat
    # end

    # """
    # Estimate pairwise interaction term for parameters j and k:
    #     f_jk(θ_j, θ_k) = y(θ_j, θ_k; θ*_rest) - f_j(θ_j) - f_k(θ_k) - y0

    # This requires first-order terms to already be computed.
    # Returns interaction term as n_outputs × (n_j*n_k) matrix.
    # """
    # function compute_pairwise_term(config::ModelConfig, j::Int, k::Int,
    #                                 y0::Vector{Float64},
    #                                 f_j_interp::Function,   # interpolated f_j
    #                                 f_k_interp::Function,   # interpolated f_k
    #                                 w_j_interp, w_k_interp,
    #                                 exp_setting::ExperimentSetting;
    #                                 n_j::Int=50, n_k::Int=50)

    #     sweep_j, sweep_k, param_matrix = make_2d_sweep(config, j, k, n_j, n_k)
    #     y_vals, se_mat = run_model_batch(config, param_matrix, exp_setting)

    #     n_total = n_j * n_k
    #     f_jk_vals = similar(y_vals)
    #     w_jk_from_1d = zeros(n_total)   # ← track 1D weight contribution

    #     idx = 1
    #     for vj in sweep_j, vk in sweep_k
    #         f_j = f_j_interp(vj)   # interpolated first-order term at vj
    #         f_k = f_k_interp(vk)   # interpolated first-order term at vk
    #         f_jk_vals[:, idx] .= y_vals[:, idx] .- f_j .- f_k .- y0
    #         # Weight from 1D reliability: a pairwise point is only as reliable
    #         # as the first-order terms subtracted from it
    #         w_jk_from_1d[idx] = min(w_j_interp(vj), w_k_interp(vk))
    #         idx += 1
    #     end

    #     return sweep_j, sweep_k, f_jk_vals, y_vals, se_mat, w_jk_from_1d
    # end

    # """
    # Simple linear interpolation of a first-order term for use in pairwise computation.
    # f_j_vals is n_outputs × n_points; sweep_vals is length n_points.
    # Returns a function: sweep_value → Vector{Float64} of length n_outputs.
    # """
    # function make_interpolator(sweep_vals::Vector{Float64}, f_j_vals::Matrix{Float64})
    #     return function(x::Float64)
    #         # Find bracketing indices
    #         idx = searchsortedfirst(sweep_vals, x)
    #         idx = clamp(idx, 2, length(sweep_vals))
    #         lo, hi = idx-1, idx
    #         # If either bracket is non-finite, return NaN (filtered downstream)
    #         if !all(isfinite, f_j_vals[:, lo]) || !all(isfinite, f_j_vals[:, hi])
    #             return fill(NaN, size(f_j_vals, 1))
    #         end
    #         t = (x - sweep_vals[lo]) / (sweep_vals[hi] - sweep_vals[lo])
    #         return (1-t) .* f_j_vals[:, lo] .+ t .* f_j_vals[:, hi]
    #     end
    # end

    # function make_scalar_interpolator(sweep_vals::Vector{Float64}, w::Vector{Float64})
    #     return function(x::Float64)
    #         idx = searchsortedfirst(sweep_vals, x)
    #         idx = clamp(idx, 2, length(sweep_vals))
    #         lo, hi = idx-1, idx
    #         t = (x - sweep_vals[lo]) / (sweep_vals[hi] - sweep_vals[lo])
    #         return (1-t) * w[lo] + t * w[hi]
    #     end
    # end

    # """
    # Compute variance-based sensitivity indices from HDMR terms.

    # Returns:
    #     S_first  : d × n_outputs  — first-order Sobol indices
    #     S_pair   : n_pairs × n_outputs — pairwise interaction indices
    #     S_total  : d × n_outputs  — total-order indices (first + all interactions)
    #     V_total  : n_outputs — total output variance (from all terms)
    # """
    # function compute_sensitivity_indices(
    #         f_first::Vector{Matrix{Float64}},   # length d, each n_outputs × n_points
    #         f_pairs::Vector{Matrix{Float64}},   # length n_pairs, each n_outputs × n_j*n_k
    #         target_pairs::Vector{Tuple{Int,Int}},
    #         d::Int, n_outputs::Int)

    #     # Variances of first-order terms
    #     V_first = zeros(d, n_outputs)
    #     for j in 1:d
    #         for o in 1:n_outputs
    #             valid = .!isnan.(f_first[j][o, :])
    #             V_first[j, o] = var(f_first[j][o, valid])
    #         end
    #     end

    #     # Variances of pairwise terms
    #     V_pair = zeros(length(target_pairs), n_outputs)
    #     for (p, (j, k)) in enumerate(target_pairs)
    #         for o in 1:n_outputs
    #             valid = .!isnan.(f_pairs[p][o, :])
    #             V_pair[p, o] = var(f_pairs[p][o, valid])
    #         end
    #     end

    #     # Total variance (sum of all terms — HDMR is an exact decomposition)
    #     V_total = vec(sum(V_first, dims=1)) .+ vec(sum(V_pair, dims=1))
    #     V_total .= max.(V_total, 1e-12)  # avoid division by zero

    #     # Normalised indices
    #     S_first = V_first ./ V_total'
    #     S_pair  = V_pair  ./ V_total'

    #     # Total-order: first-order + all interactions involving that parameter
    #     S_total = copy(S_first)
    #     for (p, (j, k)) in enumerate(target_pairs)
    #         S_total[j, :] .+= S_pair[p, :]
    #         S_total[k, :] .+= S_pair[p, :]
    #     end

    #     return S_first, S_pair, S_total, V_total
    # end

    # function weighted_var(x::Vector{Float64}, w::Vector{Float64})
    #     # Remove NaN entries
    #     valid = .!isnan.(x) .& .!isinf.(x) .& .!isnan.(w) .& (w .> 0)
    #     x_v, w_v = x[valid], w[valid]
    #     w_norm = w_v ./ sum(w_v)
    #     mean_w = dot(w_norm, x_v)
    #     return dot(w_norm, (x_v .- mean_w).^2)
    # end

    # function compute_sensitivity_indices_weighted(
    #         f_first::Vector{Matrix{Float64}}, f_pairs::Vector{Matrix{Float64}},
    #         weights_first::Vector{Vector{Float64}}, weights_pairs::Vector{Vector{Float64}},
    #         target_pairs::Vector{Tuple{Int,Int}}, d::Int, n_outputs::Int)

    #     V_first = zeros(d, n_outputs)
    #     for j in 1:d
    #         for o in 1:n_outputs
    #             V_first[j, o] = weighted_var(f_first[j][o, :], weights_first[j])
    #         end
    #     end

    #     V_pair = zeros(length(target_pairs), n_outputs)
    #     for (p, _) in enumerate(target_pairs)
    #         for o in 1:n_outputs
    #             V_pair[p, o] = weighted_var(f_pairs[p][o, :], weights_pairs[p])
    #         end
    #     end

    #     V_total = vec(sum(V_first, dims=1)) .+ vec(sum(V_pair, dims=1))
    #     V_total .= max.(V_total, 1e-12)

    #     S_first = V_first ./ V_total'
    #     S_pair  = V_pair  ./ V_total'

    #     S_total = copy(S_first)
    #     for (p, (j, k)) in enumerate(target_pairs)
    #         S_total[j, :] .+= S_pair[p, :]
    #         S_total[k, :] .+= S_pair[p, :]
    #     end

    #     return S_first, S_pair, S_total, V_total
    # end

    # """
    # Effective dimensionality: number of parameters with S_total > threshold.
    # """
    # function effective_dimensionality(S_total::Matrix{Float64};
    #                                 threshold::Float64=0.05)
    #     # Average over outputs, then count
    #     S_mean = vec(mean(S_total, dims=2))
    #     return sum(S_mean .> threshold), S_mean
    # end

    # """
    # Compute Mahalanobis-style z-score against lab data for each sweep sample.
    # lab_means: n_outputs vector
    # lab_covs:  n_outputs × n_outputs covariance matrix of lab uncertainty

    # Returns z_scores: length N vector (NaN for failed model runs).
    # """
    # function compute_predictive_scores(y_vals::Matrix{Float64},
    #                                 lab_means::Vector{Float64},
    #                                 lab_cov::Matrix{Float64})
    #     n_outputs, N = size(y_vals)
    #     Σ_inv = inv(lab_cov)
    #     z_scores = fill(NaN, N)

    #     for n in 1:N
    #         if !any(isnan.(y_vals[:, n]))
    #             diff = y_vals[:, n] .- lab_means
    #             z_scores[n] = dot(diff, Σ_inv * diff)
    #         end
    #     end
    #     return z_scores
    # end

    # """
    # Approximate log marginal likelihood from sweep samples via importance sampling.
    # Prior density cancels since sweeps are uniform on [lower, upper].
    # """
    # function approx_log_marginal_likelihood(z_scores::Vector{Float64})
    #     valid = .!isnan.(z_scores)
    #     if sum(valid) == 0
    #         return -Inf
    #     end
    #     log_likelihoods = -0.5 .* z_scores[valid]
    #     # Log-sum-exp for numerical stability
    #     lmax = maximum(log_likelihoods)
    #     return lmax + log(mean(exp.(log_likelihoods .- lmax)))
    # end

    # """
    # Compute weights based on standard error from Dantigny fitting
    # """
    # function compute_weights(se_mat::Matrix{Float64}; λ::Float64=1.0)
    #     # Use maximum SE across outputs as the plausibility signal
    #     max_se = vec(maximum(se_mat, dims=1))
    #     return 1.0 ./ (1.0 .+ λ .* max_se)
    # end

    # struct HDMRResult
    #     config          :: ModelConfig
    #     S_first         :: Matrix{Float64}   # d × n_outputs
    #     S_pair          :: Matrix{Float64}   # n_pairs × n_outputs
    #     S_total         :: Matrix{Float64}   # d × n_outputs
    #     S_mean          :: Vector{Float64}   # d — averaged over outputs
    #     d_eff           :: Int
    #     log_ml          :: Float64           # approximate log marginal likelihood
    #     z_scores_all    :: Vector{Float64}   # from all sweep samples combined
    # end

    # """
    # Perform High-Dimensional Model Representation (HDMR) on a specific model.
    # Workflow:
    #     1. Define anchor point and parameter ranges
    #     2. Compute first-order terms (single-parameter sweeps)
    #     3. Compute pairwise interaction terms (targeted 2D grids)
    #     4. Compute Sobol-style variance-based sensitivity indices
    #     5. Rank models by effective dimensionality and predictive score
    # inputs:
    #     config (ModelConfig) - model configuration
    #     exp_setting (ExperimentSetting) - experimental setting (time frames and density)
    #     lab_means (Vector{Float64}) - mean Dantigny values across the experimental settings
    #     lab_cov (Matrix{Float64}) - covariance matrix of lab Dantigny values
    #     n_first (Int) - number of points for the first-order sweeps 
    #     n_pair (Int) - number of points for the first-order sweeps 
    # """
    # function run_hdmr(config::ModelConfig,
    #                 exp_setting::ExperimentSetting,
    #                 lab_means::Vector{Float64},
    #                 lab_cov::Matrix{Float64};
    #                 n_first::Int=100,
    #                 n_pair::Int=50)

    #     d = length(config.anchor)
    #     n_outputs = length(lab_means)

    #     println("\n=== Running HDMR: $(config.name) ===")

    #     # --- Anchor ---
    #     println("Computing anchor output...")
    #     y0 = compute_anchor_output(config, exp_setting)
    #     println("  y0 = $y0")

    #     # --- First-order sweeps ---
    #     println("First-order sweeps ($d parameters × $n_first points)...")
    #     f_first   = Vector{Matrix{Float64}}(undef, d)
    #     w_first   = Vector{Vector{Float64}}(undef, d)
    #     interps   = Vector{Function}(undef, d)
    #     w_interps = Vector{Function}(undef, d)
    #     all_y_vals = Matrix{Float64}[]

    #     for j in 1:d
    #         print("  θ_$(config.param_names[j])... ")
    #         sweep_vals, f_j_vals, y_vals, se_mat = compute_first_order_term(
    #             config, j, y0, exp_setting; n_points=n_first)
    #         f_first[j] = f_j_vals

    #         λ = 1.0 / median(filter(!isnan, se_mat))
    #         # println(typeof(compute_weights(se_mat; λ)))
    #         w_first[j] = min.(compute_weights(se_mat; λ), experimental_relevance_weight.(y_vals[2, :], 168.0)) # T_max = 168 hours
    #         w_first[j][isnan.(w_first[j])] .= 0.0 # Zero NaN weights
    #         # println("1st order weights: $(w_first[j])")

    #         # Nullify f_j_vals at points with zero weight — prevents large finite
    #         # values from propagating through the interpolator into f_jk
    #         f_first[j][:, w_first[j] .== 0.0] .= NaN

    #         w_interps[j] = make_scalar_interpolator(sweep_vals, w_first[j])

    #         # large_vals = any(y_vals .> 1e6; dims=1) |> vec
    #         # println("Weights for large vals $(y_vals[:, large_vals]):\n$(w_first[j][large_vals]))")

    #         interps[j] = make_interpolator(sweep_vals, f_j_vals)
    #         push!(all_y_vals, y_vals)
    #         println("done")
    #     end

    #     # --- Pairwise sweeps ---
    #     n_pairs = length(config.target_pairs)
    #     f_pairs = Vector{Matrix{Float64}}(undef, n_pairs)
    #     w_pairs = Vector{Vector{Float64}}(undef, n_pairs)
    #     println("Pairwise sweeps ($n_pairs targeted pairs × $(n_pair^2) points each)...")

    #     for (p, (j, k)) in enumerate(config.target_pairs)
    #         print("  ($(config.param_names[j]), $(config.param_names[k]))... ")
    #         _, _, f_jk_vals, y_vals, se_mat, w_from_1d = compute_pairwise_term(
    #             config, j, k, y0, interps[j], interps[k], w_interps[j], w_interps[k], exp_setting;
    #             n_j=n_pair, n_k=n_pair)
    #         f_pairs[p] = f_jk_vals

    #         λ = 1.0 / median(filter(!isnan, se_mat))
    #         w_se    = compute_weights(se_mat; λ)
    #         w_range = experimental_relevance_weight.(y_vals[2, :], 168.0)

    #         # All three must agree: pairwise run reliable, output in range, 1D terms reliable
    #         w_pairs[p] = min.(w_se, w_range, w_from_1d)
    #         w_pairs[p][isnan.(w_pairs[p])] .= 0.0
    #         # println("2nd order weights: $(w_pairs[p])")

    #         f_pairs[p][:, w_pairs[p] .== 0.0] .= NaN

    #         # large_vals = any(y_vals .> 1e6; dims=1) |> vec
    #         # println("Weights for large vals $(y_vals[:, large_vals]):\n$(w_pairs[p][large_vals]))")

    #         push!(all_y_vals, y_vals)
    #         println("done")
    #     end

    #     # --- Sensitivity indices ---
    #     println("Computing sensitivity indices...")
    #     S_first, S_pair, S_total, V_total = compute_sensitivity_indices_weighted(
    #         f_first, f_pairs, w_first, w_pairs, config.target_pairs, d, n_outputs)
    #     println("S_first = $S_first")
    #     println("S_pair = $S_pair")
    #     println("S_total = $S_total")
    #     println("V_total = $V_total")
    #     d_eff, S_mean = effective_dimensionality(S_total)

    #     # --- Predictive scores across all sweep samples ---
    #     println("Computing predictive scores...")
    #     all_y = hcat(all_y_vals...)
    #     z_scores = compute_predictive_scores(all_y, lab_means, lab_cov)
    #     log_ml = approx_log_marginal_likelihood(z_scores)

    #     println("  Effective dimensionality: $d_eff / $d")
    #     println("  Approx. log marginal likelihood: $(round(log_ml, digits=2))")

    #     return HDMRResult(config, S_first, S_pair, S_total, S_mean,
    #                     d_eff, log_ml, z_scores)
    # end

    # """
    # Perform sensitivity analysis by:
    # 1. Analysing parameter coupling via HDMR
    # 2. Running Bayesian Model Averaging
    # """
    # function sensitivity_analysis(def_params, bounds_abs, bounds_rel, anchor_dict)

    #     # Load data
    #     aliases, combination_IDs, descriptions, param_key_sets = load_model_collection()
    #     times, sources, densities, lab_means, CIs, CI_widths, uncert_lab = unpack_ijadpanahsaravi_data()

    #     n_src = length(sources)
    #     n_dens = length(densities)
        
    #     lab_means_global = mean(lab_means; dims=(1, 2)) |> vec # Take average lab means across exp.settings as reference for HDMR
    #     lab_covs = zeros(Float64, n_dens, n_src, 3, 3)
    #     for i in 1:n_dens
    #         for j in 1:n_src
    #             lab_covs[i, j, :, :] .= diagm(uncert_lab[i, j, :].^2)
    #         end
    #     end
    #     lab_cov_global = dropdims(mean(lab_covs; dims=(1, 2)); dims=(1, 2)) # Take average covariance matrix across exp.settings as reference for HDMR
    #     densities_scaled = inverse_mL_to_cubic_um.(densities)
    #     mean_density = mean(densities_scaled) # Take average density as reference for HDMR

    #     # Experimental setting for HDMR
    #     exp_setting = ExperimentSetting(times, mean_density, def_params)

    #     # Determine relative vs absolute threshold models
    #     ids_rel = ["0", "B", "Bi"]

    #     # Determine global vs inducer-specific parameters
    #     params_glob_keys = [:b, :Pₛ, :μ_γ, :neg_δ_γ, :μ_ψ, :neg_δ_ψ]

    #     # Group parameters for couplings
    #     inducer_params = [:Pₛ_cs, :K_cC, :μ_ω, :neg_δ_ω]
    #     inhibitor_params = [:Pₛ, :μ_γ, :neg_δ_γ]
    #     coupling_types = Dict(
    #         "pure_thresholds" => [(i,j) for group in [inducer_params, inhibitor_params] for i in group for j in group if i < j]
    #     )

    #     # Iterate over models
    #     for m in 1:1#59:59#eachindex(aliases)

    #         println("Running $(aliases[m])")

    #         param_keys = param_key_sets[m]

    #         n_dims = length(param_keys)

    #         # Determine bounds
    #         if combination_IDs[m] in ids_rel
    #             bounds = bounds_rel
    #         else
    #             bounds = bounds_abs
    #         end

    #         # ----- PERFORM HDMR -----

    #         # Construct model configuration
    #         lower = zeros(Float64, n_dims)
    #         upper = similar(lower)
    #         log_scaled = zeros(Bool, n_dims)
    #         anchor = similar(lower)
    #         for (k, key) in enumerate(param_keys)
    #             if startswith(string(key), "neg_δ")
    #                 log_scaled[k] = false
    #                 # lower[k] = max(anchor_dict[key] * exp(-2), bounds[key][1])
    #                 lower[k] = bounds[key][1]
    #                 # upper[k] = min(anchor_dict[key] * exp(2), bounds[key][2])
    #                 upper[k] = bounds[key][2]
    #                 anchor[k] = anchor_dict[key]
    #                 println("$key: lower = $(bounds[key][1]) / lower (sweep) = $(lower[k]) / anchor = $(anchor[k]) / upper (sweep) = $(upper[k]) / upper = $(bounds[key][2])")
    #             else
    #                 log_scaled[k] = true
    #                 lower[k] = log(bounds[key][1])
    #                 # lower[k] = log(max(bounds[key][1], anchor_dict[key] * exp(-2)))
    #                 upper[k] = log(bounds[key][2])
    #                 # upper[k] = log(min(bounds[key][2], anchor_dict[key] * exp(2)))
    #                 anchor[k] = log(anchor_dict[key])
    #                 println("$key: lower = $(log(bounds[key][1])) / lower (sweep) = $(lower[k]) / anchor = $(anchor[k]) / upper (sweep) = $(upper[k]) / upper = $(log(bounds[key][2]))")
    #             end
                
    #         end
    #         coupling_indices = [(findfirst(==(k1), param_keys), findfirst(==(k2), param_keys)) for (k1, k2) in coupling_types["pure_thresholds"]]

    #         config = ModelConfig(
    #             aliases[m],
    #             param_key_sets[m],
    #             anchor,
    #             lower,
    #             upper,
    #             log_scaled,
    #             coupling_indices
    #         )

    #         output = run_hdmr(config, exp_setting, lab_means_global, lab_cov_global)
            
    #         println(output)

    #     end
    # end

    """
    Compute likelihoods based on Dantigny summary
    Mahalanobis distances.
    inputs:
        alias (String)              : model alias
        times (Vector)              : vector of time points (in hours)
        densities (Vector)          : unique spore densities used in the data
        sources (Vector)            : unique string identifiers of carbon sources
        param_arr (Matrix{Float64}) : matrix of parameter samples ([n_dens x n_samples] x 22)
        lab_means (Matrix{Float64}) : mean Dantigny values across the experimental settings (n_dens x n_src x 3)
        lab_covs (Matrix{Float64})  : covariance matrices of lab Dantigny values across the experimental settings (n_dens x n_src x 3 x 3)
    """
    function likelihood_scores(alias, times, densities, sources, param_arr, lab_means, lab_covs)

        n_dens = length(densities)
        n_samples_total = size(param_arr, 1)
        n_samples = div(n_samples_total, n_dens)

        l_scores = zeros(Float64, n_samples) # Log-likelihood scores for each sample

        # Run model for all densities
        @time germination = model_wrapper(alias, param_arr, times)

        # Fit Dantigny to the model outputs
        dantigny_summaries = zeros(Float64, 3, n_samples_total) # 3 parameters (p_max, tau_g, nu) X n_samples_total
        rmse_vals = Vector{Float64}(undef, n_samples_total)
        @time Threads.@threads for k in 1:n_samples_total
            p_opt, rmse = fit_dantigny_to_germination_curve(germination[k, 2:end], times[2:end]) # skip 1st time sample
            dantigny_summaries[:, k] = p_opt
            rmse_vals[k] = rmse
        end

        if any(rmse_vals .> 0.05)
            println("Warning: Some RMSE values are very large (>5%), indicating poor fits.")
            # println(rmse_vals)
        end

        # Compare model output against lab data for each source and density
        @time for (i, rho_s) in enumerate(densities)
            println("  Density: $rho_s")
            d_summaries_subset = dantigny_summaries[:, (i - 1) * n_samples + 1 : i * n_samples]
            for (j, src) in enumerate(sources)
                println("    Source: $src")

                lab_mean = lab_means[i, j, :]
                lab_cov = lab_covs[i, j, :, :]

                # Compute Mahalanobis distance for each sample
                # z_scores = Vector{Float64}(undef, n_samples)
                # for k in 1:n_samples
                #     diff = d_summaries_subset[:, k] .- lab_mean
                #     z_scores[k] = min(dot(diff, lab_cov \ diff), 1e8)
                # end
                D = d_summaries_subset .- lab_mean
                X = lab_cov \ D
                z_scores = min.(vec(sum(D .* X, dims=1)), 1e8)

                # println("    z-scores vary between $(minimum(z_scores)) and $(maximum(z_scores))")
                l_scores .+= z_scores
            end
        end

        # Run model for each density
        # for (i, rho_s) in enumerate(densities)
        #     println("  Density: $rho_s")

        #     # Update current density
        #     param_arr[:, 1] .= Float32.(rho_s)

        #     # Compute germination
        #     germination = model_wrapper(model_alias, param_arr, times)

        #     # Fit Dantigny to the model outputs
        #     dantigny_summaries = zeros(Float64, 3, n_samples) # 3 parameters (p_max, tau_g, nu) X n_samples
        #     rmse_vals = Vector{Float64}(undef, n_samples)
        #     for k in 1:n_samples
        #         p_opt, rmse = fit_dantigny_to_germination_curve(germination[k, :], times)
        #         dantigny_summaries[:, k] = p_opt
        #         rmse_vals[k] = rmse
        #     end

        #     if any(rmse_vals .> 0.05)
        #         println("Warning: Some RMSE values are very large (>5%), indicating poor fits.")
        #     end

        #     # println("    Dantigny summaries (p_max, tau_g, nu):")
        #     # for k in 1:n_samples
        #     #     println("    Sample $k: $(round.(dantigny_summaries[:, k], digits=4))")
        #     # end

        #     # Compare model output against lab data for each source
        #     for (j, src) in enumerate(sources)
        #         println("    Source: $src")

        #         lab_mean = lab_means[i, j, :]
        #         lab_cov = lab_covs[i, j, :, :]

        #         # Compute Mahalanobis distance for each sample
        #         z_scores = Vector{Float64}(undef, n_samples)
        #         for k in 1:n_samples
        #             diff = dantigny_summaries[:, k] .- lab_mean
        #             z_scores[k] = min(dot(diff, lab_cov \ diff), 1e8)
        #         end

        #         # println("    z-scores vary between $(minimum(z_scores)) and $(maximum(z_scores))")
        #         l_scores .+= z_scores
        #     end
        # end

        l_scores .= -0.5 .* l_scores # Convert to log-likelihoods

        return l_scores
    end

    """
    Calculate Gelman-Rubin PSRF for MCMC convergence.
    
    Arguments:
        chains (Matrix)         : MCMC chains of size (n_mc_steps, n_theta, n_samples)
        burn_in_frac (Float64)  : Fraction of samples to discard as burn-in
        threshold (Float54)     : Convergence threshold (default 1.1)
    
    Returns:
        PSRF: Vector of R̂ values for each parameter
        converged: Boolean indicating overall convergence
    """
    function gelman_rubin_PSRF(chains, burn_in_frac=0.5, threshold=1.1)
        
        n_steps_total = size(chains, 1)
        n_params = size(chains, 2)
        n_chains = size(chains, 3)
        
        # Burn-in
        burn_in = Int(floor(n_steps_total * burn_in_frac))
        chains_burned = chains[burn_in+1:end, :, :]
        n_steps = size(chains_burned, 1)
        
        # Within-chain variance
        W = dropdims(mean(var(chains_burned, dims=1), dims=3), dims=(1, 3))
        # W = zeros(n_params)
        # for j in 1:n_params
        #     # W[j] = mean(var(chains_burned[i][:, j]) for i in 1:n_chains)
        #     W[j] = mean(var(chains_burned[:, j, i]) for i in 1:n_chains)
        # end
        
        # Between-chain variance
        B = n_steps .* dropdims(var(mean(chains_burned, dims=1), dims=3), dims=(1,3))
        # B = zeros(n_params)
        # for j in 1:n_params
        #     chain_means = [mean(chains_burned[i][:, j]) for i in 1:n_chains]
        #     B[j] = n * var(chain_means)
        # end
        
        # Pooled variance
        V_hat = ((n_steps - 1) / n_steps) .* W .+ (1 / n_steps) .* B
        
        # PSRF
        PSRF = sqrt.(V_hat ./ W)
        
        # Convergence check
        converged = all(PSRF .< threshold)
        
        return PSRF, converged
    end

    """
    Assigns values from parameter particles
    to model input matrix.
    inputs:
        param_arr(Matrix{Float64})   : input parameter matrix of size ([n_samples x n_dens] x 22)
        theta (Matrix{Float64})      : parameter particle of size (n_dims x n_samples)
        n_dens (Int)                 : number of unique spore densities
        param_mapping (Vector{Int})  : mapping of parameter indices to model input columns
        sigma_param_map (Vector{Int})  : mapping of sigma parameter indices to mu parameter indices
    """
    function particle_to_input_params!(param_arr, theta, n_dens, param_map, sigma_param_map)
        n_samples = size(theta, 2)
        for (i, idx) in enumerate(param_map)
            for j in 1:n_dens
                start_idx = (j - 1) * n_samples + 1
                end_idx = j * (n_samples)
                param_arr[start_idx:end_idx, idx] .= theta[i, :]
                # Exponentiate if sigma parameter
                mu_idx = sigma_param_map[i]
                if mu_idx != -1
                    param_arr[start_idx:end_idx, idx] .= theta[mu_idx, :] .* exp.(theta[i, :])
                end
            end
        end
        return param_arr
    end

    """
    Perform Sequential Monte Carlo (SMC) for a specific model
    to obtain Bayesian Model Averaging weights.
    inputs:
        alias (String)      : model alias
        p0 (Dict)           : initial parameter vector (anchor) for the specific model
        priors (Vector)     : prior distributions for each parameter
        n_samples (Int)     : number of particles
        n_smc_steps (Int)   : number of SMC steps
        n_mc_steps (Int)    : number of MCMC steps (mutation)
        t_max (Float64)     : maximum time span for germination
    """
    function sequential_monte_carlo(alias, p0, priors, n_samples, n_smc_steps, n_mc_steps, t_max)

        times, sources, densities, lab_means, CIs, CI_widths, uncert_lab = unpack_ijadpanahsaravi_data()
        
        # Convert times and densities
        times = Float32.(times) # Convert to Float32 for model input
        densities = inverse_mL_to_cubic_um.(densities) # Convert to cubic micrometers for model input

        # Construct covariance matrices for lab data
        n_src = length(sources)
        n_dens = length(densities)
        lab_covs = zeros(Float64, n_dens, n_src, 3, 3)
        for i in 1:n_dens
            for j in 1:n_src
                lab_covs[i, j, :, :] = diagm(uncert_lab[i, j, :].^2) # Variance from standard error
            end
        end

        param_dict = Dict{Symbol, Vector{Float64}}()
        param_key_mapping = [:rho_s,
                            :mu_O, :neg_delta_O,
                            :mu_R, :neg_delta_R,
                            :mu_H, :neg_delta_H,
                            :mu_X, :neg_delta_X,
                            :mu_Y, :neg_delta_Y,
                            :c_ex, :P_I, :P_C, :K_s,
                            :lambda_I, :lambda_C,
                            :k_gamma, :k_omega,
                            :K_I, :n, :K_b] # mapping within param_dict_to_matrix()

        # Initialise parameter matrix with anchor values
        param_arr = zeros(Float32, n_samples * n_dens, 22)
        for (i, key) in enumerate(param_key_mapping)
            if haskey(p0, key)
                param_arr[:, i] .= p0[key]
            end
        end

        # Determine relevant (variable) parameters for current model
        relevant_key_indices = Int[]    # Indices of relevant parameters in param_key_mapping
        p_theta = Distribution[]        # Relevant parameter priors
        theta_srch_sigma = Float64[]    # Relevant parameter search sd's
        min_srch_sigma = eps(Float64)
        srch_sigma_fraction = 1e-2
        sigma_param_map = Int[] # Index mapping of sigma to mu parameters (and -1 if not applicable)
        all_var_param_keys = collect(keys(priors))
        for key in keys(priors) # All variable parameters
            if haskey(p0, key) # Variable parameters relevant for current model

                key_idx_in_mapping = findfirst(==(key), param_key_mapping)
                push!(relevant_key_indices, key_idx_in_mapping)
                push!(p_theta, priors[key])
                push!(theta_srch_sigma, max(srch_sigma_fraction * priors[key].σ, min_srch_sigma))

                # Relate neg_delta param to mu param by index
                if startswith(string(key), "neg_delta")
                    suffix = string(key)[11]  # Extract the suffix after "neg_delta"
                    mu_key = Symbol("mu_" * suffix)
                    mu_idx = findfirst(==(mu_key), all_var_param_keys)
                    push!(sigma_param_map, mu_idx)
                else
                    push!(sigma_param_map, -1)
                end
            end
        end

        # Sobol sample
        n_dims = length(relevant_key_indices)
        sobol_pts = QuasiMonteCarlo.sample(n_samples, n_dims, SobolSample())
        sobol_pts = 0.025 .+ 0.95 .* sobol_pts # Shrink samples to 95%

        # Sample priors and fill parameter-related collections
        # relevant_param_keys = Symbol[]
        # relevant_key_indices = Int[] # Indices of relevant parameters in param_key_mapping
        # p_theta = Distribution[]
        # # theta = Vector{Float64}[]
        # theta_srch_sigma = Float64[]
        # min_srch_sigma = eps(Float64)
        # srch_sigma_fraction = 1e-2
        # all_param_keys = collect(keys(p0))
        # for key in keys(priors) # All variable parameters
        #     if haskey(p0, key) # Parameters relevant for this specific model

        #         sobol_idx = findfirst(==(key), all_param_keys)
        #         param_dict[key] = quantile.(priors[key], sobol_pts[sobol_idx, :])

        #         key_idx_in_mapping = findfirst(==(key), param_key_mapping)
        #         # push!(relevant_param_keys, key)
        #         push!(relevant_key_indices, key_idx_in_mapping)
        #         push!(p_theta, priors[key])
        #         # push!(theta, param_dict[key])
        #         push!(theta_srch_sigma, max(srch_sigma_fraction * priors[key].σ, min_srch_sigma))

        #         # Determine Gaussian search widths for each parameter
        #         # based on fraction of prior standard deviation
        #         # param_srch_sigma[key] = max(sigma_fraction * priors[key].σ, min_srch_sigma)
        #     else
        #         param_dict[key] = zeros(n_samples)  # or some default value
        #     end
        # end

        # Sort by relevant_key_indices to maintain consistent order
        sorted_indices = sortperm(relevant_key_indices)
        relevant_key_indices = relevant_key_indices[sorted_indices]
        p_theta = p_theta[sorted_indices]
        theta_srch_sigma = theta_srch_sigma[sorted_indices]
        # n_theta = length(relevant_key_indices)

        isnormal = typeof.(p_theta) .== Normal{Float64}
        # println(relevant_param_keys)
        # println(theta_srch_sigma)
        # println(isnormal)

        # println("Sorted indices: $sorted_indices")
        println("Sorted keys: $(param_key_mapping[relevant_key_indices])")
        println("Normal: $isnormal")

        # Convert parameter dictionary to vector
        # theta = reduce(hcat, theta)'
        # n_theta = size(theta, 1)

        # Construct particles from sampled priors
        theta = quantile.(p_theta, sobol_pts)

        # Update input parameter matrix
        particle_to_input_params!(param_arr, theta, n_dens, relevant_key_indices, sigma_param_map)

        # Construct input parameter array
        # param_arr = zeros(Float32, n_samples * n_dens, 22)
        # for (i, idx) in enumerate(relevant_key_indices)
        #     for j in n_dens
        #         param_arr[:, idx + (j-1)*n_samples] .= theta[i, :]
        #         # Exponentiate if sigma parameter
        #         mu_idx = sigma_param_map[i]
        #         if mu_idx != -1
        #             param_arr[:, idx + (j-1)*n_samples] .= param_arr[:, mu_idx] .* exp.(theta[i, :])
        #         end
        #     end
        # end

        # Duplicate default anchors
        # for (key, value) in p0
        #     if !haskey(param_dict, key)
        #         param_dict[key] = fill(value, n_samples)
        #     end
        # end

        # Process exponents
        # sigma_param_map = fill(0, n_theta) # for mapping sigma params to mu params
        # p_ct = 1
        # for (key, prior) in priors
        #     if startswith(string(key), "neg_delta")
        #         suffix = string(key)[11]  # Extract the suffix after "neg_delta"
        #         param_dict[Symbol("sigma_" * suffix)] = param_dict[Symbol("mu_" * suffix)] .* exp.(param_dict[key])
                
        #         # Map sigma index to mu index for variable parameters (for later perturbation)
        #         if key in relevant_param_keys
        #             mu_idx = findfirst(==(Symbol("mu_" * suffix)), collect(keys(priors)))
        #             sigma_param_map[mu_idx] = p_ct
        #         end
        #     end
        #     if key in relevant_param_keys
        #         p_ct += 1
        #     end
        # end

        # Convert parameter dictionary with density values to matrix
        # param_arr = zeros(Float32, n_samples * n_dens, 22) # 22 parameters in total
        # for (i, rho_s) in enumerate(densities)
        #     start_idx = (i - 1) * n_samples + 1
        #     end_idx = i * n_samples
        #     param_arr[start_idx:end_idx, :] .= param_dict_to_matrix(param_dict, rho_s)
        # end
        
        # Parameter particles (independent of density)
        # theta = param_arr[relevant_key_indices, :]'

        # Initialise weights
        weights = fill(1.0 / n_samples, n_samples)

        # Initialise temperature
        temp = 0.0
        ess_threshold = 0.5 * n_samples

        # SMC
        for n in 1:n_smc_steps
            println("SMC step $n / $n_smc_steps")

            # --- LIKELIHOODS FOR TEMPERATURE UPDATE ---
            l_scores = likelihood_scores(alias, times, densities, sources, param_arr, lab_means, lab_covs)

            # --- TEMPERATURE UPDATE ---
            # Update temperature with largest increase that
            # keeps ESS around threshold (e.g., 0.5 * n_samples)
            temp_low = temp
            temp_high = 1.0
            weights_candidate = weights
            # println("    l-scores vary between $(minimum(l_scores)) and $(maximum(l_scores))")
            while temp_high - temp_low > 1e-8
                
                temp_mid = (temp_low + temp_high) * 0.5

                weights_candidate = weights .* exp.((temp_mid - temp_low) .* l_scores)
                weights_candidate ./= sum(weights_candidate) # Normalize weights
                ess_candidate = 1.0 / sum(weights_candidate .^ 2)

                # println("    Candidate temperature: $temp_mid, ESS: $(round(ess_candidate, digits=2))")

                if ess_candidate > ess_threshold
                    temp = temp_mid
                    break
                else
                    temp_high = temp_mid
                end
            end
            weights = weights_candidate
            println("    Updated temperature: $temp, ESS: $(round(1.0 / sum(weights .^ 2), digits=2))")

            # Resample particles
            resample_indices = wsample(collect(1:n_samples), weights, n_samples)
            theta = theta[:, resample_indices]
            # for (i, key) in enumerate(relevant_param_keys)
            #     param_dict[key] .= param_dict[key][resample_indices]
            # end

            # Evaluate priors
            l_prior = dropdims(sum(logpdf.(reshape(p_theta, :, 1), theta), dims=1), dims=1)

            return

            # --- MUTATION ---
            converged_mask = fill(false, n_samples)
            mcmc_chains = zeros(Float64, n_mc_steps, n_theta, n_samples)
            for m in 1:n_mc_steps

                println("        Running mutation step $m / $n_mc_steps")
                
                perturb_mask = .!converged_mask
                n_perturb = sum(perturb_mask)

                # Perturb (duplicate) parameter values
                theta_candidates = copy(theta)
                srch_sample = rand(Float64, n_theta, n_perturb)
                perturb_means = ifelse.(isnormal, theta[:, perturb_mask], log.(theta[:, perturb_mask])) # Use log-means for LogNormal
                theta_candidates[:, perturb_mask] .= quantile.(Normal.(perturb_means, reshape(theta_srch_sigma, :, 1)), srch_sample)
                theta_candidates[.!isnormal, perturb_mask] .= exp.(theta_candidates[.!isnormal, perturb_mask]) # Convert back to LogNormal

                println(maximum(abs.(theta .- theta_candidates), dims=2))

                # Convert to dictionary for model input
                for (i, key) in enumerate(relevant_param_keys)
                    param_dict[key] .= theta_candidates[i, :]
                    # Exponentiate sigmas
                    if startswith(string(key), "neg_delta")
                        suffix = string(key)[11]  # Extract the suffix after "neg_delta"
                        # param_dict[Symbol("sigma_" * suffix)] = param_dict[Symbol("mu_" * suffix)] .* exp.(param_dict[key])
                        param_dict[Symbol("sigma_" * suffix)] = theta_candidates[sigma_param_map[i], :] .* exp.(param_dict[key])
                    end
                end

                # println(param_dict)

                # Likelihoods for acceptance probability
                @time l_scores_candidates = likelihood_scores(alias, times, densities, sources, param_arr, lab_means, lab_covs)

                # Evaluate priors
                l_prior_candidates = dropdims(sum(logpdf.(reshape(p_theta, :, 1), theta_candidates), dims=1), dims=1)

                # Acceptance checks
                log_ratio = temp * l_scores_candidates + l_prior_candidates - temp * l_scores - l_prior
                alpha = min.(1, exp.(log_ratio))
                accept_mask = rand(Float64, n_samples) .< alpha
                theta[:, accept_mask] .= theta_candidates[:, accept_mask]
                # println("$(sum(accept_mask)) candidates accepted.")

                # if m > 10 && m % 10 == 0
                #     PSRF, converged = gelman_rubin_PSRF(mcmc_chains, 0.5, 1.1)
                #     println("PSRF = $PSRF, converged = $converged")
                # end

                mcmc_chains[m, :, :] .= theta
            end
            PSRF, converged = gelman_rubin_PSRF(mcmc_chains, 0.5, 1.1)
            println("PSRF = $PSRF, converged = $converged")
        end

        # return dantigny_summaries, rmse_vals
    end

end