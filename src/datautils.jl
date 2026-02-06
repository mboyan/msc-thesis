module DataUtils
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
    using ProgressMeter
    
    include("./conversions.jl")
    include("./germstats.jl")
    using .Conversions
    using .GermStats

    export calibrate_marginals
    export calibrate_copula
    export sample_parameters
    export parse_ijadpanahsaravi_data
    export calibrate_priors
    export dantigny
    export generate_dantigny_dataset
    export train_multioutput_nn_mixed_precision
    export predict_surrogate_gpu
    export predict_with_uncertainty_mixed_precision
    export fit_dantigny_to_germination_curve
    export fit_model_to_data
    export get_params_for_idx
    export fit_model_to_data_equilibrium

    CUDA.allowscalar(false)  # Prevent slow scalar operations


    # ===== Statistics =====
    function weighted_median(values::Vector{T}, weights::Vector{T}) where T
        """
        Compute a weighted median from a set of values and weights.
        """

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

    function weighted_quantile(x::AbstractVector, w::AbstractVector, p::Real)
        """
        Compute a quantile of a set of weighted values.
        inputs:
            x (Vector) - values
            w (Vector) - weights
            p (Real) - quantile
        output:
            the linearly interpolated value at the given quantile
        """

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

    function fit_dist(dist_current, θ, w_norm; α=0.25)
        """
        Refit a prior distribution through a set of weighted samples
        inputs:
            dist_current (Distribution) - current Distribution
            θ (Vector) - samples
            w_norm (Vector) - normalized weights
            α (Float) - learning rate
        output:
            the fitted prior distribution
        """

        μ_current = dist_current.μ
        σ_current = dist_current.σ
        dist_type = typeof(dist_current)

        ess = 1 / sum(w_norm .^ 2) # effective sample size
        ess_ratio = ess / length(w_norm)
        exploration_factor = 1.0 + ess_ratio * 0.5
        # println("ESS: $ess, exploration factor: $exploration_factor")

        if dist_type == LogNormal{Float64}

            x = log.(θ)
            m = weighted_median(vec(x), w_norm)
            iqr = weighted_quantile(x, w_norm, 0.75) - weighted_quantile(x, w_norm, 0.25)
            σ = iqr / 1.349

            μ_interp = μ_current + α * (m - μ_current)
            σ_interp = σ_current + α * (σ * exploration_factor - σ_current)

        elseif dist_type == Normal{Float64}
            
            μ = sum(w_norm .* θ)
            σ = sqrt(sum(w_norm .* (θ .- μ).^2))

            μ_interp = μ_current + α * (μ - μ_current)
            σ_interp = σ_current + α * (σ * exploration_factor - σ_current)

        else
            error("Invalid distribution type")
        end

        return dist_type(μ_interp, σ_interp)
    end

    function nearestSPD(A)
        """
        Perform near-PD projection (e.g. for when
        a matrix is not Cholesky factorizable)
        inputs:
            A (Matrix) - matrix to be corrected
        outputs:
            A3 (Matrix) - SPD matrix
        """
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

    function calibrate_marginals(marg_current, θ, weights)
        """
        Calibrate a set of distributions given a set of weighted samples.
        Optionally, condition-specific weights can also be used.
        inputs:
            marg_current (Vector{Distribution}) - the current distributions
            θ (Matrix) - parameter samples
            weights (Vector) - sample weights
        output:
            marginals (Vector{Distribution}) - the calibrated distributions
        """
        Np = size(θ, 1)

        # if isnothing(w_spec) # Global marginal calibration
        #     w_eff = w_glob ./ sum(w_glob)
        # else
        #     w_eff = w_glob .* w_spec # Specific marginal calibration
        #     w_eff ./= sum(w_eff)
        # end

        weights ./= sum(weights)

        marginals = Vector{Distribution}(undef, Np)
        for i in 1:Np
            marginals[i] = fit_dist(marg_current[i], θ[i, :], weights)
        end
        return marginals
    end

    function to_gaussian_space(Θg, Θs, marg_g, marg_s; eps_u = nothing)
        """
        Transform global and condition-specific parameter
        values from given marginal distributions
        to Gaussian space (for computing a Gaussian copula).
        inputs: 
            Θg (Matrix) - global parameter values
            Θs (Matrix) - condition-specific parameter values
            marg_g (Vector{Distribution}) - global parameter distributions
            marg_s (Vector{Distribution}) - condition-specific parameter distributions
            eps_u (Float) - tolerance avoiding the sampling of Inf
        output:
            Z (Matrix) - transformed parameter values
        """
        Ng = size(Θg, 1)
        Nc = size(Θs, 1)
        Ns = size(Θg, 2)
        Z = zeros(Ns, Ng + Nc)

        # Avoid U of 0 and 1
        if eps_u === nothing
            eps_u = 1 / (Ns + 1)
        end

        for i in 1:Ng
            u = cdf.(marg_g[i], Θg[i, :])
            u = clamp.(u, eps_u, 1 - eps_u)
            Z[:, i] = quantile.(Normal(), u)
        end
        for i in 1:Nc
            u = cdf.(marg_s[i], Θs[i, :])
            u = clamp.(u, eps_u, 1 - eps_u)
            Z[:, Ng + i] =
                quantile.(Normal(), u)
        end
        return Z
    end

    function weighted_correlation(Z, w)
        """
        Compute the correlation of weighted
        Gaussian-transformed parameter values.
        inputs:
            Z (Matrix) - transformed parameter values
            w (Vector) - weights
        output:
            correlation matrix
        """
        μ = sum(w .* Z, dims=1)
        Σ = zeros(size(Z, 2), size(Z, 2))

        for i in eachindex(w)
            δ = Z[i, :] .- μ
            Σ .+= w[i] .* (δ' * δ)
        end

        D = diagm(0 => sqrt.(diag(Σ)))
        return inv(D) * Σ * inv(D)
    end

    function calibrate_copula(Θg, Θs, marg_g, marg_s, w_glob, w_spec)
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
        output:
            multivariate distribution of weighted samples
        """

        w_eff = w_glob .* w_spec
        w_eff ./= sum(w_eff)

        Z = to_gaussian_space(Θg, Θs, marg_g, marg_s)
        R = weighted_correlation(Z, w_eff)

        # Enforce symmetry
        R = (R + R') / 2

        # Regularise
        ϵ = 1e-6 * tr(R) / size(R,1)
        R_reg = R + ϵ * I

        # Near-PD projection
        # println("Eigenvalue: $(eigmin(R_reg))")
        # println("max(|R - R'|) = $(maximum(abs.(R - R')))")
        if eigmin(R_reg) <= 0#minimum(eigen(Symmetric(R)).values) <= 0
            R_reg = nearestSPD(R_reg)
            println("Attempting near-PD projection")
        end
        # if maximum(abs.(R - R')) > 1e-12 # Enforce symmetry
        #     R = (R + R') / 2
        # end
        
        return MvNormal(zeros(size(R_reg,1)), R_reg)
    end

    function sample_parameters(p, copula, marg_g, marg_s, glob_tag)
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

        d, N = size(p)

        # Sobol → iid Gaussians
        Z0 = quantile.(Normal(), p)

        # Partition Gaussians and copula
        Zg = Z0[glob_tag, :]
        Zs = Z0[.!glob_tag, :]
        Σ = copula.Σ
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
        # U = cdf.(Normal(), Z)
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

        # Apply marginals
        # θ = zeros(d, N)
        # for i in 1:d
        #     θ[i, :] = quantile.(marginals[i], U[i, :])
        # end

        # Weave back parameters
        # θ = zeros(d, N)
        # θ[glob_tag, :] .= θg
        # θ[.!glob_tag, :] .= θs

        return θg, θs
    end
    

    function parse_ijadpanahsaravi_data()
        """
        Parses the dataset from Ijadpanahsaravi et al. (2023)
        with multiple incolulum densities and returns a DataFrame.
        """

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
            :N => df_germination_swelling[!, 8],
            :M => df_germination_swelling[!, 9]
        )

        df_germination_rebuilt
    end


    function dantigny(t, p_max, τ, ν)
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
        p = p_max * (1 - 1 / (1 + (t / τ)^ν))
        return p
    end


    function generate_dantigny_dataset(df_germination, t_max, n_pts=1000)
        """
        Generate time-dependent germination data using the Dantigny model
        and parameters from a dictionary.
        inputs:
            df_germination (DataFrame): Dantigny model parameters
            t_max (float): maximum time point in seconds
            n_pts (int): number of time points to generate
        outputs:
            dantigny_data (Matrix): matrix of time-dependent germination data
            times (Vector): vector of time points
            sources (Vector): unique string identifiers of carbon sources
            densities (Vector): unique spore densities used in the data
            errs (Matrix): negative and positive offsets of p_max CIs from the mean
            p_maxs (Matrix): maximum germination percentages
            taus (Matrix): characteristic germination times
            nus (Matrix): design parameters
        """
        
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

        return dantigny_data * 0.01, times * 3600, sources, densities, errs, p_maxs, taus, nus
    end


    # ===== Prior calibration =====
    function calibrate_priors(n_iter, n_samples, def_params, bounds_abs, bounds_rel; temp_param=0.25, use_surrogates=true)
        """
        Iteratively calibrates parameter priors
        based on lab-derived Dantigny summaries.
        inputs:
            n_iter (int) - maximum number of calibration iterations
            n_samples (int) - size of Sobol sample
            def_params (Dict) - defined parameters (not optimised)
            bounds_abs (Dict) - initial-guess distributions for the case of absolute γ-thresholds
            bounds_rel (Dict) - initial-guess distributions for the case of relative γ-thresholds
            temp_param (float) - tempering parameter of distribution updates
            use_surrogates (bool) - whether to use a surrogate NN for mapping successive parameter values to Dantigny summaries
        """
        
        # Random.seed!(1236) #1236

        t_max = 48 # hours

        # Load data
        aliases, combination_IDs, descriptions, param_key_sets = load_model_collection()
        df_germination_rebuilt = parse_ijadpanahsaravi_data()
        df_germination_rebuilt = filter(row -> row[1] != "Arg", df_germination_rebuilt) # Remove "Arg" from the dataset
        dantigny_data, times, sources, densities, _, p_maxs, taus, nus = generate_dantigny_dataset(df_germination_rebuilt, t_max)

        n_src = length(sources)
        n_dens = length(densities)

        # println(df_germination_rebuilt)

        # Determine relative vs absolute threshold models
        aliases_rel = ["0", "B", "Bi"]

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

        # Precompute lab data
        data_lookup = Dict((row[1], row[2]) => row for row in eachrow(df_germination_rebuilt))
        lab_means = zeros(Float64, n_dens, n_src, 3)
        CIs = zeros(Float64, n_dens, n_src, 3, 2)
        CI_widths = zeros(Float64, n_dens, n_src, 3)
        for i in eachindex(densities)
            for j in eachindex(sources)
                data_row = data_lookup[(sources[j], densities[i])]
                lab_means[i, j, :] = [data_row["Pmax"] * 0.01, data_row["tau"], data_row["d"]]
                CIs[i, j, :, :] = [
                    data_row["Pmax_CI_Lower"] * 0.01 data_row["Pmax_CI_Upper"] * 0.01;
                    data_row["tau_CI_Lower"] data_row["tau_CI_Upper"];
                    data_row["d_CI_Lower"] data_row["d_CI_Upper"]
                    ]
                CI_widths[i, j, :] = CIs[i, j, :, 2] .- CIs[i, j, :, 1]
            end
        end

        println("CI_widths: $CI_widths")

        input_params_all = Vector{Dict}(undef, n_src)

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
        p_out = Vector{Float64}(undef, length(times))
        diffs_dantigny = zeros(3, n_samples)
        p_maxs = zeros(Float64, n_samples)
        taus = zeros(Float64, n_samples)
        nus = zeros(Float64, n_samples)
        d = zeros(Float64, 3, n_samples)
        z_dist = zeros(Float64, n_samples)
        z_acc_specific = zeros(n_src, n_samples)
        rmses = zeros(Float64, n_dens, n_src, n_samples)

        # Iterate over models
        priors_running = Vector{Dict}(undef, n_iter)
        for m in 1:1#eachindex(aliases)

            println("Running $(aliases[m])")

            param_keys = param_key_sets[m]
            key_src_strings = [Symbol(string(key) * " " * src) for key in param_keys, src in sources]

            n_dims = length(param_keys)
            glob_tag = [key in params_glob_keys for key in param_keys]
            n_glob = sum(glob_tag)
            n_spec = n_dims - n_glob

            idx_shuffle = collect(1:n_dims)

            priors = Dict()
            surrogates = Dict()

            # Check if model uses absolute or relative bounds
            if aliases[m] in aliases_rel
                sample_dists = filter(p -> p[1] in param_keys, param_dists_rel)
            else
                sample_dists = filter(p -> p[1] in param_keys, param_dists_abs)
            end

            π_d_crit_ct = 0
            q_d_crit_ct = 0
            m_d_crit_ct = 0

            # Generate (normalized) parameter samples
            sobol_pts = QuasiMonteCarlo.sample(n_samples, n_dims, SobolSample())

            # Shrink samples to 95%
            sobol_pts = 0.025 .+ 0.95 .* sobol_pts

            # Parameter and weights placeholders
            θ = similar(sobol_pts)
            θ_glob = zeros(Float64, n_glob, n_samples)
            θ_spec = zeros(Float64, n_src, n_spec, n_samples)
            w_spec = zeros(Float64, n_src, n_samples)
            w_spec_penalized = similar(w_spec)
            w_glob = zeros(Float64, n_samples)
            w_glob_penalized = zeros(Float64, n_samples)
            w_mean = zeros(Float64, n_src)
            w_std = similar(w_mean)

            # Distribution placeholders
            marginals = Vector{Distribution}(undef, n_dims)
            marg_glob = Vector{Distribution}(undef, n_glob)
            marg_spec = Matrix{Distribution}(undef, (n_src, n_spec))

            # Generate input parameter sample
            sample_params = Dict()
            for (i, src) in enumerate(sources)
                g_ct = 1
                s_ct = 1
                for (k, key) in enumerate(param_keys)

                    sample_params[key] = quantile.(sample_dists[key], sobol_pts[k, :]) # limit to sampling to 95% within bounds

                    # Split in general and inducer-specific parameters
                    if key in params_glob_keys
                        θ_glob[g_ct, :] .= sample_params[key]
                        marg_glob[g_ct] = sample_dists[key]
                        g_ct += 1
                    else
                        θ_spec[i, s_ct, :] .= sample_params[key]
                        marg_spec[i, s_ct] = sample_dists[key]
                        s_ct += 1
                    end

                    # Initial guess priors
                    priors[key_src_strings[k, i]] = sample_dists[key]
                end
            end

            # Transform sigmas
            for key in param_keys
                if startswith(string(key), "neg_δ_")
                    suffix = string(key)[end]
                    sample_params[Symbol("σ_" * suffix)] = sample_params[Symbol("μ_" * suffix)] .* clamp.(exp.(-sample_params[key]), 1e-12, 1e6)
                end
            end

            input_params_all .= Ref(merge(sample_params, def_params)) # Merge with default parameters

            # Record data points (input parameters + Dantigny summaries)
            data_pts = zeros(Float64, n_src, n_iter, n_dims + 3, n_samples)

            # NN training set
            X_train = zeros(n_dims, n_samples, n_src)
            Y_train = zeros(3, n_samples, n_dens, n_src)

            # Calibration loop
            @inbounds for s in 1:n_iter

                # print("\rIteration $s")
                println("Iteration $s")
                
                # Shuffle Sobol points
                shuffle!(idx_shuffle)
                sobol_pts = sobol_pts[idx_shuffle, :]

                if s > 1
                    marg_glob .= calibrate_marginals(marg_glob, θ_glob, w_glob_penalized)
                    for (i, src) in enumerate(sources)

                        sample_params = Dict()

                        marg_spec[i, :] = calibrate_marginals(marg_spec[i, :], θ_spec[i, :, :], w_spec_penalized[i, :])
                        copula = calibrate_copula(θ_glob, θ_spec[i, :, :], marg_glob, marg_spec[i, :], w_glob, w_spec_penalized[i, :])

                        marginals[glob_tag] .= marg_glob
                        marginals[.!glob_tag] .= marg_spec[i, :]

                        θ_glob, θ_spec[i, :, :] = sample_parameters(sobol_pts, copula, marg_glob, marg_spec[i, :], glob_tag)
                        θ[glob_tag, :] .= θ_glob
                        θ[.!glob_tag, :] .= θ_spec[i, :, :]

                        data_pts[i, s, 1:n_dims, :] .= θ
                        X_train[:, :, i] .= θ

                        # Assign new samples
                        for (k, key) in enumerate(param_keys)
                            sample_params[key] = θ[k, :]
                            priors[key_src_strings[k, i]] = marginals[k]
                        end

                        # Transform sigmas
                        for key in param_keys
                            if startswith(string(key), "neg_δ_")
                                suffix = string(key)[end]
                                sample_params[Symbol("σ_" * suffix)] = sample_params[Symbol("μ_" * suffix)] .* clamp.(exp.(-sample_params[key]), 1e-12, 1e6)
                            end
                        end

                        input_params_all[i] = merge(sample_params, def_params) # Merge with default parameters
                    end
                end

                max_uncertainties = zeros(n_dens, n_src)

                # Iterate over experimental conditions
                @inbounds for (i, density) in enumerate(densities) # iterate over spore densities (exp. data)

                    # println("Running model $(aliases[m]) with density $density")

                    density_scaled = inverse_mL_to_cubic_um(density)

                    @inbounds for j in eachindex(sources) # Iterate over sources

                        if (use_surrogates == true && s == 1) || use_surrogates == false
                            for n in 1:n_samples # Iterate over random parameter samples
                                
                                p_out .= compute_germination_response(aliases[m], times, density_scaled, Dict(k => v[mod1(n, length(v))] for (k, v) in input_params_all[j]))
                                d[:, n], rmses[i, j, n] = fit_dantigny_to_germination_curve(p_out, times)
                                Y_train[:, n, i, j] = d[:, n]
                            end
                        else # Use surrogate model to predict Dantigny summaries
                            d, σ_pred = predict_with_uncertainty_mixed_precision(
                                surrogates[(i, j)], 
                                θ,
                                n_dropout_samples=50
                            )
                            
                            # Use uncertainty for RMSE estimate
                            rmses[i, j, :] = mean(σ_pred, dims=1) |> vec

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

                        println("Maximum RMSE: $(maximum(rmses))")

                        data_pts[j, s, (n_dims + 1):end, :] .= d

                        # Find non-identifiable τ_g and ν
                        identifiable = dropdims(any(.!isnan.(d[2:3, :]); dims=1); dims=1)
                        d_valid = d[:, identifiable]

                        # Included fraction (Criterion 1)
                        π_d_new = dropdims(mean(d_valid .> CIs[i, j, :, 1] .&& d_valid .< CIs[i, j, :, 2]; dims=2); dims=2)
                        if s > 1 # Criterion 5
                            π_diffs[i, j, :] = π_d_new ./ (π_d[i, j, :] .+ 1e-6)
                        end
                        π_d[i, j, :] .= π_d_new

                        # Total predictive probability (Criterion 6)
                        p_d[i, j] = mean(all(d_valid .> CIs[i, j, :, 1] .&& d_valid .< CIs[i, j, :, 2], dims=1))

                        # Lab mean quantiles (Criterion 2)
                        q_d[i, j, :] = dropdims(mean(d_valid .< lab_means[i, j, :]; dims=2); dims=2)

                        # Mahalanobis distance (Criterion 3)
                        μ_d = mean(d_valid, dims=2) |> vec
                        Σ_d = cov(d_valid', corrected=false)
                        diff_d = lab_means[i, j, :] .- μ_d
                        # println("Identifiable: $(sum(identifiable))")
                        # println("NaNs: $(sum(isnan.(d_valid)))")
                        # println(minimum(d_valid[3, :]), " - ", maximum(d_valid[3, :]))
                        d_mask = d_valid[3, :] .> 1e12 .|| dropdims(any(isnan.(d_valid); dims=1); dims=1)
                        if sum(d_mask) > 0
                            println("p_max: ", d_valid[1, d_mask])
                            println("tau_g: ", d_valid[2, d_mask])
                            println("nu: ", d_valid[2, d_mask])
                        end
                        if any(isnan.(Σ_d) .|| isinf.(Σ_d))
                            display(Σ_d)
                        end
                        m_d[i, j] = dot(diff_d, Σ_d \ diff_d)

                        # IQR comparison (Criterion 4)
                        d_vecs = [vec(di) for di in eachrow(d_valid)]
                        iqr_new = quantile.(d_vecs, 0.75) .- quantile.(d_vecs, 0.25)
                        if s > 1 # Criterion 5
                            iqr_diff[i, j, :] = iqr_new ./ (iqr[i, j, :] .+ 1e-6)
                        end
                        iqr[i, j, :] .= iqr_new
                        iqr_check[i, j, :] .= iqr[i, j, :] .> CI_widths[i, j, :]

                        # Standard deviation and covariance matrix
                        σ_dantigny = CI_widths[i, j, :] ./ 3.92 #[Δ/1.96 for Δ in Δ_dantigny]
                        Σ = diagm(σ_dantigny .^ 2)

                        # Compute differences to experimental data
                        diffs_dantigny[1, :] .= d[1, :] .- lab_means[i, j, 1]
                        diffs_dantigny[2, :][identifiable] .= d_valid[2, :] .- lab_means[i, j, 2]
                        diffs_dantigny[3, :][identifiable] .= d_valid[3, :] .- lab_means[i, j, 3]

                        # Compute running z's
                        for n in 1:n_samples
                            if identifiable[n]
                                z_dist[n] = dot(diffs_dantigny[:, n], Σ \ diffs_dantigny[:, n])
                                z_acc_specific[j, n] += z_dist[n]
                            else
                                z_dist[n] = diffs_dantigny[1, n] .^ 2 / σ_dantigny[1] .^ 2
                                z_acc_specific[j, n] += z_dist[n]
                            end
                        end

                        # Fraction of lab means inside joint regions (Criterion 5)
                        p_z_new = mean(z_dist .< χ_sq)
                        if s > 1
                            p_z_diffs[i, j] = p_z_new / (p_z[i, j] .+ 1e-6)
                        end
                        p_z[i, j] = p_z_new
                    end
                end

                z_acc_specific ./= n_dens
                # z_acc_specific[z_acc_specific .< χ_sq] .= 0 # Do not penalize good fits

                # Penalize weights due to RMSE
                reliabilities = exp.(-0.5 .* dropdims(mean(rmses.^2, dims=1); dims=1) ./ 0.004) # using an error scale of 2%

                # Compute condition-specific weights
                log_w_spec = -temp_param .* z_acc_specific
                w_spec .= exp.(log_w_spec)
                w_spec_penalized = w_spec .* reliabilities

                # Compute global weights as product of specific ones
                log_w_glob = dropdims(sum(log_w_spec, dims=1), dims=1)  # Sum of logs = log of product
                w_glob .= exp.(log_w_glob)
                w_glob_penalized = w_glob .* dropdims(mean(reliabilities; dims=1); dims=1)

                # Debugging
                if any(isnan.(w_glob_penalized))
                    # println("w_glob: $w_glob")
                    # println("log_w_glob: $log_w_glob")
                    # println("w_spec_penalized: $w_spec_penalized")
                    # println("w_spec: $w_spec")
                    # println("log_w_spec: $log_w_spec")
                    println("z_acc_specific: $z_acc_specific")
                    println("diffs_dantigny: $diffs_dantigny")
                end

                # Weight statistics (Criterion 5)
                w_mean_new = dropdims(mean(w_spec_penalized, dims=2); dims=2)
                w_std_new = dropdims(std(w_spec_penalized, dims=2); dims=2)
                if s > 1
                    w_mean_diff = w_mean_new / w_mean
                    w_std_diff = w_std_new / w_std
                end
                w_mean = w_mean_new
                w_std = w_std_new

                # Train separate surrogate for each (density, source) combination
                if s == 1
                    println("Building GPU-accelerated surrogate...")
                    
                    # Determine optimal batch size
                    optimal_batch_size = find_optimal_batch_size(n_dims, n_samples)
                    
                    for i in eachindex(densities)
                        for j in eachindex(sources)
                            valid_mask = .!any(isnan.(Y_train[:, :, i, j]), dims=1) |> vec
                            X_valid = X_train[:, valid_mask, j]
                            Y_valid = Y_train[:, valid_mask, i, j]
                            
                            # Train on GPU
                            surrogates[(i, j)] = train_multioutput_nn_mixed_precision(X_valid, Y_valid; batch_size=optimal_batch_size)
                        end
                    end
                end

                # Termination criteria
                π_d_crit_new = all(π_d .> 0.2 .&& π_d .< 0.8) # Criterion 1
                q_d_crit_new = all(q_d .> 0.1 .&& q_d .< 0.9) # Criterion 2
                m_d_crit_new = all(m_d .< χ_sq) # Criterion 3
                if s > 1
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
                    println("π_d_crit_ct: $π_d_crit_ct")
                    println("q_d_crit_ct: $q_d_crit_ct")
                    println("m_d_crit_ct: $m_d_crit_ct")
                    println("π_d: $π_d")
                    println("π_d: $(minimum(π_d)) - $(maximum(π_d))")
                    println("q_d: $q_d")
                    println("m_d: $m_d")
                    println("p_d: $p_d")
                    println("iqr: $iqr")
                    println("w_mean: $w_mean")
                    println("w_std: $w_std")
                    println("Max π_diffs: $(maximum(abs.(π_diffs)))")
                    println("Max iqr_diff: $(maximum(abs.(iqr_diff)))")
                    println("Max p_z_diffs: $(maximum(abs.(p_z_diffs)))")
                    println("Max w_mean_diff: $(maximum(abs.(w_mean_diff)))")
                    println("Max w_std_diff: $(maximum(abs.(w_std_diff)))")
                    println("Criterion 1: $π_d_crit_new ($(π_d .> 0.2 .&& π_d .< 0.8))")
                    println("Criterion 2: $q_d_crit_new ($(q_d .> 0.1 .&& q_d .< 0.9))")
                    println("Criterion 3: $m_d_crit_new ($(m_d .< χ_sq))")
                    println("Criterion 4: $(all(iqr_check))")
                    println("Criterion 5: $(all(abs.(1.0 .- π_diffs) .< 0.1) && all(abs.(1.0 .- iqr_diff) .< 0.1) && all(abs.(1.0 .- p_z_diffs) .< 0.1) && all(abs.(1.0 .- w_mean_diff) .< 0.1) && all(abs.(1.0 .- w_std_diff) .< 0.1))")
                    println("Criterion 6: $(all(p_d .> 1e-2))")
                    println("Criterion 1, 2, 3 (soft): $((π_d_crit && q_d_crit && m_d_crit) || (π_d_crit_ct > 5 && q_d_crit_ct > 5 && m_d_crit_ct > 5))")
                    if (all(iqr_check) # Criterion 4
                        && all(abs.(1.0 .- π_diffs) .< 0.1) && all(abs.(1.0 .- iqr_diff) .< 0.1) && all(abs.(1.0 .- p_z_diffs) .< 0.1) && all(abs.(1.0 .- w_mean_diff) .< 0.1) && all(abs.(1.0 .- w_std_diff) .< 0.1) # Criterion 5
                        && all(p_d .> 1e-2) # Criterion 6
                        && ((π_d_crit && q_d_crit && m_d_crit) || (π_d_crit_ct > 5 && q_d_crit_ct > 5 && m_d_crit_ct > 5))) # Soft criteria 1, 2 and 3
                        println("Termination criteria met at iteration $s")
                        break
                    end
                    println("===========================")
                end
                π_d_crit = π_d_crit_new
                q_d_crit = q_d_crit_new
                m_d_crit = m_d_crit_new
                
                @show priors
                priors_running[s] = copy(priors)
            end

            priors_all[aliases[m]] = priors
            surrogates_all[aliases[m]] = surrogates
            data_pts_dict[aliases[m]] = data_pts
        end

        jldsave("../src/Data/priors.jld2"; priors_all)

        return data_pts_dict
    end


    # ===== Surrogate model =====
    struct MultiOutputSurrogate
        model::Chain
        X_mean::Vector{Float64}
        X_std::Vector{Float64}
        Y_mean::Vector{Float64}
        Y_std::Vector{Float64}
        n_inputs::Int
        n_outputs::Int
    end

    function train_multioutput_nn_mixed_precision(
        X_train, Y_train;
        hidden_dims=[128, 64, 32],
        epochs=1000,
        batch_size=64,
        learning_rate=0.001,
        validation_split=0.15,
        early_stopping_patience=50,
        loss_scale=1024.0,  # Initial loss scale
        verbose=true
    )
        """
        Automatic Mixed Precision training optimized for RTX A4000
        
        Uses:
        - FP16 for forward/backward passes (Tensor Cores)
        - FP32 for loss computation and weight updates
        - Dynamic loss scaling to prevent underflow
        """
        
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
        
        Y_mean = mean(Y_log, dims=2) |> vec
        Y_std = std(Y_log, dims=2) |> vec
        Y_std[Y_std .< 1e-8] .= 1.0
        Y_normalized = (Y_log .- Y_mean) ./ Y_std
        
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
        
        push!(layers, Dense(hidden_dims[end], n_outputs))
        
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
        
        function mixed_precision_step!(model, x_fp16, y_fp16, opt_state, scale)
            """
            Single training step with mixed precision
            
            Returns: (loss, had_overflow)
            """
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
            # has_overflow = false
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
            Y_mean,
            Y_std,
            n_inputs,
            n_outputs
        )
    end
    
    function predict_surrogate_gpu(surrogate::MultiOutputSurrogate, X_new; use_gpu=false)
        """
        Predict with optional GPU acceleration for large batches
        """
        single_sample = ndims(X_new) == 1
        if single_sample
            X_new = reshape(X_new, :, 1)
        end
        
        # Normalize
        X_normalized = (X_new .- surrogate.X_mean) ./ surrogate.X_std
        
        # Move to GPU if requested and available
        if use_gpu && CUDA.functional()
            X_normalized_gpu = X_normalized |> gpu
            model_gpu = surrogate.model |> gpu
            
            Y_normalized_pred = model_gpu(X_normalized_gpu)
            Y_normalized_pred = Y_normalized_pred |> cpu  # Move back to CPU
            
            CUDA.reclaim()
        else
            Y_normalized_pred = surrogate.model(X_normalized)
        end
        
        # Denormalize
        Y_log_pred = Y_normalized_pred .* surrogate.Y_std .+ surrogate.Y_mean
        
        # Back-transform
        Y_pred = copy(Y_log_pred)
        Y_pred[2, :] = exp.(Y_log_pred[2, :])
        Y_pred[3, :] = exp.(Y_log_pred[3, :])
        
        return single_sample ? Y_pred[:, 1] : Y_pred
    end

    function predict_with_uncertainty_mixed_precision(
        surrogate::MultiOutputSurrogate,
        X_new;
        n_dropout_samples=50
    )
        """
        Mixed precision uncertainty estimation (fastest option)
        """
        
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
        Y_log_all = Y_normalized_all .* reshape(surrogate.Y_std, :, 1, 1) .+ 
                    reshape(surrogate.Y_mean, :, 1, 1)
        
        Y_all = copy(Y_log_all)
        Y_all[2, :, :] = exp.(Y_log_all[2, :, :])
        Y_all[3, :, :] = exp.(Y_log_all[3, :, :])
        
        # Compute statistics
        mean_pred = mean(Y_all, dims=3)[:, :, 1]
        std_pred = std(Y_all, dims=3)[:, :, 1]
        
        if single_sample
            return mean_pred[:, 1], std_pred[:, 1]
        else
            return mean_pred, std_pred
        end
    end
    
    function find_optimal_batch_size(n_inputs, n_samples; max_memory_gb=8)
        """
        Find optimal batch size based on GPU memory
        
        Rule of thumb:
        - Larger batches = better GPU utilization but more memory
        - Sweet spot: 32-256 for most GPUs
        """
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

    # ===== Data fitting =====
    function unresolved_step(y; atol=1e-3)
        """
        Utility function for detecting degenerate results
        (fast step-like onset)
        inputs:
            y (Vector): measurements
        output:
            Bool: whether the measurements form a rapid step
        """
        Δ = abs.(diff(y))
        return count(>(atol), Δ) ≤ 1
    end


    function fit_dantigny_to_germination_curve(germ_response, times)#; check_stderr=false)
        """
        Fit Dantigny model to simulated germination curve
        from a mechanistic model.
        inputs:
            germ_response (Vector): germination fractions per time
            times (Vector): time frames for germination fractions
            check_stderr (Bool): whether to perform a standard error check
        outputs:
            p_max (Float): maximum germination percentage
            τ (Float): half-saturation time for germination
            ν (Float): design parameter
            rmse (Float): root mean-squared error
        """

        dantigny_wrapper(t, p) = dantigny.(t, 1 / (1 + exp(-p[1])), exp(p[2]), exp(p[3])) # [p_max, τ, ν] after transformations to become strictly positive

        # Initial guesses
        p0 = [0.5, times[Int(0.5 * length(times))], 2.0]

        # Transform to strictly positive scales
        p0[1] = log(p0[1]) - log1p(-p0[1])
        p0[2] = log(p0[2])
        p0[3] = log(p0[3])

        try
            fit = curve_fit(dantigny_wrapper, times, germ_response, p0)
            params = coef(fit)
            rmse = sqrt(mean(residuals(fit) .^ 2))

            # if rmse > 0.03 # average deviation > 3 percentage points in germination
            #     println("High RMSE: $rmse")
            # end

            # Transform parameters
            params[1] = 1 / (1 + exp(-params[1]))
            params[2] = exp(params[2])
            params[3] = exp(params[3])

            # Handle degenerate (flat) curves
            if params[1] < 1e-3 || params[2] > 1e6 || isinf(params[3])
                params[2] = NaN
                params[3] = NaN
            end

            # Handle sharp immediate steps
            if params[2] < 1e-6
                params[2] = 1e-6
                params[3] = 10
            end

            return params, rmse

        catch
            # Handle sharp immediate steps
            return [germ_response[end], NaN, NaN], 0.0
        end
    end


    function fit_model_to_data(model_type, def_params, dantigny_data, times, sources, densities, bounds_dict; max_steps=10000, debug=false)
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
            max_steps (int): maximum number of steps for the optimization
            debug (bool) - whether to print additional debugging messages
        outputs:
            params_out (Dict): optimized parameters
        """

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
            wrapper = (inputs, params) -> germ_response_independent_factors_gh(
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
            # param_keys = [:Pₛ, :Pₛ_cs, :K_cC, :μ_γ, :δ_γ, :μ_ω, :δ_ω]
            # param_occurrences = [1, n_src, n_src, 1, 1, n_src, n_src]

        # elseif model_type == "inhibitor" # Inducer shifts inhibition threshold and modulates inhibitor permeability
        #         println("Model: inducer-modulated inhibitor (combined)")
        #         wrapper = (inputs, params) -> Main.germ_response_inducer_dep_inhibitor_gh(
        #             u, W,
        #             inputs[1], #t
        #             inputs[2], #ρₛ
        #             def_params[:c₀_cs],
        #             def_params[:d_hp],
        #             ξ2,
        #             κ2,
        #             params[1], #Pₛ,
        #             params[2], #Pₛ_cs,
        #             params[3], #K_cC,
        #             exp(params[4]), #k_C,
        #             params[5], #μ_γ,
        #             params[5] * exp(params[6]), # σ_γ = μ_γ * exp(δ_γ)
        #         )
        #         param_keys = [:Pₛ, :Pₛ_cs, :K_cC, :k_C, :μ_γ, :δ_γ]
        #         param_occurrences = [1, n_src, n_src, n_src, 1, 1]

        elseif model_type == "inhibitor_thresh" # Inducer shifts inhibition threshold
            println("Model: inducer-modulated inhibitor (threshold)")
            wrapper = (inputs, params) -> Main.germ_response_inducer_dep_inhibitor_thresh_gh(
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
            # param_keys = [:Pₛ, :Pₛ_cs, :k_C, :K_cC, :μ_γ, :δ_γ]
            # param_occurrences = [1, n_src, n_src, n_src, 1, 1]
            
            # elseif model_type_split[2] == "perm" # Inducer modulates inhibitor permeability
            #     println("Model: inducer-modulated inhibitor (permeability)")
            #     wrapper = (inputs, params) -> Main.germ_response_inducer_dep_inhibitor_perm_gh(
            #         u, W,
            #         inputs[1], #t
            #         inputs[2], #ρₛ
            #         def_params[:c₀_cs],
            #         def_params[:d_hp],
            #         ξ2,
            #         κ2,
            #         params[1], #Pₛ
            #         params[2], #Pₛ_cs
            #         params[3], #K_cC
            #         params[4], #μ_γ
            #         params[4] * exp(params[5]) # σ_γ = μ_γ * exp(δ_γ)
            #     )
            #     param_keys = [:Pₛ, :Pₛ_cs, :K_cC, :μ_γ, :δ_γ]
            #     param_occurrences = [1, n_src, n_src, 1, 1]
                
            # end

        elseif model_type == "inducer" # Inhibitor shifts induction threshold and modulates inducer signal strength
            println("Model: inhibitor-modulated inducer (combined)")
            wrapper = (inputs, params) -> Main.germ_response_inhibitor_dep_inducer_gh(
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
            # param_keys = [:Pₛ, :Pₛ_cs, :k_I, :K_cI, :K_cC, :K_I, :n, :μ_ω, :δ_ω, :μ_ψ, :δ_ψ]
            # param_occurrences = [1, n_src, n_src, n_src, n_src, n_src, n_src, n_src, 1, 1]
                
        elseif model_type == "inducer_thresh" # Inhibitor shifts induction threshold
            println("Model: inhibitor-modulated inducer (threshold)")
            wrapper = (inputs, params) -> Main.germ_response_inhibitor_dep_inducer_thresh_gh(
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
            # param_keys = [:Pₛ, :Pₛ_cs, :K_cI, :K_cC, :k_I, :μ_ω, :δ_ω, :μ_ψ, :δ_ψ]
            # param_occurrences = [1, n_src, n_src, n_src, n_src, n_src, 1, 1]
            
        elseif model_type == "inducer_signal" # Inhibitor shifts induction threshold
            println("Model: inhibitor-modulated inducer (signal)")
            wrapper = (inputs, params) -> Main.germ_response_inhibitor_dep_inducer_signal_gh(
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
            # param_keys = [:Pₛ, :Pₛ_cs, :K_cC, :K_I, :n, :μ_ω, :δ_ω, :μ_ψ, :δ_ψ]
            # param_occurrences = [1, n_src, n_src, n_src, n_src, n_src, n_src, 1, 1]

        elseif model_type == "combined_inhibitor"
            println("Model: inducer-modulated inhibitor (combined)")
            wrapper = (inputs, params) -> Main.germ_response_inducer_dep_inhibitor_2_factor_gh(
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
            # param_keys = [:Pₛ, :Pₛ_cs, :K_cC, :k_C, :μ_γ, :δ_γ, :μ_ω, :δ_ω]
            # param_occurrences = [1, n_src, n_src, n_src, 1, 1, n_src, n_src]

        elseif model_type == "combined_inhibitor_thresh"
            println("Model: inducer-modulated inhibitor (combined)")
            wrapper = (inputs, params) -> Main.germ_response_inducer_dep_inhibitor_thresh_2_factor_gh(
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
            # param_keys = [:Pₛ, :Pₛ_cs, :k_C, :K_cC, :μ_γ, :δ_γ, :μ_ω, :δ_ω]
            # param_occurrences = [1, n_src, n_src, n_src, 1, 1, n_src, n_src]

            # elseif model_type_split[3] == "perm"
            #     println("Model: inducer-modulated inhibitor (release)")
            #     wrapper = (inputs, params) -> Main.germ_response_inducer_dep_inhibitor_perm_2_factor_gh(
            #         u, W,
            #         inputs[1], #t
            #         inputs[2], #ρₛ
            #         def_params[:c₀_cs],
            #         def_params[:d_hp],
            #         ξ2,
            #         κ2,
            #         params[1], #Pₛ
            #         params[2], #Pₛ_cs
            #         params[3], #K_cC
            #         params[4], #μ_γ
            #         params[4] * exp(params[5]), # σ_γ = μ_γ * exp(δ_γ)
            #         params[6], #μ_ω
            #         params[6] * exp(params[7]) # σ_ω = μ_ω * exp(δ_ω)
            #     )
            #     param_keys = [:Pₛ, :Pₛ_cs, :K_cC, :μ_γ, :δ_γ, :μ_ω, :δ_ω]
            #     param_occurrences = [1, n_src, n_src, 1, 1, n_src, n_src]
            # end

        elseif model_type == "combined_inducer"
            println("Model: inhibitor-modulated inducer (combined)")
            wrapper = (inputs, params) -> Main.germ_response_inhibitor_dep_inducer_2_factor_gh(
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
            # param_keys = [:Pₛ, :Pₛ_cs, :K_cI, :K_cC, :K_I, :n, :k_I, :μ_γ, :δ_γ, :μ_ω, :δ_ω, :μ_ψ, :δ_ψ]
            # param_occurrences = [1, n_src, n_src, n_src, n_src, n_src, 1, 1, n_src, n_src, 1, 1]

        elseif model_type == "combined_inducer_thresh"
            println("Model: inhibitor-modulated inducer (threshold)")
            wrapper = (inputs, params) -> Main.germ_response_inhibitor_dep_inducer_thresh_2_factor_gh(
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
            # param_keys = [:Pₛ, :Pₛ_cs, :K_cI, :K_cC, :k_I, :μ_γ, :δ_γ, :μ_ω, :δ_ω, :μ_ψ, :δ_ψ]
            # param_occurrences = [1, n_src, n_src, n_src, 1, 1, n_src, n_src, 1, 1]

        elseif model_type == "combined_inducer_signal"
            println("Model: inhibitor-modulated inducer (signal)")
            wrapper = (inputs, params) -> Main.germ_response_inhibitor_dep_inducer_signal_2_factor_gh(
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
            wrapper = (inputs, params) -> Main.germ_response_inducer_var_perm_gh(
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
            # param_keys = [:Pₛ, :Pₛ_cs, :K_cC, :K_I, :k_I, :n, :μ_ω, :δ_ω, :μ_ψ, :δ_ψ, :μ_α, :δ_α]
            # param_occurrences = [1, n_src, n_src, n_src, n_src, n_src, n_src, n_src, 1, 1, 1, 1]

        elseif model_type == "special_independent"
            println("Model: independent factors with varying permeability")
            wrapper = (inputs, params) -> Main.germ_response_independent_factors_var_perm_gh(
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
            # param_keys = [:Pₛ, :Pₛ_cs, :K_cC, :μ_γ, :δ_γ, :μ_ω, :δ_ω, :μ_α, :δ_α]
            # param_occurrences = [1, n_src, n_src, 1, 1, n_src, n_src, 1, 1]

        elseif model_type == "special_combined"
            println("Model: 2-factor germination with inhibitor-modulated inducer (combined) and varying permeability")
            wrapper = (inputs, params) -> Main.germ_response_inducer_2_factors_var_perm_gh(
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
            # param_keys = [:Pₛ, :Pₛ_cs, :K_cC, :K_I, :k_I, :n, :μ_γ, :δ_γ, :μ_ω, :δ_ω, :μ_ψ, :δ_ψ, :μ_α, :δ_α]
            # param_occurrences = [1, n_src, n_src, n_src, n_src, n_src, 1, 1, n_src, n_src, 1, 1, 1, 1]

        elseif model_type == "special_combined_thresh"
            println("Model: 2-factor germination with inhibitor-modulated inducer (threshold) and varying permeability")
            wrapper = (inputs, params) -> Main.germ_response_inducer_thresh_2_factors_var_perm_gh(
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
            # param_keys = [:Pₛ, :Pₛ_cs, :K_cC, :k_I, :μ_γ, :δ_γ, :μ_ω, :δ_ω, :μ_ψ, :δ_ψ, :μ_α, :δ_α]
            # param_occurrences = [1, n_src, n_src, n_src, 1, 1, n_src, n_src, 1, 1, 1, 1]

        elseif model_type == "special_combined_signal"
            println("Model: 2-factor germination with inhibitor-modulated inducer (signal) and varying permeability")
            wrapper = (inputs, params) -> Main.germ_response_inducer_signal_2_factors_var_perm_gh(
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
            # param_keys = [:Pₛ, :Pₛ_cs, :K_cC, :K_I, :n, :μ_γ, :δ_γ, :μ_ω, :δ_ω, :μ_ψ, :δ_ψ, :μ_α, :δ_α]
            # param_occurrences = [1, n_src, n_src, n_src, n_src, 1, 1, n_src, n_src, 1, 1, 1, 1]
        
            # ==================== FEEDBACK MODELS ========================
        elseif model_type == "feedback_inhibitor_inducer_perm" # A
            println("Model: inhibitor-dependent germination with inducer-dependent inhibitor/inducer permeability")
            wrapper = (V_out, params) -> Main.germ_response_feedback(
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
            # param_keys = [:s_max, :Pₛ, :Pₛ_cs, :K_cC, :μ_ψ, :δ_ψ, :μ_γ, :δ_γ]
            # param_occurrences = [n_src, 1, n_src, n_src, 1, 1, 1, 1]

        elseif model_type == "feedback_combined_inducer_perm" # A
            println("Model: 2-factor germination with inducer-dependent inhibitor/inducer permeability")
            wrapper = (V_out, params) -> Main.germ_response_feedback(
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
            # param_keys = [:s_max, :Pₛ, :Pₛ_cs, :K_cC, :μ_ψ, :δ_ψ, :μ_γ, :δ_γ, :μ_ω, :δ_ω]
            # param_occurrences = [n_src, 1, n_src, n_src, 1, 1, 1, 1, n_src, n_src]
        
        elseif model_type == "feedback_inducer_inhibitor_perm" # D
            println("Model: Inducer-dependent germination with inhibitor-dependent inhibitor/inducer permeability")
            wrapper = (V_out, params) -> Main.germ_response_feedback(
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
            # param_keys = [:b_max, :Pₛ, :Pₛ_cs, :K_cI, :μ_ψ, :δ_ψ, :μ_ω, :δ_ω]
            # param_occurrences = [1, 1, n_src, 1, 1, 1, n_src, n_src]
        
        elseif model_type == "feedback_combined_inhibitor_perm" # D
            println("Model: 2-factor germination with inhibitor-dependent inhibitor/inducer permeability")
            wrapper = (V_out, params) -> Main.germ_response_feedback(
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
            # param_keys = [:b_max, :Pₛ, :Pₛ_cs, :K_cI, :K_cC, :μ_ψ, :δ_ψ, :μ_γ, :δ_γ, :μ_ω, :δ_ω]
            # param_occurrences = [1, 1, n_src, 1, 1, 1, 1, 1, n_src, n_src]
        
        elseif model_type == "feedback_inhibitor_inducer_perm_thresh" # AB
            println("Model: Inhibitor-dependent germination with inducer-dependent permeability and inhibition threshold")
            wrapper = (V_out, params) -> Main.germ_response_feedback(
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
            # param_keys = [:s_max, :Pₛ, :Pₛ_cs, :K_cC, :μ_ψ, :δ_ψ, :μ_γ, :δ_γ, :k_C]
            # param_occurrences = [n_src, 1, n_src, n_src, 1, 1, 1, 1, n_src]

        elseif model_type == "feedback_combined_inducer_perm_thresh" # AB
            println("Model: 2-factor germination with inducer-dependent permeability and inhibition threshold")
            wrapper = (V_out, params) -> Main.germ_response_feedback(
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
            # param_keys = [:s_max, :Pₛ, :Pₛ_cs, :K_cC, :μ_ψ, :δ_ψ, :μ_γ, :δ_γ, :μ_ω, :δ_ω, :k_C]
            # param_occurrences = [n_src, 1, n_src, n_src, 1, 1, 1, 1, n_src, n_src, n_src]
        
        elseif model_type == "feedback_inhibitor_inducer_perm_inhibitor_signal" # AC
            println("Model: Inhibitor-dependent germination with inducer-dependent permeability and inhibitor-dependent induction signal")
            wrapper = (V_out, params) -> Main.germ_response_feedback(
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
            # param_keys = [:s_max, :Pₛ, :Pₛ_cs, :K_cC, :K_I, :μ_ψ, :δ_ψ, :μ_γ, :δ_γ, :n]
            # param_occurrences = [n_src, 1, n_src, 1, n_src, 1, 1, 1, 1, n_src]

        elseif model_type == "feedback_inducer_inducer_perm_inhibitor_signal" # AC
            println("Model: Inducer-dependent germination with inducer-dependent permeability and inhibitor-dependent induction signal")
            wrapper = (V_out, params) -> Main.germ_response_feedback(
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
            # param_keys = [:s_max, :Pₛ, :Pₛ_cs, :K_cC, :K_I, :μ_ψ, :δ_ψ, :μ_ω, :δ_ω, :n]
            # param_occurrences = [n_src, 1, n_src, 1, n_src, 1, 1, n_src, n_src, n_src]
            
        elseif model_type == "feedback_combined_inducer_perm_inhibitor_signal" # AC
            println("Model: 2-factor germination with inducer-dependent permeability and inhibitor-dependent induction signal")
            wrapper = (V_out, params) -> Main.germ_response_feedback(
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
            # param_keys = [:s_max, :Pₛ, :Pₛ_cs, :K_cC, :K_I, :μ_ψ, :δ_ψ, :μ_γ, :δ_γ, :μ_ω, :δ_ω, :n]
            # param_occurrences = [n_src, 1, n_src, 1, n_src, 1, 1, 1, 1, n_src, n_src, n_src]
            
        elseif model_type == "feedback_inhibitor_inhibitor_inducer_perm" # AD
            println("Model: Inhibitor-dependent germination with inhibitor- and inducer-dependent inhibitor/inducer permeability")
            wrapper = (V_out, params) -> Main.germ_response_feedback(
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
            # param_keys = [:b_max, :s_max, :Pₛ, :Pₛ_cs, :K_cI, :K_cC, :μ_ψ, :δ_ψ, :μ_γ, :δ_γ]
            # param_occurrences = [1, n_src, 1, n_src, 1, n_src, 1, 1, 1, 1]
        
        elseif model_type == "feedback_inducer_inhibitor_inducer_perm" # AD
            println("Model: Inducer-dependent germination with inhibitor- and inducer-dependent inhibitor/inducer permeability")
            wrapper = (V_out, params) -> Main.germ_response_feedback(
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
            # param_keys = [:b_max, :s_max, :Pₛ, :Pₛ_cs, :K_cI, :K_cC, :μ_ψ, :δ_ψ, :μ_ω, :δ_ω]
            # param_occurrences = [1, n_src, 1, n_src, 1, n_src, 1, 1, n_src, n_src]
        
        elseif model_type == "feedback_combined_inhibitor_inducer_perm" # AD
            println("Model: 2-factor germination with inhibitor- and inducer-dependent inhibitor/inducer permeability")
            wrapper = (V_out, params) -> Main.germ_response_feedback(
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
            # param_keys = [:b_max, :s_max, :Pₛ, :Pₛ_cs, :K_cI, :K_cC, :μ_ψ, :δ_ψ, :μ_γ, :δ_γ, :μ_ω, :δ_ω]
            # param_occurrences = [1, n_src, 1, n_src, 1, n_src, 1, 1, 1, 1, n_src, n_src]
        
        elseif model_type == "feedback_inducer_inhibitor_thresh_inducer_perm" # AE
            println("Model: Inducer-dependent germination with inhibitor-dependent induction threshold and inducer-dependent inhibitor/inducer permeability")
            wrapper = (V_out, params) -> Main.germ_response_feedback(
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
            # param_keys = [:s_max, :Pₛ, :Pₛ_cs, :K_cI, :K_cC, :μ_ψ, :δ_ψ, :μ_ω, :δ_ω, :k_I]
            # param_occurrences = [n_src, 1, n_src, 1, n_src, 1, 1, n_src, n_src, n_src]

        elseif model_type == "feedback_combined_inhibitor_thresh_inducer_perm" # AE
            println("Model: 2-factor germination with inhibitor-dependent induction threshold and inducer-dependent inhibitor/inducer permeability")
            wrapper = (V_out, params) -> Main.germ_response_feedback(
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
            # param_keys = [:s_max, :Pₛ, :Pₛ_cs, :K_cI, :K_cC, :μ_ψ, :δ_ψ, :μ_γ, :δ_γ, :μ_ω, :δ_ω, :k_I]
            # param_occurrences = [n_src, 1, n_src, 1, n_src, 1, 1, 1, 1, n_src, n_src, n_src]
            
        elseif model_type == "inhibitor_thresh_inducer_signal" # BC
            println("Model: Inhibitor-dependent germination with inducer-dependent inhibition threshold and inhibitor-dependent induction signal")
            wrapper = (inputs, params) -> Main.germ_response_inh_dep_ind_signal_ind_dep_inh_thresh_gh(
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
            # param_keys = [:Pₛ, :Pₛ_cs, :k_C, :K_cC, :K_I, :n, :μ_γ, :δ_γ, :μ_ψ, :δ_ψ]
            # param_occurrences = [1, n_src, n_src, n_src, 1, n_src, 1, 1, 1, 1]

        elseif model_type == "combined_inhibitor_thresh_inducer_signal" # BC
            println("Model: 2-factor germination with inducer-dependent inhibition threshold and inhibitor-dependent induction signal")
            wrapper = (inputs, params) -> Main.germ_response_inh_dep_ind_signal_ind_dep_inh_thresh_2_factor_gh(
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
            # param_keys = [:Pₛ, :Pₛ_cs, :k_C, :K_cC, :K_I, :n, :μ_γ, :δ_γ, :μ_ω, :δ_ω, :μ_ψ, :δ_ψ]
            # param_occurrences = [1, n_src, n_src, n_src, 1, n_src, 1, 1, n_src, n_src, 1, 1]
            
        elseif model_type == "feedback_inhibitor_inducer_thresh_inhibitor_perm" # BD
            println("Model: Inhibitor-dependent germination with inducer-dependent inhibition threshold and inhibitor-dependent inhibitor/inducer permeability")
            wrapper = (V_out, params) -> Main.germ_response_feedback(
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
            # param_keys = [:b_max, :Pₛ, :Pₛ_cs, :K_cI, :K_cC, :μ_ψ, :δ_ψ, :μ_γ, :δ_γ, :k_C]
            # param_occurrences = [1, 1, n_src, 1, n_src, 1, 1, 1, 1, n_src]
        
        elseif model_type == "feedback_combined_inducer_thresh_inhibitor_perm" # BD
            println("Model: Inhibitor-dependent germination with inducer-dependent inhibition threshold and inhibitor-dependent inhibitor/inducer permeability")
            wrapper = (V_out, params) -> Main.germ_response_feedback(
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
            # param_keys = [:b_max, :Pₛ, :Pₛ_cs, :K_cI, :K_cC, :μ_ψ, :δ_ψ, :μ_γ, :δ_γ, :μ_ω, :δ_ω, :k_C]
            # param_occurrences = [1, 1, n_src, 1, n_src, 1, 1, 1, 1, n_src, n_src, n_src]
        
        elseif model_type == "combined_inhibitor_thresh_inducer_thresh" # BE
            println("Model: inhibitor-modulated induction threshold / inducer-modulated inhibition threshold")
            wrapper = (inputs, params) -> Main.germ_response_inh_dep_ind_thresh_ind_dep_inh_thresh_2_factor_gh(
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
            # param_keys = [:Pₛ, :Pₛ_cs, :K_cI, :K_cC, :k_I, :k_C, :μ_γ, :δ_γ, :μ_ω, :δ_ω, :μ_ψ, :δ_ψ]
            # param_occurrences = [1, n_src, n_src, n_src, n_src, 1, 1, n_src, n_src, 1, 1]

        elseif model_type == "feedback_inducer_inhibitor_perm_signal" # CD
            println("Model: Inhibitor-dependent germination with inducer-dependent inhibition threshold and inhibitor-dependent inhibitor/inducer permeability")
            wrapper = (V_out, params) -> Main.germ_response_feedback(
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
            # param_keys = [:b_max, :Pₛ, :Pₛ_cs, :K_cI, :K_cC, :K_I, :μ_ψ, :δ_ψ, :μ_ω, :δ_ω]
            # param_occurrences = [1, 1, n_src, n_src, 1, 1, 1, n_src, n_src]
        
        elseif model_type == "feedback_combined_inhibitor_perm_signal" # CD
            println("Model: 2-factor germination with inducer-dependent inhibition threshold and inhibitor-dependent inhibitor/inducer permeability")
            wrapper = (V_out, params) -> Main.germ_response_feedback(
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
            # param_keys = [:b_max, :Pₛ, :Pₛ_cs, :K_cI, :K_cC, :K_I, :μ_ψ, :δ_ψ, :μ_γ, :δ_γ, :μ_ω, :δ_ω]
            # param_occurrences = [1, 1, n_src, n_src, 1, 1, 1, 1, 1, n_src, n_src]
        
        elseif model_type == "feedback_inducer_inhibitor_perm_thresh" # DE
            println("Model: Inhibitor-dependent germination with inhibitor-dependent inhibitor/inducer permeability and induction threshold")
            wrapper = (V_out, params) -> Main.germ_response_feedback(
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
            # param_keys = [:b_max, :Pₛ, :Pₛ_cs, :K_cI, :K_cC, :μ_ψ, :δ_ψ, :μ_ω, :δ_ω, :k_I]
            # param_occurrences = [1, 1, n_src, 1, n_src, 1, 1, n_src, n_src, n_src]
        
        elseif model_type == "feedback_combined_inhibitor_perm_thresh" # DE
            println("Model: Inhibitor-dependent germination with inhibitor-dependent inhibitor/inducer permeability and induction threshold")
            wrapper = (V_out, params) -> Main.germ_response_feedback(
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
            # param_keys = [:b_max, :Pₛ, :Pₛ_cs, :K_cI, :K_cC, :μ_ψ, :δ_ψ, :μ_γ, :δ_γ, :μ_ω, :δ_ω, :k_I]
            # param_occurrences = [1, 1, n_src, 1, n_src, 1, 1, 1, 1, n_src, n_src, n_src]
        
        elseif model_type == "feedback_inhibitor_inducer_perm_thresh_inhibitor_signal" # ABC
            println("Model: Inducer-dependent germination with inducer-dependent permeability and inhibitor-dependent induction signal")
            wrapper = (V_out, params) -> Main.germ_response_feedback(
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
            # param_keys = [:s_max, :Pₛ, :Pₛ_cs, :K_cC, :K_I, :μ_ψ, :δ_ψ, :μ_γ, :δ_γ, :k_C, :n]
            # param_occurrences = [n_src, 1, n_src, 1, n_src, 1, 1, n_src, 1, 1, n_src]
            
        elseif model_type == "feedback_combined_inducer_perm_thresh_inhibitor_signal" # ABC
            println("Model: 2-factor germination with inducer-dependent permeability and inhibitor-dependent induction signal")
            wrapper = (V_out, params) -> Main.germ_response_feedback(
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
            # param_keys = [:s_max, :Pₛ, :Pₛ_cs, :K_cC, :K_I, :μ_ψ, :δ_ψ, :μ_γ, :δ_γ, :μ_ω, :δ_ω, :k_C, :n]
            # param_occurrences = [n_src, 1, n_src, 1, n_src, 1, 1, 1, 1, n_src, n_src, n_src, n_src]
            
        elseif model_type == "feedback_inhibitor_inducer_perm_thresh_inhibitor_perm" # ABD
            println("Model: Inhibitor-dependent germination with inducer-dependent inhibition threshold and inhibitor- and inducer-dependent inhibitor/inducer permeability")
            wrapper = (V_out, params) -> Main.germ_response_feedback(
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
            # param_keys = [:b_max, :s_max, :Pₛ, :Pₛ_cs, :K_cI, :K_cC, :μ_ψ, :δ_ψ, :μ_γ, :δ_γ, :k_C]
            # param_occurrences = [1, n_src, 1, n_src, 1, n_src, 1, 1, 1, 1, n_src]
        
        elseif model_type == "feedback_combined_inducer_perm_thresh_inhibitor_perm" # ABD
            println("Model: 2-factor germination with inducer-dependent inhibition threshold and inhibitor- and inducer-dependent inhibitor/inducer permeability")
            wrapper = (V_out, params) -> Main.germ_response_feedback(
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
            # param_keys = [:b_max, :s_max, :Pₛ, :Pₛ_cs, :K_cI, :K_cC, :μ_ψ, :δ_ψ, :μ_γ, :δ_γ, :μ_ω, :δ_ω, :k_C]
            # param_occurrences = [1, n_src, 1, n_src, 1, n_src, 1, 1, 1, 1, n_src, n_src, n_src]
        
        elseif model_type == "feedback_combined_inducer_perm_thresh_inhibitor_thresh" # ABE
            println("Model: 2-factor germination with inducer-dependent inhibitor/inducer permeability and inhibition threshold and inhibitor-dependent induction threshold")
            wrapper = (V_out, params) -> Main.germ_response_feedback(
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
            # param_keys = [:s_max, :Pₛ, :Pₛ_cs, :K_cI, :K_cC, :μ_ψ, :δ_ψ, :μ_γ, :δ_γ, :μ_ω, :δ_ω, :k_I, :k_C]
            # param_occurrences = [n_src, 1, n_src, 1, n_src, 1, 1, 1, 1, n_src, n_src, n_src, n_src]
        
        elseif model_type == "feedback_inhibitor_inhibitor_inducer_perm_inhibitor_signal" # ACD
            println("Model: Inhibitor-dependent germination with inducer/inhibitor-dependent inhibitor/inducer permeability and inhibitor-dependent induction signal")
            wrapper = (V_out, params) -> Main.germ_response_feedback(
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
            # param_keys = [:b_max, :s_max, :Pₛ, :Pₛ_cs, :K_cI, :K_cC, :K_I, :μ_ψ, :δ_ψ, :μ_γ, :δ_γ, :n]
            # param_occurrences = [1, n_src, 1, n_src, 1, n_src, n_src, 1, 1, 1, 1, n_src]
        
        elseif model_type == "feedback_inducer_inhibitor_inducer_perm_inhibitor_signal" # ACD
            println("Model: Inhibitor-dependent germination with inducer/inhibitor-dependent inhibitor/inducer permeability and inhibitor-dependent induction signal")
            wrapper = (V_out, params) -> Main.germ_response_feedback(
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
            # param_keys = [:b_max, :s_max, :Pₛ, :Pₛ_cs, :K_cI, :K_cC, :K_I, :μ_ψ, :δ_ψ, :μ_ω, :δ_ω, :n]
            # param_occurrences = [1, n_src, 1, n_src, 1, n_src, 1, 1, n_src, n_src, n_src]
        
        elseif model_type == "feedback_combined_inhibitor_inducer_perm_inhibitor_signal" # ACD
            println("Model: 2-factor germination with inducer/inhibitor-dependent inhibitor/inducer permeability and inhibitor-dependent induction signal")
            wrapper = (V_out, params) -> Main.germ_response_feedback(
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
            # param_keys = [:b_max, :s_max, :Pₛ, :Pₛ_cs, :K_cI, :K_cC, :K_I, :μ_ψ, :δ_ψ, :μ_γ, :δ_γ, :μ_ω, :δ_ω, :n]
            # param_occurrences = [1, n_src, 1, n_src, 1, n_src, 1, 1, 1, 1, n_src, n_src, n_src]
        
        elseif model_type == "feedback_inducer_inducer_perm_inhibitor_thresh_signal" # ACE
            println("Model: Inducer-dependent germination with inducer-dependent permeability and inhibitor-dependent induction threshold and signal")
            wrapper = (V_out, params) -> Main.germ_response_feedback(
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
            # param_keys = [:s_max, :Pₛ, :Pₛ_cs, :K_cI, :K_cC, :K_I, :μ_ψ, :δ_ψ, :μ_ω, :δ_ω, :k_I, :n]
            # param_occurrences = [n_src, 1, n_src, 1, n_src, 1, 1, n_src, n_src, n_src, n_src]
            
        elseif model_type == "feedback_combined_inducer_perm_inhibitor_thresh_signal" # ACE
            println("Model: 2-factor germination with inducer-dependent permeability and inhibitor-dependent induction threshold and signal")
            wrapper = (V_out, params) -> Main.germ_response_feedback(
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
            # param_keys = [:s_max, :Pₛ, :Pₛ_cs, :K_cI, :K_cC, :K_I, :μ_ψ, :δ_ψ, :μ_γ, :δ_γ, :μ_ω, :δ_ω, :k_I, :n]
            # param_occurrences = [n_src, 1, n_src, 1, n_src, 1, 1, 1, 1, n_src, n_src, n_src, n_src]
            
        elseif model_type == "feedback_inducer_inhibitor_inducer_perm_inhibitor_thresh" # ADE
            println("Model: Inhibitor-dependent germination with inhibitor- and inducer-dependent inhibitor/inducer permeability, inhibitor-dependent induction threshold")
            wrapper = (V_out, params) -> Main.germ_response_feedback(
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
            #param_keys = [:b_max, :s_max, :Pₛ, :Pₛ_cs, :K_cI, :K_cC, :μ_ψ, :δ_ψ, :μ_γ, :δ_γ, :k_I]
            # param_occurrences = [1, n_src, 1, n_src, 1, n_src, 1, 1, 1, 1, n_src]
        
        elseif model_type == "feedback_combined_inhibitor_inducer_perm_inhibitor_thresh" # ADE
            println("Model: 2-factor germination with inhibitor- and inducer-dependent inhibitor/inducer permeability, inhibitor-dependent induction threshold")
            wrapper = (V_out, params) -> Main.germ_response_feedback(
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
            #param_keys = [:b_max, :s_max, :Pₛ, :Pₛ_cs, :K_cI, :K_cC, :μ_ψ, :δ_ψ, :μ_γ, :δ_γ, :μ_ω, :δ_ω, :k_I]
            # param_occurrences = [1, n_src, 1, n_src, 1, n_src, 1, 1, 1, 1, n_src, n_src, n_src]
        
        elseif model_type == "feedback_inhibitor_inducer_thresh_inhibitor_perm_signal" # BCD
            println("Model: Inhibitor-dependent germination with inhibitor-dependent inhibitor/inducer permeability and inducer-dependent inhibition threshold")
            wrapper = (V_out, params) -> Main.germ_response_feedback(
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
            #param_keys = [:b_max, :Pₛ, :Pₛ_cs, :K_cI, :K_cC, :K_I, :μ_ψ, :δ_ψ, :μ_γ, :δ_γ, :k_C, :n]
            # param_occurrences = [1, 1, n_src, n_src, 1, 1, 1, 1, 1, n_src, n_src]
        
        elseif model_type == "feedback_combined_inducer_thresh_inhibitor_perm_signal" # BCD
            println("Model: 2-factor germination with inhibitor-dependent inhibitor/inducer permeability and inducer-dependent inhibition threshold")
            wrapper = (V_out, params) -> Main.germ_response_feedback(
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
            #param_keys = [:b_max, :Pₛ, :Pₛ_cs, :K_cI, :K_cC, :K_I, :μ_ψ, :δ_ψ, :μ_γ, :δ_γ, :μ_ω, :δ_ω, :k_C, :n]
            # param_occurrences = [1, 1, n_src, n_src, 1, 1, 1, 1, 1, n_src, n_src, n_src, n_src]
        
        elseif model_type == "combined_inhibitor_thresh_signal_inducer_thresh" # BCE
            println("Model: inhibitor-modulated induction threshold and signal / inducer-modulated inhibition threshold")
            wrapper = (inputs, params) -> Main.germ_response_inh_dep_ind_thresh_signal_ind_dep_inh_thresh_2_factor_gh(
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
            #param_keys = [:Pₛ, :Pₛ_cs, :K_cI, :K_cC, :K_I, :n, :k_I, :k_C, :μ_γ, :δ_γ, :μ_ω, :δ_ω, :μ_ψ, :δ_ψ]
            # param_occurrences = [1, n_src, n_src, 1, n_src, n_src, n_src, 1, 1, n_src, n_src, 1, 1]

        elseif model_type == "feedback_combined_inhibitor_perm_thresh_inducer_thresh" # BDE
            println("Model: 2-factor germination with inhibitor-dependent inhibitor/inducer permeability and inducer threshold, inducer-dependent inhibition threshold")
            wrapper = (V_out, params) -> Main.germ_response_feedback(
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
            #param_keys = [:b_max, :Pₛ, :Pₛ_cs, :K_cI, :K_cC, :μ_ψ, :δ_ψ, :μ_γ, :δ_γ, :μ_ω, :δ_ω, :k_I, :k_C]
            # param_occurrences = [1, 1, n_src, 1, n_src, 1, 1, 1, 1, n_src, n_src, n_src, n_src]
        
        elseif model_type == "feedback_inducer_inhibitor_perm_thresh_signal" # CDE
            println("Model: Inducer-dependent germination with inhibitor-dependent inhibitor/inducer permeability, induction threshold and signal")
            wrapper = (V_out, params) -> Main.germ_response_feedback(
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
            #param_keys = [:b_max, :Pₛ, :Pₛ_cs, :K_cI, :K_cC, :K_I, :μ_ψ, :δ_ψ, :μ_ω, :δ_ω, :k_I, :n]
            # param_occurrences = [1, 1, n_src, 1, n_src, 1, 1, 1, n_src, n_src, n_src, n_src]
        
        elseif model_type == "feedback_combined_inhibitor_perm_thresh_signal" # CDE
            println("Model: 2-factor germination with inhibitor-dependent inhibitor/inducer permeability, induction threshold and signal")
            wrapper = (V_out, params) -> Main.germ_response_feedback(
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
            #param_keys = [:b_max, :Pₛ, :Pₛ_cs, :K_cI, :K_cC, :K_I, :μ_ψ, :δ_ψ, :μ_γ, :δ_γ, :μ_ω, :δ_ω, :k_I, :n]
            # param_occurrences = [1, 1, n_src, 1, n_src, 1, 1, 1, 1, 1, n_src, n_src, n_src, n_src]
        
        elseif model_type == "feedback_inhibitor_inducer_perm_thresh_inhibitor_perm_signal" # ABCD
            println("Model: Inhibitor-dependent germination with inducer/inhibitor-dependent inhibitor/inducer permeability, inducer_dependent inhibition threshold and inhibitor-dependent induction signal")
            wrapper = (V_out, params) -> Main.germ_response_feedback(
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
            #param_keys = [:b_max, :s_max, :Pₛ, :Pₛ_cs, :K_cI, :K_cC, :K_I, :μ_ψ, :δ_ψ, :μ_γ, :δ_γ, :k_C, :n]
            # param_occurrences = [1, n_src, 1, n_src, 1, n_src, 1, 1, 1, 1, n_src, n_src]
        
        elseif model_type == "feedback_combined_inducer_perm_thresh_inhibitor_perm_signal" # ABCD
            println("Model: 2-factor germination with inducer/inhibitor-dependent inhibitor/inducer permeability, inducer_dependent inhibition threshold and inhibitor-dependent induction signal")
            wrapper = (V_out, params) -> Main.germ_response_feedback(
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
            #param_keys = [:b_max, :s_max, :Pₛ, :Pₛ_cs, :K_cI, :K_cC, :K_I, :μ_ψ, :δ_ψ, :μ_γ, :δ_γ, :μ_ω, :δ_ω, :k_C, :n]
            # param_occurrences = [1, n_src, 1, n_src, 1, n_src, 1, 1, 1, 1, n_src, n_src, n_src, n_src]
        
        elseif model_type == "feedback_combined_inhibitor_thresh_signal_inducer_perm_thresh" # ABCE
            println("Model: 2-factor germination with inducer-dependent permeability/inhibition threshold and inhibitor-dependent induction signal and threshold")
            wrapper = (V_out, params) -> Main.germ_response_feedback(
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
            #param_keys = [:s_max, :Pₛ, :Pₛ_cs, :K_cI, :K_cC, :K_I, :μ_ψ, :δ_ψ, :μ_γ, :δ_γ, :μ_ω, :δ_ω, :k_I, :k_C, :n]
            # param_occurrences = [n_src, 1, n_src, 1, n_src, 1, 1, 1, 1, n_src, n_src, n_src]
            
        elseif model_type == "feedback_combined_inhibitor_perm_thresh_inducer_perm_thresh" # ABDE
            println("Model: 2-factor germination with inhibitor- and inducer-dependent inhibitor/inducer permeability, inhibitor- and inducer-dependent thresholds")
            wrapper = (V_out, params) -> Main.germ_response_feedback(
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
            #param_keys = [:b_max, :s_max, :Pₛ, :Pₛ_cs, :K_cI, :K_cC, :μ_ψ, :δ_ψ, :μ_γ, :δ_γ, :μ_ω, :δ_ω, :k_I, :k_C]
            # param_occurrences = [1, n_src, 1, n_src, 1, n_src, 1, 1, 1, 1, n_src, n_src]
        
        elseif model_type == "feedback_inducer_inhibitor_perm_thresh_signal_inducer_perm" # ACDE
            println("Model: Inhibitor-dependent germination with inducer/inhibitor-dependent inhibitor/inducer permeability and inhibitor-dependent induction signal and threshold")
            wrapper = (V_out, params) -> Main.germ_response_feedback(
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
            #param_keys = [:b_max, :s_max, :Pₛ, :Pₛ_cs, :K_cI, :K_cC, :K_I, :μ_ψ, :δ_ψ, :μ_ω, :δ_ω, :k_I, :n]
        
        elseif model_type == "feedback_combined_inhibitor_perm_thresh_signal_inducer_perm" # ACDE
            println("Model: 2-factor germination with inducer/inhibitor-dependent inhibitor/inducer permeability and inhibitor-dependent induction signal and threshold")
            wrapper = (V_out, params) -> Main.germ_response_feedback(
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
            #param_keys = [:b_max, :s_max, :Pₛ, :Pₛ_cs, :K_cI, :K_cC, :K_I, :μ_ψ, :δ_ψ, :μ_γ, :δ_γ, :μ_ω, :δ_ω, :k_I, :n]
        
        elseif model_type == "feedback_combined_inhibitor_perm_thresh_signal_inducer_thresh" # BCDE
            println("Model: 2-factor germination with inhibitor-dependent inhibitor/inducer permeability, induction threshold and signal, and inducer-dependent inhibition threshold")
            wrapper = (V_out, params) -> Main.germ_response_feedback(
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
            #param_keys = [:b_max, :Pₛ, :Pₛ_cs, :K_cI, :K_cC, :K_I, :μ_ψ, :δ_ψ, :μ_γ, :δ_γ, :μ_ω, :δ_ω, :k_I, :k_C, :n]
        
        elseif model_type == "feedback_combined_inhibitor_perm_thresh_signal_inducer_perm_thresh" # ABCDE
            println("Model: 2-factor germination with inducer/inhibitor-dependent inhibitor/inducer permeability, inducer-dependent inhibition threshold and inhibitor-dependent induction threshold/signal")
            wrapper = (V_out, params) -> Main.germ_response_feedback(
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
            #param_keys = [:b_max, :s_max, :Pₛ, :Pₛ_cs, :K_cI, :K_cC, :K_I, :μ_ψ, :δ_ψ, :μ_γ, :δ_γ, :μ_ω, :δ_ω, :k_I, :k_C, :n]
        
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

                # println(size(times_tile))
                # println(size(densities_tile))
                # println(size(input_tuples))
                # println(size(dantigny_data))
                # println(size(dantigny_data_flat))

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
            max_steps (int): maximum number of steps for the optimization
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
            wrapper = (ρₛ, params) -> Main.germ_response_inducer_dep_inhibitor_eq(
                ρₛ,
                dist_ξ,
                params[1], #μ_γ
                params[1] * exp(params[2]) # σ_γ = μ_γ * exp(δ_γ)
            )
            param_keys = [:μ_γ, :δ_γ]
        elseif model_type in ["combined_inducer", "combined_inducer_ex"]
            println("Model: Two-factor germination with inhibitor-dependent induction threshold and signal")
            wrapper = (ρₛ, params) -> Main.germ_response_inhibitor_dep_inducer_2_factors_eq(
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
            wrapper = (ρₛ, params) -> Main.germ_response_inhibitor_dep_inducer_thresh_2_factors_eq(
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
            wrapper = (ρₛ, params) -> Main.germ_response_inhibitor_dep_inducer_signal_2_factors_eq(
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
            wrapper = (ρₛ, params) -> Main.germ_response_independent_eq(
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
            wrapper_ex = (c_ex, params) -> Main.germ_response_inducer_dep_inhibitor_eq_c_ex(
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
            wrapper_ex = (c_ex, params) -> Main.germ_response_inhibitor_dep_inducer_2_factors_eq_c_ex(
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
            wrapper_ex = (c_ex, params) -> Main.germ_response_inhibitor_dep_inducer_thresh_2_factors_eq_c_ex(
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
            wrapper_ex = (c_ex, params) -> Main.germ_response_inhibitor_dep_inducer_signal_2_factors_eq_c_ex(
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
            wrapper_ex = (c_ex, params) -> Main.germ_response_independent_eq_c_ex(
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

    function fit_feedback_model_to_data(model_type, def_params, dantigny_data, times, sources, densities, bounds_dict, max_steps=10000)

    end

end