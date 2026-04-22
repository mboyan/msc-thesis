module GermStatsGPU
__precompile__(false)
    """
    Contains GPU-accelerated tools for generating germination statistics
    """

    using CUDA
    using FastGaussQuadrature
    using DifferentialEquations
    using OrdinaryDiffEq
    using DiffEqGPU
    using Distributions
    using StaticArrays

    export compute_germination

    # """
    # Build tensor-product coords & weights
    # for Gauss-Legendre quadrature
    # """
    # function build_tensor_grid(n::Int, d::Int)
    #     xs, ws = gausslegendre(n)
    #     N = n^d
    #     coords = Array{Float32}(undef, N * d)
    #     weights = Array{Float32}(undef, N)
    #     for idx in 0:(N-1)
    #         base = idx * d
    #         tmp = idx
    #         w = 1f0
    #         for k in 1:d
    #             ik = (tmp % n) + 1
    #             tmp ÷= n
    #             coords[base + k] = Float32(xs[ik])
    #             w *= Float32(ws[ik])
    #         end
    #         weights[idx+1] = w
    #     end
    #     return coords, weights, N
    # end

    """
    Build tensor-product coords & weights
    for Gauss-Hermite quadrature
    """
    function build_hermite_tensor_grid(n::Int, d::Int)
        xs, ws = gausshermite(n)  # Gauss-hermite nodes and weights

        # Normalize weights for standard normal PDF
        ws_normalized = ws ./ sqrt(π)

        N = n^d
        coords = Array{Float32}(undef, N * d)
        weights = Array{Float32}(undef, N)
        
        for idx in 0:(N-1)
            base = idx * d
            tmp = idx
            w = 1f0
            for k in 1:d
                ik = (tmp % n) + 1
                tmp ÷= n
                coords[base + k] = Float32(xs[ik])  # Standard normal nodes
                w *= Float32(ws_normalized[ik])
            end
            weights[idx+1] = w
        end
        return coords, weights, N
    end

    # """
    # Integrand for independent inhibition/induction
    # """
    # @inline function integrand_point_0(coords, base::Int32, d::Int32, params, pbase::Int32, t::Float32)
    #     # coords: flattened coords array
    #     # params: flattened params array (M * param_dim)
    #     val = 1.0f0
    #     @inbounds for k in 0:(d-1)
    #         xi = coords[base + k + 1]
    #         val *= (1f0 - xi^2)          # example integrand
    #     end
    #     # Example param dependence: params[pbase + 1] ... params[pbase+7]
    #     # incorporate parameters into val as needed. Example multiply by sum(params):
    #     s = 0f0
    #     @inbounds for j in 0:6
    #         s += params[pbase + j + 1]
    #     end
    #     return val * (1f0 + 0.001f0 * s) * exp(-t)  # example usage
    # end

    """
    Integrand for independent inhibition/induction using Gauss-Hermite.
    Transforms standard normal nodes (u, v) to physical space (r_s, d_ps) 
    using: r_s = μ_R + σ_R * u, d_ps = μ_H + σ_H * v
    The normal PDFs are implicit in the Gauss-Hermite weights.
    Parameters layout in flattened array:
    [0:0]   = mu_R
    [1:1]   = sigma_R
    [2:2]   = mu_H
    [3:3]   = sigma_H
    [4:4]   = mu_X
    [5:5]   = sigma_X
    [5:5]   = mu_Y
    [6:6]   = sigma_Y
    [7:7]   = c_ex
    [8:8]   = P_I
    [9:9]   = P_C
    [10:10]   = K_s
    [11:11]   = rho_s
    inputs:
        coords (Array{Float32}) - standard normal nodes from Gauss-Hermite
        base (Int32) - starting index for this quadrature point's coordinates
        d (Int32) - dimension (should be 2 for your problem)
        params (Array{Float32}) - flattened parameter array
        pbase (Float32) - starting index for this parameter set
        t (Float32) - time point for evaluation
    """
    @inline function integrand_point_0(coords, base::Int32, d::Int32, params,
                                            pbase::Int32, t::Float32)
        
        
        # Extract standard normal nodes
        u = coords[base + 1]          # Standard normal for r_s
        v = coords[base + 2]          # Standard normal for d_ps
        
        # Extract transformation parameters
        mu_R = params[pbase + 1]
        sigma_R = params[pbase + 2]
        mu_H = params[pbase + 3]
        sigma_H = params[pbase + 4]
        mu_X = params[pbase + 5]
        sigma_X = params[pbase + 6]
        mu_Y = params[pbase + 7]
        sigma_Y = params[pbase + 8]
        c_ex = params[pbase + 9]
        P_I = params[pbase + 10]
        P_C = params[pbase + 11]
        K_s = params[pbase + 12]
        rho_s = params[pbase + 13]

        # Transform from standard Normal to LogNormal
        mu_R_log = log(mu_R^2 / sqrt(sigma_R^2 + mu_R^2))
        sigma_R_log = sqrt(log(sigma_R^2 / mu_R^2 + 1))
        mu_H_log = log(mu_H^2 / sqrt(sigma_H^2 + mu_H^2))
        sigma_H_log = sqrt(log(sigma_H^2 / mu_H^2 + 1))
        
        # Transform from LogNormal to physical space
        r_s = exp(mu_R_log + sigma_R_log * u)
        d_ps = exp(mu_H_log + sigma_H_log * v)
        
        # Clamp to physically meaningful values (optional, but recommended)
        r_s = max(r_s, 1f-6)      # Spore radius must be positive
        d_ps = max(d_ps, 1f-6)    # Cell wall thickness must be positive
        
        # Compute geometric variables
        V_s, A_s = calc_spore_geom(r_s)
        V_ps = calc_ps_vacant_vol(r_s, d_ps)
        
        # Compute secondary variables
        phi = rho_s * V_s
        tau_I = V_s / (P_I * A_s)
        tau_C = V_ps / (P_C * A_s)
        
        # Time-dependent signals
        beta = calc_beta(t, phi, tau_I)
        s = calc_signal(t, c_ex, K_s, tau_C)

        # Distributions
        dist_X = Normal(mu_X, sigma_X)
        dist_Y = Normal(mu_Y, sigma_Y)
        
        # CDFs
        cdf_X = cdf(dist_X, beta)
        cdf_Y = cdf(dist_Y, s)
        
        # Return integrand (NO explicit f_R, f_H—they're in the Hermite weights!)
        val = (1f0 - cdf_X) * cdf_Y
        
        return val
    end

    """
    Kernel: write per-(point,param) contribution to outbuf.
    inputs:
        coords_chunk (Array{Float36}) - coordinates of Gauss-Legendre nodes for the current data chunk
        weights_chunk (Array{Float36}) - Gauss-Legendre weights for the current data chunk
        d (Int) - dimension of integral
        params_d (CuArray) - GPU array of the parameter values
        param_dim (Int) - number of parameters
        times_d (CuArray) - GPU array of the time points for evaluation
        T (Int) - number of time points
        outbuf (CuArray) - buffer for saving preliminary values
        P (Int) - size of parameter sample
        model_idx (Int) - index of current model
    """
    function batch_kernel(coords_chunk, weights_chunk, d::Int32,
                        params_d, param_dim::Int32, times_d, T::Int32, outbuf, P::Int32, model_idx::Int32)
        tid = (blockIdx().x-1) * blockDim().x + threadIdx().x
        Ntot = Int32(length(weights_chunk)) * P * T
        if tid <= Ntot
            # Decode flat index:  tid-1 = time_idx + T*(param_idx + P*pidx)
            tmp       = tid - 1
            time_idx  = Int32(tmp % T)
            tmp       = tmp ÷ T
            param_idx = Int32(tmp % P)
            pidx      = Int32(tmp ÷ P)

            base  = Int32(pidx  * d)
            pbase = Int32(param_idx * param_dim)
            t     = times_d[time_idx + 1]
            
            if model_idx == 1 # Independent induction/inhibition
                val = integrand_point_0(coords_chunk, base, d, params_d, pbase, t)
            end
            w   = weights_chunk[pidx + 1]
            outbuf[tid] = val * w
        end
        return
    end


    """
    Host orchestration: compute germination
    probability integral by
    streaming points in chunks
    inputs:
        n (Int) - number of Gauss-Legendre nodes
        d (Int) - dimension of integral
        params (Array{Float32}) - parameters
        times (Array{Float32}) - time points for evaluation
        model_idx (Int) - index of model to use
        chunk_size (Int) - chunk size for parallel execution
    """
    function integrate_batched(n::Int, d::Int, params::Array{Float32,2}, times::Array{Float32,1}, model_idx::Int; chunk_size::Int=4096)
        coords_cpu, weights_cpu, N = build_hermite_tensor_grid(n, d)
        P = size(params, 1)
        param_dim = size(params, 2)
        T = length(times)

        params_flat = vec(params')                    # row-major flatten (M × param_dim)
        params_d = CuArray(params_flat)
        times_d = CuArray(times)

        # Host-side accumulator for reduction
        accum = zeros(Float32, P, T)

        # Iterate over chunks
        i = 1
        while i <= N
            len = min(chunk_size, N - i + 1)
            coords_chunk = CuArray(@view coords_cpu[(i-1)*d + 1 : (i-1)*d + len*d])
            weights_chunk = CuArray(@view weights_cpu[i : i+len-1])
            
            total = len * P * T
            outbuf = CUDA.zeros(Float32, total)       # per-(point,param) contributions

            threads = 256
            blocks = cld(total, threads)
            @cuda threads=threads blocks=blocks batch_kernel(coords_chunk, weights_chunk, Int32(d),
                                                        params_d, Int32(param_dim), times_d, Int32(T), outbuf, Int32(P), Int32(model_idx))
            CUDA.synchronize()
            
            # COPY OUTBUF TO HOST AND REDUCE PER-PARAM (fast for moderate chunk_size)
            host_buf = Array(outbuf)                            # length = len * P * T
            host_mat = reshape(host_buf, (T, P, len))           # T × P × len
            # sum across columns -> partial sums per parameter
            partials  = dropdims(sum(host_mat; dims=3), dims=3) # T × P
            # accum = Array(accum_d)                              # copy current accum to host
            accum .+= partials' #vec(partials)                             # update
            # accum_d .= CuArray(accum)                           # copy back to device accumulators

            i += len
        end

        return accum # P × T:  accum[p, t] = integral for param set p at time t
    end

    """
    Host orchestration: compute germination
    probability of a feedback model by running
    parallelised Monte Carlo integration for an
    ensemble of ODEs for a sample of parameters.
    Parameters layout in flattened array:
    [0:0]   = mu_O
    [1:1]   = sigma_O
    [2:2]   = mu_R
    [3:3]   = sigma_R
    [4:4]   = mu_H
    [5:5]   = sigma_H
    inputs:
        params (Array{Float32}) - parameters
        times (Array{Float32}) - time points for evaluation
        model_idx (Int) - index of model to use
        n_samples (Int) - ODE ensemble sample size
    """
    function integrate_batched_ode(params::Array{Float32,2}, times::Array{Float32,1}, model_idx::Int; n_samples::Int=2048)
        P = size(params, 1)
        param_dim = size(params, 2)
        T = length(times)

        # params_flat = vec(params')                    # row-major flatten (M × param_dim)
        # params_d = CuArray(params_flat)
        # times_d = CuArray(times)

        # Unpack parameters
        mu_O = params[:, 1]
        sigma_O = params[:, 2]
        # CONNTINUE HERE!!!!!!!!!!!!!

        # Transform from standard Normal to LogNormal
        mu_R_log = log(mu_R^2 / sqrt(sigma_R^2 + mu_R^2))
        sigma_R_log = sqrt(log(sigma_R^2 / mu_R^2 + 1))
        mu_H_log = log(mu_H^2 / sqrt(sigma_H^2 + mu_H^2))
        sigma_H_log = sqrt(log(sigma_H^2 / mu_H^2 + 1))
        mu_O_log = log(mu_O^2 / sqrt(sigma_O^2 + mu_O^2))
        sigma_O_log = sqrt(log(sigma_O^2 / mu_O^2 + 1))

        # Generate geometric samples
        r_samples = quantile(LogNormal(mu_R_log, sigma_R_log))
        dps_samples = quantile(LogNormal(mu_H_log, sigma_H_log))
        Vs_samples = 4pi/3 .* r_samples .^ 3
        As_samples = 4pi .* r_samples .^ 2
        Vps_samples = calc_ps_vacant_vol.(r_samples, dps_samples)

        # Generate initial concentration samples
        c0_samples = uantile(LogNormal(mu_O_log, sigma_O_log))

        # !!!!!!!!REVISE BELOW!!!!!!!!!!!!!!!!!!

        # Wrapper to create problem for each sample
        function prob_func(prob, i, repeat)
            
            # Parameters tuple
            p_sample = @SVector [
                Float32(P_perturbed_I), 
                Float32(P_perturbed_C), 
                Float32(P_max),
                Float32(c_ex_C),
                Float32(geom.A_s),
                Float32(geom.V_s),
                Float32(V_free),
                Float32(geom.V_ps_eff),
                Float32(K_s_C),
                Float32(λ_C)
            ]
            
            return remake(prob, u0=u0, p=p_sample)
        end
        
        # # Initial problem (dummy, will be remade by prob_func)
        # u0_dummy = @SVector [0.0f0, 1.0f0, 0.0f0]
        # p_dummy = @SVector [
        #     0.0f0, 0.0f0, Float32(P_max), Float32(c_ex_C),
        #     1.0f0, 1.0f0, 1.0f0, 1.0f0, Float32(K_s_C), Float32(λ_C)
        # ]
        
        # prob = ODEProblem{false}(germination_system!, u0_dummy, t_span, p_dummy)
        # monteprob = EnsembleProblem(prob, prob_func=prob_func, safetycopy=false)
        
        # # Solve ensemble on GPU
        # sols = solve(
        #     monteprob, 
        #     Tsit5(),
        #     DiffEqGPU.EnsembleGPUKernel(CUDA.CUDABackend()),
        #     trajectories=n_samples,
        #     adaptive=true,
        #     dt=0.001f0,
        #     save_on=false
        # )
    end

    # =====================================================================================
    # ================== MODEL-SPECIFIC KERNEL-FRIENDLY FUNCTIONS =========================
    # =====================================================================================

    """
    Compute relative decrease of inhibitor concentration
    inside the spore over time.
    inputs:
        t (Float32) - time (in seconds)
        phi (Float32) - volume fraction occupied by spores
        tau (Float32) - characteristic inhibitor decrease time (in seconds)
    """
    @inline function calc_beta(t, phi, tau)
        return phi + (1 - phi) * exp(- t / (tau * (1 - phi)))
    end

    """
    Compute the inducer concentration
    inside the inner cell wall over time.
    inputs:
        t (Float32) - time (in seconds)
        c_ex (Float32) - ambient inducer concentration (in 1e-5 M)
        tau (Float32) - characteristic inhibitor decrease time (in seconds)
    """
    @inline function calc_inducer_concentration(t, c_ex, tau)
        return c_ex * (1 - exp(-t / tau))
    end

    """
    Compute the concentration-dependent inducing signal
    inside the inner cell wall over time.
    inputs:
        t (Float32) - time (in seconds)
        c_ex (Float32) - ambient inducer concentration (in 1e-5 M)
        K_s (Float32) - half-saturation constant for inducing signal
        tau (Float32) - characteristic inhibitor decrease time (in seconds)
    """
    @inline function calc_signal(t, c_ex, K_s, tau)
        c_in = calc_inducer_concentration(t, c_ex, tau)
        return c_in / (K_s + c_in)
    end

    """
    Compute the volume and surface area of a spore from its radius.
    inputs:
        r (Float32) - spore radius (in um)
    """
    @inline function calc_spore_geom(r)
        V_s = 4/3 * π * r^3
        A_s = 4 * π * r^2
        return V_s, A_s
    end

    """
    Compute the vacant volume of the inner cell wall
    from the spore radius and the inner cell wall thickness.
    inputs:
        r (Float32) - spore radius
        d_ps (Float32) - inner cell wall (polysaccharide layer) thickness (in um)
        d_hp (Float32) - hydrophobin rodlet layer thickness (in um)
        porosity (Float32) - vacant fraction of the inner cell wall volume
    """
    @inline function calc_ps_vacant_vol(r, d_ps; d_hp=0.01, porosity=0.32)
        return porosity * π * ((r - d_hp)^3 - (r - d_hp - d_ps)^3)
    end

    # """
    # rho_s, V_s, A_s, V_ps, P_I, P_C
    # Two-factor germination criterion.
    # inputs:
    #     r (Float32) - current spore radius (in um)
    #     d_ps (Float32) - current inner cell wall thickness (in um)
    #     t (Float32) - time (in seconds)
    #     c_ex (Float32) - ambient inducer concentration (in 1e-5 M)
    #     K_s (Float32) - half-saturation constant for inducing signal
    #     f_R (Float32) - probability density function for spore radius R
    #     f_H (Float32) - probability density funciton for inner cell wall thickness H
    # """
    # @inline function two_factor_germ(r, d_ps, t, c_ex, P_I, P_C, K_s, f_R, f_H)

    #     # Geometric variables
    #     V_s, A_s = calc_spore_geom(r)
    #     V_ps = calc_ps_vacant_vol(r, d_ps)

    #     # Secondary variables
    #     phi = rho_s * V_s
    #     tau_I = V_s / (P_I * A_s)
    #     tau_C = V_ps / (P_C * A_s)

    #     # Time-dependent signals
    #     beta = calc_beta(t, phi, tau_I)
    #     s = calc_signal(t, c_ex, K_s, tau_C)

    #     # CDFs
    #     cdf_X = cdf(dist_X, beta)
    #     cdf_Y = cdf(dist_Y, s)

    #     # Probability densities
    #     # ??????

    #     return (1 - cdf_X) * cdf_Y * f_R * f_H
    # end

    # =====================================================================================
    # ====================== GERMINATION FRACTION CALCULATION =============================
    # =====================================================================================
    """
    Compute the germination fraction for
    a set of parameter values and a time series.
    inputs:
        model_alias (String) - alias of the germination model
        rho_s (Float32) - number density of spore colony (in um^(-1))
        times (Vector{Float32}) - time points to evaluate
        param_dict (Dict) - parameter dictionary, multiple values possible per key
    """
    function compute_germination(model_alias, rho_s, times, param_dict)

        n_params = length(keys(param_dict))
        sample_size = length(param_dict[:mu_X])

        if model_alias == "independent"
            # Unpack parameter dictionary into an Array P x param_dim
            param_arr = Array{Float32}(undef, sample_size, 14)
            param_arr[:, 1] .= Float32.(param_dict[:mu_R])
            param_arr[:, 2] .= Float32.(param_dict[:sigma_R])
            param_arr[:, 3] .= Float32.(param_dict[:mu_H])
            param_arr[:, 4] .= Float32.(param_dict[:sigma_H])
            param_arr[:, 5] .= Float32.(param_dict[:mu_X])
            param_arr[:, 6] .= Float32.(param_dict[:sigma_X])
            param_arr[:, 7] .= Float32.(param_dict[:mu_Y])
            param_arr[:, 8] .= Float32.(param_dict[:sigma_Y])
            param_arr[:, 9] .= Float32.(param_dict[:c_ex])
            param_arr[:, 10] .= Float32.(param_dict[:P_I])
            param_arr[:, 11] .= Float32.(param_dict[:P_C])
            param_arr[:, 12] .= Float32.(param_dict[:K_s])
            param_arr[:, 13] .= Float32.(rho_s)

            germination = integrate_batched(8, 2, param_arr, times, 1)

        elseif model_alias == "feedback_inhibitor_inducer_perm"
            # Unpack parameter dictionary into an Array P x param_dim
            param_arr = Array{Float32}(undef, sample_size, 14)
            param_arr[:, 1] .= Float32.(param_dict[:mu_O])
            param_arr[:, 2] .= Float32.(param_dict[:sigma_O])
            param_arr[:, 3] .= Float32.(param_dict[:mu_R])
            param_arr[:, 4] .= Float32.(param_dict[:sigma_R])
            param_arr[:, 5] .= Float32.(param_dict[:mu_H])
            param_arr[:, 6] .= Float32.(param_dict[:sigma_H])
        end
    end

end