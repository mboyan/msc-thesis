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
    using QuasiMonteCarlo

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
        rho_s = params[pbase + 1]
        mu_R = params[pbase + 4]
        sigma_R = params[pbase + 5]
        mu_H = params[pbase + 6]
        sigma_H = params[pbase + 7]
        mu_X = params[pbase + 8]
        sigma_X = params[pbase + 9]
        mu_Y = params[pbase + 10]
        sigma_Y = params[pbase + 11]
        c_ex = params[pbase + 12]
        P_I = params[pbase + 13]
        P_C = params[pbase + 14]
        K_s = params[pbase + 15]

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
    System of coupled ODEs for the feedback model
    of inducer-dependent cell wall permeability.
    """
    function ode_system_A(u, p, t)
        c_in_I, c_out_I, c_in_C, germ = u
    
        # Unpack parameters for this specific spore
        P_I, P_C, c_ex, A_s, V_s, V_free, V_ps, K_s, lambda_C, gamma, omega = p

        # Compute inducing signal s from c_in_C
        s = c_in_C / (K_s + c_in_C)
        
        # Update permeability constants based on signal
        P_I_pert = 1000.0f0 - (1000.0f0 - P_I) * exp(-lambda_C * s)
        P_C_pert = 1000.0f0 - (1000.0f0 - P_C) * exp(-lambda_C * s)
        
        # ODE equations
        du1 = -(P_I_pert * A_s / V_s) * (c_in_I - c_out_I)
        du2 = (P_I_pert * A_s / V_free) * (c_in_I - c_out_I)
        du3 = -(P_C_pert * A_s / V_ps) * (c_in_C - c_ex)

        # GPU-friendly germination logic (no control flow)
        thresholds_met = Float32((c_in_I < gamma) && (s > omega))
        germ_not_full = Float32(germ < 1.0f0)
        du4 = thresholds_met * germ_not_full

        return SVector{4}(du1, du2, du3, du4)
    end

    """
    Host orchestration: compute germination
    probability of a feedback model by running
    parallelised Monte Carlo integration for an
    ensemble of ODEs for a sample of parameters.
    inputs:
        params (Array{Float32}) - parameters
        times (Array{Float32}) - time points for evaluation
        model_idx (Int) - index of model to use
        n_samples (Int) - ODE ensemble sample size
    """
    function integrate_ode(params::Array{Float32,2}, times::Array{Float32,1}, model_idx::Int; n_samples::Int=256)
        P = size(params, 1)
        param_dim = size(params, 2)
        T = length(times)

        sobol_dim = 5

        # Unpack parameters
        rho_s = params[:, 1]
        mu_O = params[:, 2]
        sigma_O = params[:, 3]
        mu_R = params[:, 4]
        sigma_R = params[:, 5]
        mu_H = params[:, 6]
        sigma_H = params[:, 7]

        mu_X = params[:, 8]
        sigma_X = params[:, 9]
        mu_Y = params[:, 10]
        sigma_Y = params[:, 11]

        c_ex = params[:, 12]
        P_I = params[:, 13]
        P_C = params[:, 14]
        K_s = params[:, 15]
        lambda_C = params[:, 17]

        # Generate Sobol sample
        sobol_samples = QuasiMonteCarlo.sample(n_samples, sobol_dim, SobolSample())

        # Construct flat parameter collections and initial conditions
        n_samples_flat = P * n_samples
        # u0_matrix = zeros(Float32, 4, n_samples_flat)
        # p_matrix = zeros(Float32, 11, n_samples_flat)
        u0_vec = Vector{SVector{4, Float32}}()
        p_vec = Vector{SVector{11, Float32}}()
        gamma_samples = zeros(Float32, n_samples_flat)
        omega_samples = zeros(Float32, n_samples_flat)
        @inbounds for i in 1:P

            # Transform from standard Normal to LogNormal
            mu_R_log = log(mu_R[i]^2 / sqrt(sigma_R[i]^2 + mu_R[i]^2))
            sigma_R_log = sqrt(log(sigma_R[i]^2 / mu_R[i]^2 + 1))
            mu_H_log = log(mu_H[i]^2 / sqrt(sigma_H[i]^2 + mu_H[i]^2))
            sigma_H_log = sqrt(log(sigma_H[i]^2 / mu_H[i]^2 + 1))
            mu_O_log = log(mu_O[i]^2 / sqrt(sigma_O[i]^2 + mu_O[i]^2))
            sigma_O_log = sqrt(log(sigma_O[i]^2 / mu_O[i]^2 .+ 1))

            # Generate geometric samples
            r = quantile.(LogNormal(mu_R_log, sigma_R_log), sobol_samples[1, :])
            d_ps = quantile.(LogNormal(mu_H_log, sigma_H_log), sobol_samples[2, :])

            # Generate initial concentration samples
            c0 = quantile.(LogNormal(mu_O_log, sigma_O_log), sobol_samples[3, :])

            # Generate threshold samples
            gamma = quantile.(Normal(mu_X[i], sigma_X[i]), sobol_samples[4, :])
            omega = quantile.(Normal(mu_Y[i], sigma_Y[i]), sobol_samples[5, :])

            @inbounds for j in 1:n_samples
                
                # idx = (j - 1) * P + i
                idx = (i - 1) * n_samples + j
                
                V_s = 4pi/3 * r[j] ^ 3
                A_s = 4pi * r[j] ^ 2
                V_ps = calc_ps_vacant_vol(r[j], d_ps[j])
                V_free = 1 / rho_s[i] - V_s

                push!(u0_vec, @SVector [Float32(c0[j]), 0.0f0, 0.0f0, 0.0f0])
                push!(p_vec, @SVector [
                    P_I[i],
                    P_C[i],
                    c_ex[i],
                    Float32(A_s),
                    Float32(V_s),
                    Float32(V_free),
                    Float32(V_ps),
                    K_s[i],
                    lambda_C[i],
                    Float32(gamma[j]),
                    Float32(omega[j])
                ])

                gamma_samples[idx] = Float32(gamma[j])
                omega_samples[idx] = Float32(omega[j])
            end
        end

        # Wrapper to create problem for each sample
        function prob_func(prob, i, repeat)

            u0 = u0_vec[i]#u0_matrix[:, i]
            p_i = p_vec[i]#p_matrix[:, i]
            
            return remake(prob, u0=u0, p=p_i)
        end
        
        # Initial problem (dummy, will be remade by prob_func)
        u0_dummy = @SVector [0.0f0, 1.0f0, 0.0f0, 0.0f0]
        p_dummy = @SVector [
            0.0f0, 0.0f0, 0.0f0,
            1.0f0, 1.0f0, 1.0f0, 0.0f0, 0.0f0, 1.0f0,
            0.0f0, 1.0f0
        ]
        
        prob = ODEProblem{false}(ode_system_A, u0_dummy, times[end], p_dummy)
        monteprob = EnsembleProblem(prob, prob_func=prob_func, safetycopy=false)
        
        # Solve ensemble on GPU
        sols = solve(
            monteprob, 
            GPUTsit5(),
            DiffEqGPU.EnsembleGPUKernel(CUDA.CUDABackend()),
            trajectories=n_samples_flat,
            adaptive=true,
            dt=0.001f0,
            saveat = times
        )

        # Extract germination info from solutions
        germinated = zeros(Bool, T, n_samples_flat)
        # Threads.@threads for i in 1:n_samples_flat
        #     germ_vals = getindex.(sols[i].u, 4)  # Extract 4th component for all timepoints
        #     germinated[:, i] .= germ_vals .> 0
        # end
        Threads.@threads for i in 1:n_samples_flat
            u_trajectory = sols[i].u
            # Check threshold criterion at each timepoint, with ratchet logic
            germ_flag = false
            @inbounds for (ti, u) in enumerate(u_trajectory)
                c_in_I, c_out_I, c_in_C, germ = u
                s = c_in_C / (K_s[i] + c_in_C)
                threshold_met = (c_in_I < gamma_samples[i]) && (s > omega_samples[i])
                germ_flag = germ_flag || threshold_met
                germinated[ti, i] = germ_flag
            end
        end

        germinated = reshape(germinated, (T, n_samples, P))

        germination = mean(germinated, dims=2)
        germination = dropdims(germination; dims=2)'

        return germination
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

        # n_params = length(keys(param_dict))
        sample_size = length(param_dict[:mu_R])
        param_keys = keys(param_dict)

        # Unpack parameter dictionary into an Array P x param_dim
        param_arr = Array{Float32}(undef, sample_size, 17)
        param_arr[:, 1] .= Float32.(rho_s)
        :mu_O in param_keys ? param_arr[:, 2] .= Float32.(param_dict[:mu_O]) : nothing
        :sigma_O in param_keys ? param_arr[:, 3] .= Float32.(param_dict[:sigma_O]) : nothing
        :mu_R in param_keys ? param_arr[:, 4] .= Float32.(param_dict[:mu_R]) : nothing
        :sigma_R in param_keys ? param_arr[:, 5] .= Float32.(param_dict[:sigma_R]) : nothing
        :mu_H in param_keys ? param_arr[:, 6] .= Float32.(param_dict[:mu_H]) : nothing
        :sigma_H in param_keys ? param_arr[:, 7] .= Float32.(param_dict[:sigma_H]) : nothing
        :mu_X in param_keys ? param_arr[:, 8] .= Float32.(param_dict[:mu_X]) : nothing
        :sigma_X in param_keys ? param_arr[:, 9] .= Float32.(param_dict[:sigma_X]) : nothing
        :mu_Y in param_keys ? param_arr[:, 10] .= Float32.(param_dict[:mu_Y]) : nothing
        :sigma_Y in param_keys ? param_arr[:, 11] .= Float32.(param_dict[:sigma_Y]) : nothing
        :c_ex in param_keys ? param_arr[:, 12] .= Float32.(param_dict[:c_ex]) : nothing
        :P_I in param_keys ? param_arr[:, 13] .= Float32.(param_dict[:P_I]) : nothing
        :P_C in param_keys ? param_arr[:, 14] .= Float32.(param_dict[:P_C]) : nothing
        :K_s in param_keys ? param_arr[:, 15] .= Float32.(param_dict[:K_s]) : nothing
        :lambda_I in param_keys ? param_arr[:, 16] .= Float32.(param_dict[:lambda_I]) : nothing
        :lambda_C in param_keys ? param_arr[:, 17] .= Float32.(param_dict[:lambda_C]) : nothing

        if model_alias == "independent"

            germination = integrate_batched(8, 2, param_arr, times, 1)

        elseif model_alias == "feedback_inhibitor_inducer_perm"
            
            germination = integrate_ode(param_arr, times, 2)

        elseif model_alias == "feedback_combined_inducer_perm"
            
            germination = integrate_ode(param_arr, times, 3)
            
        end
    end

end