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
    function ode_system_A!(du, u, p, t)
        c_in_I, c_out_I, c_in_C, germ = u
    
        # Unpack parameters for this specific spore
        P_I, P_C, c_ex, A_s, V_s, V_free, V_ps, K_s, lambda_C, gamma, omega = p

        # Compute inducing signal s from c_in_C
        s = c_in_C / (K_s + c_in_C)
        
        # Update permeability constants based on signal
        P_I_pert = 1000.0f0 - (1000.0f0 - P_I) * exp(-lambda_C * s)
        P_C_pert = 1000.0f0 - (1000.0f0 - P_C) * exp(-lambda_C * s)
        
        # ODE equations
        du[1] = -(P_I_pert * A_s / V_s) * (c_in_I - c_out_I)
        du[2] = (P_I_pert * A_s / V_free) * (c_in_I - c_out_I)
        du[3] = -(P_C_pert * A_s / V_ps) * (c_in_C - c_ex)

        # Check if thresholds are reached
        if c_in_I < gamma && c_in_C > omega
            du[4] = germ >= 1.0f0 ? 0.0f0 : 1.0f0
        else
            du[4] = 0.0f0
        end
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
    function integrate_ode(params::Array{Float32,2}, times::Array{Float32,1}, model_idx::Int; n_samples::Int=2048)
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
        c_ex = params[:, 12]
        P_I = params[:, 13]
        P_C = params[:, 14]
        K_s = params[:, 15]
        lambda_C = params[:, 17]

        # Determine which thresholds are used
        if model_idx in [2, 3, 9]
            mu_X = params[:, 8]
            sigma_X = params[:, 9]
        end
        if model_idx in [3, 8, 9]
            mu_Y = params[:, 10]
            sigma_Y = params[:, 11]
        end

        # Transform from standard Normal to LogNormal
        mu_R_log = log.(mu_R.^2 ./ sqrt.(sigma_R.^2 + mu_R.^2))
        sigma_R_log = sqrt.(log.(sigma_R.^2 ./ mu_R.^2 .+ 1))
        mu_H_log = log.(mu_H.^2 ./ sqrt.(sigma_H.^2 + mu_H.^2))
        sigma_H_log = sqrt.(log.(sigma_H.^2 ./ mu_H.^2 .+ 1))
        mu_O_log = log.(mu_O.^2 ./ sqrt.(sigma_O.^2 + mu_O.^2))
        sigma_O_log = sqrt.(log.(sigma_O.^2 ./ mu_O.^2 .+ 1))

        # Generate Sobol sample
        sobol_samples = QuasiMonteCarlo.sample(n_samples, sobol_dim, SobolSample())

        # Generate geometric samples
        r_samples = quantile.(LogNormal.(mu_R_log, sigma_R_log), sobol_samples[1, :])
        dps_samples = quantile.(LogNormal.(mu_H_log, sigma_H_log), sobol_samples[2, :])
        Vs_samples = 4pi/3 .* r_samples .^ 3
        As_samples = 4pi .* r_samples .^ 2
        Vps_samples = calc_ps_vacant_vol.(r_samples, dps_samples)
        Vfree_samples = 1 ./ rho_s .- Vs_samples

        # Generate initial concentration samples
        c0_samples = quantile.(LogNormal.(mu_O_log, sigma_O_log), sobol_samples[3, :])

        # Generate threshold samples
        gamma_samples = quantile.(Normal.(mu_X, sigma_X), sobol_samples[4, :])
        omega_samples = quantile.(Normal.(mu_X, sigma_X), sobol_samples[5, :])

        # Match Sobol samples with parameter samples
        n_samples_flat = P * n_samples
        Vs_samples = repeat(Vs_samples, outer=[1, P])
        Vs_samples = reshape(Vs_samples, n_samples_flat)
        As_samples = repeat(As_samples, outer=[1, P])
        As_samples = reshape(As_samples, n_samples_flat)
        Vps_samples = repeat(Vps_samples, outer=[1, P])
        Vps_samples = reshape(Vps_samples, n_samples_flat)
        c0_samples = repeat(c0_samples, outer=[1, P])
        c0_samples = reshape(c0_samples, n_samples_flat)
        gamma_samples = repeat(gamma_samples, outer=[1, P])
        gamma_samples = reshape(gamma_samples, n_samples_flat)
        omega_samples = repeat(omega_samples, outer=[1, P])
        omega_samples = reshape(omega_samples, n_samples_flat)
        c_ex = repeat(c_ex, inner=[n_samples])
        c_ex = reshape(c_ex, n_samples_flat)
        P_I = repeat(P_I, inner=[n_samples])
        P_I = reshape(P_I, n_samples_flat)
        P_C = repeat(P_C, inner=[n_samples])
        P_C = reshape(P_C, n_samples_flat)
        K_s = repeat(K_s, inner=[n_samples])
        K_s = reshape(K_s, n_samples_flat)
        lambda_C = repeat(lambda_C, inner=[n_samples])
        lambda_C = reshape(lambda_C, n_samples_flat)

        # Save samples to device
        # Vs_samples_gpu = CuArray(Vs_samples)
        # As_samples_gpu = CuArray(As_samples)
        # Vps_samples_gpu = CuArray(Vps_samples)
        # Vfree_samples_gpu = CuArray(Vfree_samples)
        # gamma_samples_gpu = CuArray(gamma_samples)
        # omega_samples_gpu = CuArray(omega_samples)
        # c_ex_gpu = CuArray(c_ex)
        # P_I_gpu = CuArray(P_I)
        # P_C_gpu = CuArray(P_C)
        # K_s_gpu = CuArray(K_s)
        # lambda_C_gpu = CuArray(lambda_C)

        # # Initial conditions
        # c0_samples_gpu = CuArray(c0_samples)
        # coutI_gpu = CUDA.zeros(Float32, n_samples_flat)
        # cinC_gpu = CUDA.zeros(Float32, n_samples_flat)

        u0_sets = [
            [
                c0_samples[i],
                0.0f0,
                0.0f0,
                0.0f0
            ] for i in 1:n_samples_flat
        ]

        param_sets = [
            [
                P_I[i],
                P_C[i],
                c_ex[i],
                As_samples[i],
                Vs_samples[i],
                Vfree_samples[i],
                Vps_samples[i],
                K_s[i],
                lambda_C[i],
                gamma_samples[i],
                omega_samples[i]
            ] for i in 1:n_samples_flat
        ]

        # Stack into matrices (n_states × n_samples) and upload once
        u0_matrix = Float32.(hcat([collect(u) for u in u0_sets]...))
        p_matrix  = Float32.(hcat([collect(p) for p in param_sets]...))

        # Wrapper to create problem for each sample
        function prob_func(prob, i, repeat)

            u0 = u0_matrix[:, i]#u0_sets[i]
            p_i = p_matrix[:, i]#param_sets[i]
            
            return remake(prob, u0=u0, p=p_i)
        end
        
        # Initial problem (dummy, will be remade by prob_func)
        u0_dummy = @SVector [0.0f0, 1.0f0, 0.0f0, 0.0f0]
        p_dummy = @SVector [
            0.0f0, 0.0f0, 0.0f0,
            1.0f0, 1.0f0, 1.0f0, 0.0f0, 0.0f0, 1.0f0,
            0.0f0, 1.0f0
        ]
        
        prob = ODEProblem{true}(ode_system_A!, u0_dummy, times[end], p_dummy)
        monteprob = EnsembleProblem(prob, prob_func=prob_func, safetycopy=false)
        
        # Solve ensemble on GPU
        sols = solve(
            monteprob, 
            Tsit5(),
            # DiffEqGPU.EnsembleGPUKernel(CUDA.CUDABackend()),
            DiffEqGPU.EnsembleGPUArray(CUDA.CUDABackend()),
            trajectories=n_samples_flat,
            # u0=u0_matrix,   # GPU matrix, bypasses prob_func
            # p=p_matrix,    # GPU matrix, bypasses prob_func
            adaptive=true,
            dt=0.001f0,
            saveat = times
            # save_on=false
        )

        # Extract germination info from solutions
        germinated = zeros(Bool, T, n_samples_flat)
        @inbounds for i in 1:n_samples_flat
            sol = sols[i]
            for j in eachindex(sol.t)
                # println("Sol time $(sol.t[j]), input time $(times[j])")
                germinated[j, i] = sol.u[j][4] > 0
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

        elseif model_alias == "feedback_inhibitor_inducer_perm"
            
            germination = integrate_ode(param_arr, times, 3; n_samples=64)
            
        end
    end

end