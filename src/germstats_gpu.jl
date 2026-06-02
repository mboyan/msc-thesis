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
    using Adapt
    using Distributions
    using StaticArrays
    using QuasiMonteCarlo

    export compute_germination


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

    """
    Utility function for unpacking the parameters
    commonly used in all integrand functions.
    inputs:
        params (Array{Float32}) - flattened parameter array
        pbase (Float32) - starting index for this parameter set
    outputs:
        [rho_s, mu_R, sigma_R, mu_H, sigma_H, mu_X, sigma_X, mu_Y, sigma_Y, c_ex, P_I, P_C, K_s]
    """
    @inline function unpack_standard_parameters(params, pbase)
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
        return rho_s, mu_R, sigma_R, mu_H, sigma_H, mu_X, sigma_X, mu_Y, sigma_Y, c_ex, P_I, P_C, K_s
    end

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
        rho_s, mu_R, sigma_R, mu_H, sigma_H, mu_X, sigma_X, mu_Y, sigma_Y,
        c_ex, P_I, P_C, K_s = unpack_standard_parameters(params, pbase)

        # Time-dependent signals
        beta, s = calc_signals(u, v, t, rho_s, c_ex,
                                mu_R, sigma_R, mu_H, sigma_H,
                                K_s, P_I, P_C)

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
    Integrand for inducer-dependent inhibition threshold using Gauss-Hermite.
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
        thresh_mode (Int32) - inhibitor-triggered (0) or 2-factor-triggered (1) germination
    """
    @inline function integrand_point_B(coords, base::Int32, d::Int32, params,
                                            pbase::Int32, t::Float32, thresh_mode::Int32)
        
        
        # Extract standard normal nodes
        u = coords[base + 1]          # Standard normal for r_s
        v = coords[base + 2]          # Standard normal for d_ps
        
        # Extract transformation parameters
        rho_s, mu_R, sigma_R, mu_H, sigma_H, mu_X, sigma_X, mu_Y, sigma_Y,
        c_ex, P_I, P_C, K_s = unpack_standard_parameters(params, pbase)
        k_gamma = params[pbase + 18]

        # Time-dependent signals
        beta, s = calc_signals(u, v, t, rho_s, c_ex,
                                mu_R, sigma_R, mu_H, sigma_H,
                                K_s, P_I, P_C)

        # Distributions
        dist_X = Normal(mu_X, sigma_X)
        dist_Y = Normal(mu_Y, sigma_Y)
        
        # CDFs
        cdf_X = cdf(dist_X, beta - k_gamma * s)
        if thresh_mode == 1
            cdf_Y = cdf(dist_Y, s)
        else
            cdf_Y = 1f0
        end
        
        # Return integrand (NO explicit f_R, f_H—they're in the Hermite weights!)
        val = (1f0 - cdf_X) * cdf_Y
        
        return val
    end

    """
    Integrand for inhibitor-dependent induction signal using Gauss-Hermite.
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
        thresh_mode (Int32) - inhibitor-triggered (0) or 2-factor-triggered (1) germination
    """
    @inline function integrand_point_C(coords, base::Int32, d::Int32, params,
                                            pbase::Int32, t::Float32, thresh_mode::Int32)
        
        # Extract standard normal nodes
        u = coords[base + 1]          # Standard normal for r_s
        v = coords[base + 2]          # Standard normal for d_ps
        w = coords[base + 3]          # Standard normal for c_0
        
        # Extract transformation parameters
        rho_s, mu_R, sigma_R, mu_H, sigma_H, mu_X, sigma_X, mu_Y, sigma_Y,
        c_ex, P_I, P_C, K_s = unpack_standard_parameters(params, pbase)
        k_gamma = params[pbase + 18]
        K_I = params[pbase + 20]
        n = params[pbase + 21]

        # Sample initial inhibitor concentration
        c_0 = sample_c0(w, mu_R, sigma_R)

        # Time-dependent signals
        beta, s = calc_signals(u, v, t, rho_s, c_ex,
                                mu_R, sigma_R, mu_H, sigma_H,
                                K_s, P_I, P_C)
        s_mod = s / (1 + (beta * c_0 / K_I)^n)

        # Distributions
        dist_X = Normal(mu_X, sigma_X)
        dist_Y = Normal(mu_Y, sigma_Y)
        
        # CDFs
        if thresh_mode == 1                         # Normal inhibition threshold
            cdf_X = cdf(dist_X, beta)
        elseif thresh_mode == 2 || thresh_mode == 3 # Shifted inhibition threshold
            cdf_X = cdf(dist_X, beta - k_gamma * s_mod)
        else                                        # No inhibition threshold
            cdf_X = 0f0
        end
        if thresh_mode ==2 || thresh_mode == 3      # No induction threshold
            cdf_Y = 1f0
        else                                        # Normal induction threshold
            cdf_Y = cdf(dist_Y, s_mod)
        end
        
        # Return integrand (NO explicit f_R, f_H—they're in the Hermite weights!)
        val = (1f0 - cdf_X) * cdf_Y
        
        return val
    end

    """
    Integrand for inhibitor-dependent induction threshold using Gauss-Hermite.
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
        thresh_mode (Int32) - inhibitor-triggered (0) or 2-factor-triggered (1) germination
    """
    @inline function integrand_point_E(coords, base::Int32, d::Int32, params,
                                            pbase::Int32, t::Float32, thresh_mode::Int32)
        
        # Extract standard normal nodes
        u = coords[base + 1]          # Standard normal for r_s
        v = coords[base + 2]          # Standard normal for d_ps
        w = coords[base + 3]          # Standard normal for c_0
        
        # Extract transformation parameters
        rho_s, mu_R, sigma_R, mu_H, sigma_H, mu_X, sigma_X, mu_Y, sigma_Y,
        c_ex, P_I, P_C, K_s = unpack_standard_parameters(params, pbase)
        k_gamma = params[pbase + 18]
        k_omega = params[pbase + 19]
        K_b = params[pbase + 22]

        # Sample initial inhibitor concentration
        c_0 = sample_c0(w, mu_R, sigma_R)

        # Time-dependent signals
        beta, s = calc_signals(u, v, t, rho_s, c_ex,
                                mu_R, sigma_R, mu_H, sigma_H,
                                K_s, P_I, P_C)
        c_in_I = beta * c_0
        b = c_in_I / (K_b + c_in_I)

        # Distributions
        dist_X = Normal(mu_X, sigma_X)
        dist_Y = Normal(mu_Y, sigma_Y)
        
        # CDFs
        if thresh_mode == 1                         # Normal inhibition threshold
            cdf_X = cdf(dist_X, beta)
        elseif thresh_mode == 2                     # Shifted inhibition threshold
            cdf_X = cdf(dist_X, beta - k_gamma * s)
        else
            cdf_X = 0f0                             # No inhibition threshold
        end
        cdf_Y = cdf(dist_Y, s - k_omega * b)
        
        # Return integrand (NO explicit f_R, f_H—they're in the Hermite weights!)
        val = (1f0 - cdf_X) * cdf_Y
        
        return val
    end

    """
    Kernel: write per-(point,param) contribution to outbuf.
    inputs:
        coords_chunk (CuArray{Float32}) - coordinates of Gauss-Legendre nodes for the current data chunk
        weights_chunk (CuArray{Float32}) - Gauss-Legendre weights for the current data chunk
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
            elseif model_idx == 4
                val = integrand_point_B(coords_chunk, base, d, params_d, pbase, t, Int32(0))
            elseif model_idx == 5
                val = integrand_point_B(coords_chunk, base, d, params_d, pbase, t, Int32(1))
            elseif model_idx == 6
                val = integrand_point_C(coords_chunk, base, d, params_d, pbase, t, Int32(0))
            elseif model_idx == 7
                val = integrand_point_C(coords_chunk, base, d, params_d, pbase, t, Int32(1))
            elseif model_idx == 10
                val = integrand_point_E(coords_chunk, base, d, params_d, pbase, t, Int32(0))
            elseif model_idx == 11
                val = integrand_point_E(coords_chunk, base, d, params_d, pbase, t, Int32(1))
            elseif model_idx == 22
                val = integrand_point_C(coords_chunk, base, d, params_d, pbase, t, Int32(2))
            elseif model_idx == 23
                val = integrand_point_C(coords_chunk, base, d, params_d, pbase, t, Int32(3))
            elseif model_idx == 26
                val = integrand_point_E(coords_chunk, base, d, params_d, pbase, t, Int32(2))
            else
                val = 0.0f0
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
            accum .+= partials'                                 # update

            i += len
        end

        return accum # P × T:  accum[p, t] = integral for param set p at time t
    end

    """
    Kernel for testing threshold crossings.
    inputs:
        germinated (CuArray{Bool}) - output array to mark if germination occurred
        thresh_mode (Int32) - threshold mode
        params_d (CuArray{Float32}) - GPU array of the parameter values
        sols (CuArray{Float32}) - ODE solutions for all samples and time points
        P (Int32) - number of parameter samples
        T (Int32) - number of time steps
    """
    function eval_thresholds_gpu(germinated, thresh_mode, params, sols, P, T)
        tid = (blockIdx().x-1) * blockDim().x + threadIdx().x
        Ntot = P * T

        if tid <= Ntot
            # Decode flat index: tid-1 = time_idx + T*param_idx
            tmp = tid - 1
            time_idx = Int32(tmp % T) + 1
            param_idx = Int32(tmp ÷ T) + 1

            # Unpack concentrations
            c_in_I = sols[time_idx, param_idx][1]
            c_in_C = sols[time_idx, param_idx][3]

            #1 P_I[i],
            #2 P_C[i],
            #3 c_ex[i],
            #4 K_s[i],
            #5 K_b[i],
            #6 K_I[i],
            #7 n[i],
            #8 lambda_I[i],
            #9 lambda_C[i],
            #10 k_gamma[i],
            #11 k_omega[i],
            #12 A_s,
            #13 V_s,
            #14 V_free,
            #15 V_ps,
            #16 gamma[j],
            #17 omega[j],
            #18 c0[j]

            # Unpack parameters for this sample
            params_sample = @view params[param_idx, :]
            K_s = params_sample[4]
            K_b = params_sample[5]
            k_gamma = params_sample[10]
            k_omega = params_sample[11]
            gamma = params_sample[16]
            omega = params_sample[17]
            c0 = params_sample[18]

            # Compute signals
            s = c_in_C / (K_s + c_in_C)
            b = c_in_I / (K_b + c_in_I)

            # Germination logic
            condition = false
            if thresh_mode == 0
                # Inhibitor-dependent germination triggering
                condition = c_in_I < gamma * c0
            elseif thresh_mode == 1
                # Inducer-dependent germination triggering
                condition = s > omega
            elseif thresh_mode == 2
                # 2-factor germination triggering
                condition = (c_in_I < gamma * c0) && (s > omega)
            elseif thresh_mode == 3
                # Shifted inhibitor-dependent germination triggering
                condition = c_in_I < (gamma + k_gamma * s) * c0
            elseif thresh_mode == 4
                # Shifted inducer-dependent germination triggering
                condition = s > omega + k_omega * b
            elseif thresh_mode == 5
                # 2-factor germination triggering with shifted gamma
                condition = (c_in_I < (gamma + k_gamma * s) * c0) && (s > omega)
            elseif thresh_mode == 6
                # 2-factor germination triggering with shifted omega
                condition = (c_in_I < gamma * c0) && (s > omega + k_omega * b)
            end

            # Check if germination occurs for this sample at this time point
            if condition
                germinated[time_idx, param_idx] = true
            end
        end

        return
    end

    """
    System of coupled ODEs for the feedback model
    of inducer-dependent cell wall permeability.
    """
    function ode_system_A(u, p, t)
        c_in_I, c_out_I, c_in_C, germ = u
    
        # Unpack parameters for this specific spore
        P_I, P_C, c_ex, K_s, K_b, K_I, n, lambda_I, lambda_C, k_gamma, k_omega,
        A_s, V_s, V_free, V_ps, gamma, omega, c0 = p

        # Compute inducing signal s from c_in_C
        s = c_in_C / (K_s + c_in_C + 1f-10)
        s = min(max(s, 0.0f0), 1.0f0)

        exponent = -(1 + lambda_C * s) * 0.001f0

        # Limit permeability to Pmax = 1000 μm/s
        PmaxA = 1000.0f0 * A_s
        # rateI = PmaxA * (1.0f0 - exp(exponent * P_I))
        # rateC = PmaxA * (1.0f0 - exp(exponent * P_C))
        rateI = PmaxA * (-expm1(exponent * P_I))
        rateC = PmaxA * (-expm1(exponent * P_C))
        # expr1 = 1000.0f0 * (6.9077553f0 + log(A_s))
        # expr2 = (1 + lambda_C * s)
        # rateI = max(PmaxA - exp(0.001f0 * expr1 - P_I * expr2), 0.0f0)
        # rateC = max(PmaxA - exp(0.001f0 * expr1 - P_C * expr2), 0.0f0)
        
        # Update permeability constants based on signal
        # P_I_pert = 1000.0f0 * (1.0f0 - exp_factor) + P_I * exp_factor
        # P_C_pert = 1000.0f0 * (1.0f0 - exp_factor) + P_C * exp_factor

        # Concentration differences
        diffI = c_in_I - c_out_I
        diffC = c_in_C - c_ex
        
        # ODE equations
        # du1 = -(P_I_pert * A_s / V_s) * diffI
        # du2 = (P_I_pert * A_s / V_free) * diffI
        # du3 = -(P_C_pert * A_s / V_ps) * diffC
        du1 = -(rateI / V_s) * diffI
        du2 = (rateI / V_free) * diffI
        du3 = -(rateC / V_ps) * diffC

        du4 = 0.0f0

        return SVector{4}(du1, du2, du3, du4)
    end

    """
    System of coupled ODEs for the feedback model
    of inhibitor-dependent cell wall permeability.
    """
    function ode_system_D(u, p, t)
        c_in_I, c_out_I, c_in_C, germ = u
    
        # Unpack parameters for this specific spore
        P_I, P_C, c_ex, K_s, K_b, K_I, n, lambda_I, lambda_C, k_gamma, k_omega,
        A_s, V_s, V_free, V_ps, gamma, omega, c0 = p

        # Compute inducing signal s from c_in_C
        s = c_in_C / (K_s + c_in_C + 1f-10)
        s = min(max(s, 0.0f0), 1.0f0)

        # exponent = -(1 + lambda_I * c_in_I) * 0.001f0
        exp_factor = exp(-lambda_I * c_in_I)
        
        # Update permeability constants based on signal
        P_I_pert = P_I * exp_factor
        P_C_pert = P_C * exp_factor

        rateI = P_I_pert * A_s
        rateC = P_C_pert * A_s

        # Concentration differences
        diffI = c_in_I - c_out_I
        diffC = c_in_C - c_ex
        
        # ODE equations
        du1 = -(rateI / V_s) * diffI
        du2 = (rateI / V_free) * diffI
        du3 = -(rateC / V_ps) * diffC

        # GPU-friendly germination logic (no control flow)
        if thresh_mode == 1.0f0
            # Inducer-dependent germination triggering
            thresholds_met = Float32(s > omega)
        elseif thresh_mode == 2.0f0
            # 2-factor germination triggering
            thresholds_met = Float32((c_in_I < gamma * c0) && (s > omega))
        elseif thresh_mode == 3.0f0
            # Shifted inhibitor-dependent germination triggering
            thresholds_met = Float32(c_in_I < (gamma + k_gamma * s) * c0)
        else
            # 2-factor germination triggering with shifted gamma
            thresholds_met = Float32((c_in_I < (gamma + k_gamma * s) * c0) && (s > omega))
        end
        germ_not_full = Float32(germ < 1.0f0)
        du4 = thresholds_met * germ_not_full

        return SVector{4}(du1, du2, du3, du4)
    end

    """
    System of coupled ODEs for the feedback model
    of inducer-dependent cell wall permeability
    and inhibitor-suppressed inducing signal.
    """
    function ode_system_AC(u, p, t)
        c_in_I, c_out_I, c_in_C, germ = u
    
        # Unpack parameters for this specific spore
        thresh_mode, P_I, P_C, c_ex, K_s, K_b, K_I, n, lambda_I, lambda_C, k_gamma, k_omega, A_s, V_s, V_free, V_ps, gamma, omega, c0 = p

        # Compute inducing signal s from c_in_C
        s = c_in_C / (K_s + c_in_C + 1f-10)
        s = min(max(s, 0.0f0), 1.0f0)
        s = s / (1 + (c_in_I / K_I)^n)

        exp_factor = exp(-lambda_C * s)
        
        # Update permeability constants based on signal
        P_I_pert = 1000.0f0 * (1.0f0 - exp_factor) + P_I * exp_factor
        P_C_pert = 1000.0f0 * (1.0f0 - exp_factor) + P_C * exp_factor
        
        # ODE equations
        du1 = -(P_I_pert * A_s / V_s) * (c_in_I - c_out_I)
        du2 = (P_I_pert * A_s / V_free) * (c_in_I - c_out_I)
        du3 = -(P_C_pert * A_s / V_ps) * (c_in_C - c_ex)

        # GPU-friendly germination logic (no control flow)
        if thresh_mode == 0.0f0
            # Inhibitor-dependent germination triggering
            thresholds_met = Float32(c_in_I < gamma)
        elseif thresh_mode == 1.0f0
            # 2-factor germination triggering
            thresholds_met = Float32(s > omega)
        else
            # 2-factor germination triggering with shifted gamma
            thresholds_met = Float32((c_in_I < gamma * c0) && (s > omega))
        end
        germ_not_full = Float32(germ < 1.0f0)
        du4 = thresholds_met * germ_not_full

        return SVector{4}(du1, du2, du3, du4)
    end

    """
    System of coupled ODEs for the feedback model
    of inducer and inhibitor-dependent cell wall permeability.
    """
    function ode_system_AD(u, p, t)
        c_in_I, c_out_I, c_in_C, germ = u
    
        # Unpack parameters for this specific spore
        thresh_mode, P_I, P_C, c_ex, K_s, K_b, K_I, n, lambda_I, lambda_C, k_gamma, k_omega, A_s, V_s, V_free, V_ps, gamma, omega, c0 = p

        # Compute inducing signal s from c_in_C
        s = c_in_C / (K_s + c_in_C + 1f-10)
        s = min(max(s, 0.0f0), 1.0f0)

        exp_factor_I = exp(-lambda_I * s)
        exp_factor_C = exp(-lambda_C * c_in_I)
        
        # Update permeability constants based on signal
        P_I_pert = (1000.0f0 * (1.0f0 - exp_factor_C) + P_I * exp_factor_C) * exp_factor_I
        P_C_pert = (1000.0f0 * (1.0f0 - exp_factor_C) + P_C * exp_factor_C) * exp_factor_I
        
        # ODE equations
        du1 = -(P_I_pert * A_s / V_s) * (c_in_I - c_out_I)
        du2 = (P_I_pert * A_s / V_free) * (c_in_I - c_out_I)
        du3 = -(P_C_pert * A_s / V_ps) * (c_in_C - c_ex)

        # GPU-friendly germination logic (no control flow)
        if thresh_mode == 0.0f0
            # Inducer-dependent germination triggering
            thresholds_met = Float32(c_in_I < gamma * c0)
        elseif thresh_mode == 1.0f0
            # 2-factor germination triggering
            thresholds_met = Float32(s > omega)
        else
            # 2-factor germination triggering with shifted gamma
            thresholds_met = Float32((c_in_I < gamma * c0) && (s > omega))
        end
        germ_not_full = Float32(germ < 1.0f0)
        du4 = thresholds_met * germ_not_full

        return SVector{4}(du1, du2, du3, du4)
    end

    """
    System of coupled ODEs for the feedback model
    of inhibitor-dependent cell wall permeability
    and inhibitor-suppressed inducing signal.
    """
    function ode_system_CD(u, p, t)
        c_in_I, c_out_I, c_in_C, germ = u
    
        # Unpack parameters for this specific spore
        thresh_mode, P_I, P_C, c_ex, K_s, K_b, K_I, n, lambda_I, lambda_C, k_gamma, k_omega, A_s, V_s, V_free, V_ps, gamma, omega, c0 = p

        # Compute inducing signal s from c_in_C
        s = c_in_C / (K_s + c_in_C + 1f-10)
        s = min(max(s, 0.0f0), 1.0f0)
        s = s / (1 + (c_in_I / K_I)^n)

        exp_factor = exp(-lambda_I * c_in_I)
        
        # Update permeability constants based on signal
        P_I_pert = P_I * exp_factor
        P_C_pert = P_C * exp_factor
        
        # ODE equations
        du1 = -(P_I_pert * A_s / V_s) * (c_in_I - c_out_I)
        du2 = (P_I_pert * A_s / V_free) * (c_in_I - c_out_I)
        du3 = -(P_C_pert * A_s / V_ps) * (c_in_C - c_ex)

        # GPU-friendly germination logic (no control flow)
        if thresh_mode == 1.0f0
            # Inducer-dependent germination triggering
            thresholds_met = Float32(s > omega)
        elseif thresh_mode == 2.0f0
            # 2-factor germination triggering
            thresholds_met = Float32((c_in_I < gamma * c0) && (s > omega))
        elseif thresh_mode == 3.0f0
            # Shifted inhibitor-dependent germination triggering
            thresholds_met = Float32(c_in_I < (gamma + k_gamma * s) * c0)
        else
            # 2-factor germination triggering with shifted gamma
            thresholds_met = Float32((c_in_I < (gamma + k_gamma * s) * c0) && (s > omega))
        end
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
    function integrate_ode(params::Array{Float32,2}, times::Array{Float32,1}, model_idx::Int, ode_system; n_samples::Int=1024)
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
        lambda_I = params[:, 16]
        lambda_C = params[:, 17]
        k_gamma = params[:, 18]
        k_omega = params[:, 19]
        K_I = params[:, 20]
        n = params[:, 21]
        K_b = params[:, 22]

        # Determine threshold mode
        if model_idx in [2, 14, 17]
            thresh_mode = 0 # Inhibitor threshold
        elseif model_idx in [8, 15, 18, 27]
            thresh_mode = 1 # Inducer threshold
        elseif model_idx in [3, 9, 16, 19, 28]
            thresh_mode = 2 # Both thresholds
        elseif model_idx in [12, 24] # Shifted inhibitor threshold
            thresh_mode = 3
        elseif model_idx in [20] # Shifted inducer threshold
            thresh_mode = 4
        elseif model_idx in [13, 25] # Both thresholds + shifted inhibitor threshold
            thresh_mode = 5
        elseif model_idx in [21] # Both thresholds + shifted inducer threshold
            thresh_mode = 6
        end

        println(thresh_mode)

        # Generate Sobol sample
        sobol_samples = QuasiMonteCarlo.sample(n_samples, sobol_dim, SobolSample())

        # Construct flat parameter collections and initial conditions
        n_samples_flat = P * n_samples
        u0_vec = Vector{SVector{4, Float32}}()
        p_vec = Vector{SVector{18, Float32}}()
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

            # Generate initial inhibitor concentration samples
            c0 = quantile.(LogNormal(mu_O_log, sigma_O_log), sobol_samples[3, :])

            # Generate threshold samples
            gamma = quantile.(Normal(mu_X[i], sigma_X[i]), sobol_samples[4, :])
            omega = quantile.(Normal(mu_Y[i], sigma_Y[i]), sobol_samples[5, :])

            @inbounds for j in 1:n_samples
                
                V_s = 4pi/3 * r[j] ^ 3
                A_s = 4pi * r[j] ^ 2
                V_ps = calc_ps_vacant_vol(r[j], d_ps[j])
                V_free = 1 / rho_s[i] - V_s

                push!(u0_vec, @SVector [Float32(c0[j]), 0.0f0, 0.0f0, 0.0f0])
                push!(p_vec, @SVector [
                    P_I[i],
                    P_C[i],
                    c_ex[i],
                    K_s[i],
                    K_b[i],
                    K_I[i],
                    n[i],
                    lambda_I[i],
                    lambda_C[i],
                    k_gamma[i],
                    k_omega[i],
                    Float32(A_s),
                    Float32(V_s),
                    Float32(V_free),
                    Float32(V_ps),
                    Float32(gamma[j]),
                    Float32(omega[j]),
                    Float32(c0[j])
                ])
            end
        end

        # Wrapper to create problem for each sample
        function prob_func(prob, i, repeat)

            u0 = u0_vec[i]
            p_i = p_vec[i]
            
            return remake(prob, u0=u0, p=p_i)
        end
        
        # Initial problem (dummy, will be remade by prob_func)
        u0_dummy = @SVector [0.0f0, 1.0f0, 0.0f0, 0.0f0]
        p_dummy = @SVector [
            1.0f0, 1.0f0, 1.0f0,
            1.0f0, 1.0f0, 1.0f0, 1.0f0, 1.0f0, 1.0f0,
            1.0f0, 1.0f0, 1.0f0, 1.0f0, 1.0f0, 1.0f0,
            1.0f0, 1.0f0, 1.0f0
        ]
        
        prob = ODEProblem{false}(ode_system, u0_dummy, times[end], p_dummy)
        monteprob = EnsembleProblem(prob, prob_func=prob_func, safetycopy=false)

        dt = Float32(min(maximum(diff(times)), 10.0))

        # function threshold_condition(u, t, integrator)
        #     # Simple comparison, no complex operations
        #     return u[1] > 5.0f0
        # end

        # function threshold_affect!(integrator)
        #     # Simple state modification only
        #     integrator.u = integrator.u .* 0.5f0
        # end

        # Building different problems for different parameters
        batch = 1:n_samples_flat
        probs = map(batch) do i
            DiffEqGPU.make_prob_compatible(remake(prob, u0=u0_vec[i], p=p_vec[i]))
        end
        gpu_probs = adapt(CUDA.CUDABackend(), probs)

        sols_gpu = DiffEqGPU.vectorized_solve(
            gpu_probs,
            prob,
            GPURosenbrock23(),
            dt=dt,
            saveat=times
        )

        # Extract germination info from solutions
        germinated = zeros(Bool, T, n_samples_flat)

        # Concatenate p_vec into a matrix for GPU access
        params_gpu = CuArray(stack(p_vec)')
        
        germinated_gpu = CuArray(germinated)
        n_threads = 256
        n_blocks = cld(n_samples_flat * T, n_threads)
        @cuda threads=n_threads blocks=n_blocks eval_thresholds_gpu(germinated_gpu, thresh_mode, params_gpu, sols_gpu[2], n_samples_flat, T)
        germinated = Array(germinated_gpu)

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

    """
    Transform the R and H means and sd's
    from standard Normal to LogNormal
    and sample using Gauss-Hermite nodes,
    then compute spore and vacant cell wall volume
    and spore surface area.
    """
    @inline function calc_geom_variables(u, v, mu_R, sigma_R, mu_H, sigma_H)

        mu_R_log = log(mu_R^2 / sqrt(sigma_R^2 + mu_R^2))
        sigma_R_log = sqrt(log(sigma_R^2 / mu_R^2 + 1))
        mu_H_log = log(mu_H^2 / sqrt(sigma_H^2 + mu_H^2))
        sigma_H_log = sqrt(log(sigma_H^2 / mu_H^2 + 1))

        # Transform from LogNormal to physical space
        r_s = exp(mu_R_log + sigma_R_log * u)
        d_ps = exp(mu_H_log + sigma_H_log * v)

        # Compute geometric variables
        V_s, A_s = calc_spore_geom(r_s)
        V_ps = calc_ps_vacant_vol(r_s, d_ps)

        return V_s, A_s, V_ps
    end

    """
    Compute secondary variables relevant
    for germination.
    """
    @inline function calc_secondary_variables(rho_s, V_s, V_ps, A_s, P_I, P_C)
        phi = rho_s * V_s
        tau_I = V_s / (P_I * A_s)
        tau_C = V_ps / (P_C * A_s)
        return phi, tau_I, tau_C
    end

    """
    Compute time-dependent
    inhibitory and inductory signals.
    """
    @inline function calc_signals(u, v, t, rho_s, c_ex, mu_R, sigma_R, mu_H, sigma_H, K_s, P_I, P_C)
        V_s, A_s, V_ps = calc_geom_variables(u, v, mu_R, sigma_R, mu_H, sigma_H)
        phi, tau_I, tau_C = calc_secondary_variables(rho_s, V_s, V_ps, A_s, P_I, P_C)
        beta = calc_beta(t, phi, tau_I)
        s = calc_signal(t, c_ex, K_s, tau_C)
        return beta, s
    end

    """
    Sample the initial inhibitor concentration values
    using Gauss-Hermite nodes.
    """
    @inline function sample_c0(w, mu_R, sigma_R)
        mu_O_log = log(mu_R^2 / sqrt(sigma_R^2 + mu_R^2))
        sigma_O_log = sqrt(log(sigma_R^2 / mu_R^2 + 1))
        return exp(mu_O_log + sigma_O_log * w)
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
        param_arr = Array{Float32}(undef, sample_size, 22)
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
        :k_gamma in param_keys ? param_arr[:, 18] .= Float32.(param_dict[:k_gamma]) : nothing
        :k_omega in param_keys ? param_arr[:, 19] .= Float32.(param_dict[:k_omega]) : nothing
        :K_I in param_keys ? param_arr[:, 20] .= Float32.(param_dict[:K_I]) : nothing
        :n in param_keys ? param_arr[:, 21] .= Float32.(param_dict[:n]) : nothing
        :K_b in param_keys ? param_arr[:, 22] .= Float32.(param_dict[:K_b]) : nothing

        if model_alias == "independent" # 0
            germination = integrate_batched(8, 2, param_arr, times, 1)
        elseif model_alias == "feedback_inhibitor_inducer_perm" # Ai
            germination = integrate_ode(param_arr, times, 2, ode_system_A)
        elseif model_alias == "feedback_combined_inducer_perm" # A
            germination = integrate_ode(param_arr, times, 3, ode_system_A)
        elseif model_alias == "inhibitor_thresh" # Bi
            germination = integrate_batched(8, 2, param_arr, times, 4)
        elseif model_alias == "combined_inhibitor_thresh" # B
            germination = integrate_batched(8, 2, param_arr, times, 5)
        elseif model_alias == "inducer_signal" # Cc
            germination = integrate_batched(8, 3, param_arr, times, 6)
        elseif model_alias == "combined_inducer_signal" # C
            germination = integrate_batched(8, 3, param_arr, times, 7)
        elseif model_alias == "feedback_inducer_inhibitor_perm" # Dc
            germination = integrate_ode(param_arr, times, 8, ode_system_D)
        elseif model_alias == "feedback_combined_inhibitor_perm" # D
            germination = integrate_ode(param_arr, times, 9, ode_system_D)
        elseif model_alias == "inducer_thresh" # Ec
            germination = integrate_batched(8, 3, param_arr, times, 10)
        elseif model_alias == "combined_inducer_thresh" # E
            germination = integrate_batched(8, 3, param_arr, times, 11)
        elseif model_alias == "feedback_inhibitor_inducer_perm_thresh" # ABi
            germination = integrate_ode(param_arr, times, 12, ode_system_A) 
        elseif model_alias == "feedback_combined_inducer_perm_thresh" # AB
            germination = integrate_ode(param_arr, times, 13, ode_system_A)
        elseif model_alias == "feedback_inhibitor_inducer_perm_inhibitor_signal" # ACi
            germination = integrate_ode(param_arr, times, 14, ode_system_AC)
        elseif model_alias == "feedback_inducer_inducer_perm_inhibitor_signal" # ACc !!!!!
            germination = integrate_ode(param_arr, times, 15, ode_system_AC)
        elseif model_alias == "feedback_combined_inducer_perm_inhibitor_signal" # AC !!!!!
            germination = integrate_ode(param_arr, times, 16, ode_system_AC)
        elseif model_alias == "feedback_inhibitor_inhibitor_inducer_perm" # ADi !!!!!!!!!!!!
            germination = integrate_ode(param_arr, times, 17, ode_system_AD)
        elseif model_alias == "feedback_inducer_inhibitor_inducer_perm" # ADc !!!!!!!!!!!!
            germination = integrate_ode(param_arr, times, 18, ode_system_AD)
        elseif model_alias == "feedback_combined_inhibitor_inducer_perm" # AD !!!!!!!!!!!!
            germination = integrate_ode(param_arr, times, 19, ode_system_AD)
        elseif model_alias == "feedback_inducer_inhibitor_thresh_inducer_perm" # AEc
            germination = integrate_ode(param_arr, times, 20, ode_system_A)
        elseif model_alias == "feedback_combined_inhibitor_thresh_inducer_perm" # AE
            germination = integrate_ode(param_arr, times, 21, ode_system_A)
        elseif model_alias == "inhibitor_thresh_inducer_signal" # BCi
            germination = integrate_batched(8, 3, param_arr, times, 22)
        elseif model_alias == "combined_inhibitor_thresh_inducer_signal" # BC
            germination = integrate_batched(8, 3, param_arr, times, 23)
        elseif model_alias == "feedback_inhibitor_inducer_thresh_inhibitor_perm" # BDc
            germination = integrate_ode(param_arr, times, 24, ode_system_D)
        elseif model_alias == "feedback_combined_inducer_thresh_inhibitor_perm" # BD
            germination = integrate_ode(param_arr, times, 25, ode_system_D)
        elseif model_alias == "combined_inhibitor_thresh_inducer_thresh" # BE
            germination = integrate_batched(8, 3, param_arr, times, 26)
        elseif model_alias == "feedback_inducer_inhibitor_perm_signal" # CDc
            germination = integrate_ode(param_arr, times, 27, ode_system_CD)
        elseif model_alias == "feedback_combined_inhibitor_perm_signal" # CD ???????
            germination = integrate_ode(param_arr, times, 28, ode_system_CD)
        end
    end

end