module GermStatsGPU
__precompile__(false)
    """
    Contains GPU-accelerated tools for generating germination statistics
    """

    using CUDA
    using FastGaussQuadrature

    """
    Build tensor-product coords & weights
    for Gauss-Legendre quadrature
    """
    function build_tensor_grid(n::Int, d::Int)
        xs, ws = gausslegendre(n)
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
                coords[base + k] = Float32(xs[ik])
                w *= Float32(ws[ik])
            end
            weights[idx+1] = w
        end
        return coords, weights, N
    end

    """
    Example integrand: replace body with your param-dependent function
    """
    @inline function integrand_point(coords, base::Int32, d::Int32, params, pbase::Int32, t::Float32)
        # coords: flattened coords array
        # params: flattened params array (M * param_dim)
        val = 1.0f0
        @inbounds for k in 0:(d-1)
            xi = coords[base + k + 1]
            val *= (1f0 - xi^2)          # example integrand
        end
        # Example param dependence: params[pbase + 1] ... params[pbase+7]
        # incorporate parameters into val as needed. Example multiply by sum(params):
        s = 0f0
        @inbounds for j in 0:6
            s += params[pbase + j + 1]
        end
        return val * (1f0 + 0.001f0 * s) * exp(-t)  # example usage
    end

    """
    Kernel: write per-(point,param) contribution to outbuf
    """
    function batch_kernel(coords_chunk, weights_chunk, d::Int32,
                        params_d, param_dim::Int32, times_d, T::Int32, outbuf, P::Int32)
        tid = (blockIdx().x-1) * blockDim().x + threadIdx().x
        Ntot = Int32(length(weights_chunk)) * P
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

            val = integrand_point(coords_chunk, base, d, params_d, pbase, t)
            w   = weights_chunk[pidx + 1]
            outbuf[tid] = val * w
        end
        return
    end

    """
    Host orchestration: compute integral by
    streaming points in chunks
    inputs:
        n (Int) - number of Gauss-Legendre nodes
        d (Int) - dimension of integral
        params (Array{Float32}) - parameters
        chunk_size (Int) - chunk size for parallel execution
    """
    function integrate_batched(n::Int, d::Int, params::Array{Float32,2}, times::Array{Float32,1}; chunk_size::Int=4096)
        coords_cpu, weights_cpu, N = build_tensor_grid(n, d)
        P = size(params, 1); param_dim = size(params, 2)
        T = length(times)

        @assert param_dim == 7
        params_flat = vec(params')                    # row-major flatten (M × param_dim)
        params_d = CuArray(params_flat)
        times_d = CuArray(times)

        accum_d = CUDA.zeros(Float32, P)              # final accumulators

        i = 1
        while i <= N
            len = min(chunk_size, N - i + 1)
            coords_chunk = CuArray(@view coords_cpu[(i-1)*d + 1 : (i-1)*d + len*d])
            weights_chunk = CuArray(@view weights_cpu[i : i+len-1])
            
            total = len * P
            outbuf = CUDA.zeros(Float32, total)       # per-(point,param) contributions

            threads = 256
            blocks = cld(total, threads)
            @cuda threads=threads blocks=blocks batch_kernel(coords_chunk, weights_chunk, Int32(d),
                                                        params_d, Int32(param_dim), times_d, Int32(T), outbuf, Int32(P))
            CUDA.synchronize()
            
            # COPY OUTBUF TO HOST AND REDUCE PER-PARAM (fast for moderate chunk_size)
            host_buf = Array(outbuf)                            # length = len * P
            host_mat = reshape(host_buf, (T, P, len))           # T × P × len
            # sum across columns -> partial sums per parameter
            partials  = dropdims(sum(host_vol; dims=3), dims=3) # T × P
            accum = Array(accum_d)                              # copy current accum to host
            accum .+= vec(partials)                             # update
            accum_d .= CuArray(accum)                           # copy back to device accumulators

            i += len
        end

        return Array(accum_d)
    end

    # =====================================================================================
    # ================== MODEL-SPECIFIC KERNEL-FRIENDLY FUNCTIONS =========================
    # =====================================================================================

    """
    Compute relative decrease of inhibitor concentration
    inside the spore over time.
    inputs:
        t - time (in seconds)
        phi - volume fraction occupied by spores
        tau - characteristic inhibitor decrease time (in seconds)
    """
    @inline function calc_beta(t, phi, tau)
        return phi + (1 - phi) * exp(- t / (tau * (1 - phi)))
    end

    """
    Compute the inducer concentration
    inside the inner cell wall over time.
    inputs:
        t - time (in seconds)
        c_ex - ambient inducer concentration (in 1e-5 M)
        tau - characteristic inhibitor decrease time (in seconds)
    """
    @inline function calc_inducer_concentration(t, c_ex, tau)
        return c_ex * (1 - exp(-t / tau))
    end

    """
    Compute the concentration-dependent inducing signal
    inside the inner cell wall over time.
    inputs:
        t - time (in seconds)
        c_ex - ambient inducer concentration (in 1e-5 M)
        K_s - half-saturation constant for inducing signal
        tau - characteristic inhibitor decrease time (in seconds)
    """
    @inline function calc_signal(t, c_ex, K_s, tau)
        c_in = calc_inducer_concentration(t, c_ex, tau)
        return c_in / (K_s + c_in)
    end

    """
    Two-factor germination criterion.
    inputs:
        t - time (in seconds)
        rho_s - number density of spore colony (in um^(-1))
        c_ex - ambient inducer concentration (in 1e-5 M)
        V_s - spore volume (in um^3)
        V_ps - vacant inner cell volume (in um^3)
        P_I - inhibitor permeation constant (in um/s)
        P_C - inhibitor permeation constant (in um/s)
        K_s - half-saturation constant for inducing signal
        f_R - probability density function for spore radius R
        f_H - probability density funciton for inner cell wall thickness H
    """
    @inline function two_factor_germ(t, rho_s, c_ex, V_s, A_s, V_ps, P_I, P_C, K_C, f_R, f_H)
        # Secondary variables
        phi = rho_s * V_s
        tau_I = V_s / (P_I * A_s)
        tau_C = V_ps / (P_C * A_s)

        # Time-dependent signals
        beta = calc_beta(t, phi, tau)
        s = calc_signal(t, c_ex, K_s, tau)

        cdf_X = cdf(dist_X, beta)
        cdf_Y = cdf(dist_Y, s)

        f_R = 
        f_H = 
    end

    # =====================================================================================
    # ====================== GERMINATION FRACTION CALCULATION =============================
    # =====================================================================================
    function compute_germination(model_alias; n_times=100)
        asd
    end

end