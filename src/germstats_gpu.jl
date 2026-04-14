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
    @inline function integrand_point(coords, base::Int32, d::Int32, params, pbase::Int32)
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
        return val * (1f0 + 0.001f0 * s)  # example usage
    end

    """
    Kernel: write per-(point,param) contribution to outbuf
    """
    function batch_kernel(coords_chunk, weights_chunk, d::Int32,
                        params_d, param_dim::Int32, outbuf, P::Int32)
        tid = (blockIdx().x-1) * blockDim().x + threadIdx().x
        Ntot = Int32(length(weights_chunk)) * P
        if tid <= Ntot
            pi = Int32((tid - 1) ÷ P)        # 0-based point index in chunk
            param_idx = Int32((tid - 1) % P) # 0-based param index
            base = Int32(pi * d)
            pbase = Int32(param_idx * param_dim)
            val = integrand_point(coords_chunk, base, d, params_d, pbase)
            w = weights_chunk[pi + 1]
            outbuf[tid] = val * w
        end
        return
    end

    """
    Host orchestration: stream points in chunks
    """
    function integrate_batched(n::Int, d::Int, params::Array{Float32,2}; chunk_size::Int=4096)
        coords_cpu, weights_cpu, N = build_tensor_grid(n, d)
        P = size(params, 1); param_dim = size(params, 2)
        @assert param_dim == 7
        params_flat = vec(params')                    # row-major flatten (M × param_dim)
        params_d = CuArray(params_flat)
        accum_d = CUDA.zeros(Float32, P)              # final accumulators

        i = 1
        while i <= N
            len = min(chunk_size, N - i + 1)
            coords_chunk = CuArray(@view coords_cpu[(i-1)*d + 1 : (i-1)*d + len*d])
            weights_chunk = CuArray(@view weights_cpu[i : i+len-1])
            P32 = Int32(P)
            total = len * P
            outbuf = CUDA.zeros(Float32, total)       # per-(point,param) contributions
            threads = 256
            blocks = cld(total, threads)
            @cuda threads=threads blocks=blocks batch_kernel(coords_chunk, weights_chunk, Int32(d),
                                                        params_d, Int32(param_dim), outbuf, P32)
            CUDA.synchronize()
            
            # COPY OUTBUF TO HOST AND REDUCE PER-PARAM (fast for moderate chunk_size)
            host_buf = Array(outbuf)               # length = len * P
            host_mat = reshape(host_buf, (P, len)) # P rows, len columns
            # sum across columns -> partial sums per parameter
            partials = sum(host_mat; dims=2)       # returns P×1 array
            accum = Array(accum_d)                 # copy current accum to host
            accum .+= vec(partials)                # update
            accum_d .= CuArray(accum)              # copy back to device accumulators

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
    Two-factor germination criterion.
    inputs:
        cdf_X - cumulative distribution function for inhibitor criterion
        cdf_X - cumulative distribution function for inducer criterion
        f_R - probability density function for spore radius R
        f_H - probability density funciton for inner cell wall thickness H
    """
    @inline function two_factor_germ(rho_s, V_s, A_s, P_I, f_R, f_H)
        phi = rho_s * V_s
        tau = V_s / (P_I * A_s)
        beta = calc_beta(t, phi, tau)
        cdf_X = cdf(dist_X, beta)
        cdf_Y = cdf(dist_Y, s)
    end

end