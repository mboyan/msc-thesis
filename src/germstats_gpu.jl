module GermStatsGPU
__precompile__(false)
    """
    Contains GPU-accelerated tools for generating germination statistics
    """

    using CUDA
    using FastGaussQuadrature

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
                w *= Float32(ws[ik])
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
    Parameters layout in flattened array (param_dim ≥ 11):
    [0:0]   = mu_R
    [1:1]   = sigma_R
    [2:2]   = mu_H
    [3:3]   = sigma_H
    [4:4]   = c_ex
    [5:5]   = P_I
    [6:6]   = P_C
    [7:7]   = K_s
    [8:8]   = rho_s
    [9:9]   = (reserved or additional params)
    [10:10] = (reserved or additional params)
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
        μ_R = params[pbase + 1]
        σ_R = params[pbase + 2]
        μ_H = params[pbase + 3]
        σ_H = params[pbase + 4]
        c_ex = params[pbase + 5]
        P_I = params[pbase + 6]
        P_C = params[pbase + 7]
        K_s = params[pbase + 8]
        rho_s = params[pbase + 9]
        
        # Transform from standard normal to physical space
        r_s = mu_R + sigma_R * u
        d_ps = mu_H + sigma_H * v
        
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
        
        # CDFs
        cdf_X = cdf(dist_X, beta)
        cdf_Y = cdf(dist_Y, s)
        
        # Return integrand (NO explicit f_R, f_H—they're in the Hermite weights!)
        val = (1f0 - cdf_X) * cdf_Y
        
        return val
    end

    """
    Kernel: write per-(point,param) contribution to outbuf
    """
    function batch_kernel(coords_chunk, weights_chunk, d::Int32,
                        params_d, param_dim::Int32, times_d, T::Int32, outbuf, P::Int32, model_idx::Int32)
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
            
            if model_idx == 1 # Independent induction/inhibition
                val = integrand_point_0(coords_chunk, base, d, params_d, pbase, t)
            end
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
    function integrate_batched(n::Int, d::Int, params::Array{Float32,2}, times::Array{Float32,1}, model_idx::Int; chunk_size::Int=4096)
        coords_cpu, weights_cpu, N = build_hermite_tensor_grid(n, d)
        P = size(params, 1); param_dim = size(params, 2)
        T = length(times)

        # @assert param_dim == 7
        params_flat = vec(params')                    # row-major flatten (M × param_dim)
        params_d = CuArray(params_flat)
        times_d = CuArray(times)

        accum_d = CUDA.zeros(Float32, P, T)              # final accumulators

        # Iterate over chunks
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
                                                        params_d, Int32(param_dim), times_d, Int32(T), outbuf, Int32(P), Int32(model_idx))
            CUDA.synchronize()
            
            # COPY OUTBUF TO HOST AND REDUCE PER-PARAM (fast for moderate chunk_size)
            host_buf = Array(outbuf)                            # length = len * P
            host_mat = reshape(host_buf, (T, P, len))           # T × P × len
            # sum across columns -> partial sums per parameter
            partials  = dropdims(sum(host_vol; dims=3), dims=3) # T × P
            # accum = Array(accum_d)                              # copy current accum to host
            accum .+= partials' #vec(partials)                             # update
            # accum_d .= CuArray(accum)                           # copy back to device accumulators

            i += len
        end

        return accum # P × T:  accum[p, t] = integral for param set p at time t
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
        times (Array{Float32}) - time points to evaluate
        param_dict (Dict) - parameter dictionary, multiple values possible per key
    """
    function compute_germination(model_alias, rho_s, times, param_dict)

        n_params = length(keys(param_dict))
        sample_size = length(param_dict[:mu_X])

        if model_alias == "independent"
            # Unpack parameter dictionary into an Array P x param_dim
            param_arr = Array{Float32}(undef, n_params, sample_size)
            param_arr[:, 1] .= Float32.(param_dict[:mu_R])
            param_arr[:, 2] .= Float32.(param_dict[:sigma_R])
            param_arr[:, 3] .= Float32.(param_dict[:mu_H])
            param_arr[:, 4] .= Float32.(param_dict[:sigma_H])
            param_arr[:, 5] .= Float32.(param_dict[:c_ex])
            param_arr[:, 6] .= Float32.(param_dict[:P_I])
            param_arr[:, 7] .= Float32.(param_dict[:P_C])
            param_arr[:, 8] .= Float32.(param_dict[:K_s])
            param_arr[:, 9] .= Float32.(param_dict[:rho_s])

            germination = integrate_batched(8, 3, param_arr, times, 1)
        end
    end

end