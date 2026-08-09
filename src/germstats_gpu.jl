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
    export load_model_collection_ordered
    export param_dict_to_matrix
    export model_wrapper


    """
    Loads the entire collection of germination models,
    including their aliases, combination IDs and textual descriptions.
    output:
        aliases - model identifier strings
        combination_IDs - a string containing the interaction identifiers (A, B, C, D and E)
        descriptions - textual descriptions of the models
        param_key_sets - arrays of keys of the parameters used in each model
    """
    function load_model_collection_ordered()
        
        models = Dict(
            # "T" => ["test",                                                                     "Toy model for testing\n",
            #             [:P_I, :neg_delta_Y, :mu_Y]],
            "0" => [1, "independent",                                                              "Independent inducer/inhibitor model\n",
                        [:K_s, :P_I, :P_C, :neg_delta_X, :neg_delta_Y, :mu_X, :mu_Y]],
            "Ai" => [2, "feedback_inhibitor_inducer_perm",                                         "Inhibitor-dependent germination with\ninducer-dependent permeability",
                        [:K_s, :P_I, :P_C, :lambda_C, :neg_delta_X, :neg_delta_O, :mu_X, :mu_O]],
            "A" => [3, "feedback_combined_inducer_perm",                                           "2-factor germination with\ninducer-dependent permeability",
                        [:K_s, :P_I, :P_C, :lambda_C, :neg_delta_X, :neg_delta_O, :neg_delta_Y, :mu_X, :mu_O, :mu_Y]],
            "Bi" => [4, "inhibitor_thresh",                                                        "Inhibitor-dependent induction threshold\n",
                        [:K_s, :P_I, :P_C, :k_gamma, :neg_delta_X, :mu_X]],
            "B" => [5, "combined_inhibitor_thresh",                                                "2-factor germination with inducer-dependent\ninhibitor threshold",
                        [:K_s, :P_I, :P_C, :k_gamma, :neg_delta_X, :neg_delta_Y, :mu_X, :mu_Y]],
            "Cc" => [6, "inducer_signal" ,                                                         "Inhibitor-dependent induction signal\n",
                        [:K_I, :K_s, :P_I, :P_C, :n, :neg_delta_O, :neg_delta_Y, :mu_O, :mu_Y]],
            "C" => [7, "combined_inducer_signal" ,                                                 "2-factor germination with inhibitor-dependent\ninduction signal",
                        [:K_I, :K_s, :P_I, :P_C, :n, :neg_delta_X, :neg_delta_O, :neg_delta_Y, :mu_X, :mu_O, :mu_Y]],
            "Dc" => [8, "feedback_inducer_inhibitor_perm",                                         "Inducer-dependent germination with\ninhibitor-dependent permeability",
                        [:K_s, :K_b, :P_I, :P_C, :lambda_I, :neg_delta_O, :neg_delta_Y, :mu_O, :mu_Y]],
            "D" => [9, "feedback_combined_inhibitor_perm",                                         "2-factor germination with\ninhibitor-dependent permeability",
                        [:K_s, :K_b, :P_I, :P_C, :lambda_I, :neg_delta_X, :neg_delta_O, :neg_delta_Y, :mu_X, :mu_O, :mu_Y]],
            "Ec" => [10, "inducer_thresh",                                                          "Inhibitor-dependent induction threshold\n",
                        [:K_s, :K_b, :P_I, :P_C, :k_omega, :neg_delta_O, :neg_delta_Y, :mu_O, :mu_Y]],
            "E" => [11, "combined_inducer_thresh",                                                  "2-factor germination with inhibitor-dependent\ninduction threshold",
                        [:K_s, :K_b, :P_I, :P_C, :k_omega, :neg_delta_X, :neg_delta_O, :neg_delta_Y, :mu_X, :mu_O, :mu_Y]],
            "ABi" => [12, "feedback_inhibitor_inducer_perm_thresh",                                 "Inhibitor-dependent germination with\ninducer-dependent permeability/threshold",
                        [:K_s, :P_I, :P_C, :k_gamma, :lambda_C, :neg_delta_X, :neg_delta_O, :mu_X, :mu_O]],
            "AB" => [13, "feedback_combined_inducer_perm_thresh",                                   "2-factor germination with\ninducer-dependent permeability/threshold",
                        [:K_s, :P_I, :P_C, :k_gamma, :lambda_C, :neg_delta_X, :neg_delta_O, :neg_delta_Y, :mu_X, :mu_O, :mu_Y]],
            "ACi" => [14, "feedback_inhibitor_inducer_perm_inhibitor_signal",                       "Inhibitor-dependent germination with\ninducer-dep. permeability / inhibitor-dep. signal",
                        [:K_I, :K_s, :P_I, :P_C, :n, :lambda_C, :neg_delta_X, :neg_delta_O, :mu_X, :mu_O]],
            "ACc" => [15, "feedback_inducer_inducer_perm_inhibitor_signal",                         "Inducer-dependent germination with\ninducer-dep. permeability / inhibitor-dep. signal",
                        [:K_I, :K_s, :P_I, :P_C, :n, :lambda_C, :neg_delta_O, :neg_delta_Y, :mu_O, :mu_Y]],
            "AC" => [16, "feedback_combined_inducer_perm_inhibitor_signal",                         "2-factor germination with\ninducer-dep. permeability / inhibitor-dep. signal",
                        [:K_I, :K_s, :P_I, :P_C, :n, :lambda_C, :neg_delta_X, :neg_delta_O, :neg_delta_Y, :mu_X, :mu_O, :mu_Y]],
            "ADi" => [17, "feedback_inhibitor_inhibitor_inducer_perm",                              "Inhibitor-dependent germination with\ninhibitor+inducer-dependent permeability",
                        [:K_s, :K_b, :P_I, :P_C, :lambda_I, :lambda_C, :neg_delta_X, :neg_delta_O, :mu_X, :mu_O]],
            "ADc" => [18, "feedback_inducer_inhibitor_inducer_perm",                                "Inducer-dependent germination with\ninhibitor+inducer-dependent permeability",
                        [:K_s, :K_b, :P_I, :P_C, :lambda_I, :lambda_C, :neg_delta_O, :neg_delta_Y, :mu_O, :mu_Y]],
            "AD" => [19, "feedback_combined_inhibitor_inducer_perm",                                "2-factor germination with\ninhibitor+inducer-dependent permeability",
                        [:K_s, :K_b, :P_I, :P_C, :lambda_I, :lambda_C, :neg_delta_X, :neg_delta_O, :neg_delta_Y, :mu_X, :mu_O, :mu_Y]],
            "AEc" => [20, "feedback_inducer_inhibitor_thresh_inducer_perm",                         "Inducer-dependent germination with\ninhibitor-dep. thresh., inducer-dep. perm.",
                        [:K_s, :K_b, :P_I, :P_C, :k_omega, :lambda_C, :neg_delta_O, :neg_delta_Y, :mu_O, :mu_Y]],
            "AE" => [21, "feedback_combined_inhibitor_thresh_inducer_perm",                         "2-factor germination with\ninhibitor-dep. thresh., inducer-dep. perm.",
                        [:K_s, :K_b, :P_I, :P_C, :k_omega, :lambda_C, :neg_delta_X, :neg_delta_O, :neg_delta_Y, :mu_X, :mu_O, :mu_Y]],
            "BCi" => [22, "inhibitor_thresh_inducer_signal",                                        "Inhibitor-dependent germination with\ninhibitor-dep. signal, inducer-dep. thresh",
                        [:K_I, :K_s, :P_I, :P_C, :k_gamma, :n, :neg_delta_X, :neg_delta_O, :mu_X, :mu_O]],
            "BC" => [23, "combined_inhibitor_thresh_inducer_signal",                                "2-factor germination with\ninhibitor-dep. signal, inducer-dep. thresh",
                        [:K_I, :K_s, :P_I, :P_C, :k_gamma, :n, :neg_delta_X, :neg_delta_O, :neg_delta_Y, :mu_X, :mu_O, :mu_Y]],
            "BDi" => [24, "feedback_inhibitor_inducer_thresh_inhibitor_perm",                       "Inhibitor-dependent germination with\ninducer-dep. thresh, inhibitor-dep. perm.",
                        [:K_s, :K_b, :P_I, :P_C, :lambda_I, :k_gamma, :neg_delta_X, :neg_delta_O, :mu_X, :mu_O]],
            "BD" => [25, "feedback_combined_inducer_thresh_inhibitor_perm",                         "2-factor germination with\ninducer-dep. thresh., inhibitor-dep. perm.",
                        [:K_s, :K_b, :P_I, :P_C, :lambda_I, :k_gamma, :neg_delta_X, :neg_delta_O, :neg_delta_Y, :mu_X, :mu_O, :mu_Y]],
            "BE" => [26, "combined_inhibitor_thresh_inducer_thresh",                                "2-factor germination with\ninducer-dep. thresh., inhibitor-dep. thresh.",
                        [:K_s, :K_b, :P_I, :P_C, :k_gamma, :k_omega, :neg_delta_X, :neg_delta_O, :neg_delta_Y, :mu_X, :mu_O, :mu_Y]],
            "CDc" => [27, "feedback_inducer_inhibitor_perm_signal",                                 "Inducer-dependent germination with\ninhibitor-dependent perm. and induction signal",
                        [:K_I, :K_s, :K_b, :P_I, :P_C, :lambda_I, :neg_delta_O, :neg_delta_Y, :mu_O, :mu_Y, :n]],
            "CD" => [28, "feedback_combined_inhibitor_perm_signal",                                 "2-factor germination with\ninhibitor-dependent perm. and induction signal",
                        [:K_I, :K_s, :K_b, :P_I, :P_C, :lambda_I, :neg_delta_X, :neg_delta_O, :neg_delta_Y, :mu_X, :mu_O, :mu_Y, :n]],
            "CEc" => [29, "inducer",                                                                "Inhibitor-dependent induction threshold and signal\n",
                        [:K_I, :K_s, :K_b, :P_I, :P_C, :k_omega, :n, :neg_delta_O, :neg_delta_Y, :mu_O, :mu_Y]],
            "CE" => [30, "combined_inducer",                                                        "Combined model with inhibitor-dependent\ninduction threshold and signal",
                        [:K_I, :K_s, :K_b, :P_I, :P_C, :k_omega, :n, :neg_delta_X, :neg_delta_O, :neg_delta_Y, :mu_X, :mu_O, :mu_Y]],
            "DEc" => [31, "feedback_inducer_inhibitor_perm_thresh",                                 "Inducer-dependent germination with\ninhibitor-dependent perm. and induction thresh.",
                        [:K_s, :K_b, :P_I, :P_C, :lambda_I, :k_omega, :neg_delta_O, :neg_delta_Y, :mu_O, :mu_Y]],
            "DE" => [32, "feedback_combined_inhibitor_perm_thresh",                                 "2-factor germination with\ninhibitor-dependent perm. and induction thresh.",
                        [:K_s, :K_b, :P_I, :P_C, :lambda_I, :k_omega, :neg_delta_X, :neg_delta_O, :neg_delta_Y, :mu_X, :mu_O, :mu_Y]],
            "ABCi" => [33, "feedback_inhibitor_inducer_perm_thresh_inhibitor_signal",               "Inhibitor-dependent germination with\ninducer-dep. perm./thresh., inhibitor-dep. signal",
                        [:K_I, :K_s, :P_I, :P_C, :k_gamma, :n, :lambda_C, :neg_delta_X, :neg_delta_O, :mu_X, :mu_O]],
            "ABC" => [34, "feedback_combined_inducer_perm_thresh_inhibitor_signal",                 "2-factor germination with\ninducer-dep. perm./thresh., inhibitor-dep. signal",
                        [:K_I, :K_s, :P_I, :P_C, :k_gamma, :n, :lambda_C, :neg_delta_X, :neg_delta_O, :neg_delta_Y, :mu_X, :mu_O, :mu_Y]],
            "ABDi" => [35, "feedback_inhibitor_inducer_perm_thresh_inhibitor_perm",                 "Inhibitor-dependent germination with\ninducer-dep. perm./thresh., inhibitor-dep. perm.",
                        [:K_s, :K_b, :P_I, :P_C, :lambda_I, :k_gamma, :lambda_C, :neg_delta_X, :neg_delta_O, :mu_X, :mu_O]],
            "ABD" => [36, "feedback_combined_inducer_perm_thresh_inhibitor_perm",                   "2-factor germination with\ninducer-dep. perm./thresh., inhibitor-dep. perm.",
                        [:K_s, :K_b, :P_I, :P_C, :lambda_I, :k_gamma, :lambda_C, :neg_delta_X, :neg_delta_O, :neg_delta_Y, :mu_X, :mu_O, :mu_Y]],
            "ABE" => [37, "feedback_combined_inducer_perm_thresh_inhibitor_thresh",                 "2-factor germination with\ninducer-dep. pern./thresh., inhibitor-dep. thresh.",
                        [:K_s, :K_b, :P_I, :P_C, :k_gamma, :k_omega, :lambda_C, :neg_delta_X, :neg_delta_O, :neg_delta_Y, :mu_X, :mu_O, :mu_Y]],
            "ACDi" => [38, "feedback_inhibitor_inhibitor_inducer_perm_inhibitor_signal",            "Inhibitor-dependent germination with\ninhibitor/inducer-dep. perm, inhibitor-dep. signal",
                        [:K_I, :K_s, :K_b, :P_I, :P_C, :lambda_I, :n, :lambda_C, :neg_delta_X, :neg_delta_O, :mu_X, :mu_O]],
            "ACDc" => [39, "feedback_inducer_inhibitor_inducer_perm_inhibitor_signal",              "Inducer-dependent germination with\ninhibitor/inducer-dep. perm, inhibitor-dep. signal",
                        [:K_I, :K_s, :K_b, :P_I, :P_C, :lambda_I, :n, :lambda_C, :neg_delta_O, :neg_delta_Y, :mu_O, :mu_Y]],
            "ACD" => [40, "feedback_combined_inhibitor_inducer_perm_inhibitor_signal",              "2-factor germination with\ninhibitor/inducer-dep. perm, inhibitor-dep. signal",
                        [:K_I, :K_s, :K_b, :P_I, :P_C, :lambda_I, :n, :lambda_C, :neg_delta_X, :neg_delta_O, :neg_delta_Y, :mu_X, :mu_O, :mu_Y]],
            "ACEc" => [41, "feedback_inducer_inducer_perm_inhibitor_thresh_signal",                 "Inducer-dependent germination with\ninducer-dep. perm., inhbitor-dep. thresh./signal",
                        [:K_I, :K_s, :K_b, :P_I, :P_C, :k_omega, :n, :lambda_C, :neg_delta_O, :neg_delta_Y, :mu_O, :mu_Y]],
            "ACE" => [42, "feedback_combined_inducer_perm_inhibitor_thresh_signal",                 "2-factor germination with\ninducer-dep. perm., inhbitor-dep. thresh./signal",
                        [:K_I, :K_s, :K_b, :P_I, :P_C, :k_omega, :n, :lambda_C, :neg_delta_X, :neg_delta_O, :neg_delta_Y, :mu_X, :mu_O, :mu_Y]],
            "ADEc" => [43, "feedback_inducer_inhibitor_inducer_perm_inhibitor_thresh",              "Inhibitor-dependent germination with\ninhibitor/inducer-dep. perm, inhibitor-dep thresh.",
                        [:K_s, :K_b, :P_I, :P_C, :lambda_I, :k_omega, :lambda_C, :neg_delta_X, :neg_delta_O, :mu_X, :mu_O]],
            "ADE" => [44, "feedback_combined_inhibitor_inducer_perm_inhibitor_thresh",              "2-factor germination with\ninhibitor/inducer-dep. perm, inhibitor-dep thresh.",
                        [:K_s, :K_b, :P_I, :P_C, :lambda_I, :k_omega, :lambda_C, :neg_delta_X, :neg_delta_O, :neg_delta_Y, :mu_X, :mu_O, :mu_Y]],
            "BCDi" => [45, "feedback_inhibitor_inducer_thresh_inhibitor_perm_signal",               "Inhibitor-dependent germination with\ninducer-dep. thresh., inhibitor-dep. perm./signal",
                        [:K_I, :K_s, :K_b, :P_I, :P_C, :lambda_I, :k_gamma, :n, :neg_delta_X, :neg_delta_O, :mu_X, :mu_O]],
            "BCD" => [46, "feedback_combined_inducer_thresh_inhibitor_perm_signal",                 "2-factor germination with\ninducer-dep. thresh., inhibitor-dep. perm./signal",
                        [:K_I, :K_s, :K_b, :P_I, :P_C, :lambda_I, :k_gamma, :n, :neg_delta_X, :neg_delta_O, :neg_delta_Y, :mu_X, :mu_O, :mu_Y]],
            "BCE" => [47, "combined_inhibitor_thresh_signal_inducer_thresh",                        "2-factor germination with\ninhibitor-dep. thresh./signal, inducer-dep. thresh",
                        [:K_I, :K_s, :K_b, :P_I, :P_C, :k_gamma, :k_omega, :n, :neg_delta_X, :neg_delta_O, :neg_delta_Y, :mu_X, :mu_O, :mu_Y]],
            "BDE" => [48, "feedback_combined_inhibitor_perm_thresh_inducer_thresh",                 "2-factor germination with\ninhibitor-dep. perm./thresh., inducer-dep. thresh.",
                        [:K_s, :K_b, :P_I, :P_C, :lambda_I, :k_gamma, :k_omega, :neg_delta_X, :neg_delta_O, :neg_delta_Y, :mu_X, :mu_O, :mu_Y]],
            "CDEc" => [49, "feedback_inducer_inhibitor_perm_thresh_signal",                         "Inducer-dependent germination with\ninhibitor-dep. perm./thresh./signal",
                        [:K_I, :K_s, :K_b, :P_I, :P_C, :lambda_I, :k_omega, :n, :neg_delta_O, :neg_delta_Y, :mu_O, :mu_Y]],
            "CDE" => [50, "feedback_combined_inhibitor_perm_thresh_signal",                         "2-factor germination with\ninhibitor-dep. perm./thresh./signal",
                        [:K_I, :K_s, :K_b, :P_I, :P_C, :lambda_I, :k_omega, :n, :neg_delta_X, :neg_delta_O, :neg_delta_Y, :mu_X, :mu_O, :mu_Y]],
            "ABCDi" => [51, "feedback_inhibitor_inducer_perm_thresh_inhibitor_perm_signal",         "Inhibitor-dependent germination with\ninducer-dep. perm./thresh., inhibitor-dep. perm./signal",
                        [:K_I, :K_s, :K_b, :P_I, :P_C, :lambda_I, :k_gamma, :n, :lambda_C, :neg_delta_X, :neg_delta_O, :mu_X, :mu_O]],
            "ABCD" => [52, "feedback_combined_inducer_perm_thresh_inhibitor_perm_signal",           "2-factor germination with\ninducer-dep. perm./thresh., inhibitor-dep. perm./signal",
                        [:K_I, :K_s, :K_b, :P_I, :P_C, :lambda_I, :k_gamma, :n, :lambda_C, :neg_delta_X, :neg_delta_O, :neg_delta_Y, :mu_X, :mu_O, :mu_Y]],
            "ABCE" => [53, "feedback_combined_inhibitor_thresh_signal_inducer_perm_thresh",         "2-factor germination with inhibitor-dep.\nthresh./signal, inducer-dep. perm./thresh.",
                        [:K_I, :K_s, :K_b, :P_I, :P_C, :k_gamma, :k_omega, :n, :lambda_C, :neg_delta_X, :neg_delta_O, :neg_delta_Y, :mu_X, :mu_O, :mu_Y]],
            "ABDE" => [54,"feedback_combined_inhibitor_perm_thresh_inducer_perm_thresh",           "2-factor germination with inhibitor/inducer-dep. perm,\ninhibitor/inducer-dep. thresholds",
                        [:K_s, :K_b, :P_I, :P_C, :lambda_I, :k_gamma, :k_omega, :lambda_C, :neg_delta_X, :neg_delta_O, :neg_delta_Y, :mu_X, :mu_O, :mu_Y]],
            "ACDEc" => [55, "feedback_inducer_inhibitor_perm_thresh_signal_inducer_perm",           "Inducer-dependent germination with inhibitor/inducer-dep.\nperm., inhibitor-dep. thresh./signal",
                        [:K_I, :K_s, :K_b, :P_I, :P_C, :lambda_I, :k_omega, :n, :lambda_C, :neg_delta_O, :neg_delta_Y, :mu_O, :mu_Y]],
            "ACDE" => [56, "feedback_combined_inhibitor_perm_thresh_signal_inducer_perm",            "2-factor germination with inhibitor/inducer-dep. perm.,\ninhibitor-dep. thresh./signal",
                        [:K_I, :K_s, :K_b, :P_I, :P_C, :lambda_I, :k_omega, :n, :lambda_C, :neg_delta_X, :neg_delta_O, :neg_delta_Y, :mu_X, :mu_O, :mu_Y]],
            "BCDE" => [57, "feedback_combined_inhibitor_perm_thresh_signal_inducer_thresh",         "2-factor germination with inhibitor/inducer-dep. thresh.,\ninhibitor-dep. perm./signal",
                        [:K_I, :K_s, :K_b, :P_I, :P_C, :lambda_I, :k_gamma, :k_omega, :n, :neg_delta_X, :neg_delta_O, :neg_delta_Y, :mu_X, :mu_O, :mu_Y]],
            "ABCDE" => [58, "feedback_combined_inhibitor_perm_thresh_signal_inducer_perm_thresh",   "2-factor germination with inhibitor/inducer-dep.\nperm./thresh., inhibitor-dep. signal",
                        [:K_I, :K_s, :K_b, :P_I, :P_C, :lambda_I, :k_gamma, :k_omega, :n, :lambda_C, :neg_delta_X, :neg_delta_O, :neg_delta_Y, :mu_X, :mu_O, :mu_Y]] 
        )

        combination_IDs = collect(keys(models))
        indices = [models[id][1] for id in combination_IDs]
        indices_sorted = sortperm(indices)
        combination_IDs = combination_IDs[indices_sorted]

        aliases = [models[id][2] for id in combination_IDs]
        descriptions = [models[id][3] for id in combination_IDs]
        param_key_sets = [models[id][4] for id in combination_IDs]

        return aliases, combination_IDs, descriptions, param_key_sets
    end

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
    GPU-optimized approximation of the normal CDF using the Wichura method.
    This function computes the cumulative distribution function (CDF) of a normal distribution
    with mean μ and standard deviation σ at a given point x. The implementation is based on
    the Abramowitz and Stegun approximation (7.1.26) and is designed for efficient execution on GPUs.
    inputs:
        x (Float32) - the point at which to evaluate the CDF
        μ (Float32) - the mean of the normal distribution
        σ (Float32) - the standard deviation of the normal distribution
    """
    @inline function normal_cdf_wichura(x, μ, σ)
        # Standardize: convert to standard normal (z-score)
        z = (x - μ) / σ
        
        # Abramowitz and Stegun approximation (7.1.26)
        # Maximum error: 2.5 × 10⁻⁵
        b1 = Float32(0.319381530f0)
        b2 = Float32(-0.356563782f0)
        b3 = Float32(1.781477937f0)
        b4 = Float32(-1.821255978f0)
        b5 = Float32(1.330274429f0)
        p  = Float32(0.2316419f0)
        c  = Float32(0.39894228f0)
        
        abs_z = abs(z)
        t = 1.0f0 / (1.0f0 + p * abs_z)
        
        # Compute the CDF value
        cdf_val = 1.0f0 - c * exp(-0.5f0 * z * z) * t * 
                (b1 + t * (b2 + t * (b3 + t * (b4 + t * b5))))
        
        # Handle the sign
        return z < 0 ? 1.0f0 - cdf_val : cdf_val
    end

    """
    Threshold evaluation for integrand function
    """
    @inline function eval_threshold_integrand(thresh_mode, beta, s, b, mu_X, sigma_X, mu_Y, sigma_Y, k_gamma, k_omega)
        
        # dist_X = Normal(mu_X, sigma_X)
        # dist_Y = Normal(mu_Y, sigma_Y)

        cdf_X = 1.0f0
        cdf_Y = 0.0f0
        if thresh_mode == 0
            # Inhibitor-dependent germination triggering
            # cdf_X = cdf(dist_X, beta)
            cdf_X = normal_cdf_wichura(beta, mu_X, sigma_X)
            cdf_Y = 1.0f0
        elseif thresh_mode == 1 || thresh_mode == 8 # s_mod supplied if thresh_mode == 8
            # Inducer-dependent germination triggering
            cdf_X = 0.0f0
            # cdf_Y = cdf(dist_Y, s)
            cdf_Y = normal_cdf_wichura(s, mu_Y, sigma_Y)
        elseif thresh_mode == 2 || thresh_mode == 9 # s_mod supplied if thresh_mode == 9
            # 2-factor germination triggering
            # cdf_X = cdf(dist_X, beta)
            # cdf_Y = cdf(dist_Y, s)
            cdf_X = normal_cdf_wichura(beta, mu_X, sigma_X)
            cdf_Y = normal_cdf_wichura(s, mu_Y, sigma_Y)
        elseif thresh_mode == 3 || thresh_mode == 10 # s_mod supplied if thresh_mode == 10
            # Shifted inhibitor-dependent germination triggering
            # cdf_X = cdf(dist_X, beta - k_gamma * s)
            cdf_X = normal_cdf_wichura(beta - k_gamma * s, mu_X, sigma_X)
            cdf_Y = 1.0f0
        elseif thresh_mode == 4 || thresh_mode == 11 # s_mod supplied if thresh_mode == 11
            # Shifted inducer-dependent germination triggering
            cdf_X = 0.0f0
            # cdf_Y = cdf(dist_Y, s - k_omega * b)
            cdf_Y = normal_cdf_wichura(s - k_omega * b, mu_Y, sigma_Y)
        elseif thresh_mode == 5 || thresh_mode == 12 # s_mod supplied if thresh_mode == 12
            # 2-factor germination triggering with shifted gamma
            # cdf_X = cdf(dist_X, beta - k_gamma * s)
            # cdf_Y = cdf(dist_Y, s)
            cdf_X = normal_cdf_wichura(beta - k_gamma * s, mu_X, sigma_X)
            cdf_Y = normal_cdf_wichura(s, mu_Y, sigma_Y)
        elseif thresh_mode == 6 || thresh_mode == 13 # s_mod supplied if thresh_mode == 13
            # 2-factor germination triggering with shifted omega
            # cdf_X = cdf(dist_X, beta)
            # cdf_Y = cdf(dist_Y, s - k_omega * b)
            cdf_X = normal_cdf_wichura(beta, mu_X, sigma_X)
            cdf_Y = normal_cdf_wichura(s - k_omega * b, mu_Y, sigma_Y)
        elseif thresh_mode == 7 || thresh_mode == 14 # s_mod supplied if thresh_mode == 14
            # 2-factor germination triggering with shifted gamma and omega
            # cdf_X = cdf(dist_X, beta - k_gamma * s)
            # cdf_Y = cdf(dist_Y, s - k_omega * b)
            cdf_X = normal_cdf_wichura(beta - k_gamma * s, mu_X, sigma_X)
            cdf_Y = normal_cdf_wichura(s - k_omega * b, mu_Y, sigma_Y)
        end

        return (1f0 - cdf_X) * cdf_Y

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
                                            pbase::Int32, t::Float32, thresh_mode::Int32)
        
        
        # Extract standard normal nodes
        u = coords[base + 1]          # Standard normal for r_s
        v = coords[base + 2]          # Standard normal for d_ps
        
        # Extract parameters
        rho_s, mu_R, sigma_R, mu_H, sigma_H, mu_X, sigma_X, mu_Y, sigma_Y,
        c_ex, P_I, P_C, K_s = unpack_standard_parameters(params, pbase)
        k_gamma = params[pbase + 18]
        k_omega = params[pbase + 19]

        # Time-dependent signals
        beta, s = calc_signals(u, v, t, rho_s, c_ex,
                                mu_R, sigma_R, mu_H, sigma_H,
                                K_s, P_I, P_C)
        
        return eval_threshold_integrand(thresh_mode, beta, s, 0.0f0, mu_X, sigma_X, mu_Y, sigma_Y, k_gamma, k_omega)
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
        
        # Extract parameters
        rho_s, mu_R, sigma_R, mu_H, sigma_H, mu_X, sigma_X, mu_Y, sigma_Y,
        c_ex, P_I, P_C, K_s = unpack_standard_parameters(params, pbase)
        k_gamma = params[pbase + 18]
        k_omega = params[pbase + 19]

        # Time-dependent signals
        beta, s = calc_signals(u, v, t, rho_s, c_ex,
                                mu_R, sigma_R, mu_H, sigma_H,
                                K_s, P_I, P_C)
        
        return eval_threshold_integrand(thresh_mode, beta, s, 0.0f0, mu_X, sigma_X, mu_Y, sigma_Y, k_gamma, k_omega)
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
        
        # Extract parameters
        rho_s, mu_R, sigma_R, mu_H, sigma_H, mu_X, sigma_X, mu_Y, sigma_Y,
        c_ex, P_I, P_C, K_s = unpack_standard_parameters(params, pbase)
        k_gamma = params[pbase + 18]
        k_omega = params[pbase + 19]
        K_I = params[pbase + 20]
        n = params[pbase + 21]

        # Sample initial inhibitor concentration
        c_0 = sample_c0(w, mu_R, sigma_R)

        # Time-dependent signals
        beta, s = calc_signals(u, v, t, rho_s, c_ex,
                                mu_R, sigma_R, mu_H, sigma_H,
                                K_s, P_I, P_C)

        if thresh_mode == 8 || thresh_mode == 9 || thresh_mode == 10 || thresh_mode == 12
            s_mod = s / (1 + (beta * c_0 / K_I)^n)
            val = eval_threshold_integrand(thresh_mode, beta, s_mod, 0.0f0, mu_X, sigma_X, mu_Y, sigma_Y, k_gamma, k_omega)
        else
            val = eval_threshold_integrand(thresh_mode, beta, s, 0.0f0, mu_X, sigma_X, mu_Y, sigma_Y, k_gamma, k_omega)
        end
        
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
        
        # Extract parameters
        rho_s, mu_R, sigma_R, mu_H, sigma_H, mu_X, sigma_X, mu_Y, sigma_Y,
        c_ex, P_I, P_C, K_s = unpack_standard_parameters(params, pbase)
        k_gamma = params[pbase + 18]
        k_omega = params[pbase + 19]
        K_I = params[pbase + 20]
        n = params[pbase + 21]
        K_b = params[pbase + 22]

        # Sample initial inhibitor concentration
        c_0 = sample_c0(w, mu_R, sigma_R)

        # Time-dependent signals
        beta, s = calc_signals(u, v, t, rho_s, c_ex,
                                mu_R, sigma_R, mu_H, sigma_H,
                                K_s, P_I, P_C)
        c_in_I = beta * c_0
        b = c_in_I / (K_b + c_in_I)

        if thresh_mode == 11 || thresh_mode == 13 || thresh_mode == 14
            s_mod = s / (1 + (beta * c_0 / K_I)^n)
            val = eval_threshold_integrand(thresh_mode, beta, s_mod, 0.0f0, mu_X, sigma_X, mu_Y, sigma_Y, k_gamma, k_omega)
        else
            val = eval_threshold_integrand(thresh_mode, beta, s, 0.0f0, mu_X, sigma_X, mu_Y, sigma_Y, k_gamma, k_omega)
        end
        
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
                val = integrand_point_0(coords_chunk, base, d, params_d, pbase, t, Int32(2))
            elseif model_idx == 4 # Inducer-dependent inhibition threshold
                val = integrand_point_B(coords_chunk, base, d, params_d, pbase, t, Int32(3))
            elseif model_idx == 5 # 2-factor germination with inducer-dependent inhibition threshold
                val = integrand_point_B(coords_chunk, base, d, params_d, pbase, t, Int32(5))
            elseif model_idx == 6 # Inhibited inducer-dependent germination
                val = integrand_point_C(coords_chunk, base, d, params_d, pbase, t, Int32(8))
            elseif model_idx == 7 # 2-factor germination with inhibited inducer
                val = integrand_point_C(coords_chunk, base, d, params_d, pbase, t, Int32(9))
            elseif model_idx == 10 # Shifted induction threshold
                val = integrand_point_E(coords_chunk, base, d, params_d, pbase, t, Int32(4))
            elseif model_idx == 11 # 2-factor germination with shifted inhibitor-dependent induction threshold
                val = integrand_point_E(coords_chunk, base, d, params_d, pbase, t, Int32(6))
            elseif model_idx == 22 # Shifted inhibition threshold with inhibited inducer
                val = integrand_point_C(coords_chunk, base, d, params_d, pbase, t, Int32(10))
            elseif model_idx == 23 # 2-factor germination with shifted inhibitor-dependent induction threshold and inhibited inducer
                val = integrand_point_C(coords_chunk, base, d, params_d, pbase, t, Int32(12))
            elseif model_idx == 26 # Shifted inhibition and induction thresholds
                val = integrand_point_E(coords_chunk, base, d, params_d, pbase, t, Int32(7))
            elseif model_idx == 29 # Shifted induction threshold with inhibited inducer
                val = integrand_point_E(coords_chunk, base, d, params_d, pbase, t, Int32(11))
            elseif model_idx == 30 # 2-factor germination with shifted induction threshold and inhibited inducer
                val = integrand_point_E(coords_chunk, base, d, params_d, pbase, t, Int32(13))
            elseif model_idx == 47 # 2-factor germination with shifted thresholds and inhibited inducer
                val = integrand_point_E(coords_chunk, base, d, params_d, pbase, t, Int32(14))
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
        params (Union{Matrix{Float32}, Array{Float32,2}}) - parameters
        times (Array{Float32}) - time points for evaluation
        model_idx (Int) - index of model to use
        chunk_size (Int) - chunk size for parallel execution
    """
    function integrate_batched(n::Int, d::Int, params::Union{Matrix{Float32}, Array{Float32,2}}, times::Array{Float32,1}, model_idx::Int; chunk_size::Int=4096)
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
            K_I = params_sample[6]
            n = params_sample[7]
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
            elseif thresh_mode == 7
                # 2-factor germination triggering with shifted gamma and omega
                condition = (c_in_I < (gamma + k_gamma * s) * c0) && (s > omega + k_omega * b)
            elseif thresh_mode == 8
                # Inhibited inducer triggering germination
                condition = s / (1 + (c_in_I / K_I)^n) > omega
            elseif thresh_mode == 9
                # 2-factor germination triggering with inhibited inducer
                condition = (c_in_I < gamma * c0) && (s / (1 + (c_in_I / K_I)^n) > omega)
            elseif thresh_mode == 10
                # Shifted inhibitor-dependent germination triggering with inhibited inducer
                condition = c_in_I < (gamma + k_gamma * (s / (1 + (c_in_I / K_I)^n))) * c0
            elseif thresh_mode == 11
                # Shifted inducer-dependent germination triggering with inhibited inducer
                condition = s / (1 + (c_in_I / K_I)^n) > omega + k_omega * b
            elseif thresh_mode == 12
                # 2-factor germination triggering with shifted inhibitor-dependent induction threshold and inhibited inducer
                condition = (c_in_I < (gamma + k_gamma * (s / (1 + (c_in_I / K_I)^n))) * c0) && (s / (1 + (c_in_I / K_I)^n) > omega)
            elseif thresh_mode == 13
                # 2-factor germination triggering with shifted inducer-dependent induction threshold and inhibited inducer
                condition = (c_in_I < gamma * c0) && (s / (1 + (c_in_I / K_I)^n) > omega + k_omega * b)
            elseif thresh_mode == 14
                # 2-factor germination triggering with both shifted thresholds and inhibited inducer
                condition = (c_in_I < (gamma + k_gamma * (s / (1 + (c_in_I / K_I)^n))) * c0) && (s / (1 + (c_in_I / K_I)^n) > omega + k_omega * b)
            end

            # Check if germination occurs for this sample at this time point
            if condition
                germinated[time_idx, param_idx] = true
            end
        end

        return
    end

    """
    Generic function for computing concentration
    differences from permeation rates.
    """
    @inline function calc_diffs(c_in_I, c_in_C, c_out_I, c_ex, rateI, rateC, V_s, V_free, V_ps)
        
        # Concentration differences
        diffI = c_in_I - c_out_I
        diffC = c_in_C - c_ex
        
        # ODE equations
        du1 = -(rateI / V_s) * diffI
        du2 = (rateI / V_free) * diffI
        du3 = -(rateC / V_ps) * diffC

        return SVector{3}(du1, du2, du3)
    end

    """
    System of coupled ODEs for the feedback model
    of inducer-dependent cell wall permeability.
    """
    function ode_system_A(u, p, t)
        c_in_I, c_out_I, c_in_C = u
    
        # Unpack parameters for this specific spore
        P_I, P_C, c_ex, K_s, K_b, K_I, n, lambda_I, lambda_C, k_gamma, k_omega,
        A_s, V_s, V_free, V_ps, gamma, omega, c0 = p

        # Compute inducing signal s from c_in_C
        s = c_in_C / (K_s + c_in_C)

        exponent = -(1 + lambda_C * s) * 0.001f0

        # Limit permeability to Pmax = 1000 μm/s
        PmaxA = 1000.0f0 * A_s
        rateI = PmaxA * (-expm1(exponent * P_I))
        rateC = PmaxA * (-expm1(exponent * P_C))

        return calc_diffs(c_in_I, c_in_C, c_out_I, c_ex, rateI, rateC, V_s, V_free, V_ps)
    end

    """
    System of coupled ODEs for the feedback model
    of inhibitor-dependent cell wall permeability.
    """
    function ode_system_D(u, p, t)
        c_in_I, c_out_I, c_in_C = u
    
        # Unpack parameters for this specific spore
        P_I, P_C, c_ex, K_s, K_b, K_I, n, lambda_I, lambda_C, k_gamma, k_omega,
        A_s, V_s, V_free, V_ps, gamma, omega, c0 = p

        # Compute inhibitory signal b from c_in_I
        b = c_in_I / (K_b + c_in_I)

        exp_factor = exp(-lambda_I * b)
        
        # Update permeability constants based on signal
        P_I_pert = P_I * exp_factor
        P_C_pert = P_C * exp_factor

        rateI = P_I_pert * A_s
        rateC = P_C_pert * A_s

        return calc_diffs(c_in_I, c_in_C, c_out_I, c_ex, rateI, rateC, V_s, V_free, V_ps)
    end

    """
    System of coupled ODEs for the feedback model
    of inducer-dependent cell wall permeability
    and inhibitor-suppressed inducing signal.
    """
    function ode_system_AC(u, p, t)
        c_in_I, c_out_I, c_in_C = u
    
        # Unpack parameters for this specific spore
        P_I, P_C, c_ex, K_s, K_b, K_I, n, lambda_I, lambda_C, k_gamma, k_omega, A_s, V_s, V_free, V_ps, gamma, omega, c0 = p

        # Compute inducing signal s from c_in_C
        s = c_in_C / (K_s + c_in_C)
        s = s / (1 + (c_in_I / K_I)^n)

        exponent = -(1 + lambda_C * s) * 0.001f0

        # Limit permeability to Pmax = 1000 μm/s
        PmaxA = 1000.0f0 * A_s
        rateI = PmaxA * (-expm1(exponent * P_I))
        rateC = PmaxA * (-expm1(exponent * P_C))

        return calc_diffs(c_in_I, c_in_C, c_out_I, c_ex, rateI, rateC, V_s, V_free, V_ps)
    end

    """
    System of coupled ODEs for the feedback model
    of inducer and inhibitor-dependent cell wall permeability.
    """
    function ode_system_AD(u, p, t)
        c_in_I, c_out_I, c_in_C = u
    
        # Unpack parameters for this specific spore
        P_I, P_C, c_ex, K_s, K_b, K_I, n, lambda_I, lambda_C, k_gamma, k_omega, A_s, V_s, V_free, V_ps, gamma, omega, c0 = p

        # Compute inducing signal s from c_in_C and inhibitory signal b from c_in_I
        s = c_in_C / (K_s + c_in_C)
        b = c_in_I / (K_b + c_in_I)

        exponent = -(1 + lambda_C * s) * 0.001f0
        exp_factor = exp(-lambda_I * b)

        # Limit permeability to Pmax = 1000 μm/s
        PmaxA = 1000.0f0 * A_s
        rateI = PmaxA * (-expm1(exponent * P_I)) * exp_factor
        rateC = PmaxA * (-expm1(exponent * P_C)) * exp_factor

        return calc_diffs(c_in_I, c_in_C, c_out_I, c_ex, rateI, rateC, V_s, V_free, V_ps)
    end

    """
    System of coupled ODEs for the feedback model
    of inducer and inhibitor-dependent cell wall permeability
    and inhibitor-suppressed inducing signal.
    """
    function ode_system_ACD(u, p, t)
        c_in_I, c_out_I, c_in_C = u
    
        # Unpack parameters for this specific spore
        P_I, P_C, c_ex, K_s, K_b, K_I, n, lambda_I, lambda_C, k_gamma, k_omega, A_s, V_s, V_free, V_ps, gamma, omega, c0 = p

        # Compute inducing signal s from c_in_C and inhibitory signal b from c_in_I
        s = c_in_C / (K_s + c_in_C)
        s = s / (1 + (c_in_I / K_I)^n)
        b = c_in_I / (K_b + c_in_I)

        exponent = -(1 + lambda_C * s) * 0.001f0
        exp_factor = exp(-lambda_I * b)

        # Limit permeability to Pmax = 1000 μm/s
        PmaxA = 1000.0f0 * A_s
        rateI = PmaxA * (-expm1(exponent * P_I)) * exp_factor
        rateC = PmaxA * (-expm1(exponent * P_C)) * exp_factor

        return calc_diffs(c_in_I, c_in_C, c_out_I, c_ex, rateI, rateC, V_s, V_free, V_ps)
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
    function integrate_ode(params::Union{Matrix{Float32}, Array{Float32,2}}, times::Array{Float32,1}, model_idx::Int, ode_system; n_samples::Int=1024)
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
        if model_idx in [2, 14, 17, 38]
            thresh_mode = 0 # Inhibitor threshold
        elseif model_idx in [8, 18]
            thresh_mode = 1 # Inducer threshold
        elseif model_idx in [3, 9, 19]
            thresh_mode = 2 # Both thresholds
        elseif model_idx in [12, 24, 35]
            thresh_mode = 3 # Shifted inhibitor threshold
        elseif model_idx in [20, 31, 43]
            thresh_mode = 4 # Shifted inducer threshold
        elseif model_idx in [13, 25, 36]
            thresh_mode = 5 # Both thresholds + shifted inhibitor threshold
        elseif model_idx in [21, 32, 44]
            thresh_mode = 6 # Both thresholds + shifted inducer threshold
        elseif model_idx in [37, 48, 54]
            thresh_mode = 7 # Both shifted thresholds
        elseif model_idx in [15, 27, 39]
            thresh_mode = 8 # Inhibited inducer threshold
        elseif model_idx in [16, 28, 40]
            thresh_mode = 9 # Both thresholds + inhibited inducer threshold
        elseif model_idx in [33, 45, 51]
            thresh_mode = 10 # Shifted inhibitor threshold with inhibited inducer
        elseif model_idx in [41, 49, 55]
            thresh_mode = 11 # Shifted inducer threshold with inhibited inducer
        elseif model_idx in [34, 42, 46, 52]
            thresh_mode = 12 # Both thresholds + shifted inhibitor threshold with inhibited inducer
        elseif model_idx in [50, 56]
            thresh_mode = 13 # Both thresholds + shifted inducer threshold with inhibited inducer
        elseif model_idx in [53, 57, 58]
            thresh_mode = 14 # Both shifted thresholds with inhibited inducer
        else
            error("Unknown model index: $model_idx")
        end

        println("Model index: $model_idx")
        println("Threshold mode: $thresh_mode")

        # Generate Sobol sample
        sobol_samples = QuasiMonteCarlo.sample(n_samples, sobol_dim, SobolSample())

        # Construct flat parameter collections and initial conditions
        n_samples_flat = P * n_samples
        u0_vec = Vector{SVector{3, Float32}}()
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

                push!(u0_vec, @SVector [Float32(c0[j]), 0.0f0, 0.0f0])
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
        u0_dummy = @SVector [0.0f0, 1.0f0, 0.0f0]
        p_dummy = @SVector [
            1.0f0, 1.0f0, 1.0f0,
            1.0f0, 1.0f0, 1.0f0, 1.0f0, 1.0f0, 1.0f0,
            1.0f0, 1.0f0, 1.0f0, 1.0f0, 1.0f0, 1.0f0,
            1.0f0, 1.0f0, 1.0f0
        ]

        println("Constructing problem...")
        
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

        println("Starting solver...")

        sols_gpu = DiffEqGPU.vectorized_solve(
            gpu_probs,
            prob,
            GPURosenbrock23(),
            dt=dt,
            saveat=times
        )

        println("Solutions complete.")

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
    Convert a parameter dictionary to a matrix
    for GPU processing.
    inputs:
        param_dict (Dict) - parameter dictionary, multiple values possible per key
    outputs:
        param_arr (Array{Float32}) - parameter matrix of size P x param_dim
    """
    function param_dict_to_matrix(param_dict, rho_s)
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

        return param_arr
    end

    """
    Compute germination fraction over time
    for a given model and parameter values.
    inputs:
        model_alias (String) - model name
        param_arr (Array{Float32}) - parameter matrix of size P x param_dim
        times (Array{Float32}) - time points for evaluation
    outputs:
        germination (Array{Float32}) - germination fraction over time for each parameter set
    """
    function model_wrapper(model_alias, param_arr, times)
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
        elseif model_alias == "feedback_inducer_inducer_perm_inhibitor_signal" # ACc
            germination = integrate_ode(param_arr, times, 15, ode_system_AC)
        elseif model_alias == "feedback_combined_inducer_perm_inhibitor_signal" # AC
            germination = integrate_ode(param_arr, times, 16, ode_system_AC)
        elseif model_alias == "feedback_inhibitor_inhibitor_inducer_perm" # ADi
            germination = integrate_ode(param_arr, times, 17, ode_system_AD)
        elseif model_alias == "feedback_inducer_inhibitor_inducer_perm" # ADc
            germination = integrate_ode(param_arr, times, 18, ode_system_AD)
        elseif model_alias == "feedback_combined_inhibitor_inducer_perm" # AD
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
            germination = integrate_ode(param_arr, times, 27, ode_system_D)
        elseif model_alias == "feedback_combined_inhibitor_perm_signal" # CD
            germination = integrate_ode(param_arr, times, 28, ode_system_D)
        elseif model_alias == "inducer" # CEc
            germination = integrate_batched(8, 3, param_arr, times, 29)
        elseif model_alias == "combined_inducer" # CE
            germination = integrate_batched(8, 3, param_arr, times, 30)
        elseif model_alias == "feedback_inducer_inhibitor_perm_thresh" # DEc
            germination = integrate_ode(param_arr, times, 31, ode_system_D)
        elseif model_alias == "feedback_combined_inhibitor_perm_thresh" # DE
            germination = integrate_ode(param_arr, times, 32, ode_system_D)
        elseif model_alias == "feedback_inhibitor_inducer_perm_thresh_inhibitor_signal" # ABCi
            germination = integrate_ode(param_arr, times, 33, ode_system_AC)
        elseif model_alias == "feedback_combined_inducer_perm_thresh_inhibitor_signal" # ABC
            germination = integrate_ode(param_arr, times, 34, ode_system_AC)
        elseif model_alias == "feedback_inhibitor_inducer_perm_thresh_inhibitor_perm" # ABDi
            germination = integrate_ode(param_arr, times, 35, ode_system_AD)
        elseif model_alias == "feedback_combined_inducer_perm_thresh_inhibitor_perm" # ABD
            germination = integrate_ode(param_arr, times, 36, ode_system_AD)
        elseif model_alias == "feedback_combined_inducer_perm_thresh_inhibitor_thresh" # ABE
            germination = integrate_ode(param_arr, times, 37, ode_system_A)
        elseif model_alias == "feedback_inhibitor_inhibitor_inducer_perm_inhibitor_signal" # ACDi
            germination = integrate_ode(param_arr, times, 38, ode_system_ACD)
        elseif model_alias == "feedback_inducer_inhibitor_inducer_perm_inhibitor_signal" # ACDc
            germination = integrate_ode(param_arr, times, 39, ode_system_ACD)
        elseif model_alias == "feedback_combined_inhibitor_inducer_perm_inhibitor_signal" # ACD
            germination = integrate_ode(param_arr, times, 40, ode_system_ACD)
        elseif model_alias == "feedback_inducer_inducer_perm_inhibitor_thresh_signal" # ACEc
            germination = integrate_ode(param_arr, times, 41, ode_system_AC)
        elseif model_alias == "feedback_combined_inducer_perm_inhibitor_thresh_signal" # ACE
            germination = integrate_ode(param_arr, times, 42, ode_system_AC)
        elseif model_alias == "feedback_inducer_inhibitor_inducer_perm_inhibitor_thresh" # ADEc
            germination = integrate_ode(param_arr, times, 43, ode_system_AD)
        elseif model_alias == "feedback_combined_inhibitor_inducer_perm_inhibitor_thresh" # ADE
            germination = integrate_ode(param_arr, times, 44, ode_system_AD)
        elseif model_alias == "feedback_inhibitor_inducer_thresh_inhibitor_perm_signal" # BCDi
            germination = integrate_ode(param_arr, times, 45, ode_system_D)
        elseif model_alias == "feedback_combined_inducer_thresh_inhibitor_perm_signal" # BCD
            germination = integrate_ode(param_arr, times, 46, ode_system_D)
        elseif model_alias == "combined_inhibitor_thresh_signal_inducer_thresh" # BCE
            germination = integrate_batched(8, 3, param_arr, times, 47)
        elseif model_alias == "feedback_combined_inhibitor_perm_thresh_inducer_thresh" # BDE
            germination = integrate_ode(param_arr, times, 48, ode_system_D)
        elseif model_alias == "feedback_inducer_inhibitor_perm_thresh_signal" # CDEc
            germination = integrate_ode(param_arr, times, 49, ode_system_D)
        elseif model_alias == "feedback_combined_inhibitor_perm_thresh_signal" # CDE
            germination = integrate_ode(param_arr, times, 50, ode_system_D)
        elseif model_alias == "feedback_inhibitor_inducer_perm_thresh_inhibitor_perm_signal" # ABCDi
            germination = integrate_ode(param_arr, times, 51, ode_system_ACD)
        elseif model_alias == "feedback_combined_inducer_perm_thresh_inhibitor_perm_signal" # ABCD
            germination = integrate_ode(param_arr, times, 52, ode_system_ACD)
        elseif model_alias == "feedback_combined_inhibitor_thresh_signal_inducer_perm_thresh" # ABCE
            germination = integrate_ode(param_arr, times, 53, ode_system_AC)
        elseif model_alias == "feedback_combined_inhibitor_perm_thresh_inducer_perm_thresh" # ABDE
            germination = integrate_ode(param_arr, times, 54, ode_system_AD)
        elseif model_alias == "feedback_inducer_inhibitor_perm_thresh_signal_inducer_perm" # ACDEc
            germination = integrate_ode(param_arr, times, 55, ode_system_ACD)
        elseif model_alias == "feedback_combined_inhibitor_perm_thresh_signal_inducer_perm" # ACDE
            germination = integrate_ode(param_arr, times, 56, ode_system_ACD)
        elseif model_alias == "feedback_combined_inhibitor_perm_thresh_signal_inducer_thresh" # BCDE
            germination = integrate_ode(param_arr, times, 57, ode_system_D)
        elseif model_alias == "feedback_combined_inhibitor_perm_thresh_signal_inducer_perm_thresh" # ABCDE
            germination = integrate_ode(param_arr, times, 58, ode_system_ACD)
        else
            error("Unknown model alias: $model_alias")
        end

        return germination
    end

    """
    Compute the germination fraction for
    a dictionary of parameter values and a time series.
    inputs:
        model_alias (String) - alias of the germination model
        rho_s (Float32) - number density of spore colony (in um^(-1))
        times (Vector{Float32}) - time points to evaluate
        param_dict (Dict) - parameter dictionary, multiple values possible per key
    outputs:
        germination (Array{Float32}) - germination fraction over time for each parameter set
    """
    function compute_germination(model_alias, rho_s, times, param_dict)
        param_arr = param_dict_to_matrix(param_dict, rho_s)
        germination = model_wrapper(model_alias, param_arr, times)
        return germination
    end

end