module Diffusion
    """
    Functions for solving the diffusion equation.
    """

    export diffusion_time_dependent_analytical_src, permeation_time_dependent_analytical, compute_permeation_constant
    
    # ===== ANALYTICAL SOLUTIONS =====
    function permeation_time_dependent_analytical(c_in, c_out, t, Ps, A, V; alpha=1.0)
        """
        Compute the concentration of a solute in a spore given the initial and external concentrations.
        inputs:
            c_in (float) - the initial concentration at the spore
            c_out (float) - the initial external concentration
            t (float) - time
            Ps (float) - the spore membrane permeation constant
            A (float) - the surface area of the spore
            V (float) - the volume of the spore
            alpha (float) - permeable fraction of the area; defaults to 1
        """
        tau = V / (alpha * A * Ps)
        c = c_out - (c_out - c_in) .* exp(-t / tau)
        return c
    end


    function diffusion_time_dependent_analytical_src(c_init, D, time, vol; dims=3)
        """
        Compute the analytical solution of the time-dependent diffusion equation at the source.
        inputs:
            c_init (float) - the initial concentration
            D (float) - the diffusion constant
            time (float) - the time at which the concentration is to be computed
            vol (float) - the volume of the initial concentration source
        """
        if dims == 2
            result = (vol * c_init) ^ (2/3) / (4*π*D*time)
        elseif dims == 3
            result = vol * c_init / np.power(4*π*D*time, 1.5)
        end
        result = [isnan(x) || isinf(x) ? c_init[i] : x for (i, x) in enumerate(result)]
        return result
    end


    function compute_permeation_constant(c_in_target, c_out, c_0, t, A, V; alpha=1.0)
        """
        Compute the permeation constant P_s given the constant external concentration,
        the initial concentration of the solute, the concentration of the solute at time t
        and the surface area and volume of the spore.
        inputs:
            c_in (float) - the concentration at the spore at time t
            c_out (float) - the constant external concentration
            c_0 (float) - the initial concentration of the solute at the spore
            t (float) - time
            A (float) - the surface area of the spore
            V (float) - the volume of the spore
            alpha (float) - permeable fraction of the area; defaults to 1
        """
        Ps = V / (alpha * A * t) * log((c_out - c_0) / (c_out - c_in_target))
        return Ps
    end

    # ===== NUMERICAL SOLUTIONS =====
end