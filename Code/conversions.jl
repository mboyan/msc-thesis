module Conversions
    """
    Contains conversion utilites
    """

    export mL_to_cubic_um
    export inverse_mL_to_cubic_um
    export inverse_cubic_um_to_mL
    export inverse_uL_to_mL
    export convert_D_to_Ps
    export convert_Ps_to_D

    function mL_to_cubic_um(mL)
        """
        Convert milliliters to micrometers cubed.
        inputs:
            mL (float): volume in milliliters
        """
        return mL * 1e12
    end

    function inverse_mL_to_cubic_um(mL_inv)
        """
        Convert inverse milliliters to inverse micrometers cubed.
        inputs:
            mL_inv (float): volume in inverse milliliters
        """
        return mL_inv * 1e-12
    end

    function inverse_cubic_um_to_mL(cubic_um_inv)
        """
        Convert inverse micrometers cubed to inverse milliliters.
        inputs:
            microns_cubed_inv (float): number density in inverse micrometers cubed
        """
        return cubic_um_inv * 1e12
    end

    function inverse_uL_to_mL(uL_inv)
        """
        Convert inverse milliliters to inverse micrometers cubed.
        inputs:
            uL_inv (float): number density in inverse microliters
        """
        return uL_inv * 1000
    end

    function convert_D_to_Ps(D, K, d)
        """
        Convert diffusion coefficient to permeability.
        inputs:
            D (float): diffusion coefficient in micrometers squared per second
            K (float): partition coefficient
            d (float): thickness of the membrane in micrometers
        """
        return D * K / d
    end

    function convert_Ps_to_D(Ps, K, d)
        """
        Convert permeability to diffusion coefficient.
        inputs:
            Ps (float): permeability in micrometers per second
            K (float): partition coefficient
            d (float): thickness of the membrane in micrometers
        """
        return Ps * d / K
    end
end