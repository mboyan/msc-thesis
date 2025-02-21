module Setup
__precompile__(false)
    """
    Contains setup functions
    """

    using ArgCheck
    using IterTools

    # using .Conversions
    # using .Diffusion

    export setup_spore_cluster

    function setup_spore_cluster(n_nbrs::Int, N::Int, spore_rad::Float64, cut_half::Bool=false)
        """
        Compute the centers of a cluster of spheres,
        one placed in the center of the lattice and the rest
        placed at the vertices of a regular n_nbrs-gon.
        inputs:
            n_nbrs (int): number of neighbors
            N (int): number of spheres
            spore_rad (float): radius of the spores
            cut_half (bool): whether to cut the cluster in half
        outputs:
            spore_centers (Array): centers of the spores
        """

        @argcheck n_nbrs in [2, 3, 4, 6, 8, 12] "n_nbrs must be in [2, 3, 4, 6, 8, 12]"
        @argcheck N > 0 "N must be positive"

        spore_dia = 2 * spore_rad
        center = [N ÷ 2, N ÷ 2, N ÷ 2]

        spore_centers = zeros(n_nbrs + 1, 3)
        spore_centers[1, :] = center

        if n_nbrs == 2
            # Two neighbours at poles
            spore_centers[2, :] = [center[1], center[2], center[3] + spore_dia]
            spore_centers[3, :] = [center[1], center[2], center[3] - spore_dia]
        elseif n_nbrs == 3
            # Three neighbours at equilateral triangle
            for i in 1:3
                spore_centers[i + 1, :] = [center[1] + spore_dia * cos(2 * π * i / 3),
                                           center[2],
                                           center[3] + spore_dia * sin(2 * π * i / 3)]
            end
        elseif n_nbrs == 4
            # Four neighbours at tetrahedron vertices
            cp = spore_dia/sqrt(3)
            cm = -cp
            spore_centers[2, :] = [center[1] + cp, center[2] + cp, center[2] + cp]
            spore_centers[3, :] = [center[1] + cm, center[2] + cm, center[2] + cp]
            spore_centers[4, :] = [center[1] + cp, center[2] + cm, center[2] + cm]
            spore_centers[5, :] = [center[1] + cm, center[2] + cp, center[2] + cm]
        elseif n_nbrs == 6
            # Six neighbours at octahedron vertices
            coords = [spore_dia, 0, 0]
            for i in 1:3
                spore_centers[2*i, :] = center .+ coords
                spore_centers[2*i + 1, :] = center .- coords
                coords = circshift(coords, 1)
            end
        elseif n_nbrs == 8
            # Eight neighbours at cube vertices
            for i in 1:4
                spore_centers[2*i, :] = [center[1] + spore_dia * sqrt(2/3) * cos(2 * π * i / 4),
                                         center[2] + spore_dia * sqrt(2/3) * sin(2 * π * i / 4),
                                         center[3] + spore_dia * sqrt(1/3)]
                spore_centers[2*i + 1, :] = [center[1] + spore_dia * sqrt(2/3) * cos(2 * π * i / 4),
                                             center[2] + spore_dia * sqrt(2/3) * sin(2 * π * i / 4),
                                             center[3] - spore_dia * sqrt(1/3)]
            end
        elseif n_nbrs == 12
            # Twelve neighbours at icosahedron vertices
            phi = (1 + sqrt(5)) / 2
            norm_coeff = spore_dia / sqrt(phi + 2)
            coords = [0, norm_coeff, phi * norm_coeff]
            for i in 1:12
                coords[1 + mod1(i, 2)] = -coords[1 + mod1(i, 2)]
                coordshift = circshift(coords, floor(Int, i/4))
                spore_centers[i + 1, :] = center .+ coordshift
            end
        end

        # Take only the upper half of the cluster`
        if cut_half
            mask = spore_centers[:, 3] .≤ center[3]
            spore_centers = spore_centers[mask, :]
        end

        return spore_centers
    end

end