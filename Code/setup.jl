module Setup
__precompile__(false)
        """
    Contains setup functions
    """

    using IterTools

    # using .Conversions
    # using .Diffusion

    export setup_spore_cluster

    function setup_spore_cluster(n_nbrs::Int, N::Int, spore_rad::Float64)
        """
        Compute the centers of a cluster of spheres,
        one placed in the center of the lattice and the rest
        placed at the vertices of a regular n_nbrs-gon.
        inputs:
            n_nbrs (int): number of neighbors
            N (int): number of spheres
        """

        @assert n_nbrs in [2, 3, 4, 6, 8, 12], "n_nbrs must be in [2, 3, 4, 6, 8, 12]"
        @assert N > 0, "N must be positive"

        spore_dia = 2 * spore_rad
        center = [N ÷ 2, N ÷ 2, N ÷ 2]

        spore_centers = zeros(N + 1, 3)
        spore_centers[1, :] = center

        if n_nbrs == 2
            # Two neighbours at poles
            spore_centers[2, :] = [center[1], center[2], center[3] + spore_rad]
            spore_centers[3, :] = [center[1], center[2], center[3] - spore_rad]
        elseif n_nbrs == 3
            # Three neighbours at equilateral triangle
            for i in 1:3
                spore_centers[i + 1, :] = [center[1] + spore_rad * cos(2 * π * i / 3),
                                           center[2] + spore_rad * sin(2 * π * i / 3),
                                           center[3]]
            end
        elseif n_nbrs == 4
            # Four neighbours at tetrahedron vertices
            cp = spore_rad/sqrt(3)
            cm = -cp
            spore_centers[2, :] = [cp, cp, cp]
            spore_centers[3, :] = [cm, cm, cp]
            spore_centers[4, :] = [cp, cm, cm]
            spore_centers[5, :] = [cm, cp, cm]
        elseif n_nbrs == 6
            # Six neighbours at octahedron vertices
            coords = [1, 0, 0]
            for i in 1:3
                spore_centers[2*i + 1, :] = coords
                spore_centers[2*(i + 1), :] = -coords
                coords = circshift(coords, 1)
            end
        elseif n_nbrs == 8
            # Eight neighbours at cube vertices
            for i in 1:4
                spore_centers[2*i + 1, :] = [center[1] + spore_rad * cos(2 * π * i / 4),
                                         center[2] + spore_rad * sin(2 * π * i / 4),
                                         center[3] + spore_rad]
                spore_centers[2*(i + 1), :] = [center[1] + spore_rad * cos(2 * π * i / 4),
                                             center[2] + spore_rad * sin(2 * π * i / 4),
                                             center[3] - spore_rad]
            end
        elseif n_nbrs == 12
            # Twelve neighbours at icosahedron vertices
            phi = (1 + sqrt(5)) / 2
            pm = product([1, -1], 2)
            coords = [0, spore_rad, phi * spore_rad]
            for i in 1:12
                coords[floor(Int, i/4) + 1 + mod1(i, 2)] = -coords[floor(Int, i/4) + 1 + mod1(i, 2)]
                coordshift = circshift(coords, floor(Int, i/4))
                spore_centers[i + 1, :] = coordshift
            end
        end

        return spore_centers
    end

end