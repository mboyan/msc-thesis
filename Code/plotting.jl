module Plotting
__precompile__(false)
    """
    Contains plotting functions
    """

    using ArgCheck
    using PyPlot
    using GLMakie
    using GeometryBasics
    using Revise

    include("./setup.jl")
    include("./conversions.jl")
    # using .Setup
    # using .Conversions
    Revise.includet("./conversions.jl")
    Revise.includet("./setup.jl")
    using .Conversions
    using .Setup

    export generate_ax_grid_pyplot
    export generate_grid_layout_glmakie
    export plot_spheres!
    export plot_spore_clusters
    export plot_concentration_lattice
    export plot_concentration_evolution_hi_res
    

    function generate_ax_grid_pyplot(n_rows, n_cols, figsize=(8, 4))
        """
        Generates a grid of axes for PyPlot.
        inputs:
            n_rows (int): number of rows
            n_cols (int): number of columns
            figsize (Tuple): figure size
        """
        fig, axs = subplots(n_rows, n_cols, figsize=figsize)
        axs = reshape(axs, length(axs))
        return fig, axs
    end


    function generate_grid_layout_glmakie(n_rows, n_cols, figsize, _3D=true)
        """
        Generates a grid layout for GLMakie.
        inputs:
            n_rows (int): number of rows
            n_cols (int): number of columns
            figsize (Tuple): figure size
            _3D (bool): whether to plot in 3D
        """
        fig = GLMakie.Figure(size=figsize)
        if _3D
            axs = Array{Axis3}(undef, n_rows, n_cols)
        else
            axs = Array{Axis}(undef, n_rows, n_cols)
        end
        for i in 1:n_rows
            for j in 1:n_cols
                if _3D
                    axs[i, j] = Axis3(fig[i, j], viewmode=:fit, aspect=(1.0, 1.0, 1.0))
                    scale!(axs[i, j].scene, 1.0, 1.0, 2.5)
                else
                    axs[i, j] = Axis(fig[i, j])
                end
            end
        end
        axs = reshape(axs, length(axs))
        return fig, axs
    end


    function plot_spheres!(centers, rad, L; inline=true, title=nothing, ax=nothing::Union{Axis, Nothing})
        """
        Plots spheres in an interactive 3D plot.
        inputs:
            centers (Array{Float64, 2}): centers of the spheres
            rads (Float64): radius of the spheres
            L (int): the size of the domain
            inline (bool): whether to display the plot inline
            title (str): title of the plot
            ax (Axis): axis to plot on
        """

        if inline
            GLMakie.activate!()
            Makie.inline!(true)
        else
            GLMakie.activate!()
            Makie.inline!(false)
        end

        if isnothing(ax)
            fig = GLMakie.Figure()
            ax = Axis3(fig[1, 1], viewmode=:fit, aspect=(1.0, 1.0, 1.0), title=title)
            scale!(ax.scene, 1.0, 1.0, 2.5)
            plotself = true
        else
            ax.title = title
            fig = ax.scene
            plotself = false
        end
        
        ax.xgridvisible = true
        ax.ygridvisible = true
        ax.zgridvisible = true

        xlims!(ax, 0, L)
        ylims!(ax, 0, L)
        zlims!(ax, 0, L)
        
        cam = cam3d!(ax.scene, eyeposition=Vec3f(2*L, L÷2, L÷2), lookat=Vec3f(L÷2, L÷2, L÷2))

        for center in eachrow(centers)
            sphere = Sphere(Point3f(center[1], center[2], center[3]), rad)
            mesh!(ax.scene, sphere, color = RGBAf(0.993, 0.906, 0.145, 0.5))
        end
        
        if plotself
            display(fig)
        end

        return ax
    end


    function plot_spore_clusters(cluster_sizes, spore_rad, L, per_row=2; cut_half=false)
        """
        Create multiple plots for a range of spore cluster sizes.
        inputs:
            cluster_sizes (Array{Int}): sizes of the clusters
            spore_rad (float): radius of the spores
            L (int): size of the domain
            per_row (int): number of plots per row
        """

        n_rows = ceil(Int, length(cluster_sizes) / per_row)

        fig, axs = generate_grid_layout_glmakie(n_rows, per_row, (800, 400*n_rows), true)

        for (i, cluster_size) in enumerate(cluster_sizes)
            centers = setup_spore_cluster(cluster_size, L, spore_rad, cut_half)
            sample_sphere_center = centers[1, :]
            nbr_sphere_centers = centers[2:end, :]
            coverage = measure_coverage(sample_sphere_center, nbr_sphere_centers, spore_rad)
            plot_spheres!(centers, spore_rad, L, inline=true, title="Cluster size: $(size(nbr_sphere_centers)[1]) + 1, Q = $(round(coverage,digits=5))", ax=axs[i])
        end

        display(fig)
    end


    function plot_concentration_lattice(c_frames::Array{Float64}, dx; frame_indices=nothing, times=nothing, title=nothing)
        """
        Plots a 2D section of the concentration lattice.
        inputs:
            c_frames (Array{Float64}): concentration lattice frames
            dx (float): lattice spacing
            frame_indices (Array{Int}): indices of the frames to plot
            title (str): title of the plot
        """

        if !isnothing(frame_indices)
            @argcheck typeof(frame_indices) in [Array{Int}, Vector{Int}] "frame_indices must be an array of integers"
            println("Plotting frames: ", frame_indices)
            c_frames = c_frames[frame_indices, :, :]
        else
            c_frames = c_frames[[1, end], :, :]
        end

        n_rows = ceil(Int, size(c_frames)[1] / 2)
        N = size(c_frames)[2]
        H = size(c_frames)[3]

        vmin = minimum(c_frames)
        vmax = maximum(c_frames)
        
        fig, axs = subplots(n_rows, 2, figsize=(8, 4*n_rows))
        # Flatten axes
        axs = reshape(axs, length(axs))
        img = nothing
        for i in 1:size(c_frames)[1]
            if !isnothing(times) && !isnothing(frame_indices)
                time = round(times[frame_indices[i]], digits=4)
                ax_title = "t = $time s"
            else
                ax_title = "Frame $(frame_indices[i])"
            end
            img = axs[i].imshow(c_frames[i, :, :], cmap="viridis", interpolation="nearest", extent=[0, N*dx, 0, H*dx], vmin=vmin, vmax=vmax)
            axs[i].set_title(ax_title)
            axs[i].set_xlabel(@L_str"x [\\mu m]")
            axs[i].set_ylabel(@L_str"y [\\mu m]")
        end
        
        cax = fig.add_axes([0.1, -0.05, 0.85, 0.05])
        fig.colorbar(img, cax=cax, orientation="horizontal")
        fig.tight_layout()
        fig.suptitle(title)

        gcf()
    end


    function plot_concentration_evolution_hi_res(c_frames::Array{Float64}, times::Vector{Float64}, region_ids::Matrix{Int64})
        """
        Plots the evolution of the concentration in a high resolution simulation
        by identifying the cell wall region and taking the average concentration.
        inputs:
            c_frames (Array{Float64}): concentration lattice frames
            times (Array{Float64}): times
            region_ids (Array{Float64}): region IDs
        """

        @argcheck size(c_frames)[1] == size(times)[1] "c_frames and times must have the same lengths"

        # Add new axis to region_ids
        region_ids = reshape(region_ids, 1, size(region_ids)[1], size(region_ids)[2])

        # Mask the cell wall region and take the average concentration
        c_cell_wall = c_frames .* (region_ids .== 1)
        c_avg = sum(c_cell_wall, dims=(2, 3)) ./ sum(region_ids .== 1)

        fig, ax = subplots(1, 1, figsize=(8, 4))
        ax.plot(times, c_avg[:], label="Cell wall region")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Average concentration")
        ax.set_title("Average concentration in the cell wall region")
        ax.grid()

        gcf()
    end
end