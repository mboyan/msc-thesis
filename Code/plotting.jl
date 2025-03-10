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
    using CurveFit

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
    export plot_concentration_evolution
    export compare_concentration_evolutions
    export compare_concentration_evolution_groups
    export plot_lattice_regions
    export plot_functional_relationship
    export compare_functional_relationships
    

    function generate_ax_grid_pyplot(n_rows, n_cols, figsize=(8, 4))
        """
        Generates a grid of axes for PyPlot.
        inputs:
            n_rows (int): number of rows
            n_cols (int): number of columns
            figsize (Tuple): figure size
        outputs:
            fig (Figure): figure object
            axs (Array): axes objects
        """
        fig, axs = subplots(n_rows, n_cols, figsize=figsize)
        axs = reshape(permutedims(axs, (2, 1)), length(axs))
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
        outputs:
            fig (Figure): figure object
            axs (Array): axes objects
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
        axs = reshape(permutedims(axs, (2, 1)), length(axs))
        return fig, axs
    end


    function plot_spheres!(centers, rad, L; inline=true, title=nothing, ax=nothing::Union{Axis, Nothing})
        """
        Plots spheres in an interactive 3D plot.
        inputs:
            centers (Array{Tuple}): centers of the spheres
            rads (Float64): radius of the spheres
            L (int): the size of the domain
            inline (bool): whether to display the plot inline
            title (str): title of the plot
            ax (Axis): axis to plot on
        outputs:
            ax (Axis): axis object
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

        for center in centers
            sphere = Sphere(Point3f(center[1], center[2], center[3]), rad)
            mesh!(ax.scene, sphere, color = RGBAf(0.993, 0.906, 0.145, 0.5))
        end
        
        if plotself
            display(fig)
        end

        return ax
    end


    function plot_spore_clusters(cluster_sizes, spore_rad, L, per_row=2; cut_half=false, spore_spacings=nothing)
        """
        Create multiple plots for a range of spore cluster sizes.
        inputs:
            cluster_sizes (Array{Int}): sizes of the clusters
            spore_rad (float): radius of the spores
            L (int): size of the domain
            per_row (int): number of plots per row
            cut_half (bool): whether to cut the cluster in half
            spore_spacings (Array{Float64}): distances between spores, if unspecified, spore diameter is used
        """

        n_rows = ceil(Int, length(cluster_sizes) / per_row)

        if isnothing(spore_spacings)
            spore_spacings = [spore_rad for s in cluster_sizes]
        end

        fig, axs = generate_grid_layout_glmakie(n_rows, per_row, (400*per_row, 400*n_rows), true)

        for (i, cluster_size) in enumerate(cluster_sizes)
            centers = setup_spore_cluster(cluster_size, L, spore_spacings[i]*0.5, cut_half)
            sample_sphere_center = centers[1]
            nbr_sphere_centers = centers[2:end]
            coverage = measure_coverage(sample_sphere_center, nbr_sphere_centers, spore_spacings[i]*0.5)
            plot_spheres!(centers, spore_rad, L, inline=true, title="Cluster size: $(size(nbr_sphere_centers)[1]) + 1, Q = $(round(coverage,digits=5))", ax=axs[i])
        end

        display(fig)
    end


    function plot_concentration_lattice(c_frames::Array{Float64}, dx; frame_indices=nothing, times=nothing, title=nothing, logscale=false)
        """
        Plots a 2D section of the concentration lattice.
        inputs:
            c_frames (Array{Float64}): concentration lattice frames
            dx (float): lattice spacing
            frame_indices (Array{Int}): indices of the frames to plot
            times (Array{Float64}): times
            title (str): title of the plot
            logscale (bool): whether to plot the colorbar in log scale
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
        axs = reshape(permutedims(axs, (2, 1)), length(axs))
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


    function plot_concentration_evolution(c_vals::Array{Float64}, times::Vector{Float64}, label=nothing, ax=nothing, logy=false, fit_exp=false, cmap=nothing, cmap_idx=1)
        """
        Plots the time-series of a calculated concentration.
        inputs:
            c_vals (Array{Float64}): concentration lattice frames
            times (Array{Float64}): times
            region_ids (Array{Float64}): region IDs
            spore_rad (float): spore radius
            cw_thickness (float): cell wall thickness
            dx (float): lattice spacing
            ax (Axis): axis to plot on
            logy (bool): whether to plot the y-axis in log scale
            fit_exp (bool): whether to fit an exponential to the data
            cmap (str): colormap
            cmap_idx (int): index of the colormap
        """
        
        if isnothing(ax)
            fig, ax = subplots(1, 1, figsize=(8, 4))
            plotself = true
        else
            plotself = false
        end

        if fit_exp
            fit = exp_fit(times, c_vals)
            println("Fitted exponential: ", fit)
            fit_vals = fit[1] .* exp.(fit[2] .* times)
            ax.plot(times, fit_vals, color="red", linestyle="--")
        end

        if isnothing(cmap)
            ax.plot(times, c_vals, label=label)
        else
            ax.plot(times, c_vals, label=label, color=cmap(cmap_idx))
        end

        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Concentration [M]")
        ax.set_title("Total concentration in the spore")
        ax.grid(true)

        if logy
            ax.set_yscale("log")
        end

        if plotself
            gcf()
        end
    end

    
    function compare_concentration_evolutions(c_vals_array:: Vector{Vector{Float64}}, times_array::Vector{Vector{Float64}}, labels=nothing, ax=nothing; logy=false, fit_exp=false, cmap=nothing, cmap_idx_base=0)
        """
        Plot multiple concentration evolutions on the same axis.
        inputs:
            c_vals_array (Array{Float64, 2}): concentration lattice frames
            times_array (Array{Vector{Float64}}): times
            labels (Array{String}): labels for the plots
            ax (Axis): axis to plot on
            logy (bool): whether to plot the y-axis in log scale
            fit_exp (bool): whether to fit an exponential to the data
            cmap (str): colormap
            cmap_idx_base (int): base index of the colormap
        """

        # Check labels
        if !isnothing(labels)
            @argcheck length(labels) == size(c_vals_array)[1] "Number of labels must match the number of concentration arrays"
        else
            labels = ["$i" for i in 1:size(c_vals_array)[1]]
        end

        if isnothing(ax)
            plotself = true
            fig, ax = subplots(1, 1, figsize=(8, 4))
        else
            plotself = false
        end
        
        for i in 1:size(c_vals_array)[1]
            println(labels[i])
            plot_concentration_evolution(c_vals_array[i], times_array[i], labels[i], ax, logy, fit_exp, cmap, cmap_idx_base+i)
        end

        if plotself
            ax.legend(fontsize="small")
            gcf()
        end

    end


    function compare_concentration_evolution_groups(c_groups, times_groups, group_labels=nothing, ax=nothing; logy=false, fit_exp=false)
        """
        Compare the concentration evolutions from groups of simulations
        on the same axis, with corresponding colors.
        inputs:
            c_groups (Array{Array{Float64, 2}}): concentration lattice frames
            times_groups (Array{Array{Float64}}): times
            group_labels (Array{String}): labels for the groups
            logy (bool): whether to plot the y-axis in log scale
            fit_exp (bool): whether to fit an exponential to the data
        """

        # Check labels
        if !isnothing(group_labels)
            @argcheck length(group_labels) == size(c_groups)[1] "Number of labels must match the number of concentration groups"
        else
            group_labels = [["Group $i"] for i in 1:size(c_groups)[1]]
        end

        if isnothing(ax)
            plotself = true
            fig, ax = subplots(1, 1, figsize=(8, 4))
        else
            plotself = false
        end

        cmap = get_cmap("tab20c")
        for i in 1:size(c_groups)[1]
            println(group_labels[i])
            compare_concentration_evolutions(c_groups[i], times_groups[i], group_labels[i], ax; logy, fit_exp, cmap, cmap_idx_base=i*4)
        end

        if plotself
            ax.legend(fontsize="small")
            gcf()
        end
    end


    function plot_lattice_regions(region_ids)
        """
        Plots the exterior, cell wall and interior lattice regions.
        inputs
            region_ids (Array{Int}): region IDs
        """

        N, H = size(region_ids)
        fig, ax = subplots(1, 1, figsize=(8, 4))
        ax.imshow(region_ids, cmap="viridis", interpolation="nearest")
        ax.set_title("Lattice regions")
        ax.set_xlabel(@L_str"i")
        ax.set_ylabel(@L_str"j")
        
        gcf()
    end


    function plot_functional_relationship(input, response, axlabels, title=nothing, label=nothing; ax=nothing, logx=false, logy=false, fit=nothing)
        """
        Plots the functional relationship between input and response.
        inputs:
            input (Array{Float64}): input values
            response (Array{Float64}): response values
            axlabels (Array{String}): axis labels
            title (str): title of the plot
            label (str): label for the plot
            ax (Axis): axis to plot on
            logx (bool): whether to plot the x-axis in log scale
            logy (bool): whether to plot the y-axis in log scale
            fit (str): type of fit to perform
        """

        if isnothing(ax)
            fig, ax = subplots(1, 1, figsize=(6, 4))
        end
        
        ax.plot(input, response, marker="o", label=label)

        if fit == "lin"
            fit = linear_fit(input, response)
            println("Fitted linear: ", fit)
            fit_vals = fit[1] .* input .+ fit[2]
        elseif fit == "exp"
            fit = exp_fit(input, response)
            println("Fitted exponential: ", fit)
            fit_vals = fit[1] .* exp.(fit[2] .* input)
        elseif fit == "pow"
            input_clean = input[input .> 0]
            response_clean = response[input .> 0]
            fit = power_fit(input_clean, response_clean)
            println("Fitted power: ", fit)
            fit_vals = fit[1] .* input .^ fit[2]
        elseif fit == "log"
            fit = log_fit(input, response)
            println("Fitted logarithmic: ", fit)
            fit_vals = fit[1] .+ fit[2] .* log.(input)
        elseif !isnothing(fit)
            println("Invalid fit type, must be one of: lin, exp, pow, log")
        end
        if !isnothing(fit)
            ax.plot(input, fit_vals, color="red", linestyle="--")
        end

        ax.set_xlabel(axlabels[1])
        ax.set_ylabel(axlabels[2])
        ax.grid(true)

        if logx
            ax.set_xscale("log")
        end

        if logy
            ax.set_yscale("log")
        end

        if !isnothing(title)
            ax.set_title(title)
        end

        gcf()
    end

    
    function compare_functional_relationships(inputs, responses, axlabels, plotlabels=nothing, title=nothing; logx=false, logy=false, fit=nothing)
        """
        Compare multiple functional relationships on the same axis.
        inputs:
            inputs (Array{Array{Float64}}): input values
            responses (Array{Array{Float64}}): response values
            axlabels (Array{String}): axis labels
            plotlabels (Array{String}): labels for the plots
            titles (Array{String}): titles for the plots
            logx (bool): whether to plot the x-axis in log scale
            logy (bool): whether to plot the y-axis in log scale
            fits (Array{String}): types of fits to perform
        """

        # Check labels
        if !isnothing(plotlabels)
            @argcheck length(plotlabels) == size(inputs)[1] "Number of titles must match the number of input-response pairs"
        else
            titles = ["$i" for i in 1:size(inputs)[1]]
        end

        plotself = true
        fig, ax = subplots(figsize=(8, 4))

        for i in 1:size(inputs)[1]
            println(responses[i])
            plot_functional_relationship(inputs[i], responses[i], axlabels, title, plotlabels[i]; ax, logx, logy, fit)
        end

        ax.legend(fontsize="small")
        gcf()
    end
end