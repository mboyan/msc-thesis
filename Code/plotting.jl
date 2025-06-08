module Plotting
__precompile__(false)
    """
    Contains plotting functions
    """

    using ArgCheck
    using PyPlot
    using PyCall
    using GLMakie
    using GeometryBasics
    using Revise
    using CurveFit
    using Printf

    include("./setup.jl")
    include("./conversions.jl")
    include("./datautils.jl")
    include("./germstats.jl")
    Revise.includet("./conversions.jl")
    Revise.includet("./setup.jl")
    Revise.includet("./datautils.jl")
    Revise.includet("./germstats.jl")
    using .Conversions
    using .Setup
    using .DataUtils
    using .GermStats

    export generate_ax_grid_pyplot
    export generate_grid_layout_glmakie
    export plot_spheres!
    export plot_spore_clusters
    export plot_concentration_lattice
    export compare_concentration_lattice
    export plot_concentration_evolution
    export compare_concentration_evolutions
    export compare_concentration_evolution_groups
    export plot_lattice_regions
    export plot_functional_relationship
    export compare_functional_relationships
    export compare_functional_relationships_groups
    export plot_spore_positions
    export plot_spore_arrangements
    export plot_dantigny_time_course
    export compare_time_course_to_dantigny
    export plot_germination_data_fit
    export plot_germination_data_fit_all


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


    function plot_spore_clusters(cluster_sizes, spore_rad, L, per_row=2; cut_half=false, spore_spacings=nothing, measure_shielding=false)
        """
        Create multiple plots for a range of spore cluster sizes.
        inputs:
            cluster_sizes (Array{Int}): sizes of the clusters
            spore_rad (float): radius of the spores
            L (int): size of the domain
            per_row (int): number of plots per row
            cut_half (bool): whether to cut the cluster in half
            spore_spacings (Array{Float64}): distances between spores, if unspecified, spore diameter is used
            shielding (bool): whether to label the shielding index, otherwise labels the coverage
        """

        n_rows = ceil(Int, length(cluster_sizes) / per_row)

        if isnothing(spore_spacings)
            spore_spacings = [spore_rad*2 for s in cluster_sizes]
        end

        fig, axs = generate_grid_layout_glmakie(n_rows, per_row, (400*per_row, 400*n_rows), true)

        for (i, cluster_size) in enumerate(cluster_sizes)
            centers = setup_spore_cluster(cluster_size, L, spore_spacings[i]*0.5, cut_half)
            sample_sphere_center = centers[1]
            nbr_sphere_centers = centers[2:end]
            if measure_shielding
                shielding = measure_shielding_index(sample_sphere_center, Array{Tuple}(nbr_sphere_centers))
                label = "Cluster size: $(size(nbr_sphere_centers)[1]) + 1, S = $(round(shielding,digits=5))"
            else
                coverage = measure_coverage(sample_sphere_center, Array{Tuple}(nbr_sphere_centers); rad=spore_spacings[i]*0.5)
                label = "Cluster size: $(size(nbr_sphere_centers)[1]) + 1, Q = $(round(coverage,digits=5))"
            end
            plot_spheres!(centers, spore_rad, L, inline=true, title=label, ax=axs[i])
        end

        display(fig)
    end


    function plot_concentration_lattice(c_frames::Array{Float64}, dx; frame_indices=nothing, times=nothing, title=nothing, zoom=1.0)
        """
        Plots a 2D section of the concentration lattice.
        inputs:
            c_frames (Array{Float64}): concentration lattice frames
            dx (float): lattice spacing
            frame_indices (Array{Int}): indices of the frames to plot
            times (Array{Float64}): time labels
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
        
        n_rows = ceil(Int, size(c_frames, 1) / 2)
        N = size(c_frames, 2)
        H = size(c_frames, 3)

        vmin = minimum(c_frames)
        vmax = maximum(c_frames)
        
        fig, axs = subplots(n_rows, 2, figsize=(8, 4*n_rows))
        if ndims(axs) == 2
            axs = reshape(permutedims(axs, (2, 1)), length(axs))
        else
            axs = reshape(axs, (1, length(axs)))
        end
        img = nothing
        for i in 1:size(c_frames, 1)
            if !isnothing(times) && !isnothing(frame_indices)
                time = round(times[frame_indices[i]], digits=4)
                ax_title = "t = $time s"
            else
                ax_title = "Frame $i"
            end
            img = axs[i].imshow(c_frames[i, :, :], cmap="viridis", interpolation="nearest", extent=[0, N*dx, 0, H*dx], vmin=vmin, vmax=vmax)
            axs[i].set_title(ax_title)
            axs[i].set_xlabel(@L_str"x [\\mu m]")
            axs[i].set_ylabel(@L_str"y [\\mu m]")
            axs[i].set_xlim((1-zoom)*0.5*N*dx, (1+zoom)*0.5*N*dx)
            axs[i].set_ylim((1-zoom)*0.5*H*dx, (1+zoom)*0.5*H*dx)
        end
        
        cax = fig.add_axes([0.1, -0.05, 0.85, 0.05])
        fig.colorbar(img, cax=cax, orientation="horizontal")
        fig.tight_layout()
        fig.suptitle(title)

        gcf()
    end


    function compare_concentration_lattice(c_frames_compare, dx_compare; frame_indices=nothing, times=nothing, title=nothing, zoom=1.0, lognorm=false)
        """
        Compares snapshots of the concentration lattice from two simulations.
        The simulations need to share the same time labels.
        inputs:
            c_frames_array (Vector{Array{Float64}}): concentration lattice frames for each simulation
            dx_compare (float): lattice spacings for each simulation
            frame_indices (Array{Int}): indices of the frames to plot for each simulation
            times (Array{Float64}): time labels for each simulation
            title (str): title of the plot
            zoom (float): zoom factor for the plot
            lognorm (bool): whether to use logarithmic normalization for the color scale
        """
        
        c_frames_filtered = []
        for c_frames in c_frames_compare
            if !isnothing(frame_indices)
                @argcheck typeof(frame_indices) in [Array{Int}, Vector{Int}] "frame_indices must be an array of integers"
                println("Plotting frames: ", frame_indices)
                push!(c_frames_filtered, c_frames[frame_indices, :, :])
            else
                frame_indices = [1, size(c_frames, 1)]
                println("Plotting frames: ", frame_indices)
                push!(c_frames_filtered, c_frames[frame_indices, :, :])
            end
        end

        # n_rows = size(c_frames_filtered, 1)
        n_rows = length(c_frames_filtered)
        n_cols = size(c_frames_filtered[1], 1)

        # fig, axs = subplots(n_rows, size(c_frames_filtered[1], 1), figsize=(10, 5*n_rows))
        fig, axs = subplots(n_rows, n_cols, figsize=(8, (8/n_cols)*n_rows), sharey=true)
        if n_rows < 2
            axs = reshape(axs, (1, length(axs)))
        end
        img = nothing
        for i in 1:n_rows
            N = size(c_frames_filtered[i], 2)
            H = size(c_frames_filtered[i], 3)
            vmin = minimum(c_frames_filtered[i])
            vmax = maximum(c_frames_filtered[i])
            if lognorm
                norm = "log"
                vmin = max(vmin, 1e-8)  # Avoid log(0)
                c_frames_filtered[i] .= max.(1e-8, c_frames_filtered[i])  # Avoid log(0) in the data
            else
                norm = "linear"
            end
            for j in 1:size(c_frames_filtered[i], 1)
                if !isnothing(times) && !isnothing(frame_indices)
                    time = round(times[frame_indices[j]], digits=4)
                    # ax_title = "Configuration $i, t = $time s"
                    ax_title = L"t = %$time\ \text{s}"
                else
                    ax_title = "Frame $j"
                end
                img = axs[i, j].imshow(c_frames_filtered[i][j, :, :], cmap="viridis", interpolation="nearest", extent=[0, N*dx_compare[i], 0, H*dx_compare[i]], vmin=vmin, vmax=vmax, norm=norm)
                axs[i, j].set_title(ax_title)
                if i == n_rows 
                    axs[i, j].set_xlabel(@L_str"x\\ [\\mu m]")
                end
                if j == 1
                    axs[i, j].set_ylabel(@L_str"y\\ [\\mu m]")
                end
                axs[i, j].set_xlim((1-zoom)*0.5*N*dx_compare[i], (1+zoom)*0.5*N*dx_compare[i])
                axs[i, j].set_ylim((1-zoom)*0.5*H*dx_compare[i], (1+zoom)*0.5*H*dx_compare[i])
            end
            cax = fig.add_axes([
                axs[i, end].get_position().x1 + 0.02, 
                axs[i, end].get_position().y0 + 0.005, 
                0.02, 
                axs[i, end].get_position().height - 0.01
            ])
            fig.colorbar(img, cax=cax, orientation="vertical", label=L"\mu\text{M}")
        end

        fig.suptitle(title)
        fig.subplots_adjust(wspace=0.2)
        fig.subplots_adjust(hspace=0.5)

        # fig.tight_layout()

        gcf()
    end


    function plot_concentration_evolution(c_vals::Array{Float64}, times::Vector{Float64};
                                            label=nothing, ax=nothing, logx=false, logy=false, fit_lim=nothing, cmap=nothing,
                                            cmap_idx=1, time_cutoff=nothing, title=nothing, ylim=nothing, dashed=true)
        """
        Plots the time-series of a calculated concentration.
        inputs:
            c_vals (Array{Float64}): concentrations
            times (Array{Float64}): measurement times
            label (string): plot label
            ax (Axis): axis to plot on
            logx (bool): whether to plot the x-axis in log scale
            logy (bool): whether to plot the y-axis in log scale
            fit_lim (tuple): time interval for exponential fit
            cmap (str): colormap
            cmap_idx (int): index of colour in colormap
            time_cutoff (float): time cutoff for the plot
            title (str): title of the plot
            ylim (Tuple): y-axis limits
            dashed (bool): whether to plot with dashed lines
        """
        
        if isnothing(ax)
            fig, ax = subplots(1, 1, figsize=(8, 4))
            plotself = true
        else
            plotself = false
        end

        if !isnothing(time_cutoff)
            time_cutoff_idx = findfirst(x -> x > time_cutoff, times)
            c_vals = c_vals[1:time_cutoff_idx]
            times = times[1:time_cutoff_idx]
        end

        if !isnothing(fit_lim)
            times_mask = (times .> fit_lim[1]) .& (times .< fit_lim[2])
            fit = exp_fit(times[times_mask], c_vals[times_mask])
            println("Fitted exponential: ", fit)
            fit_vals = fit[1] .* exp.(fit[2] .* times)
            ax.plot(times, fit_vals, color="red", linestyle="--")
        end

        if isnothing(cmap)
            if dashed
                ax.plot(times, c_vals, label=label, linestyle="--", dashes=(rand(1:5), rand(1:5)), alpha=0.9)
            else
                ax.plot(times, c_vals, label=label, alpha=0.9)
            end
        else
            ax.plot(times, c_vals, label=label, color=cmap(cmap_idx), alpha=0.75)#, marker="o")
        end

        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Concentration [M]")
        if isnothing(title)
            ax.set_title("Concentration evolution")
        else
            ax.set_title(title)
        end
        ax.grid(true)
        ax.legend()

        if logx
            ax.set_xscale("log")
            ax.set_xlim(1e-2, maximum(times))
        end

        if logy
            ax.set_yscale("log")
        end

        if !isnothing(ylim)
            ax.set_ylim(ylim[1], ylim[2])
        end

        if plotself
            gcf()
        end
    end


    function to_nested(collection)
        if ndims(collection) == 1
            return collect(collection)
        elseif ndims(collection) == 2
            return [to_nested(collection[i, :]) for i in 1:size(collection, 1)]
        elseif ndims(collection) == 3
            return [to_nested(collection[i, :, :]) for i in 1:size(collection, 1)]
        else
            println("Maximum depth of 3 reached")
            return collection
        end
    end

    
    function compare_concentration_evolutions(c_vals_array, times_array, labels=nothing, ax=nothing;
                                                logx=false, logy=false, fit_lim=nothing, cmap=nothing, cmap_idx_base=0, title=nothing,
                                                time_cutoff=nothing, ylim=nothing, legend_loc=nothing, dashed=true)
        """
        Plot multiple concentration evolutions on the same axis.
        inputs:
            c_vals_array (Array{Float64, 2}): concentration lattice frames
            times_array (Array{Vector{Float64}}): times
            labels (Array{String}): labels for the plots
            ax (Axis): axis to plot on
            logx (bool): whether to plot the x-axis in log scale
            logy (bool): whether to plot the y-axis in log scale
            fit_lim (tuple): time interval for exponential fit
            cmap (str): colormap
            cmap_idx_base (int): base index of the colormap
            title (str): title of the plot
            time_cutoff (float): time cutoff for the plot
            ylim (Tuple): y-axis limits
            legend_loc (str): location of the legend
            dashed (bool): whether to plot with dashed lines
        """

        # @argcheck (typeof(c_vals_array) in [Vector{Vector{Float64}}, Matrix{Float64}]) "c_groups must be a vector of matrices or a matrix"
        # @argcheck (typeof(times_array) in [Vector{Vector{Float64}}, Matrix{Float64}]) "times_groups must be a vector of matrices or a matrix"

        # Convert to nested arrays
        c_vals_array = to_nested(c_vals_array)
        times_array = to_nested(times_array)

        # Check labels
        if !isnothing(labels)
            @argcheck length(labels) == size(c_vals_array, 1) "Number of labels must match the number of concentration arrays"
        else
            labels = ["$i" for i in 1:size(c_vals_array)[1]]
        end

        if isnothing(ax)
            plotself = true
            fig, ax = subplots(1, 1, figsize=(8, 4))
        else
            plotself = false
        end
        
        for i in eachindex(c_vals_array)
            plot_concentration_evolution(c_vals_array[i], times_array[i]; label=labels[i], ax, logx, logy, fit_lim, cmap,
                                        cmap_idx=cmap_idx_base+i-1, time_cutoff, title, ylim, dashed)
        end

        if plotself
            ax.legend(fontsize="small", loc=legend_loc)
            gcf()
        end

    end


    function compare_concentration_evolution_groups(c_groups, times_groups, group_labels=nothing, ax=nothing;
                                                    logx=false, logy=false, fit_lim=nothing, title=nothing, time_cutoff=nothing, ylim=nothing, legend_loc=nothing)
        """
        Compare the concentration evolutions from groups of simulations
        on the same axis, with corresponding colors.
        inputs:
            c_groups (Array{Array{Float64, 2}}): concentration lattice frames
            times_groups (Array{Array{Float64}}): times
            group_labels (Array{String}): labels for the groups
            ax (Axis): axis to plot on
            logx (bool): whether to plot the x-axis in log scale
            logy (bool): whether to plot the y-axis in log scale
            fit_lim (tuple): time interval for exponential fit
            title (str): title of the plot
            time_cutoff (float): time cutoff for the plot
            ylim (Tuple): y-axis limits
            legend_loc (str): location of the legend
        """

        # @argcheck (typeof(c_groups) in [Vector{Vector{Vector{Float64}}}, Matrix{Float64}, Array{Float64, 3}]) "c_groups must be a vector of matrices or a matrix"
        # @argcheck (typeof(times_groups) in [Vector{Vector{Vector{Float64}}}, Matrix{Float64}, Array{Float64, 3}]) "times_groups must be a vector of matrices or a matrix"

        # Convert to nested arrays
        c_groups = to_nested(c_groups)
        times_groups = to_nested(times_groups)

        # Check labels
        if isnothing(group_labels)
            group_labels = [["Group $i" for j in 1:size(c_groups[i], 1)] for i in 1:size(c_groups, 1)]
        else
            group_labels = to_nested(group_labels)
        end

        if isnothing(ax)
            plotself = true
            fig, ax = subplots(1, 1, figsize=(8, 4))
        else
            plotself = false
        end

        cmap = get_cmap("tab20c")
        for i in eachindex(c_groups)
            compare_concentration_evolutions(c_groups[i], times_groups[i], group_labels[i], ax; logx, logy, fit_lim, cmap, cmap_idx_base=(i - 1)*4, title=title, time_cutoff=time_cutoff, ylim=ylim, legend_loc=legend_loc)
        end

        if plotself
            if length(collect(group_labels)) > 4
                ax.legend(fontsize="small", loc="upper left", bbox_to_anchor=(1.05, 1))
            else
                ax.legend(fontsize="small", loc=legend_loc)
            end
            gcf()
        end
    end


    function plot_lattice_regions(region_ids; ax=nothing, zoom=1.0)
        """
        Plots the exterior, cell wall and interior lattice regions.
        inputs
            region_ids (Array{Int}): region IDs
            ax (Axis): axis to plot on
            zoom (float): zoom factor for the plot
        """

        N, H = size(region_ids)

        if isnothing(ax)
            plotself = true
            fig, ax = subplots(1, 1, figsize=(8, 4))
            ax.set_title("Lattice regions")
        else
            plotself = false
        end

        img = ax.imshow(region_ids, cmap="rainbow", interpolation="nearest", vmin=0, vmax=2)
        ax.set_xlabel(@L_str"i")
        ax.set_ylabel(@L_str"j")
        ax.set_xlim((1-zoom)*0.5*N, (1+zoom)*0.5*N)
        ax.set_ylim((1-zoom)*0.5*H, (1+zoom)*0.5*H)

        # Create a discrete color legend
        Patch = pyimport("matplotlib.patches").Patch
        legend_labels = ["Exterior", "Inhibitor space", "Interior"]
        unique_ids = unique(region_ids)
        cmap = get_cmap("rainbow")
        patches = [Patch(color=cmap((i-1)/2.0), label=legend_labels[i]) for i in eachindex(unique_ids)]
        ax.legend(handles=patches, loc="upper right", fontsize="small")
        
        if plotself
            gcf()
        end
    end


    function plot_functional_relationship(input, response, axlabels, title=nothing, label=nothing; ax=nothing, logx=false, logy=false, fit=nothing, cmap=nothing, cmap_idx=1, scatter=false)
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
            cmap (str): colormap
            cmap_idx (int): index of colour in colormap
            scatter (bool): whether to plot as scatter points
        """

        if isnothing(ax)
            fig, ax = subplots(1, 1, figsize=(6, 4))
        end
        
        if isnothing(cmap)
            if scatter
                ax.scatter(input, response, label=label)
            else
                ax.plot(input, response, marker="o", label=label)
            end
        else
            if scatter
                ax.scatter(input, response, label=label, color=cmap(cmap_idx), alpha=0.75)
            else
                ax.plot(input, response, marker="o", label=label, color=cmap(cmap_idx), alpha=0.75)
            end
        end

        if !isnothing(label)
            println(label)
        end
        
        if fit == "lin"
            fit = linear_fit(input, response)
            println("Fitted linear: ", fit)
            fit_vals = fit[2] .* input .+ fit[1]
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
        ax.grid(true, which="both")

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

    
    function compare_functional_relationships(inputs, responses, axlabels, plotlabels=nothing, title=nothing; ax=nothing, logx=false, logy=false, fit=nothing, cmap=nothing, cmap_idx_base=0)
        """
        Compare multiple functional relationships on the same axis.
        inputs:
            inputs (Array{Array{Float64}}): input values
            responses (Array{Array{Float64}}): response values
            axlabels (Array{String}): axis labels
            plotlabels (Array{String}): labels for the plots
            titles (Array{String}): titles for the plots
            ax (Axis): axis to plot on
            logx (bool): whether to plot the x-axis in log scale
            logy (bool): whether to plot the y-axis in log scale
            fit (String): type of fit to perform
            cmap (str): colormap
            cmap_idx_base (int): base index of the colormap
        """

        # Check labels
        if !isnothing(plotlabels)
            @argcheck length(plotlabels) == size(inputs)[1] "Number of titles must match the number of input-response pairs"
        else
            titles = ["$i" for i in 1:size(inputs)[1]]
        end

        if isnothing(ax)
            plotself = true
            fig, ax = subplots(1, 1, figsize=(6, 4))
        else
            plotself = false
            
        end

        for i in eachindex(inputs)
            plot_functional_relationship(inputs[i], responses[i], axlabels, title, plotlabels[i]; ax=ax, logx=logx, logy=logy, fit=fit, cmap=cmap, cmap_idx=cmap_idx_base+i-1)
        end

        if plotself
            ax.legend(fontsize="small")
            gcf()
        end
    end


    function compare_functional_relationships_groups(in_groups, res_groups, axlabels, group_labels=nothing, title=nothing; ax=nothing, logx=false, logy=false, fit=nothing)
        """
        Compare the functional relationships from groups of simulations
        on the same axis, with corresponding colors.
        inputs:
            in_groups (Array{Array{Float64}}): input values
            res_groups (Array{Array{Float64}}): response values
            axlabels (Array{String}): axis labels
            group_labels (Array{String}): labels for the groups
            title (str): title of the plot
            ax (Axis): axis to plot on
            logx (bool): whether to plot the x-axis in log scale
            logy (bool): whether to plot the y-axis in log scale
            fit (str): type of fit to perform
        """

        # Convert to nested arrays
        in_groups = to_nested(in_groups)
        res_groups = to_nested(res_groups)

        # Check labels
        if isnothing(group_labels)
            group_labels = [["Group $i, configuration $j" for j in 1:size(in_groups[i])[1]] for i in 1:size(in_groups)[1]]
        else
            group_labels = to_nested(group_labels)
        end

        if isnothing(ax)
            plotself = true
            fig, ax = subplots(1, 1, figsize=(6, 4))
        else
            plotself = false
        end

        cmap = get_cmap("tab20c")
        for i in eachindex(in_groups)
            compare_functional_relationships(in_groups[i], res_groups[i], axlabels, group_labels[i], title; ax, logx, logy, fit, cmap=cmap, cmap_idx_base=(i - 1)*4)
        end

        if plotself
            ax.legend(fontsize="small")
            gcf()
        end
    end


    function plot_spore_positions(Lx, Lz, spore_positions; ax=nothing, title=nothing, top_view=false)
        """
        Creates a scatter plot of the spore positions.
        inputs:
            Lx (float): length of the x/y-axis
            Lz (float): length of the z-axis
            spore_positions (Array{Tuple}): spore positions
            ax (Axis): axis to plot on
            title (str): title of the plot
            top_view (bool): whether to plot in top view
        """

        if isnothing(ax)
            if top_view 
                fig, ax = subplots(1, 1, figsize=(6, 6))
            else
                fig = figure(figsize=(6, 6))
                ax = fig[:add_subplot](projection="3d")
                ax.scatter(spore_positions[:, 1], spore_positions[:, 2], spore_positions[:, 3], color="blue", alpha=0.5, s=1)
            end
            plotself = true
        else
            plotself = false
        end

        if top_view
            ax.scatter(spore_positions[:, 1], spore_positions[:, 2], color="blue", alpha=0.5, s=1)
        else
            ax.scatter(spore_positions[:, 1], spore_positions[:, 2], spore_positions[:, 3], color="blue", alpha=0.5, s=1)
        end
        
        ax.set_xlim(0, Lx)
        ax.set_ylim(0, Lx)
        ax.set_xlabel(L"x\ [μm]")
        ax.set_ylabel(L"y\ [μm]")
        if !top_view
            ax.set_zlim(0, Lz)
            ax.set_zlabel(L"z\ [μm]")
        end

        if !isnothing(title)
            ax.set_title(title)
        end

        if plotself
            gcf()
        end
    end


    function plot_spore_arrangements(Lx, Lz, spore_arrangements, labels=nothing; title=nothing, top_view=false, figsize=6)
        """
        Creates multiple scatter plots for the different spore arrangements.
        inputs:
            Lx (float): length of the x/y-axis
            Lz (float): length of the z-axis
            spore_arrangements (Array{Array{Tuple}}): spore arrangements
            labels (Array{String}): labels for the plots
            title (str): title of the plot
            top_view (bool): whether to plot in top view
            figsize (float): vertical size of figure
        """
        if isnothing(labels)
            labels = ["Arrangement $i" for i in 1:length(spore_arrangements)]
        end

        # Convert to nested arrays
        spore_arrangements = to_nested(spore_arrangements)
        
        if top_view 
            fig, axs = subplots(1, 1, figsize=(figsize, figsize))
        else
            fig, axs = subplots(1, length(spore_arrangements), figsize=(figsize*length(spore_arrangements), figsize), subplot_kw=Dict("projection"=>"3d",))
        end

        for (i, arrangement) in enumerate(spore_arrangements)
            plot_spore_positions(Lx, Lz, arrangement; ax=axs[i], title=labels[i], top_view=top_view)
        end
        fig.tight_layout()
        fig.suptitle(title)

        gcf()
    end


    function plot_dantigny_time_course(p_max, τ_g, ν; germination_responses=nothing, times=nothing, ax=nothing, title=nothing)
        """
        Plots the time course of the Dantigny germination responses.
        inputs:
            p_max (float): maximum germination response in %
            τ_g (float): half-saturation time in hours
            ν (float): design parameter
            t_max (float): maximum time in hours
            germination_responses (Array{Float64}): germination responses
            times (Array{Float64}): time labels in seconds
            ax (Axis): axis to plot on
            title (str): title of the plot
        outputs:
            germination_responses (Array{Float64}): germination responses according to the Dantigny model
        """
        if isnothing(ax)
            fig, ax = subplots(1, 1, figsize=(6, 4))
            plotself = true
        else
            plotself = false
        end

        if isnothing(times)
            times = collect(LinRange(0, τ_g * 5, 100)) .* 3600
        end

        if isnothing(germination_responses)
            germination_responses = dantigny.(times ./3600, p_max, τ_g, ν) * 0.01
        end

        ax.plot(times ./ 3600, germination_responses .* 100, label="Dantigny model")
        ax.axhline(p_max, color="red", linestyle="--", label=L"p_{\text{max}}")
        ax.axvline(τ_g, color="green", linestyle="--", label=L"\tau_g")
        ax.set_xlabel("Time [h]")
        ax.set_ylabel("Germination response [%]")
        ax.grid(true)
        
        if !isnothing(title)
            ax.set_title(title)
        end

        if plotself
            ax.legend(fontsize="small")
            gcf()
        end

        return germination_responses
    end


    function compare_time_course_to_dantigny(germination_responses, times, p_max, τ_g, ν; ax=nothing, title=nothing)
        """
        Compare simulated germination responses to the Dantigny model.
        inputs:
            germination_responses (Array{Float64}): simulated germination responses
            times (Array{Float64}): time labels in hours
            p_max (float): maximum germination response fraction
            τ_g (float): half-saturation time in hours
            ν (float): design parameter
            ax (Axis): axis to plot on
            title (str): title of the plot
        outputs:
            dantigny_responses (Array{Float64}): germination responses according to the Dantigny model
        """
        if isnothing(ax)
            fig, ax = subplots(1, 1, figsize=(6, 4))
            plotself = true
        else
            plotself = false
        end

        dantigny_responses = plot_dantigny_time_course(p_max, τ_g, ν, ax=ax, times=times)
        ax.plot(times ./ 3600, germination_responses .* 100, label="Volume-based model")
        ax.legend(fontsize=10, loc="lower right")#"small")

        if !isnothing(title)
            ax.set_title(title)
        end

        if plotself
            gcf()
        end

        return dantigny_responses
    end


    function plot_germination_data_fit(data_inputs, data_responses, model_inputs, model_responses, sources; yerr=nothing, ax=nothing, title=nothing, c_ex=false)
        """
        Plots the germination data and the fitted model
        as functions of the spore senisity.
        inputs:
            data_inputs (Array{Float64}): germination data inputs
            data_responses (Array{Float64}): germination data response fractions
            model_inputs (Array{Float64}): model inputs
            model_responses (Array{Float64}): model response fractions
            sources (Array{String}): carbon sources
            yerr (Array{Float64}): error in the data responses
            ax (Axis): axis to plot on
            title (str): title of the plot
            c_ex (bool): whether the inputs are exogenous concentrations
        """
        if isnothing(ax)
            fig, ax = subplots(1, 1, figsize=(6, 4))
            plotself = true
        else
            plotself = false
        end

        for (i, src) in enumerate(sources)
            ax.plot(model_inputs, model_responses[i, :], label=src)
            ax.errorbar(data_inputs, data_responses[i, :], yerr=yerr[i, :, :]', fmt="o", markersize=5, label="Data ($(src))")
        end

        if !c_ex
            ax.set_xscale("log")
            ax.set_xlabel("Spore Density [spores/mL]")
        else
            ax.set_xlabel("Exogenous inhibitor concentration [M]")
        end
        
        ax.set_ylabel("Long-term germination Response [%]")
        ax.set_ylim(0, 110)
        ax.grid()
        ax.legend(fontsize=11)#"small")

        if !isnothing(title)
            ax.set_title(title)
        end

        if plotself
            gcf()
        end
    end


    function plot_germination_data_fit_all(model_type, st, params_opt, times, sources_data, densities_data, p_maxs_data, taus_data, nus_data, p_max_errs, dens_exp_limits=(4, 6); n_nodes=nothing)
        """
        Creates a combined plot with density-dependent
        and time-dependent germination data and model fits.
        inputs:
            model_type (str): type of model to use
            st (Bool): whether to use a time-dependent inducer
            params_opt (Array{Float64}): optimal parameters
            times (Array{Float64}): time labels in hours
            sources_data (Array{String}): carbon sources
            densities_data (Array{Float64}): germination data spore densities in spores/mL
            p_maxs_data (Array{Float64}): germination data asymptotic response fractions
            taus_data (Array{Float64}): germination data half-saturation times in hours
            nus_data (Array{Float64}): germination data design parameters
            p_max_errs (Array{Float64}): error in the data responses
            dens_exp_limits (Tuple): limits of density exponents to plot
            n_nodes (int): number of Gauss-Hermite nodes for computing the visualised germination response
        """

        @argcheck model_type in ["independent",
                                "inhibitor", "inhibitor_thresh", "inhibitor_perm",
                                "inducer", "inducer_thresh", "inducer_signal",
                                "combined_inhibitor", "combined_inhibitor_thresh", "combined_inhibitor_perm",
                                "combined_inducer", "combined_inducer_thresh", "combined_inducer_signal",
                                "special_inhibitor", "special_inducer", "special_independent", "special_combined"]

        # Create figure and subfigures
        fig = figure(figsize=(8, 2 + 2*length(densities_data)))
        topfig, bottomfig = fig.subfigures(2, 1, height_ratios=(0.75, 1.25*length(densities_data)/2), hspace=0.15)
        top_axs = topfig.subplots(1, 1)
        bottom_axs = bottomfig.subplots(length(densities_data), length(sources_data), sharex=true, sharey=true)

        # Compute germination responses using the fitted parameters
        density_exp_range = LinRange(dens_exp_limits[1], dens_exp_limits[2], 1000)
        density_range = 10 .^ density_exp_range
        germ_resp_final = zeros(length(sources_data), length(density_range))
        error_total = 0

        for (i, src) in enumerate(sources_data)
            # Smooth curve of density-dependent germination response
            for (j, density) in enumerate(density_range)
                germ_resp_final[i, j] = compute_germination_response(model_type, st, times[end], inverse_mL_to_cubic_um(density), get_params_for_idx(params_opt, i), n_nodes=n_nodes)[1]
            end
            # Time-dependent germination responses
            for (j, density) in enumerate(densities_data)
                germ_resp_sample = compute_germination_response(model_type, st, times, inverse_mL_to_cubic_um(density), get_params_for_idx(params_opt, i), n_nodes=n_nodes)
                
                dantigny_responses = compare_time_course_to_dantigny(germ_resp_sample, times, p_maxs_data[i, j], taus_data[i, j], nus_data[i, j],
                                                                        ax=bottom_axs[j, i], title="$(sources_data[i]), " * @sprintf("%.3e", round(Int, densities_data[j])) * " spores/mL")
                
                error_total += sum(abs2, germ_resp_sample .- dantigny_responses)
                
                if i > 0
                    bottom_axs[j, i].set_ylabel("")
                end
                if j < length(densities_data)
                    bottom_axs[j, i].set_xlabel("")
                end
            end
        end

        # Compute RMSE
        rmse = sqrt(error_total / (length(sources_data) * length(densities_data) * length(times)))

        # Model labels
        model_labels = Dict(
            "independent" => "Independent inducer/inhibitor model",
            "inhibitor" => "Inducer-dependent inhibitor threshold and release",
            "inhibitor_thresh" => "Inducer-dependent inhibition threshold",
            "inhibitor_perm" => "Inducer-dependent inhibitor release",
            "inducer" => "Inhibitor-dependent induction threshold and signal",
            "inducer_thresh" => "Inhibitor-dependent induction threshold",
            "inducer_signal" => "Inhibitor-dependent induction signal",
            "combined_inhibitor" => "Combined model with inducer-dependent inhibitor threshold and release",
            "combined_inhibitor_thresh" => "Combined model with inducer-dependent inhibitor threshold",
            "combined_inhibitor_perm" => "Combined model with inducer-dependent inhibitor release",
            "combined_inducer" => "Combined model with inhibitor-dependent induction threshold and signal",
            "combined_inducer_thresh" => "Combined model with inhibitor-dependent induction threshold",
            "combined_inducer_signal" => "Combined model with inhibitor-dependent induction signal",
            "special_inhibitor" => "Inducer-dependent inhibition (varying permeability)",
            "special_inducer" => "Inhibitor-dependent induction (varying permeability)",
            "special_independent" => "Independent inducer/inhibitor model (varying permeability)",
            "special_combined" => "2-factor germination with inhibitor-dependent induction (var. permeability)"
        )

        plot_germination_data_fit(densities_data, p_maxs_data, density_range, germ_resp_final .* 100, sources_data, yerr=p_max_errs,
                                    ax=top_axs, title=model_labels[model_type] * ", RMSE: $(round(rmse, sigdigits=3))")

        tight_layout()
        gcf()

    end
end