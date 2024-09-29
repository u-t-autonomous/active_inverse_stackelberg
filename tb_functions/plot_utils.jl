function plot_splines(
    leader_splines::Vector{Spline},
    follower_spline::Spline
)
    
    fig = Figure(resolution=(1150, 850))
    ax = Axis(fig[1, 1], aspect=DataAspect(), title ="Trajectories", 
                xlabel="x (m)", ylabel="y (m)")
    colormaps = [:Blues_9, :Greens_9, :Reds_9, :Purples_9, :Oranges_9, :Greys_9]

    for (j, spl) in enumerate(leader_splines)
        t = range(0, stop=spl.ts[end], length=3000)
        T = length(t)-1
        xs = zeros(T)
        ys = zeros(T)
        xdots = zeros(T)
        ydots = zeros(T)
        vs = zeros(T)
        for i in 1:T
            xs[i], ys[i], xdots[i], ydots[i] = evaluate(spl, t[i])
            vs[i] = norm([xdots[i], ydots[i]])
        end
        
        s = scatter!(ax, xs, ys, color=vs, markersize=6,
                        colormap=colormaps[j])
        Colorbar(fig[1, 1+j], s, label="Planned Velocity (m/s), Leader $(j)")
        println("Leader $(j) planned max velocity: $(maximum(vs)) m/s")
    end

    d = length(leader_splines)
    t = range(0, stop=follower_spline.ts[end], length=3000)
    T = length(t)-1
    xs = zeros(T)
    ys = zeros(T)
    xdots = zeros(T)
    ydots = zeros(T)
    vs = zeros(T)
    for i in 1:T
        xs[i], ys[i], xdots[i], ydots[i] = evaluate(follower_spline, t[i])
        vs[i] = norm([xdots[i], ydots[i]])
    end
    s = scatter!(ax, xs, ys, color=vs, markersize=6,
                    colormap=colormaps[d+1])
    Colorbar(fig[1, d+2], s, label="Planned Velocity (m/s), Follower")
    println("Follower planned max velocity: $(maximum(vs)) m/s")

    display(fig)
    return fig, ax
end

function plot_splines_followers(
    follower_splines::Vector{Spline}
)
    
    fig = Figure(resolution=(1150, 850))
    ax = Axis(fig[1, 1], title ="Trajectories", 
                xlabel="x (m)", ylabel="y (m)")
    colormaps = [:Blues_9, :Greens_9, :Reds_9, :Purples_9, :Oranges_9, :Greys_9]

    for (j, spl) in enumerate(follower_splines)
        t = range(0, stop=spl.ts[end], length=3000)
        T = length(t)-1
        xs = zeros(T)
        ys = zeros(T)
        xdots = zeros(T)
        ydots = zeros(T)
        vs = zeros(T)
        for i in 1:T
            xs[i], ys[i], xdots[i], ydots[i] = evaluate(spl, t[i])
            vs[i] = norm([xdots[i], ydots[i]])
        end
        
        s = scatter!(ax, xs, ys, color=vs, markersize=6,
                        colormap=colormaps[j])
        Colorbar(fig[1, 1+j], s, label="Planned Velocity (m/s), Follower $(j)")
        println("Follower $(j) planned max velocity: $(maximum(vs)) m/s")
    end

    display(fig)
    return fig, ax
end

function plot_rollouts(fig, ax, rs::Vector{RolloutData}, r_f::RolloutData)
    for r in rs
        scatter!(ax, r.xs[1,:], r.xs[2,:], color=:black, markersize=3)
    end
    scatter!(ax, r_f.xs[1,:], r_f.xs[2,:], color=:black, markersize=3)
    display(fig)
end

function plot_rollouts_follower(fig, ax, r_f::RolloutData)
    scatter!(ax, r_f.xs[1,:], r_f.xs[2,:], color=:black, markersize=3)
    display(fig)
end