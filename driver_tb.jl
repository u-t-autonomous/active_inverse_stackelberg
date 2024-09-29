using LinearAlgebra: I, kron, diagm, pinv, norm
using QuadGK: quadgk
using Combinatorics: combinations
using JuMP
using MosekTools
using Ipopt
using CairoMakie: Figure, Axis, lines!, scatter!, DataAspect, Colorbar, save
using RosSockets
import JSON
using Rotations: QuatRotation
using Distributions
using Random
using Plots

include("tb_functions/types.jl")
include("tb_functions/communication.jl")
include("tb_functions/plot_utils.jl")
include("tb_functions/spline.jl")

Random.seed!(123) # Setting the seed
connections = open_tbf_connections()

# ----------------------------------------------------------------------------------------
# Step 1: Define Constants and Set Up the Problem
# -----------------------------------------------------------------------------------------

d = 3  # number of follower types
dt = 2  # discretization step size (delta)
T = 30
tau = Int(round(T/dt))  # number of trajectory waypoints
setP = collect(combinations(1:d, 2))  # set of all hypothsis pairs
n_p = size(setP, 1)  # number of hypotheses
road_width = 1.5 - 0.2  # width of road minus the width of a turtlebot
road1_length = 3  # length of the first road segment

# Dynamics
Ac0 = [
    0  0  1  0
    0  0  0  1
    0  0  0  0
    0  0  0  0
]
Bc0 = [
    0  0
    0  0
    1  0
    0  1
]
Af = exp(dt * Ac0)  # nf x nf
integral, _ = quadgk(t -> exp(t*Ac0), 0, dt)
Bf = integral * Bc0  # nf x mf
Al = Af
Bl = Bf

nl, ml = size(Bl)  # number of leader states and inputs
nf, mf = size(Bf)  # number of follwer states and inputs

x1_f = zeros(nf)  # follower initial condition 
x1_l = zeros(nl)  # leader initial condition 

Rl = Matrix(I, ml, ml)
Qi = zeros(nf, nf, d) 
Ri = zeros(mf, mf, d) 
M = zeros(nf, nl, d)  # driver type matrix
yvel_weights = [0.85, 1, 1.15]
xvel_weights = [0.95, 1, 1.05]
for i in 1:d
    Qi[:, :, i] = diagm([1000,100,100,100]) 
    Ri[:, :, i] = diagm([10000,1000])
    M[:, :, i] = diagm([xvel_weights[i], yvel_weights[i], xvel_weights[i], yvel_weights[i]])
end

Ωf = 0.0001 * Matrix(I, nf, nf)  # follower distrubance covar matrix
Ωl = 0.0000001 * Matrix(I, nl, nl)  # leader distrubance covar matrix



# ----------------------------------------------------------------------------------------
# Step 2: Perform Dynamic Programming to Set Up the Leader's Problem
# -----------------------------------------------------------------------------------------
# Define arrays for recording matrix values for each hypothesis
Ff = zeros(nf, nf, tau, d)  
Ei = zeros(nf, nf, tau, d)    
Pi = zeros(nf, nf, tau+1, d)    
Lambda = zeros(nf, nf, tau+1, d)    

for i in 1:d  # for each hypothesis
    # Set initial conditions
    Pi[:,:,end,i] = copy(Qi[:,:,i])  # Eq (12c)
    Lambda[:,:,1,i] = zeros(nf, nf)   # Eq (12d)

    for t in (tau):-1:1
        # Loop backwards through time and calculate Ff, 
        # Ef, and Pf values for each time step, t.
        Ff[:,:,t,i] = Bf * inv(Ri[:,:,i] + (Bf' * Pi[:,:,t+1,i] * Bf)) * Bf'   # Eq (12b)
        Ei[:,:,t,i] = Af - (Ff[:,:,t,i] * Pi[:,:,t+1,i] * Af)           # Eq (12a)
        Pi[:,:,t,i] = Qi[:,:,i] + (Af' * Pi[:,:,t+1,i] * Ei[:,:,t,i])   # Eq (12c)    
    end

    for t in 1:tau
        # Loop forwards through time and calcualte
        # Lambda for each time step. 
        Lambda[:,:,t+1,i] = (Ei[:,:,t,i] * Lambda[:,:,t,i] * Ei[:,:,t,i]') + Ff[:,:,t,i]' + Ωf # Eq (12d)
    end
end



# ----------------------------------------------------------------------------------------
# Step 3: Solve the Leader's Optimization Problem using Convex-Concave Procedure
# -----------------------------------------------------------------------------------------
beta = 0.05  # upper bound on inf-norm of leader inputs
max_iter = 50  # maximum number of iterations
ϵ = 0.0001  # convergence tolerance
θt = [0.0040629963879748095 0.0024567338113939408 -0.0024615091105849043 -0.0007267211912640082 -0.004008663851563958 0.0019220866205473907 -0.004679033266472528 0.004303323763821094 0.0027409169357750385 -0.002029771655340663 0.003935369466209786 -0.0036820170876587744 -0.004428997725373357 0.0030593101028508654 0.003236563299357046 0.003913769757127503 -0.0013941709968182659 -0.0010998696815647413 0.0043401558790130855 0.0022959972829948794 0.0013707725579207432 -0.0011645148240492964 -0.00015710673185313385 -0.00046306634985859185 0.0001004093779672699 0.0036975458228329593 0.0019223495388637423 -0.004033213098959351 0.0008191234238764567 -0.0037885247948187305;
     -0.0005650626754039545 0.00012083040036614312 -0.0016584846361808114 0.00367547200255958 -0.003747125923084497 -0.0036344852486254265 -0.0014945417854117338 0.0045943359940715375 -0.00316445271163198 -0.003498450535020272 -0.0014513234334710878 0.004411330896079792 -0.002543502986917816 -0.0016217603883107656 -0.0004951635736964977 0.0021103854980051083 -0.002404371740582322 -0.00038137772266173765 0.0025327784999104497 -0.003376195748510206 0.0049140513617811745 0.001182065026677851 0.000997019277391098 -0.0017509657975838012 0.0015609828413433923 -0.0012662409299324796 0.0024685405191074527 -0.0004089746475283773 -0.001885524992949471 -0.0029547018267964053]

# Initialize the CCP with this warm-up optimization procedure.
pre_model = Model(Mosek.Optimizer)
set_silent(pre_model)
@variables(pre_model, begin
    ul[1:ml, 1:tau]  # leader's input
    ηl[1:nl, 1:tau+1]  # leader's state
    ξ[1:nf, 1:tau+1, 1:d]  # hypothesis agent state
    qt[1:nf, 1:tau+1, 1:d]  # hypothesis agent co-state
end)

# Build the minimization objective
@objective(pre_model, Min, sum((ul[:,t]-θt[:,t])'*(ul[:,t]-θt[:,t]) for t in 1:tau))

# Build the constraints on the leader trajectory.
@constraint(pre_model, ηl[:, 1] == x1_l)  # leader initial condition
# Constraints for road segment 1.
for t in 1:Int(round(tau/2)) 
    @constraint(pre_model, ηl[:, t+1] == Al*ηl[:, t] + Bl*ul[:, t])
    @constraint(pre_model, ηl[2, t+1] >= 0.0)  # drive in the lane
    @constraint(pre_model, ηl[4, t+1] >= 0.0)  # don't drive backwards
end
# Constraints for road segment 2.
for t in Int(round(tau/2)):tau
    @constraint(pre_model, ηl[:, t+1] == Al*ηl[:, t] + Bl*ul[:, t])
    @constraint(pre_model, ηl[1, t+1] >= -road_width/2)  # drive in the lane
    @constraint(pre_model, ηl[3, t+1] >= 0.0)  # don't drive backwards
end

# Build the constraints on q and ξ
# For each follwer type:
for i in 1:d
    q_taup1 = -Qi[:,:,i] * M[:,:,i]*ηl[:, tau+1]
    @constraint(pre_model, qt[:, tau+1, i] == q_taup1)
    for t in tau:-1:1
        @constraint(pre_model, qt[:, t, i] == Ei[:,:,t,i]'*qt[:, t+1, i] - Qi[:,:,i] * M[:,:,i]*ηl[:, t])
    end

    # Build constraints on the follower's trajectory
    @constraint(pre_model, ξ[:, 1, i] == x1_f)  # follower initial condition
    # Constaints for road segment 1.
    for t in 1:Int(round(tau/2))
        @constraint(pre_model, ξ[:, t+1, i] == Ei[:,:,t,i]*ξ[:, t, i] - Ff[:,:,t,i]*qt[:, t+1, i])
        @constraint(pre_model, ξ[1, t+1, i] <= (road_width)/2)  # drive in the lane
        @constraint(pre_model, ξ[1, t+1, i] >= (-road_width)/2) # drive in the lane
    end
    # Constraints for road segment 2.
    for t in Int(round(tau/2)):tau
        @constraint(pre_model, ξ[:, t+1, i] == Ei[:,:,t,i]*ξ[:, t, i] - Ff[:,:,t,i]*qt[:, t+1, i])
        @constraint(pre_model, ξ[2, t+1, i] <= road1_length + (road_width))  # drive in the lane
        @constraint(pre_model, ξ[2, t+1, i] >= road1_length) # drive in the lane
    end
end

# Build the constraints for ul
for t in 1:tau
    for j in 1:ml
        @constraint(pre_model, -beta <= ul[j, t] <= beta)
    end
end

optimize!(pre_model)


# Function for testing convergence. 
function test_convergence(u, ξ)
    sum_u = sum((u[:,t+1]-u[:,t])'*Rl*(u[:,t+1]-u[:,t]) for t in 1:tau-1)
    sum_ξ = zeros(n_p)
    h = 1  # hytpothesis number
    for i in 1:d-1
        for j in i+1:d
            sum_ξ[h] = sum((ξ[:,t,i]-ξ[:,t,j])'*(pinv(Lambda[:,:,t,i])+pinv(Lambda[:,:,t,j]))*(ξ[:,t,i]-ξ[:,t,j]) for t in 2:tau+1)
            h = h + 1  # go to the next hypothesis
        end
    end
    min_ξ = min(sum_ξ...)
    return(sum_u - min_ξ)
end

# Set parameters for the main optimization procedure.
val_min = test_convergence(value.(ul), value.(ξ))
val_max = typemax(Float64)

# Initialize the optimized ξtil and ul
ξtil = value.(ξ)
ul_opt = value.(ul)



# Main optimization procedure.
for iter in 1:max_iter

    model = Model(Mosek.Optimizer)
    set_silent(model)
    @variables(model, begin
        ul2[1:ml, 1:tau]            
        ηl2[1:nl, 1:tau+1]          
        ξ2[1:nf, 1:tau+1, 1:d]      
        qt2[1:nf, 1:tau+1, 1:d]     
        ζ[1:nf, 2:tau+1, 1:n_p]    
        sig_t[2:tau+1, 1:n_p]  
        helper[2:tau+1, 1:n_p]       
        rho               
    end)

    @objective(model, Min, rho + sum((ul2[:,t+1]-ul2[:,t])'*Rl*(ul2[:,t+1]-ul2[:,t]) for t in 1:tau-1) - 2*sum(helper))
    
    # Define constraints for the helper variable.
    for t in 2:tau+1 # notice, start indexing time at 2 -- we lose nothing by starting here
        k = 1 # pair index
        for i in 1:d-1
            for j in i+1:d
                @constraint(model, helper[t, k] == (ξtil[:,t,i] - ξtil[:,t,j])' * (pinv(Lambda[:,:,t,i]) + pinv(Lambda[:,:,t,j])) * (ξ2[:,t,i]-ξ2[:,t,j]))
                k = k + 1
            end
        end
    end        

    # Build the constraints on the leader trajectory
    @constraint(model, ηl2[:, 1] == x1_l)  # leader initial condition
    # Constraints for road segment 1.
    for t in 1:Int(round(tau/2))
        @constraint(model, ηl2[:, t+1] == Al*ηl2[:, t] + Bl*ul2[:, t])
        @constraint(model, ηl2[2, t+1] >= 0.0)  # drive in the lane
        @constraint(model, ηl2[4, t+1] >= 0.0)  # don't drive backwards
    end
    # Constraints for road segment 2.
    for t in Int(round(tau/2)):tau
        @constraint(model, ηl2[:, t+1] == Al*ηl2[:, t] + Bl*ul2[:, t])
        @constraint(model, ηl2[1, t+1] >= -road_width/2)  # drive in the lane
        @constraint(model, ηl2[3, t+1] >= 0.0)  # don't drive backwards
    end

    # Build the constraints on q and ξ
    for i in 1:d
        q_taup1 = -Qi[:,:,i] * M[:,:,i]*ηl2[:, tau+1]
        @constraint(model, qt2[:, tau+1, i] == q_taup1)
        for t in tau:-1:1
            @constraint(model, qt2[:, t, i] == Ei[:,:,t,i]'*qt2[:, t+1, i] - Qi[:,:,i] * M[:,:,i]*ηl2[:, t])
        end

        # Build constraints on the follower's trajectory
        @constraint(model, ξ2[:, 1, i] == x1_f)  # follower initial condition
        # Constraints for road segment 1.
        for t in 1:Int(round(tau/2))
            @constraint(model, ξ2[:, t+1, i] == Ei[:,:,t,i]*ξ2[:, t, i] - Ff[:,:,t,i]*qt2[:, t+1, i])
            @constraint(model, ξ2[1, t+1, i] <= (road_width)/2)  # drive in the lane
            @constraint(model, ξ2[1, t+1, i] >= (-road_width)/2)  # drive in the lane
        end
        # Constraints for road segment 2.
        for t in Int(round(tau/2)):tau
            @constraint(model, ξ2[:, t+1, i] == Ei[:,:,t,i]*ξ2[:, t, i] - Ff[:,:,t,i]*qt2[:, t+1, i])
            @constraint(model, ξ2[2, t+1, i] <= road1_length + (road_width))  # drive in the lane
            @constraint(model, ξ2[2, t+1, i] >= road1_length)  # drive in the lane
        end

    end

    # Build the constraint on ζ
    for t in 2:tau+1
        h = 1 # hypothesis #
        for i in 1:d-1
            for j in i+1:d
                @constraint(model, ζ[:, t, h] == sqrt(pinv(Lambda[:,:,t,i]) + pinv(Lambda[:,:,t,j]))*(ξ2[:,t,i] - ξ2[:,t,j]))
                h = h + 1 # keep track of hypotheses
            end
        end
    end

    # Build the constraints for ||ζ||^2
    for t in 2:tau+1
        for h in 1:n_p
            @constraint(model, ζ[:, t, h]'*ζ[:, t, h] <= sig_t[t, h])
        end
    end

    # Build the constraints for σ and ρ
    for h in 1:n_p
        @constraint(model, sum(sig_t[:, :]) - sum(sig_t[:, h]) <= rho)
    end

    # Build the constraints for ul
    for t in 1:tau
        for j in 1:ml
            @constraint(model, -beta <= ul2[j, t] <= beta) # work aroudn for inf-norm
        end
    end

    optimize!(model)

    global ul_opt = value.(ul2)
    global ξtil = value.(ξ2)
    global val_max = test_convergence(ul_opt, ξtil)
    if abs(val_max - val_min) <= ϵ
        break
    else
        global val_min = val_max
    end       
end



# ----------------------------------------------------------------------------------------
# Step 4: Calculate the Leader's Trajectory and Plot
# -----------------------------------------------------------------------------------------
xl_opt = zeros(nl, tau+1)  
xl_opt[:, 1] = x1_l

for t in 1:tau 
    # Calculate the next state
    xl_opt[:,t+1] = (Al * xl_opt[:,t]) + (Bl * ul_opt[:, t])
end

# Calculate each follower type's reference trajectory
xf_ref1 = M[:,:,1]*xl_opt
xf_ref2 = M[:,:,2]*xl_opt
xf_ref3 = M[:,:,3]*xl_opt



# ----------------------------------------------------------------------------------------
# Step 5: Solve the Follower's Optimization Problem
#-----------------------------------------------------------------------------------------
fig = Figure(resolution = (1080, 1080))
ax = Axis(fig[1,1], limits=((-1,12), (-0.25,12.75)), title="Follower Bundles", xlabel="x", ylabel="y")

# Generate the follower's random noise distrubition.
df = MvNormal(zeros(nf), Ωf)

xf_all = zeros(d*nf, tau+1)
v = 0.01 # small initial velocity for tb orientation

for ii in 1:d
    target = ii  # the follower type

    Σf = zeros(mf, mf, tau)
    Kf = zeros(mf, nf, tau)
    bf = zeros(mf, tau)
    qf = zeros(nf, tau+1)

    Pf = Pi[:,:,:,target]
    Ef = Ei[:,:,:,target]
    Qf = Qi[:, :, target] # follower cost parameters
    Rf = Ri[:, :, target] # follower cost parameters
    Mf = M[:, :, target] # maps from the leader's state x_l to an output reference observable by the follower (Mf*x_l)

    # Set final condition
    qf[:, tau+1] = -1 * Qf * (Mf * xl_opt[:, tau+1])   # Eq (5d)

    for t in (tau):-1:1
        # Loop backwards through time and calculate values
        global Σf[:,:,t] = inv(Rf + (Bf' * Pf[:,:,t+1] * Bf))  # Eq (6a) 
        global Kf[:,:,t] = -Σf[:,:,t] * Bf' * Pf[:,:,t+1] * Af  # Eq (6b) 
        global bf[:,t]   = -Σf[:,:,t] * Bf' * qf[:,t+1]  # Eq (6b) 
        global qf[:,t]   = (Ef[:,:,t]' * qf[:,t+1]) - (Qf * (Mf * xl_opt[:, t]))  # Eq (5d)  
    end



    # ----------------------------------------------------------------------------------------
    # Step 6: Calculate the Follower's Trajectory
    # -----------------------------------------------------------------------------------------
    xf = zeros(nf, tau+1)  
    xf[:,1] = copy(xf_all[(4*ii-3):(4*ii),1])
    global wf = rand(df, tau) # generate (nf row x tau) disturbance vector

    for t in 1:tau
        # Calculate the next follower state
        mu = vec(Kf[:,:,t]*xf[:,t] + bf[:,t]) # mean input
        d_u = MvNormal(mu, Σf[:,:,t]) # normal distribution about the mean
        local u_f = rand(d_u, 1) # sample an input from the distribution
        global xf[:,t+1] = (Af * xf[:,t]) + (Bf * u_f) + wf[:, t]
    end

    # Plot the follower's trajectory
    if target==1
        lines!(ax, xf[1,:], xf[2,:], color=:blue, linewidth=2, label="Follower")
    elseif target==2
        lines!(ax, xf[1,:], xf[2,:], color=:green, linewidth=2, label="Follower")
    elseif target==3
        lines!(ax, xf[1,:], xf[2,:], color=:red, linewidth=2, label="Follower")
    end

    # Save follower trajectory to global array
    xf_all[(4*ii-3):(4*ii),:] = xf
end



# ----------------------------------------------------------------------------------------
# Step 8: Plot the Leader's Trajectories (Followers' reference trajectories)
# -----------------------------------------------------------------------------------------
lines!(ax, xf_ref1[1,:], xf_ref1[2,:], color=:black, label = "Leader 1")
scatter!(ax, xf_ref1[1,:], xf_ref1[2,:], color=:black, label = "Leader 1")
lines!(ax, xf_ref2[1,:], xf_ref2[2,:], color=:black, label = "Leader 2")
scatter!(ax, xf_ref2[1,:], xf_ref2[2,:], color=:black, label = "Leader 2")
lines!(ax, xf_ref3[1,:], xf_ref3[2,:], color=:black, label = "Leader 3")
scatter!(ax, xf_ref3[1,:], xf_ref3[2,:], color=:black, label = "Leader 3")

# Plot the road margins
lines!(ax, (-road_width/2)*ones(11), ((road1_length+road_width)/10)*collect(0:10), color=:black, linestyle=:dash)
lines!(ax, (road_width/2)*ones(11), (road1_length/10)*collect(0:10), color=:black, linestyle=:dash)
lines!(ax, (5/10)*collect(0:10) .+ (road_width/2), road1_length*ones(11), color=:black, linestyle=:dash)
lines!(ax, ((5+road_width)/10)*collect(0:10) .- (road_width/2), road1_length*ones(11) .+ (road_width), color=:black, linestyle=:dash)
display(fig)



# ----------------------------------------------------------------------------------------
# Step 9: Create, Plot, and Send Splines
# -----------------------------------------------------------------------------------------
follower_splines = make_splines(d, dt, xf_all)
fig, ax = plot_splines_followers(follower_splines)

# Choose which to follow (1, 2, or 3) using follower_splines[#]
send_follower_spline(connections, follower_splines[3])
sleep(0.5)
start_robots(connections)
t = 0.0
while t < T
    sleep(0.5)
    global t = time_elapsed(connections)
    @info "Running experiemnt. Time elapsed: $t out of $T"
end
stop_robots(connections)
sleep(1.0)
_, follower_rs = all_rollout_data(connections)
close_tb_connections(connections)

plot_rollouts_follower(fig, ax, follower_rs)