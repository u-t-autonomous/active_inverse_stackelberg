using LinearAlgebra
using QuadGK: quadgk
using Random, Distributions
using Combinatorics: combinations
using Ipopt
using CairoMakie: Figure, Axis, lines!, scatter!, DataAspect, Colorbar
using QuadGK: quadgk
using JuMP
using MosekTools
using RosSockets
import JSON
using Rotations: QuatRotation

include("tb_functions/types.jl")
include("tb_functions/communication.jl")
include("tb_functions/plot_utils.jl")
include("tb_functions/spline.jl")

Random.seed!(123) # Setting the seed

# ----------------------------------------------------------------------------------------
# Step 1: Define Constants and Set Up the Problem
# -----------------------------------------------------------------------------------------

# Parameters
d = 3  # number of ground rovers controlled by the leader
dt = 2.0  # discretization step size (delta)
T = 30
tau = Int(round(T/dt))  # number of trajectory waypoints
setP = collect(combinations(1:d, 2))  # set of all hypothsis pairs
n_p = size(setP, 1)  # number of hypotheses


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
Al = kron(Matrix(I, d, d), Af)  # nl x nl (kron means tensor product)
Bl = kron(Matrix(I, d, d), Bf)  # nl × ml 

nl, ml = size(Bl)  # number of leader states and inputs
nf, mf = size(Bf)  # number of follwer states and inputs

Rl = Matrix(I, ml, ml)
Qi = zeros(nf, nf, d) 
Ri = zeros(mf, mf, d) 
M = zeros(nf, nl, d)
for i in 1:d
    Qi[:, :, i] = 20 * diagm([1, 1, 0, 0]) 
    Ri[:, :, i] = 30000 * Matrix(I, mf, mf) 
    M[:, :, i] = [zeros(nf, nf*(i-1));; Matrix(I, nf, nf);; zeros(nf, nf*(d-i))]
end

Ωf = 0.00001 * Matrix(I, nf, nf)  # follower distrubance covar matrix
Ωl = 0.00001 * Matrix(I, nl, nl)  # leader distrubance covar matrix



# ----------------------------------------------------------------------------------------
# Step 2: Get Initial States from Turtlebots
# -----------------------------------------------------------------------------------------
connections = open_tb_connections()

x1_l = Vector{Real}() # initial condition of leader states
# small initial velocity for each leader so that they initially "point" the
# same way as the actual robot
v = 0.01
for i in 1:d
    s = state(connections.leader_tbs[i])
    x = s[1]
    y = s[2]
    θ = s[4]
    xdot = v*cos(θ)
    ydot = v*sin(θ)
    append!(x1_l, [x, y, xdot, ydot])
end

x1_f = zeros(nf) # initial condition of follower state
s_follower = state(connections.follower_tb)
x1_f[1] = s_follower[1]
x1_f[2] = s_follower[2]
x1_f[3] = v*cos(s_follower[4])
x1_f[4] = v*sin(s_follower[4])



# ----------------------------------------------------------------------------------------
# Step 2: Perform Dynamic Programming to Set Up the Leader's Problem
# -----------------------------------------------------------------------------------------
# Define arrays for recording matrix values for each hypothesis
Ff = zeros(nf, nf, tau, d)  
Ei = zeros(nf, nf, tau, d)    
Pi = zeros(nf, nf, tau+1, d)    
Lambda = zeros(nf, nf, tau+1, d)    

for i in 1:d  # for each hypothesis (leader rover)
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
beta = 5e-3  # upper bound on inf-norm of leader inputs
max_iter = 50  # maximum number of iterations
ϵ = 0.0001  # convergence tolerance
θt = 2*beta*(rand(ml, tau).-0.5)  # rand values b/w -beta and beta

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

# Build the constraints on the leader trajectory
@constraint(pre_model, ηl[:, 1] == x1_l) # set leader init condition
@constraint(pre_model, ηl[3, :] .>= -0.1) # limit max x vel
@constraint(pre_model, ηl[3, :] .<= 0.1) # limit max x vel
@constraint(pre_model, ηl[4, :] .>= -0.1) # limit max y vel
@constraint(pre_model, ηl[4, :] .<= 0.1) # limit max y vel
@constraint(pre_model, ηl[7, :] .>= -0.1) # limit max x vel
@constraint(pre_model, ηl[7, :] .<= 0.1) # limit max x vel
@constraint(pre_model, ηl[8, :] .>= -0.1) # limit max y vel
@constraint(pre_model, ηl[8, :] .<= 0.1) # limit max y vel
@constraint(pre_model, ηl[11, :] .>= -0.1) # limit max x vel
@constraint(pre_model, ηl[11, :] .<= 0.1) # limit max x vel
@constraint(pre_model, ηl[12, :] .>= -0.1) # limit max y vel
@constraint(pre_model, ηl[12, :] .<= 0.1) # limit max y vel
for t in 1:tau 
    @constraint(pre_model, ηl[:, t+1] == Al*ηl[:, t] + Bl*ul[:, t])
end

# Build the constraints on q and ξ
for i in 1:d
    q_taup1 = -Qi[:,:,i]*M[:,:,i]*ηl[:, tau+1]
    @constraint(pre_model, qt[:, tau+1, i] == q_taup1)
    for t in tau:-1:1
        @constraint(pre_model, qt[:, t, i] == Ei[:,:,t,i]'*qt[:, t+1, i] - Qi[:,:,i]*M[:,:,i]*ηl[:, t])
    end

    @constraint(pre_model, ξ[:, 1, i] == x1_f)
    for t in 1:tau
        @constraint(pre_model, ξ[:, t+1, i] == Ei[:,:,t,i]*ξ[:, t, i] - Ff[:,:,t,i]*qt[:, t+1, i])
    end
end

# Build the constraints for ul
for t in 1:tau
    for j in 1:ml
        @constraint(pre_model, -beta <= ul[j, t] <= beta) # work aroudn for inf-norm
    end
end

optimize!(pre_model)


# Function for testing convergence. 
function test_convergence(u, ξ)
    sum_u = sum((u[:,t+1]-u[:,t])'*Rl*(u[:,t+1]-u[:,t]) for t in 1:tau-1)
    sum_ξ = zeros(n_p)
    h = 1 # hypothesis number
    for i in 1:d-1
        for j in i+1:d
            sum_ξ[h] = sum((ξ[:,t,i]-ξ[:,t,j])'*(pinv(Lambda[:,:,t,i])+pinv(Lambda[:,:,t,j]))*(ξ[:,t,i]-ξ[:,t,j]) for t in 2:tau+1)
            h = h + 1 # go to the next hypothesis
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

    # Define constraints for helper variable (for the minimizer objective)
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
    @constraint(model, ηl2[:, 1] == x1_l) # set leader init condition
    @constraint(model, ηl2[3, :] .>= -0.1) # limit max x vel
    @constraint(model, ηl2[3, :] .<= 0.1) # limit max x vel
    @constraint(model, ηl2[4, :] .>= -0.1) # limit max y vel
    @constraint(model, ηl2[4, :] .<= 0.1) # limit max y vel
    @constraint(model, ηl2[7, :] .>= -0.1) # limit max x vel
    @constraint(model, ηl2[7, :] .<= 0.1) # limit max x vel
    @constraint(model, ηl2[8, :] .>= -0.1) # limit max y vel
    @constraint(model, ηl2[8, :] .<= 0.1) # limit max y vel
    @constraint(model, ηl2[11, :] .>= -0.1) # limit max x vel
    @constraint(model, ηl2[11, :] .<= 0.1) # limit max x vel
    @constraint(model, ηl2[12, :] .>= -0.1) # limit max y vel
    @constraint(model, ηl2[12, :] .<= 0.1) # limit max y vel
    for t in 1:tau 
        @constraint(model, ηl2[:, t+1] == Al*ηl2[:, t] + Bl*ul2[:, t])
    end

    # Build the constraints on q and ξ
    for i in 1:d
        q_taup1 = -Qi[:,:,i]*M[:,:,i]*ηl2[:, tau+1]
        @constraint(model, qt2[:, tau+1, i] == q_taup1)
        for t in tau:-1:1
            @constraint(model, qt2[:, t, i] == Ei[:,:,t,i]'*qt2[:, t+1, i] - Qi[:,:,i]*M[:,:,i]*ηl2[:, t])
        end

        @constraint(model, ξ2[:, 1, i] == x1_f)
        for t in 1:tau
            @constraint(model, ξ2[:, t+1, i] == Ei[:,:,t,i]*ξ2[:, t, i] - Ff[:,:,t,i]*qt2[:, t+1, i])
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
# Step 4: Calculate the Leaders' Trajectories
# -----------------------------------------------------------------------------------------
xl_opt = zeros(nl, tau+1)  
xl_opt[:, 1] = x1_l

# Generate a random vector distribution with mean 
# zeros(nl) and covariance Ωl. Sample vectors 
# (size nl) from the distribution using rand().
dl = MvNormal(zeros(nl), Ωl) # Create distribution.
wl = rand(dl, tau) # Create (nl x tau) disturbance matrix.

for t in 1:tau 
    # Calculate the next state
    xl_opt[:,t+1] = (Al * xl_opt[:,t]) + (Bl * ul_opt[:, t]) + wl[:, t]
end



# ----------------------------------------------------------------------------------------
# Step 5: Solve the Follower's Optimization Problem
#-----------------------------------------------------------------------------------------
target = 3 # true hypothesis (the rover that the follower tracks)
df = MvNormal(zeros(nf), Ωf) # generate the distribution for disturbance vectors

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
xf[:,1] = copy(x1_f)
global mu_all = zeros(mf, tau)
wf = rand(df, tau) # generate (nf row x tau) disturbance vector

for t in 1:tau
    # Calculate the next follower state
    mu = vec(Kf[:,:,t]*xf[:,t] + bf[:,t])
    d_u = MvNormal(mu, Σf[:,:,t])
    local u_f = rand(d_u, 1)
    global xf[:,t+1] = (Af * xf[:,t]) + (Bf * u_f) + wf[:, t]
end



# ----------------------------------------------------------------------------------------
# Step 7: Plot the Results
# -----------------------------------------------------------------------------------------
fig = Figure(resolution = (800, 800))
ax = Axis(fig[1,1], title="Leader and Follower Trajectories", xlabel="x", ylabel="y")
for k = 1:d
    num = (k-1)*nf
    lines!(ax, xl_opt[num+1,:], xl_opt[num+2,:], label = "Leader $k")
    scatter!(ax, xl_opt[num+1,:], xl_opt[num+2,:], label = "Leader $k")
end
lines!(ax, xf[1,:], xf[2,:], linestyle=:dash, color=:black, label="Follower")
display(fig)


# ----------------------------------------------------------------------------------------
# Step 8: Create and Plot Splines
# -----------------------------------------------------------------------------------------
leader_splines = make_splines(d, dt, xl_opt)
follower_spline = make_spline(dt, xf)
fig, ax = plot_splines(leader_splines, follower_spline)

send_leader_splines(connections, leader_splines)
send_follower_spline(connections, follower_spline)
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
leader_rs, follwer_r = all_rollout_data(connections)
close_tb_connections(connections)

plot_rollouts(fig, ax, leader_rs, follwer_r)