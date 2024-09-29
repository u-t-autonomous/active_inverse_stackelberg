# active_inverse_stackelberg
Optimization code for active inverse learning in stackelberg trajectory games.

# Usage
pursuit.jl and driver.jl run the pursuit game and driving assistant optimization games in Julia. pursuit_tb.jl and driver_tb.jl run the optimizations, then send the trajectories to simulated turtlebots in Gazebo. Ensure that the corresponding Gazebo simulation is running before starting the Julia scripts. See [willward20/active_inverse_stackelberg_ros](https://github.com/willward20/active_inverse_stackelberg_ros) for details on running the Gazebo sim.

# Depdendencies
Note that you will need a MOSEK license to run all code in this repository. 
