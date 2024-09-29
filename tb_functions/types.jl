struct TurtlebotConnection
    feedback::Connection
    rollout::Connection
    ts::Connection
    coeffs_x::Connection
    coeffs_y::Connection
end

struct Connections
    leader_tbs::Vector{TurtlebotConnection}
    follower_tb::TurtlebotConnection
    timing::Connection
end

struct ConnectionsFollowers
    follower_tbs::Vector{TurtlebotConnection}
    timing::Connection
end

struct SplineSegment
    coeffs_x::Vector{<:Real}
    coeffs_y::Vector{<:Real}
    t0::Real
    tf::Real
end

struct Spline
    ts::Vector{<:Real}
    all_x_coeffs::Matrix{<:Real}
    all_y_coeffs::Matrix{<:Real}
end

struct RolloutData
    ts::Vector{<:Real}
    xs::Matrix{<:Real}
    xds::Matrix{<:Real}
    us::Matrix{<:Real}
end