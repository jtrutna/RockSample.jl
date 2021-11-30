module RockSample

using LinearAlgebra
using POMDPs
using POMDPModelTools
using StaticArrays
using Parameters
using Random
using Compose
using Combinatorics
using ParticleFilters
using DiscreteValueIteration
using POMDPPolicies

export
    RockSamplePOMDP,
    RSPos,
    RSState,
    RSExit,
    RSExitSolver,
    RSMDPSolver,
    RSQMDPSolver

const RSPos = SVector{2, Int}

"""
    RSState{K}
Represents the state in a RockSamplePOMDP problem. 
`K` is an integer representing the number of rocks

# Fields
- `pos::RPos` position of the robot
- `rocks::SVector{K, Bool}` the status of the rocks (false=bad, true=good)
"""
struct RSState{K}
    pos::RSPos 
    rocks::SVector{K, Bool}
end

@with_kw struct RockSamplePOMDP{K} <: POMDP{RSState{K}, Int, Int}
    map_size::Tuple{Int, Int} = (5,5)
    rocks_positions::SVector{K,RSPos} = @SVector([(1,1), (3,3), (4,4)])
    init_pos::RSPos = (1,1)
    sensor_efficiency::Float64 = 10.0
    bad_rock_penalty::Float64 = -10
    good_rock_reward::Float64 = 10.
    exit_reward::Float64 = 10.
    terminal_state::RSState{K} = RSState(RSPos(-1,-1),
                                         SVector{length(rocks_positions),Bool}(falses(length(rocks_positions))))
    # Some special indices for quickly retrieving the stateindex of any state
    indices::Vector{Int} = cumprod([map_size[1], map_size[2], fill(2, length(rocks_positions))...][1:end-1])
    discount_factor::Float64 = 0.95
end

# to handle the case where rocks_positions is not a StaticArray
function RockSamplePOMDP(map_size,
                         rocks_positions,
                         args...
                        )

    k = length(rocks_positions)
    return RockSamplePOMDP{k}(map_size,
                              SVector{k,RSPos}(rocks_positions),
                              args...
                             )
end

# Generate a random instance of RockSample(n,m) with a n×n square map and m rocks
RockSamplePOMDP(map_size::Int, rocknum::Int, rng::AbstractRNG=Random.GLOBAL_RNG) = RockSamplePOMDP((map_size,map_size), rocknum, rng)

# Generate a random instance of RockSample with a n×m map and l rocks
function RockSamplePOMDP(map_size::Tuple{Int, Int}, rocknum::Int, rng::AbstractRNG=Random.GLOBAL_RNG)
    possible_ps = [(i, j) for i in 1:map_size[1], j in 1:map_size[2]]
    selected = unique(rand(rng, possible_ps, rocknum))
    while length(selected) != rocknum
        push!(selected, rand(rng, possible_ps))
        selected = unique!(selected)
    end
    return RockSamplePOMDP(map_size=map_size, rocks_positions=selected)
end

# To handle the case where the `rocks_positions` is specified
RockSamplePOMDP(map_size::Tuple{Int, Int}, rocks_positions::AbstractVector) = RockSamplePOMDP(map_size=map_size, rocks_positions=rocks_positions)

POMDPs.isterminal(pomdp::RockSamplePOMDP, s::RSState) = s.pos == pomdp.terminal_state.pos 
POMDPs.discount(pomdp::RockSamplePOMDP) = pomdp.discount_factor

## States
function POMDPs.stateindex(pomdp::RockSamplePOMDP{K}, s::RSState{K}) where K
    if isterminal(pomdp, s)
        return length(pomdp)
    end
    return s.pos[1] + pomdp.indices[1] * (s.pos[2]-1) + dot(view(pomdp.indices, 2:(K+1)), s.rocks)
end

function state_from_index(pomdp::RockSamplePOMDP{K}, si::Int) where K
    if si == length(pomdp)
        return pomdp.terminal_state
    end
    rocks_dim = fill(2, K)
    nx, ny = pomdp.map_size
    s = CartesianIndices((nx, ny, rocks_dim...))[si]
    pos = RSPos(s[1], s[2])
    rocks = SVector{K, Bool}([(s[i] - 1) for i=3:K+2])
    return RSState{K}(pos, rocks)
end

# the state space is the pomdp itself
POMDPs.states(pomdp::RockSamplePOMDP) = pomdp

Base.length(pomdp::RockSamplePOMDP) = pomdp.map_size[1]*pomdp.map_size[2]*2^length(pomdp.rocks_positions) + 1

# we define an iterator over it 
function Base.iterate(pomdp::RockSamplePOMDP, i::Int=1)
    if i > length(pomdp)
        return nothing
    end
    s = state_from_index(pomdp, i)
    return (s, i+1)
end

function POMDPs.initialstate(pomdp::RockSamplePOMDP{K}) where K 
    probs = normalize!(ones(2^K), 1)
    states = Vector{RSState{K}}(undef, 2^K)
    for (i,rocks) in enumerate(Iterators.product(ntuple(x->[false, true], K)...))
        states[i] = RSState{K}(pomdp.init_pos, SVector(rocks))
    end
    return SparseCat(states, probs)
end


## Actions
const N_BASIC_ACTIONS = 5
const BASIC_ACTIONS_DICT = Dict(:north => 1, 
                                :east => 2,
                                :south => 3,
                                :west => 4,
                                :sample => 5)

const ACTION_DIRS = (RSPos(0,1),
                    RSPos(1,0),
                    RSPos(0,-1),
                    RSPos(-1,0),
                    RSPos(0,0))

POMDPs.actions(pomdp::RockSamplePOMDP{K}) where K = 1:N_BASIC_ACTIONS+K
POMDPs.actionindex(pomdp::RockSamplePOMDP, a::Int) = a

function POMDPs.actions(pomdp::RockSamplePOMDP{K}, s::RSState) where K
    if in(s.pos, pomdp.rocks_positions) # slow? pomdp.rock_pos is a vec 
        return actions(pomdp)
    else
        # sample not available
        return 2:N_BASIC_ACTIONS+K
    end
end

## Transitions
function POMDPs.transition(pomdp::RockSamplePOMDP{K}, s::RSState{K}, a::Int) where K
    if isterminal(pomdp, s)
        return Deterministic(pomdp.terminal_state)
    end
    new_pos = next_position(s, a)
    if a == BASIC_ACTIONS_DICT[:sample] && in(s.pos, pomdp.rocks_positions)
        rock_ind = findfirst(isequal(s.pos), pomdp.rocks_positions) # slow ?
        # set the new rock to bad
        new_rocks = MVector{K, Bool}(undef)
        for r=1:K
            new_rocks[r] = r == rock_ind ? false : s.rocks[r]
        end
        new_rocks = SVector(new_rocks)
    else 
        new_rocks = s.rocks
    end
    if new_pos[1] > pomdp.map_size[1]
        # the robot reached the exit area
        new_state = pomdp.terminal_state
    else
        new_pos = RSPos(clamp(new_pos[1], 1, pomdp.map_size[1]), 
                        clamp(new_pos[2], 1, pomdp.map_size[2]))
        new_state = RSState{K}(new_pos, new_rocks)
    end
    return Deterministic(new_state)
end

function next_position(s::RSState, a::Int)
    if a < N_BASIC_ACTIONS
        # the robot moves 
        return s.pos + ACTION_DIRS[a]
    elseif a >= N_BASIC_ACTIONS 
        # robot check rocks or samples
        return s.pos
    else
        throw("ROCKSAMPLE ERROR: action $a not valid")
    end
end

## Observations
const OBSERVATION_NAME = (:good, :bad, :none)

POMDPs.observations(pomdp::RockSamplePOMDP) = 1:3
POMDPs.obsindex(pomdp::RockSamplePOMDP, o::Int) = o

function POMDPs.observation(pomdp::RockSamplePOMDP, a::Int, s::RSState)
    if a <= N_BASIC_ACTIONS
        # no obs
        return SparseCat((1,2,3), (0.0,0.0,1.0)) # for type stability
    else
        rock_ind = a - N_BASIC_ACTIONS 
        rock_pos = pomdp.rocks_positions[rock_ind]
        dist = norm(rock_pos - s.pos)
        efficiency = 0.5*(1.0 + exp(-dist/pomdp.sensor_efficiency))
        rock_state = s.rocks[rock_ind]
        if rock_state
            return SparseCat((1,2,3), (efficiency, 1.0 - efficiency, 0.0))
        else
            return SparseCat((1,2,3), (1.0 - efficiency, efficiency, 0.0))
        end
    end
end

## Rewards
function POMDPs.reward(pomdp::RockSamplePOMDP, s::RSState, a::Int)
    if next_position(s, a)[1] > pomdp.map_size[1]
        return pomdp.exit_reward
    end

    if a == BASIC_ACTIONS_DICT[:sample] && in(s.pos, pomdp.rocks_positions) # sample 
        rock_ind = findfirst(isequal(s.pos), pomdp.rocks_positions) # slow ?
        return s.rocks[rock_ind] ? pomdp.good_rock_reward : pomdp.bad_rock_penalty 
    end
    return 0.
end

include("visualization.jl")
include("heuristics.jl")

end # module
