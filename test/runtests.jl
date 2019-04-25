using Random
using RockSample
using POMDPs
using POMDPTesting
using POMDPModelTools
using POMDPPolicies
using POMDPSimulators
using BeliefUpdaters
using Test

function test_state_indexing(pomdp::RockSamplePOMDP{K}, ss::Vector{RSState{K}}) where K
    for (i,s) in enumerate(states(pomdp))
        if s != ss[i]
            return false
        end
    end
    return true
end

@testset "state space" begin 
    pomdp = RockSamplePOMDP{3}()
    state_iterator =  states(pomdp)
    ss = ordered_states(pomdp)
    @test length(ss) == n_states(pomdp)
    @test test_state_indexing(pomdp, ss)
    pomdp = RockSamplePOMDP{3}(map_size=(7, 10))
    state_iterator =  states(pomdp)
    ss = ordered_states(pomdp)
    @test length(ss) == n_states(pomdp)
    @test test_state_indexing(pomdp, ss)
end

@testset "action space" begin 
    pomdp = RockSamplePOMDP{3}()
    acts = actions(pomdp)
    @test acts == ordered_actions(pomdp)
    @test length(acts) == n_actions(pomdp)
    @test length(acts) == RockSample.N_BASIC_ACTIONS + 3
end

@testset "transition" begin
    rng = MersenneTwister(1)
    pomdp = RockSamplePOMDP{3}(init_pos=(1,1))
    s0 = initialstate(pomdp, rng)
    @test s0.pos == pomdp.init_pos
    d = transition(pomdp, s0, 1) # move up
    sp = rand(rng, d)
    spp = rand(rng, d)
    @test spp == sp
    @test sp.pos == [1, 2]
    @test sp.rocks == s0.rocks
    s = RSState{3}((pomdp.map_size[1], 1), s0.rocks)
    d = transition(pomdp, s, 2) # move right
    sp = rand(rng, d)
    @test isterminal(pomdp, sp)
    @test sp == pomdp.terminal_state
    @inferred transition(pomdp, s0, 3)
    @inferred rand(rng, transition(pomdp, s0, 3))
    trans_prob_consistency_check(pomdp)
end

@testset "observation" begin 
    rng = MersenneTwister(1)
    pomdp = RockSamplePOMDP{3}(init_pos=(1,1))
    obs = observations(pomdp)
    @test obs == ordered_observations(pomdp)
    s0 = initialstate(pomdp, rng)
    od = observation(pomdp, 1, s0)
    o = rand(rng, od)
    @test o == 3
    @inferred observation(pomdp, 6, s0)
    @inferred observation(pomdp, 1, s0)
    o = rand(rng, observation(pomdp, 6, s0))
    @test o == 1
    o = rand(rng, observation(pomdp, 7, s0))
    @test o == 1
    obs_prob_consistency_check(pomdp)
end

@testset "reward" begin
    pomdp = RockSamplePOMDP{3}(init_pos=(1,1))
    rng = MersenneTwister(3)
    s = initialstate(pomdp, rng)
    @test reward(pomdp, s, 5, s) == pomdp.good_rock_reward
    @test reward(pomdp, s, 1, s) == 0.0
    s = RSState(RSPos(3,3), s.rocks)
    @test reward(pomdp, s, 5, s) == pomdp.bad_rock_penalty
    @test reward(pomdp, s, 2, s) == 0.0
    s = RSState(RSPos(5,4), s.rocks)
    sp = rand(rng, transition(pomdp, s, RockSample.BASIC_ACTIONS_DICT[:east]))
    @test reward(pomdp, s, RockSample.BASIC_ACTIONS_DICT[:east], sp) == pomdp.exit_reward
end

@testset "simulation" begin 
    pomdp = RockSamplePOMDP{3}(init_pos=(1,1))
    rng = MersenneTwister(3)
    up = DiscreteUpdater(pomdp)

    # go straight to the exit
    policy = FunctionPolicy(s->RockSample.BASIC_ACTIONS_DICT[:east]) 
    hr = HistoryRecorder(rng=rng)
    hist = simulate(hr, pomdp, policy, up)
    @test undiscounted_reward(hist) == pomdp.exit_reward
    @test discounted_reward(hist) ≈ discount(pomdp)^(n_steps(hist) - 1) * pomdp.exit_reward

    # random policy
    policy = RandomPolicy(pomdp, rng=rng)
    hr = HistoryRecorder(rng=rng)
    hist = simulate(hr, pomdp, policy, up)
    @test n_steps(hist) > pomdp.map_size[1]
end

@testset "rendering" begin 
    pomdp = RockSamplePOMDP{3}(init_pos=(1,1))
    render(pomdp, Dict(:s => (3,2)))
end