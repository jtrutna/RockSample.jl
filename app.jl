import Cairo, Fontconfig
using POMDPs
using RockSample 
using SARSOP # load a  POMDP Solver
using POMDPGifs # to make gifs

pomdp = RockSamplePOMDP(sensor_efficiency=20.0, discount_factor=0.95, good_rock_reward = 20.0)

solver = SARSOPSolver(precision=1e-3)

policy = solve(solver, pomdp)

sim = GifSimulator(filename="test.gif", max_steps=30)
simulate(sim, pomdp, policy)