#
# Correctness Tests
#

using Optim
using Base.Test
using Compat

my_tests = [
    "api.jl",
    "types.jl",
    "gradient_descent.jl",
    "accelerated_gradient_descent.jl",
    "momentum_gradient_descent.jl",
    "grid_search.jl",
    "cg.jl",
    "bfgs.jl",
    "l_bfgs.jl",
    "nelder_mead.jl",
    "newton.jl",
    "newton_trust_region.jl",
    "particle_swarm.jl",
    "simulated_annealing.jl",
    "levenberg_marquardt.jl",
    "optimize.jl",
    "golden_section.jl",
    "brent.jl",
    "type_stability.jl",
    "array.jl",
    "constrained.jl",
    "callbacks.jl",
    "precon.jl",
    "initial_convergence.jl",
    "extrapolate.jl"
]

println("Running tests:")

for my_test in my_tests
    println(" * $(my_test)")
    include(my_test)
end
