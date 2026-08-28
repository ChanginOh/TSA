include("algTSA.jl")

# ======================================================
# List of benchmark functions for testing TSA
# ======================================================

# Sphere
# Global minimum: [0, 0, ..., 0]
# Global minimum value: 0
# Range: -100 ≤ xᵢ ≤ 100
sphere(x) = sum(abs2, x)

# Rastrigin
# Global minimum: [0, 0, ..., 0]
# Global minimum value: 0
# Range: -5.12 ≤ xᵢ ≤ 5.12
function rastrigin(x)
    n = length(x)
    s = 0.0
    @inbounds for i in eachindex(x)
        xi = x[i]
        s += xi^2 - 10 * cos(2π * xi)
    end
    return 10 * n + s
end

# Griewank
# Global minimum: [0, 0, ..., 0]
# Global minimum value: 0
# Range: -600 ≤ xᵢ ≤ 600
function griewank(x)
    s = 0.0
    p = 1.0
    @inbounds for i in eachindex(x)
        xi = x[i]
        s += xi * xi
        p *= cos(xi / sqrt(i))
    end
    return s / 4000 - p + 1
end

# ======================================================
# Set-up for testing TSA on benchmark functions
# ======================================================

# Define lower bounds and upper bounds for 100-dimensional search space
bounds = [fill(-600.0, 100)'; fill(600.0, 100)']

# Set population size, number of iterations, and search balance constant
alg = TSA(N = 5, k = 2.0, information = Information(), options = Options(iterations = 500, seed = 1, store_convergence = true))

# Run TSA and store results
result = optimize(griewank, bounds, alg)

# Report best minimum value, minimum point, and number of function evaluations
println("Minimum: ", minimum(result))
println("Minimizer: ", minimizer(result))
println("Function calls: ", result.f_calls)