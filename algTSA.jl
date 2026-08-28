using Metaheuristics

import Metaheuristics: AbstractParameters, Algorithm, BoxConstrainedSpace, AbstractProblem
import Metaheuristics: initialize!, update_state!, final_stage!
import Metaheuristics: gen_initial_state, get_position, optimize
import Metaheuristics: create_solution, is_better

mutable struct TSA <: AbstractParameters
    N::Int
    k::Float64
    W::Matrix{Float64}
end

function TSA(; N::Int=10, k::Real=2.0, kargs...)
    N >= 2 || throw(ArgumentError("TSA requires a population size N ≥ 2."))
    k > 0 || throw(ArgumentError("The sigmoid parameter k must be positive."))

    parameters = TSA(N, Float64(k), zeros(Float64, 0, 0))

    return Algorithm(parameters; kargs...)
end

function initialize!(status, parameters::TSA, problem::AbstractProblem, information::Information, options::Options, args...; kargs...)
    if !(problem.search_space isa BoxConstrainedSpace)
        throw(ArgumentError("TSA only supports box-constrained search spaces."))
    end

    lb = problem.search_space.lb
    ub = problem.search_space.ub
    
    all(isfinite, lb) || throw(ArgumentError("All lower bounds must be finite."))
    all(isfinite, ub) || throw(ArgumentError("All upper bounds must be finite."))
    all(ub .> lb) || throw(ArgumentError("Each upper bound must be strictly greater than its lower bound."))
    
    if options.iterations == 0
        options.iterations = 500
    end

    if options.f_calls_limit == 0
        options.f_calls_limit = options.iterations * parameters.N + 1
    end

    status = gen_initial_state(problem, parameters, information, options, status)

    D = length(lb)
    parameters.W = zeros(Float64, D, length(status.population))

    return status
end

function update_state!(status, parameters::TSA, problem::AbstractProblem, information::Information, options::Options, args...; kargs...)
    population = status.population
    N = length(population)

    lb = problem.search_space.lb
    ub = problem.search_space.ub
    ul_diff = ub .- lb
    D = length(lb)

    rng = options.rng

    t = status.iteration - 1
    T = options.iterations

    s = 1.0 / (1.0 + exp(100.0 * (parameters.k * t / T - 1.0)))

    Pbest = copy(get_position(status.best_sol))

    kwrap = Vector{Int}(undef, D)
    Pnew = Vector{Float64}(undef, D)

    for i in 1:N
        solᵢ = population[i]
        Pᵢ = get_position(solᵢ)
        Wᵢ = @view parameters.W[:, i]

        if rand(rng) < 0.8
            j = rand(rng, 1:N-1)
            j = (j >= i) ? j + 1 : j

            Pⱼ = get_position(population[j])

            winding_norm = 0.0
            @inbounds for d in 1:D
                winding_norm += Wᵢ[d]^2
            end
            scaling = 1.0 / (1.0 + winding_norm)

            r₁ = rand(rng)

            @inbounds for d in 1:D
                Pᵢd = Pᵢ[d]
                Pⱼd = Pⱼ[d]
                Pbd = Pbest[d]

                cosθ = cospi(rand(rng))

                width = ul_diff[d]
                direct_diff = abs(Pbd - r₁ * Pᵢd)
                distance = max(direct_diff, width - direct_diff)

                Pnew[d] = Pᵢd + s * abs(cosθ) * sign(Pᵢd) * distance +
                          (1 - s) * scaling * cosθ * (Pᵢd - Pⱼd)
            end
        else
            r₂ = 1.0 + rand(rng)
            @inbounds for d in 1:D
                Pᵢd = Pᵢ[d]
                Pbd = Pbest[d]
                cosθ = cospi(rand(rng))
                Pnew[d] = Pᵢd + abs(cosθ) * (Pbd - r₂ * Pᵢd)
            end
        end

        @inbounds for d in 1:D
            Pnd = Pnew[d]
            lower = lb[d]
            width = ul_diff[d]
            kwrap[d] = floor(Int, (Pnd - lower) / width)
            Pnew[d] = lower + mod(Pnd - lower, width)
        end

        offspring = create_solution(copy(Pnew), problem)

        if is_better(offspring, solᵢ)
            @inbounds for d in 1:D
                parameters.W[d, i] += kwrap[d]
            end

            population[i] = offspring

            if is_better(offspring, status.best_sol)
                status.best_sol = offspring
                Pbest = copy(get_position(offspring))
            end
        end
    end

    return
end

function final_stage!(status, parameters::TSA, problem, information, options, args...; kargs...)
    status.final_time = time()
    return
end