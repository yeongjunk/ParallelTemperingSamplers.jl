using Random
using Test
using ParallelTemperingSamplers

import ParallelTemperingSamplers:
    AbstractReplicas,
    step!,
    step_slot!,
    getenergy,
    getstate,
    getbeta,
    getwalkerid,
    swapwalkers!

mutable struct GaussianReplicas{R<:AbstractRNG} <: AbstractReplicas
    states::Vector{Vector{Float64}}
    betas::Vector{Float64}
    walkerids::Vector{Int}
    proposal_width::Float64
    rng::R
end

function GaussianReplicas(
    betas::AbstractVector;
    proposal_width=0.5,
    rng=Xoshiro(1234),
)
    K = length(betas)
    states = [[randn(rng)] for _ in 1:K]

    return GaussianReplicas(
        states,
        Float64.(betas),
        collect(1:K),
        proposal_width,
        rng,
    )
end

Base.length(reps::GaussianReplicas) = length(reps.betas)

getwalkerid(reps::GaussianReplicas, slot::Int) =
    reps.walkerids[slot]

getbeta(reps::GaussianReplicas, slot::Int) =
    reps.betas[slot]

getstate(reps::GaussianReplicas, slot::Int) =
    reps.states[getwalkerid(reps, slot)]

function getenergy(reps::GaussianReplicas, slot::Int)
    x = getstate(reps, slot)[1]
    return x^2 / 2
end

function swapwalkers!(
    reps::GaussianReplicas,
    slot_i::Int,
    slot_j::Int,
)
    reps.walkerids[slot_i], reps.walkerids[slot_j] =
        reps.walkerids[slot_j], reps.walkerids[slot_i]

    return nothing
end

function step_slot!(reps::GaussianReplicas, slot::Int)
    walker = getwalkerid(reps, slot)
    x = reps.states[walker][1]
    x_proposed = x + reps.proposal_width * randn(reps.rng)

    energy = x^2 / 2
    energy_proposed = x_proposed^2 / 2
    beta = getbeta(reps, slot)
    log_acceptance = -beta * (energy_proposed - energy)

    if log(rand(reps.rng)) < min(0.0, log_acceptance)
        reps.states[walker][1] = x_proposed
        return true
    end

    return false
end

function step!(reps::GaussianReplicas)
    return [step_slot!(reps, slot) for slot in 1:length(reps)]
end

@testset "Gaussian parallel tempering" begin
    betas = [0.5, 1.0, 2.0, 4.0]
    reps = GaussianReplicas(betas)

    edge_groups = [
        [(1, 2), (3, 4)],
        [(2, 3)],
    ]

    exchange_params = ExchangeParams(edge_groups, 10)
    sampling_params = SamplingParams(
        100_000,
        10,
        10_000,
        [length(betas)],
    )

    result = sample_replicas!(
        reps,
        sampling_params,
        exchange_params;
        rng=Xoshiro(5678),
        sample_eltype=Float64,
    )

    samples = vec(result.samples[1, 1, :])
    sample_mean = sum(samples) / length(samples)
    sample_variance =
        sum(x -> abs2(x - sample_mean), samples) / (length(samples) - 1)

    @test size(result.samples) == (1, 1, 10_000)
    @test length(result.exchange_history) == 10
    @test length(result.acceptance_history) == 10
    @test all(isfinite, samples)

    @test abs(sample_mean) < 0.08
    @test isapprox(sample_variance, 1 / betas[end]; rtol=0.15)

    @test all(
        status -> sum(sum, status.n_attempts) > 0,
        result.exchange_history,
    )
    @test all(
        status -> sum(sum, status.n_accepts) > 0,
        result.exchange_history,
    )
end

@testset "Sampling within a swap block" begin
    betas = [0.5, 1.0, 2.0, 4.0]
    reps = GaussianReplicas(betas)

    exchange_params = ExchangeParams([[(1, 2), (3, 4)]], 100)
    sampling_params = SamplingParams(100, 10, 100, [length(betas)])

    result = sample_replicas!(
        reps,
        sampling_params,
        exchange_params;
        rng=Xoshiro(5678),
        sample_eltype=Float64,
    )

    @test size(result.samples) == (1, 1, 10)
    @test all(isfinite, result.samples)
    @test sum(result.exchange_history[1].n_attempts[1]) == 2
end

@testset "Equilibration event blocks" begin
    betas = [0.5, 1.0, 2.0, 4.0]
    exchange_params = ExchangeParams([[(1, 2), (3, 4)]], 100)
    equilibration_params = EquilibrationParams(105, 10)

    reps = GaussianReplicas(betas)
    result = equilibrate!(
        reps,
        equilibration_params;
        ex_params=exchange_params,
        rng=Xoshiro(5678),
    )

    @test result.acceptance.n_attempts == 105
    @test sum(result.exchange.n_attempts[1]) == 2

    reps = GaussianReplicas(betas)
    result = monitor_equilibration!(
        reps,
        equilibration_params,
        exchange_params;
        rng=Xoshiro(5678),
    )

    @test size(result.energies) == (length(betas), 10)
    @test all(isfinite, result.energies)
    @test all(status -> status.n_attempts == 10, result.acceptance)
    @test sum(status -> sum(sum, status.n_attempts), result.exchange) == 2
end
