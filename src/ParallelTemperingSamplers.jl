module ParallelTemperingSamplers

using Random, LinearAlgebra

export AbstractReplicas, step!, step_slot!, try_exchange,
    ExchangeParams, ExchangeStatus, WalkerStatus,
    AcceptanceStatus, EquilibrationParams, EquilibrationStats,
    equilibrate!, monitor_equilibration!,
    replica_exchanges!, compute_exchange_rates, compute_acceptance_rates,
    check_edge_groups,
    getstate, getenergy, getenergies,
    getbeta, getbetas, getwalkerid, getwalkerids, swapwalkers!,
    SamplingParams, sample_replicas!

export Edge, EdgeGroupsOf, ExchangeRates, AcceptanceRates

## AbstractReplicas interface
abstract type AbstractReplicas end

step!(reps::AbstractReplicas) = error("step!(reps) is not implemented.")
step_slot!(reps::AbstractReplicas, slot::Int) = error("step_slot!(reps, slot) is not implemented.")
getenergy(::AbstractReplicas, slot::Int) = error("getenergy(reps, slot) is not implemented.")
getstate(::AbstractReplicas, slot::Int) = error("getstate(reps, slot) is not implemented.")
getbeta(::AbstractReplicas, slot::Int) = error("getbeta(reps, slot) is not implemented.")
getwalkerid(::AbstractReplicas, slot::Int) = error("getwalkerid(reps, slot) is not implemented.")
swapwalkers!(::AbstractReplicas, slot_i::Int, slot_j::Int) = error("swapwalkers!(reps, slot_i, slot_j) is not implemented.")

Base.length(::AbstractReplicas) = error("length(reps) is not implemented.")

getenergies(reps::AbstractReplicas)  = [getenergy(reps, slot) for slot in 1:length(reps)]
getbetas(reps::AbstractReplicas)     = [getbeta(reps, slot) for slot in 1:length(reps)]
getwalkerids(reps::AbstractReplicas) = [getwalkerid(reps, slot) for slot in 1:length(reps)]

function steps!(reps::AbstractReplicas, n_steps::Int)
    @warn "Using fallback steps!(reps, n_steps) implementation." maxlog=1

    n_steps > 0 || error("n_steps must be positive.")

    n_accepts = Int.(step!(reps))

    for _ in 2:n_steps
        n_accepts .+= step!(reps)
    end

    return n_accepts
end

function getenergies!(out, reps::AbstractReplicas)
    @inbounds @simd for i in 1:length(reps)
        out[i] = getenergy(reps, i)
    end
end

## Type alias
"ParallelTemperingSamplers.Edge = Tuple{Int, Int}"
const Edge = Tuple{Int, Int}

"""
ParallelTemperingSamplers.EdgeGroupsOf{T} = Vector{Vector{T}}. The outer vector contains edge groups, and each inner vector contains the edges attempted together.
"""
const EdgeGroupsOf{T} = Vector{Vector{T}}

"CONST: ParallelTemperingSamplers.ExchangeRates = EdgeGroupsOf{Float64}"
const ExchangeRates   = EdgeGroupsOf{Float64}

"CONST: ParallelTemperingSamplers.AcceptanceRates = Vector{Float64}"
const AcceptanceRates = Vector{Float64}


function check_replica_implementation(reps::AbstractReplicas)
    K = length(reps)
    energies = getenergies(reps)
    betas = getbetas(reps)
    walkerids = getwalkerids(reps)

    length(energies) == K || error("energies must have length K.")
    length(betas) == K || error("betas must have length K.")
    length(walkerids) == K || error("walkerids must have length K.")
    sort(walkerids) == collect(1:K) || error("walkerids must be a permutation of 1:K.")

    return nothing
end

function replica_exchanges!(reps::AbstractReplicas, edges::Vector{Edge}, n_attempts::Vector{Int}, n_accepts::Vector{Int}; rng = Random.GLOBAL_RNG)
    length(n_attempts) == length(edges) || error("n_attempts has wrong length.")
    length(n_accepts) == length(edges) || error("n_accepts has wrong length.")

    @inbounds for e in eachindex(edges)
        slot_i, slot_j = edges[e]

        E_i = getenergy(reps, slot_i)
        E_j = getenergy(reps, slot_j)
        beta_i = getbeta(reps, slot_i)
        beta_j = getbeta(reps, slot_j)

        n_attempts[e] += 1

        if try_exchange(E_i, E_j, beta_i, beta_j; rng=rng)
            swapwalkers!(reps, slot_i, slot_j)
            n_accepts[e] += 1
        end
    end

    return nothing
end

function try_exchange(E_i, E_j, beta_i, beta_j; rng = Random.GLOBAL_RNG)
    log_alpha = (beta_i - beta_j) * (E_i - E_j)

    log_alpha >= 0 && return true
    isfinite(log_alpha) || return false

    return log(rand(rng)) < log_alpha
end

## Exchange parameters and status updates

"""
    ExchangeParams(edge_groups, swap_every)

Configure replica exchanges. `edge_groups` is a vector of groups of slot pairs;
one group is used per exchange round in cyclic order. `swap_every` is the number
of local Monte Carlo sweeps between exchange rounds.
"""
struct ExchangeParams
    edge_groups::EdgeGroupsOf{Edge}
    swap_every::Int
end

struct ExchangeStatus
    edge_groups::EdgeGroupsOf{Edge}
    n_attempts::EdgeGroupsOf{Int}
    n_accepts::EdgeGroupsOf{Int}
end

function ExchangeStatus(params::ExchangeParams)
    n_groups = length(params.edge_groups)

    n_attempts = EdgeGroupsOf{Int}(undef, n_groups)
    n_accepts  = EdgeGroupsOf{Int}(undef, n_groups)

    for g in 1:n_groups
        n_edges = length(params.edge_groups[g])
        n_attempts[g] = zeros(Int, n_edges)
        n_accepts[g]  = zeros(Int, n_edges)
    end

    return ExchangeStatus(params.edge_groups, n_attempts, n_accepts)
end

function compute_exchange_rates(status::ExchangeStatus)
    n_groups = length(status.n_attempts)

    length(status.n_accepts) == n_groups || error("n_attempts and n_accepts have inconsistent numbers of groups.")

    rates = EdgeGroupsOf{Float64}(undef, n_groups)

    for g in 1:n_groups
        n_edges = length(status.n_attempts[g])

        length(status.n_accepts[g]) == n_edges || error("n_attempts[$g] and n_accepts[$g] have inconsistent lengths.")

        rates[g] = Vector{Float64}(undef, n_edges)

        for e in 1:n_edges
            n_attempts = status.n_attempts[g][e]
            n_accepts = status.n_accepts[g][e]

            rates[g][e] = n_attempts == 0 ? NaN : n_accepts / n_attempts
        end
    end

    return rates
end


mutable struct AcceptanceStatus
    n_accepts::Vector{Int}
    n_attempts::Int
end

AcceptanceStatus(K::Int) = AcceptanceStatus(zeros(Int, K), 0)

function update_acceptance!(status::AcceptanceStatus, accepted::Union{Nothing, AbstractVector{Bool}})
    status.n_attempts += 1

    if accepted !== nothing
        length(accepted) == length(status.n_accepts) || error("accepted has wrong length.")
        status.n_accepts .+= accepted
    end

    return nothing
end

function update_acceptance!(status::AcceptanceStatus, n_accepts::AbstractVector{<:Integer}, n_attempts::Int)
    status.n_attempts += n_attempts
    status.n_accepts .+= n_accepts
    return nothing
end

function compute_acceptance_rates(status::AcceptanceStatus)
    return status.n_accepts ./ max(status.n_attempts, 1)
end

function check_edge_groups(edge_groups::Vector{Vector{Tuple{Int,Int}}}, len_replica::Int)
    K = len_replica

    length(edge_groups) > 0 || error("edge_groups must be nonempty.")

    for edges in edge_groups
        used = falses(K)

        for (i, j) in edges
            1 <= i <= K || error("edge index out of range.")
            1 <= j <= K || error("edge index out of range.")
            i != j || error("self edge is not allowed.")
            !used[i] || error("edge group has duplicated vertex.")
            !used[j] || error("edge group has duplicated vertex.")

            used[i] = true
            used[j] = true
        end
    end

    return nothing
end

## Walker interface

mutable struct WalkerStatus
    slot_counts::Matrix{Int}
    last_endpoint::Vector{Int}
    endpoint_crossings::Vector{Int}
    hot_visits::Vector{Int}
    cold_visits::Vector{Int}
end

function WalkerStatus(K::Int)
    K > 0 || error("K must be positive.")

    slot_counts = zeros(Int, K, K)
    last_endpoint = zeros(Int, K)
    endpoint_crossings = zeros(Int, K)
    hot_visits = zeros(Int, K)
    cold_visits = zeros(Int, K)

    return WalkerStatus(slot_counts, last_endpoint, endpoint_crossings, hot_visits, cold_visits)
end

function update_walker_status!(mon::WalkerStatus, walkerids::AbstractVector{Int})
    K = length(walkerids)
    all(w -> 1 <= w <= K, walkerids) || error("walkerids out of range.")
    size(mon.slot_counts) == (K, K) || error("slot_counts must have size K x K.")
    length(mon.last_endpoint) == K || error("last_endpoint must have length K.")
    length(mon.endpoint_crossings) == K || error("endpoint_crossings must have length K.")
    length(mon.hot_visits) == K || error("hot_visits must have length K.")
    length(mon.cold_visits) == K || error("cold_visits must have length K.")

    hot_slot = 1
    cold_slot = K

    @inbounds for slot in 1:K
        walker = walkerids[slot]

        mon.slot_counts[walker, slot] += 1

        is_hot = (slot == hot_slot)
        is_cold = (slot == cold_slot)

        is_hot && (mon.hot_visits[walker] += 1)
        is_hot && (mon.last_endpoint[walker] == 2) && (mon.endpoint_crossings[walker] += 1)
        is_hot && (mon.last_endpoint[walker] = 1)

        is_cold && (mon.cold_visits[walker] += 1)
        is_cold && (mon.last_endpoint[walker] == 1) && (mon.endpoint_crossings[walker] += 1)
        is_cold && (mon.last_endpoint[walker] = 2)
    end

    return nothing
end

function Base.copy(status::WalkerStatus)
    return WalkerStatus(copy(status.slot_counts), copy(status.last_endpoint), copy(status.endpoint_crossings), copy(status.hot_visits), copy(status.cold_visits))
end

## Equilibration


"""
    EquilibrationParams(n_sweeps, partition_every)

Configure equilibration. `n_sweeps` is the total number of local Monte Carlo
sweeps, and `partition_every` sets the interval used to record diagnostics in
[`monitor_equilibration!`](@ref).
"""
struct EquilibrationParams
    n_sweeps::Int
    partition_every::Int
end

mutable struct EquilibrationStats
    exchange::ExchangeStatus
    walker::WalkerStatus
    acceptance::AcceptanceStatus
end

function EquilibrationStats(params::ExchangeParams, K::Int)
    return EquilibrationStats(ExchangeStatus(params), WalkerStatus(K), AcceptanceStatus(K))
end

function replica_sweep!(reps::AbstractReplicas, n_steps::Int, acceptance_status::AcceptanceStatus)
    accepted = steps!(reps, n_steps)
    update_acceptance!(acceptance_status, accepted)
    return nothing
end

"""
    equilibrate!(reps, eq_params; ex_params, rng=Random.GLOBAL_RNG)

Equilibrate `reps` using the local update supplied by its `AbstractReplicas`
implementation and the exchange schedule in `ex_params`. Return aggregate
exchange and local-acceptance statistics.
"""
function equilibrate!(reps::AbstractReplicas, eq_params::EquilibrationParams; ex_params::ExchangeParams, rng = Random.GLOBAL_RNG)
    K = length(reps)

    eq_params.n_sweeps > 0 || error("n_sweeps must be positive.")
    eq_params.partition_every > 0 || error("partition_every must be positive.")
    ex_params.swap_every > 0 || error("swap_every must be positive.")
    check_edge_groups(ex_params.edge_groups, K)

    eq_params.n_sweeps % ex_params.swap_every == 0 || error("n_sweeps must be divisible by swap_every.")

    stats = EquilibrationStats(ex_params, K)
    n_blocks = eq_params.n_sweeps ÷ ex_params.swap_every

    for block in 1:n_blocks
        accepted = steps!(reps, ex_params.swap_every)
        update_acceptance!(stats.acceptance, accepted, ex_params.swap_every)

        g = mod1(block, length(ex_params.edge_groups))
        replica_exchanges!(
            reps,
            ex_params.edge_groups[g],
            stats.exchange.n_attempts[g],
            stats.exchange.n_accepts[g];
            rng=rng,
        )

        update_walker_status!(stats.walker, getwalkerids(reps))
    end

    return (exchange=stats.exchange, acceptance=stats.acceptance)
end

"""
    monitor_equilibration!(reps, eq_params, ex_params; rng=Random.GLOBAL_RNG)

Equilibrate `reps` while recording energies, exchange statistics, local
acceptance statistics, and walker round-trip diagnostics at each partition.
"""
function monitor_equilibration!(reps::AbstractReplicas, eq_params::EquilibrationParams, ex_params::ExchangeParams; rng = Random.GLOBAL_RNG)
    K = length(reps)

    eq_params.n_sweeps > 0 || error("n_sweeps must be positive.")
    eq_params.partition_every > 0 || error("partition_every must be positive.")
    ex_params.swap_every > 0 || error("swap_every must be positive.")
    check_edge_groups(ex_params.edge_groups, K)
    
    eq_params.n_sweeps % ex_params.swap_every == 0 || error("n_sweeps must be divisible by swap_every.")
    
    eq_params.partition_every % ex_params.swap_every == 0 || error("partition_every must be divisible by swap_every.")

    n_blocks = eq_params.n_sweeps ÷ ex_params.swap_every
    n_partitions = eq_params.n_sweeps ÷ eq_params.partition_every


    energies = Matrix{Float64}(undef, K, n_partitions)
    exchange_history = Vector{ExchangeStatus}(undef, n_partitions)
    acceptance_history = Vector{AcceptanceStatus}(undef, n_partitions)
    walker_history = Vector{WalkerStatus}(undef, n_partitions)

    exchange_status = ExchangeStatus(ex_params)
    acceptance_status = AcceptanceStatus(K)
    walker_status = WalkerStatus(K)
    partition = 0

    swap_every = ex_params.swap_every
    edge_groups= ex_params.edge_groups

    for block in 1:n_blocks
        accepted = steps!(reps, swap_every)
        update_acceptance!(acceptance_status, accepted, swap_every)

        g = mod1(block, length(ex_params.edge_groups))
        replica_exchanges!(reps, edge_groups[g], exchange_status.n_attempts[g], exchange_status.n_accepts[g]; rng=rng)

        update_walker_status!(walker_status, getwalkerids(reps))

        sweep = block * ex_params.swap_every

        if sweep % eq_params.partition_every == 0
            partition += 1
    
            @views getenergies!(energies[:, partition], reps)
            exchange_history[partition] = exchange_status
            acceptance_history[partition] = acceptance_status

            walker_history[partition] = copy(walker_status)

            exchange_status = ExchangeStatus(ex_params)
            acceptance_status = AcceptanceStatus(K)
        end
    end
    return (energies=energies, exchange=exchange_history, acceptance=acceptance_history, walker=walker_history)
end

## Sampling


"""
    SamplingParams(n_sweeps, sample_every, partition_every, beta_indices)

Configure production sampling. States are saved every `sample_every` sweeps
from the slots in `beta_indices`; exchange and acceptance statistics are
collected every `partition_every` sweeps.
"""
struct SamplingParams
    n_sweeps::Int
    sample_every::Int
    partition_every::Int
    beta_indices::Vector{Int}
end

SamplingParams(n_sweeps::Int, sample_every::Int, partition_every::Int, K::Int) = SamplingParams(n_sweeps, sample_every, partition_every, collect(1:K))

"""
    sample_replicas!(reps, sampling_params, exchange_params;
                     rng=Random.GLOBAL_RNG, sample_eltype=ComplexF64)

Run production parallel-tempering sampling and return saved states, their
inverse temperatures, and partitioned exchange and acceptance histories.
"""
function sample_replicas!(reps::AbstractReplicas, sampling_params::SamplingParams, exchange_params::ExchangeParams; rng = Random.GLOBAL_RNG, sample_eltype = ComplexF64)
    K = length(reps)

    sampling_params.n_sweeps > 0 || error("n_sweeps must be positive.")
    sampling_params.sample_every > 0 || error("sample_every must be positive.")
    sampling_params.partition_every > 0 || error("partition_every must be positive.")
    exchange_params.swap_every > 0 || error("swap_every must be positive.")
    check_edge_groups(exchange_params.edge_groups, K)

    sample_indices = sampling_params.beta_indices
    all(k -> 1 <= k <= K, sample_indices) || error("beta_indices out of range.")

    n_samples = sampling_params.n_sweeps ÷ sampling_params.sample_every
    n_partitions = sampling_params.n_sweeps ÷ sampling_params.partition_every
    n_slots = length(sample_indices)

    state_dim = length(getstate(reps, first(sample_indices)))
    all(slot -> length(getstate(reps, slot)) == state_dim, sample_indices) || error("All sampled states must have the same dimension.")

    samples = Array{sample_eltype}(undef, state_dim, n_slots, n_samples)
    sampled_betas = Matrix{Float64}(undef, n_slots, n_samples)
    exchange_history = Vector{ExchangeStatus}(undef, n_partitions)
    acceptance_history = Vector{AcceptanceStatus}(undef, n_partitions)

    exchange_status = ExchangeStatus(exchange_params)
    acceptance_status = AcceptanceStatus(K)
    
    sample = 0
    partition = 0

    swap_every = exchange_params.swap_every
    edge_groups = exchange_params.edge_groups

    sweep = 0
    swap_round = 0

    while sweep < sampling_params.n_sweeps
        block_size = min(
            swap_every - sweep % swap_every,
            sampling_params.sample_every - sweep % sampling_params.sample_every,
            sampling_params.partition_every - sweep % sampling_params.partition_every,
            sampling_params.n_sweeps - sweep,
        )

        # 1. Local updates up to the next swap, sample, or partition boundary
        accepted = steps!(reps, block_size)
        update_acceptance!(acceptance_status, accepted, block_size)
        sweep += block_size

        # 2. Replica exchanges
        if sweep % swap_every == 0
            swap_round += 1
            g = mod1(swap_round, length(edge_groups))
            replica_exchanges!(reps, edge_groups[g], exchange_status.n_attempts[g], exchange_status.n_accepts[g]; rng=rng)
        end

        # 3. Sampling
        if sweep % sampling_params.sample_every == 0
            sample += 1

            @inbounds for (j, slot) in enumerate(sample_indices)
                copyto!(@view(samples[:, j, sample]), getstate(reps, slot))
                sampled_betas[j, sample] = getbeta(reps, slot)
            end
        end

        # 4. Partitioning
        if sweep % sampling_params.partition_every == 0
            partition += 1

            exchange_history[partition] = exchange_status
            acceptance_history[partition] = acceptance_status

            exchange_status = ExchangeStatus(exchange_params)
            acceptance_status = AcceptanceStatus(K)
        end
    end

    return (
        samples=samples,
        betas=sampled_betas,
        exchange_history=exchange_history,
        acceptance_history=acceptance_history,
    )
end

end # module
