# ParallelTemperingSamplers.jl

[![CI](https://github.com/yeongjunk/ParallelTemperingSamplers.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/yeongjunk/ParallelTemperingSamplers.jl/actions/workflows/CI.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A lightweight, algorithm-independent implementation of parallel tempering
(replica-exchange Monte Carlo) for Julia.

This package provides the replica-exchange workflow while leaving the local
Monte Carlo update to the user. It can therefore be combined with algorithms
such as random-walk Metropolis, Langevin Monte Carlo (LMC), or Hamiltonian
Monte Carlo (HMC).

## Installation

You can install it through the following command in julia REPL:

```julia
using Pkg
Pkg.add("ParallelTemperingSamplers")
```

## Interface

Users must define the following methods for their own subtype of
`AbstractReplicas`:

```julia
step!(reps::AbstractReplicas) =
    error("step!(reps) is not implemented.")

step_slot!(reps::AbstractReplicas, slot::Int) =
    error("step_slot!(reps, slot) is not implemented.")

getenergy(reps::AbstractReplicas, slot::Int) =
    error("getenergy(reps, slot) is not implemented.")

getstate(reps::AbstractReplicas, slot::Int) =
    error("getstate(reps, slot) is not implemented.")

getbeta(reps::AbstractReplicas, slot::Int) =
    error("getbeta(reps, slot) is not implemented.")

getwalkerid(reps::AbstractReplicas, slot::Int) =
    error("getwalkerid(reps, slot) is not implemented.")

swapwalkers!(reps::AbstractReplicas, slot_i::Int, slot_j::Int) =
    error("swapwalkers!(reps, slot_i, slot_j) is not implemented.")

Base.length(reps::AbstractReplicas) =
    error("length(reps) is not implemented.")
```

Temperature parameters belong to fixed slots, while states belong to walkers.
Replica exchange is implemented by changing the mapping from slots to walkers.

## Parameters and main functions

First define the exchange schedule and equilibration parameters:

```julia
edge_groups = [
    [(1, 2), (3, 4)],
    [(2, 3)],
]

exchange_params = ExchangeParams(edge_groups, 10)
equilibration_params = EquilibrationParams(100_000, 10_000)
```

`ExchangeParams(edge_groups, swap_every)` specifies which pairs of temperature
slots may exchange and how often exchanges are attempted. Each edge group is
used in cyclic order. Edges within a group must not share a slot.

`EquilibrationParams(n_sweeps, partition_every)` specifies the total number of
equilibration sweeps and the interval used to collect diagnostic histories.

The main driver functions are:

- `equilibrate!(reps, equilibration_params; ex_params=exchange_params)` runs
  equilibration and returns aggregate exchange and local-acceptance statistics.
- `monitor_equilibration!(reps, equilibration_params, exchange_params)` runs
  equilibration while recording energies, acceptance statistics, and walker
  diagnostics for each partition.
- `sample_replicas!(reps, sampling_params, exchange_params)` performs production
  sampling. Construct `sampling_params` with `SamplingParams(n_sweeps,
  sample_every, partition_every, beta_indices)`.

Equilibration, sampling, partitioning, and replica exchanges may use
independent intervals. Local updates are grouped into blocks ending at the next
scheduled event.

## Example

A self-contained [Gaussian example](examples/gaussian.jl) implements a minimal
`AbstractReplicas` subtype using random-walk Metropolis updates. It runs
parallel tempering and compares the sampled variance with the exact result
`1 / beta`.

Run it from the repository root with:

```bash
julia --project=. examples/gaussian.jl
```

For an optimized Langevin Monte Carlo implementation, see
[LMC.jl](https://github.com/yeongjunk/LMC.jl).

## Testing

Run the test suite with:

```julia
using Pkg
Pkg.test()
```

## Releases

- `v0.1.0`: Initial release.
- `v0.1.1`: Allow sampling and equilibration diagnostics within exchange blocks
  (divisibility conditions between event intervals are no longer required).

## License

ParallelTemperingSamplers.jl is released under the [MIT License](LICENSE).
