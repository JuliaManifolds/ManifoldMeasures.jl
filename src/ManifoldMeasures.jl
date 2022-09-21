module ManifoldMeasures

using KeywordCalls,
    LinearAlgebra,
    Manifolds,
    MeasureTheory,
    Random,
    SpecialFunctions,
    StaticArrays,
    StatsBase,
    StatsFuns
using HypergeometricFunctions: HypergeometricFunctions
using Manifolds: ℝ
using RealDot: realdot

const MAX_LENGTH_SIZED = 100

# experimental implementation of Option 3 in
# https://github.com/cscherrer/MeasureTheory.jl/issues/129#issue-962733685
default_number_type(::typeof(ℝ), T) = real(T)
default_number_type(::typeof(ℂ), T) = complex(real(T))

function default_point(μ, S::Type)
    M = base_manifold(μ)
    T = default_number_type(number_system(M), S)
    sz = representation_size(M)
    return prod(sz) ≤ MAX_LENGTH_SIZED ? MArray{Tuple{sz...},T}(undef) : Array{T}(undef, sz)
end

Manifolds.base_manifold(μ::AbstractMeasure) = base_manifold(basemeasure(μ))

include("utils.jl")
include("specialfunctions.jl")
include("factorizations.jl")
include("normalized.jl")
include("haar.jl")
include("hausdorff.jl")

include("vonmisesfisher.jl")
include("bingham.jl")
include("angularcentralgaussian.jl")

# primitive
export Normalized, Hausdorff, Haar, LeftHaar, RightHaar

export AngularCentralGaussian, Bingham, VonMises, VonMisesFisher

export mode, normalize

end
