module ManifoldMeasures

using HypergeometricFunctions: HypergeometricFunctions
using KeywordCalls: KeywordCalls
using LinearAlgebra
using Manifolds:
    Manifolds,
    ℝ,
    ℂ,
    ℍ,
    AbstractProjectiveSpace,
    AbstractSphere,
    Circle,
    Grassmann,
    Rotations,
    SpecialOrthogonal,
    Sphere,
    Stiefel
using ManifoldsBase: ManifoldsBase
using MeasureBase: MeasureBase
using MeasureTheory: Bernoulli, Beta
using Random: Random
using RealDot: realdot
using SpecialFunctions: SpecialFunctions
using StaticArraysCore: StaticArraysCore
using StatsBase: StatsBase, mode
using StatsFuns

const MAX_LENGTH_SIZED = 100

# experimental implementation of Option 3 in
# https://github.com/cscherrer/MeasureTheory.jl/issues/129#issue-962733685
default_number_type(::typeof(ℝ), T) = real(T)
default_number_type(::typeof(ℂ), T) = complex(real(T))

function default_point(μ, S::Type)
    M = ManifoldsBase.base_manifold(μ)
    T = default_number_type(ManifoldsBase.number_system(M), S)
    sz = ManifoldsBase.representation_size(M)
    if prod(sz) ≤ MAX_LENGTH_SIZED
        return StaticArraysCore.MArray{Tuple{sz...},T}(undef)
    else
        return Array{T}(undef, sz)
    end
end

function ManifoldsBase.base_manifold(μ::MeasureBase.AbstractMeasure)
    return ManifoldsBase.base_manifold(MeasureBase.basemeasure(μ))
end

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
