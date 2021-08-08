module ManifoldMeasures

using KeywordCalls,
    LinearAlgebra, Manifolds, MeasureTheory, Random, SpecialFunctions, StatsFuns
using Manifolds: ℝ

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

export AngularCentralGaussian, Bingham, Langevin, VonMisesFisher

export normalize

end
