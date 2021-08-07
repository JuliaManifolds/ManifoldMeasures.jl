"""
    logmass(μ::AbstractMeasure)

Compute the logarithm of the total mass of the measure `μ` over its manifold `M`, that is
`μ(M) = ∫_M dμ(x)`.
"""
function logmass end

mass(μ) = MeasureTheory.Exp(-logmass(μ))

# Warning! Type-piracy! ☠️
LinearAlgebra.normalize(μ::AbstractMeasure) = mass(μ) * μ

Manifolds.base_manifold(μ::AbstractMeasure) = base_manifold(basemeasure(μ))
Manifolds.manifold_dimension(μ::AbstractMeasure) = manifold_dimension(base_manifold(μ))
MeasureTheory.testvalue(μ::PrimitiveMeasure) = μ.point
