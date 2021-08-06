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

# The Hausdorff measure generalizes the notion of volume or area to a manifold.
# For manifolds embedded in real space, it can be intuitively understoof as the addition
# of an infinitesimal dimension and the limit of the volume in the embedded space as the
# infinitesimal dimension goes to 0.
# A more useful description comes from the area formula. Given an m-
# dimensional manifold M and a Lipschitz map f: ℝᵐ → M ⊆ ℝⁿ that embeds M in ℝⁿ, then the Hausdorff
# measure dμ(x) on M is is the pushforward of the Lebesgue measure through f, given as
# dμ(x) = det(G(x))^(-1/2) dλ(f⁻¹(x)), where G(x) is the matrix representation of the metric tensor
# in M, given by G(x) = J(x)ᵀ J(x), where J is the Jacobian of f.
# This is the un-normalized Hausdorff measure, which integrates to the area/volume of M.
struct Hausdorff{M<:AbstractManifold,A} <: PrimitiveMeasure
    manifold::M
    atlas::A  # optional, when the Hausdorff measure is constructed from a Lebesgue measure
end
Hausdorff(M::AbstractManifold) = Hausdorff(M, nothing)

function Base.show(io::IO, ::MIME"text/plain", μ::Hausdorff)
    return print(io, "Hausdorff", (μ.manifold, μ.atlas))
end

Manifolds.base_manifold(μ::Hausdorff) = μ.manifold

MeasureTheory.logdensity(::Hausdorff, x) = zero(eltype(x))

function Random.rand!(rng::AbstractRNG, p, μ::WeightedMeasure{S,<:Hausdorff}) where {S}
    return rand!(rng, p, μ.base)
end

# The Haar measure is a left- or right-invariant measure on a group manifold.
# That is, given p ∈ G and q ∈ H ⊆ G, where τ(H, p) is the set of all q τ-translated
# (left- or right-) by p, then μ(τ(H, p)) = μ(H). This is the τ-invariant Haar measure
# on G.
struct Haar{M<:AbstractManifold,D<:ActionDirection} <: PrimitiveMeasure
    group::M
    direction::D
end
Haar(G::AbstractManifold) = Haar(G, LeftAction())

const LeftHaar{M} = Haar{M,LeftAction}
LeftHaar(G) = Haar(G, LeftAction())

const RightHaar{M} = Haar{M,RightAction}
RightHaar(G) = Haar(G, RightAction())

function Base.show(io::IO, ::MIME"text/plain", μ::Haar)
    return print(io, "Haar", (μ.group, μ.direction))
end

Manifolds.base_manifold(μ::Haar) = μ.group
Manifolds.base_group(μ::Haar) = μ.group
Manifolds.direction(μ::Haar) = μ.direction

MeasureTheory.logdensity(::Haar, x) = zero(eltype(x))
