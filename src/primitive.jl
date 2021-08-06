"""
    log_total_mass(μ::AbstractMeasure)

Compute the logarithm of the total mass of the measure `μ` over its manifold `M`, that is
`μ(M) = ∫_M dμ(x)`.
"""
function log_total_mass end

# Warning! Type-piracy! ☠️
LinearAlgebra.normalize(μ::AbstractMeasure) = MeasureTheory.Exp(-log_total_mass(μ)) * μ

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
struct Hausdorff{M<:AbstractManifold,P,A} <: PrimitiveMeasure
    manifold::M
    point::P  # for random sampling
    atlas::A  # optional, when the Hausdorff measure is constructed from a Lebesgue measure
end
Hausdorff(M::AbstractManifold, point) = Hausdorff(M, point, nothing)

function Base.show(io::IO, ::MIME"text/plain", μ::Hausdorff)
    return print(io, "Hausdorff", (μ.manifold, μ.point, μ.atlas))
end

Manifolds.base_manifold(μ::Hausdorff) = μ.manifold

MeasureTheory.sampletype(::Hausdorff{M,P}) where {M,P} = P

MeasureTheory.logdensity(::Hausdorff, x) = zero(eltype(x))

function Base.rand(rng::AbstractRNG, T::Type, μ::WeightedMeasure{S,<:Hausdorff}) where {S}
    return rand(rng, T, μ.base)
end

# The Haar measure is a left- or right-invariant measure on a group manifold.
# That is, given p ∈ G and q ∈ H ⊆ G, where τ(H, p) is the set of all q τ-translated
# (left- or right-) by p, then μ(τ(H, p)) = μ(H). This is the τ-invariant Haar measure
# on G.
struct Haar{M<:AbstractManifold,P,D<:ActionDirection} <: PrimitiveMeasure
    group::M
    point::P
    direction::D
end
Haar(G::AbstractManifold, point) = Haar(G, point, LeftAction())

const LeftHaar{M,P} = Haar{M,P,LeftAction}
LeftHaar(G, point) = Haar(G, point, LeftAction())

const RightHaar{M,P} = Haar{M,P,RightAction}
RightHaar(G, point) = Haar(G, point, RightAction())

function Base.show(io::IO, ::MIME"text/plain", μ::Haar)
    return print(io, "Haar", (μ.group, μ.point, μ.direction))
end

Manifolds.base_manifold(μ::Haar) = μ.group
Manifolds.base_group(μ::Haar) = μ.group
Manifolds.direction(μ::Haar) = μ.direction

MeasureTheory.sampletype(::Haar{M,P}) where {M,P} = P

MeasureTheory.logdensity(::Haar, x) = zero(eltype(x))
