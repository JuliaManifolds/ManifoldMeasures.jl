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

function Base.show(io::IO, mime::MIME"text/plain", μ::LeftHaar)
    print(io, "LeftHaar(")
    show(io, mime, base_manifold(μ))
    return print(io, ")")
end
function Base.show(io::IO, mime::MIME"text/plain", μ::RightHaar)
    print(io, "RightHaar(")
    show(io, mime, base_manifold(μ))
    return print(io, ")")
end

Manifolds.base_manifold(μ::Haar) = μ.group
Manifolds.base_group(μ::Haar) = μ.group
Manifolds.direction(μ::Haar) = μ.direction

MeasureTheory.logdensity(::Haar, x) = zero(eltype(x))
