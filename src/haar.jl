# The Haar measure is a left- or right-invariant measure on a group manifold.
# That is, given p ∈ G and q ∈ H ⊆ G, where τ(H, p) is the set of all q τ-translated
# (left- or right-) by p, then μ(τ(H, p)) = μ(H). This is the τ-invariant Haar measure
# on G.

"""
    Haar{G,D}

The Haar measure on a group manifold.

The Haar measure on a group ``G`` is a measure that is invariant to all left- and/or right-
group translations on the manifold. That is, given the ``τ``-invariant Haar measure ``μ``
and some subset ``H ⊆ G``, then ``∫_H \\mathrm{d}μ(p) = ∫_H \\mathrm{d}μ(τ_q(p))`` for all
``q ∈ G``, where ``τ_q(p)`` is group translation of ``p`` by ``q`` in the ``τ`` direction
(left- or right-).

The convenient aliases [`LeftHaar`](@ref) and [`RightHaar`](@ref) are also provided.

# Constructors

    Haar(G::AbstractManifold, D::ActionDirection = LeftAction())

Construct the `D`-invariant Haar measure on group manifold `G`.
"""
struct Haar{G,D} <: MeasureBase.PrimitiveMeasure
    group::G
    direction::D
end
Haar(G) = Haar(G, Manifolds.LeftAction())

"""
    LeftHaar{G<:AbstractManifold}

Alias for a left-[`Haar`](@ref) measure.
"""
const LeftHaar{G} = Haar{G,Manifolds.LeftAction}
LeftHaar(G) = Haar(G, Manifolds.LeftAction())

"""
RightHaar{G<:AbstractManifold}

Alias for a right-[`Haar`](@ref) measure.
"""
const RightHaar{M} = Haar{M,Manifolds.RightAction}
RightHaar(G) = Haar(G, Manifolds.RightAction())

function Base.show(io::IO, mime::MIME"text/plain", μ::LeftHaar)
    print(io, "LeftHaar(")
    show(io, mime, ManifoldsBase.base_manifold(μ))
    return print(io, ")")
end
function Base.show(io::IO, mime::MIME"text/plain", μ::RightHaar)
    print(io, "RightHaar(")
    show(io, mime, ManifoldsBase.base_manifold(μ))
    return print(io, ")")
end

ManifoldsBase.base_manifold(μ::Haar) = μ.group
Manifolds.base_group(μ::Haar) = μ.group
Manifolds.direction(μ::Haar) = μ.direction

function MeasureBase.insupport(μ::Haar, x)
    return ManifoldsBase.is_point(ManifoldsBase.base_manifold(μ), x)
end

MeasureBase.logdensity_def(::Haar, x) = zero(real(eltype(x)))
