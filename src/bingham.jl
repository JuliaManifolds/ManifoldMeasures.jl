const BinghamManifolds = Union{
    Sphere,ProjectiveSpace,Stiefel,Grassmann,Rotations,SpecialOrthogonal
}

"""
    Bingham(M; params...)

The Bingham distribution on the manifold ``M``.

Accepted manifolds are `Sphere`, `ProjectiveSpace`, `Stiefel`, `Grassmann`, `Rotations`
and `SpecialOrthogonal`.

For manifolds with matrix points, this is also called the Matrix Bingham distribution.

# Constructors

    Bingham(M; B)

For a manifold ``M ⊂ 𝔽^{n × k}``, construct the Bingham distribution parameterized by
``B ∈ 𝔽^{n × k}``.

The density function with respect to the normalized [`Hausdorff`](@ref) measure on ``M`` is

```math
p(x | B) = \\frac{\\exp(⟨x, Bx⟩)}{_1 F_1(\\frac{k}{2}, \\frac{n}{2}; B)},
```

where ``⟨⋅,⋅⟩`` is the Frobenius inner product, and ``_1 F_1(a, b; B)``
is hypergeometric function with matrix arguments.
"""
struct Bingham{M,N,T} <: ParameterizedMeasure{N}
    manifold::M
    par::NamedTuple{N,T}
end
Bingham(M::AbstractManifold; kwargs...) = Bingham(M, NamedTuple(kwargs))

Manifolds.base_manifold(d::Bingham) = getfield(d, :manifold)

function MeasureTheory.basemeasure(μ::Bingham{<:BinghamManifolds})
    return normalize(Hausdorff(base_manifold(μ)))
end

function MeasureTheory.logdensity(d::Bingham{<:BinghamManifolds,(:B,)}, x::AbstractArray)
    s = representation_size(base_manifold(d))
    n, k = length(s) == 1 ? (first(s), 1) : s
    B = d.B
    return real(dot(x, B, x)) - logpFq((k//2,), (n//2,), B)
end
