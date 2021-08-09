"""
    Bingham(M; params...)

The Bingham distribution on the manifold ``M``.

Accepted manifolds are `Sphere`, `ProjectiveSpace`, `Stiefel`, `Grassmann`, `Rotations`
and `SpecialOrthogonal`.

For manifolds with matrix points, this is also called the Matrix Bingham distribution.

# Constructors

    Bingham(M; B)

For a manifold ``M ⊂ 𝔽^{n × k}``, construct the Bingham distribution parameterized by
some positive definite matrix ``B ∈ 𝔽^{n × n}``.

The density function with respect to the normalized [`Hausdorff`](@ref) measure on ``M`` is

```math
p(x | B) = \\frac{\\exp(\\Re⟨x, Bx⟩)}{_1 F_1(\\frac{k}{2}, \\frac{n}{2}; B)},
```

where ``⟨⋅,⋅⟩`` is the Frobenius inner product, and ``_1 F_1(a, b; B)``
is a hypergeometric function with matrix argument ``B``.
"""
struct Bingham{M,N,T} <: ParameterizedMeasure{N}
    manifold::M
    par::NamedTuple{N,T}
end
Bingham(M; params...) = Bingham(M, NamedTuple(params))

Manifolds.base_manifold(d::Bingham) = getfield(d, :manifold)

function MeasureTheory.basemeasure(μ::Bingham)
    return normalize(Hausdorff(base_manifold(μ)))
end

function MeasureTheory.logdensity(d::Bingham{M,(:B,)}, x) where {M}
    n = size(x, 1)
    k = size(x, 2)
    B = d.B
    return real(dot(x, B, x)) - logpFq((k//2,), (n//2,), B)
end
