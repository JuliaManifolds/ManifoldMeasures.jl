"""
    Bingham(M; params...)

The Bingham distribution on the manifold ``M``.

Accepted manifolds are `Sphere`, `ProjectiveSpace`, `Stiefel`, `Grassmann`, `Rotations`
and `SpecialOrthogonal`.

For manifolds with matrix points, this is also called the Matrix Bingham distribution.

# Constructors

    Bingham(M; A)

For a manifold ``M ⊂ 𝔽^{n × k}``, construct the Bingham distribution parameterized by
some positive definite matrix ``A ∈ 𝔽^{n × n}``.

The density function with respect to the normalized [`Hausdorff`](@ref) measure on ``M`` is

```math
p(x | A) = \\frac{\\exp(\\Re⟨x, Ax⟩)}{_1 F_1(\\frac{k}{2}, \\frac{n}{2}; A)},
```

where ``⟨⋅,⋅⟩`` is the Frobenius inner product, and ``_1 F_1(a, b; A)``
is a hypergeometric function with matrix argument ``A``.

Note that ``p(x | A + α I) = p(x | A)`` for all scalars ``α ∈ 𝔽``.
Hence, ``A`` can not be uniquely identified from draws from the Bingham distribution.
"""
struct Bingham{M,N,T} <: MeasureBase.ParameterizedMeasure{N}
    manifold::M
    par::NamedTuple{N,T}
end
Bingham(M; params...) = Bingham(M, NamedTuple(params))

function Base.show(io::IO, mime::MIME"text/plain", μ::Bingham)
    return show_manifold_measure(io, mime, μ)
end

ManifoldsBase.base_manifold(d::Bingham) = getfield(d, :manifold)

function MeasureBase.insupport(μ::Bingham, x)
    return ManifoldsBase.is_point(ManifoldsBase.base_manifold(μ), x)
end

function MeasureBase.basemeasure(μ::Bingham)
    return normalize(Hausdorff(ManifoldsBase.base_manifold(μ)))
end

function MeasureBase.logdensity_def(d::Bingham{M,(:A,)}, x) where {M}
    n = size(x, 1)
    k = size(x, 2)
    A = d.A
    return real(dot(x, A, x)) - logpFq((k//2,), (n//2,), A)
end
