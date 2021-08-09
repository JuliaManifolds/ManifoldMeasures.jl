"""
    VonMisesFisher(M; params...)

The von Mises-Fisher (vMF) distribution on the `Sphere`, `Stiefel`, `Rotations`,
or `SpecialOrthogonal` manifold `M`.

# Constructors

    VonMisesFisher(M::AbstractSphere{𝔽}; c)
    VonMisesFisher(M::AbstractSphere{𝔽}; μ, κ)

Construct the vMF distribution on the `Sphere` parameterized either by the mean direction
``μ ∈ 𝔽𝕊ⁿ`` and concentration ``κ ∈ ℝ⁺`` or by the single vector ``c = κμ``.

The density of the vMF distribution on `Sphere(n, 𝔽)` with respect to the normalized
Hausdorff measure is

```math
p(x | μ, κ) = \\frac{κ^{(n-1)/2}}{I_{(n-1)/2}(κ)} \\exp(κ \\Re⟨μ, x⟩)),
```

where ``⟨⋅,⋅⟩`` is the Frobenius inner product, and ``I_ν(z)`` is the modified Bessel
function of the first kind.

    VonMisesFisher(M::Stiefel{n,k,𝔽}; F)
    VonMisesFisher(M::Stiefel{n,k,𝔽}; M, D, Vt)

Construct the (Matrix) vMF distribution on the `Stiefel(n,k,𝔽)` manifold parameterized
either by the matrix ``F ∈ 𝔽^{n × k}`` or by its SVD decomposition ``F = M * D * Vt``.

Because `Stiefel(n, n) = \\mathrm{SO}(n)`, these constructors also apply to `Rotations(n)`
and `SpecialOrthogonal(n)`.

The density of the vMF distribution on `Stiefel(n, k, 𝔽)` with respect to the normalized
Hausdorff measure is

```math
p(x | F) = \\frac{\\exp(\\Re⟨F, x⟩)}{_0 F_1(\\frac{n}{2}; \\frac{1}{4} F^\\mathrm{H}F)},
```

where ``_0 F_1(b; B)`` is a hypergeometric function with matrix argument ``B``.
"""
struct VonMisesFisher{M,N,T} <: ParameterizedMeasure{N}
    manifold::M
    par::NamedTuple{N,T}
end
VonMisesFisher(M; params...) = VonMisesFisher(M, NamedTuple(params))

const Langevin = VonMisesFisher

function Base.show(io::IO, mime::MIME"text/plain", μ::VonMisesFisher)
    return show_manifold_measure(io, mime, μ)
end

Manifolds.base_manifold(d::VonMisesFisher) = getfield(d, :manifold)

function MeasureTheory.basemeasure(μ::VonMisesFisher)
    return normalize(Hausdorff(base_manifold(μ)))
end

function MeasureTheory.logdensity(d::VonMisesFisher{M,(:μ, :κ)}, x) where {M}
    p = length(x)
    κ = d.κ
    return κ * real(dot(d.μ, x)) - logvmfnorm(p, κ)
end
function MeasureTheory.logdensity(d::VonMisesFisher{M,(:c,)}, x) where {M}
    p = length(x)
    c = d.c
    return real(dot(c, x)) - logvmfnorm(p, norm(c))
end
function MeasureTheory.logdensity(d::VonMisesFisher{M,(:F,)}, x) where {M}
    n = size(x, 1)
    F = d.F
    return real(dot(F, x)) - logpFq((), (n//2,), (F'F) / 4)
end
function MeasureTheory.logdensity(d::VonMisesFisher{M,(:M, :D, :Vt)}, x) where {M}
    n = size(x, 1)
    D = d.D
    return real(dot(D .* d.Vt, d.M' * x)) - logpFq((), (n//2,), Diagonal((D .^ 2) ./ 4))
end

# TODO: handle potential under/overflow
function logvmfnorm(p, κ)
    ν = p//2 - 1
    lognorm = ν * log(κ) - logbesseli(ν, κ)
    return ifelse(iszero(κ), zero(lognorm), lognorm)
end
