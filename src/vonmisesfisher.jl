"""
    VonMisesFisher(M; params...)

The von Mises-Fisher (vMF) distribution on the `Sphere` or `Stiefel` manifold `M`.

Given a multivariate normal distribution with mean ``μ`` and identity covariance in
the Euclidean space in which ``M`` is embedded, the von Mises-Fisher distribution with
paramater ``μ`` is the restriction of this distribution to points on ``M``.

# Parameterizations

    VonMisesFisher(M::AbstractSphere{𝔽}; params...)
    VonMisesFisher(n::Int[, 𝔽]; params...)

Construct the vMF distribution on `Sphere(n-1,𝔽)==` ``𝔽𝕊^{n-1}```.

Implemented parameterizations are:

  - `(μ, κ)`: the modal direction ``μ ∈ 𝔽𝕊^{n-1}`` and concentration ``κ ∈ ℝ⁺``
  - `(c,)`: ``c = κ μ ∈ 𝔽^n``, the mean of the normal distribution in the embedded space.

The density of the vMF distribution on ``𝔽𝕊^{n-1}`` with respect to the normalized
Hausdorff measure is

```math
p(x | μ, κ) = \\frac{Γ(ν + 1)κ^ν}{2^ν I_ν(κ)} \\exp(κ \\Re⟨μ, x⟩)),
```

where ``ν = n/2-1``,  ``⟨⋅,⋅⟩`` is the Frobenius inner product, and ``I_ν(z)``
is the modified Bessel function of the first kind.

    VonMisesFisher(M::Stiefel{n,k,𝔽}; params...)
    VonMisesFisher(n::Int, k::Int[, 𝔽]; params...)

Construct the matrix vMF distribution on `Stiefel(n, k, 𝔽)=` ``\\mathrm{St}(n, k, 𝔽)``.

Implemented parameterizations are:

  - `(F,)`: a parameter matrix ``F ∈ 𝔽^{n × k}``, the mean of the normal distribution in the
    embedded space.
  - `(U, D, V)`: The SVD decomposition of ``F = U D V``, where ``U ∈ \\mathrm{St}(n, k, 𝔽)`` and
    ``V ∈ \\mathrm{U}(k, 𝔽)``.
  - `(H, P)`: The polar decomposition of ``F = H P``, where ``H ∈ \\mathrm{St}(n, k, 𝔽)`` is the
    mode, and ``P ∈ 𝔽^{k × k}`` is a Hermitian positive definite matrix.

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
VonMisesFisher(p::Int, 𝔽=ℝ; params...) = VonMisesFisher(Sphere(p - 1, 𝔽); params...)
function VonMisesFisher(n::Int, k::Int, 𝔽::AbstractNumbers=ℝ; params...)
    return VonMisesFisher(Stiefel(n, k, 𝔽), params...)
end

# common aliases

"""
    VonMises(𝔽=ℝ; params...)

The von Mises distribution on the real or complex `Circle`.

This is just a convenient alias for [`VonMisesFisher(Circle(𝔽); params...)`](@ref).

# Constructors

    VonMises(μ=π, κ=1)
    VonMises(ℂ; μ=im, κ=1)
    VonMises(ℂ; c=3+4im)
"""
const VonMises{𝔽,N,T} = VonMisesFisher{Circle{𝔽},N,T}
VonMises(𝔽=ℝ; params...) = VonMisesFisher(Circle(𝔽); params...)

"""
    Fisher(; params...) = VonMisesFisher(Sphere(2); params...)

The Fisher distribution on the 2-`Sphere`.
"""
const Fisher{N,T} = VonMisesFisher{Sphere{2,ℝ},N,T}
Fisher(; params...) = VonMisesFisher(Sphere(2); params...)

const Langevin = VonMisesFisher

function Base.show(io::IO, mime::MIME"text/plain", μ::VonMisesFisher)
    return show_manifold_measure(io, mime, μ)
end

Manifolds.base_manifold(d::VonMisesFisher) = getfield(d, :manifold)

function MeasureTheory.basemeasure(μ::VonMisesFisher)
    return normalize(Hausdorff(base_manifold(μ)))
end

function MeasureTheory.logdensity(d::VonMises{ℝ,(:μ, :κ)}, x)
    κ = d.κ
    return κ * cos(only(x) - only(d.μ)) - logbesseli(0, κ)
end

function MeasureTheory.logdensity(d::VonMisesFisher{M,(:μ, :κ)}, x) where {M}
    p = size(x, 1)
    κ = d.κ
    return κ * real(dot(d.μ, x)) - lognorm_vmf(p, κ)
end
function MeasureTheory.logdensity(d::VonMisesFisher{M,(:c,)}, x) where {M}
    p = size(x, 1)
    c = d.c
    κ = norm(c)
    return real(dot(c, x)) - lognorm_vmf(p, κ)
end
function MeasureTheory.logdensity(d::VonMisesFisher{M,(:F,)}, x) where {M}
    n = size(x, 1)
    F = d.F
    return real(dot(F, x)) - logpFq((), (n//2,), (F'F) / 4)
end
function MeasureTheory.logdensity(d::VonMisesFisher{M,(:U, :D, :V)}, x) where {M}
    n = size(x, 1)
    D = Diagonal(d.D)
    return real(dot(D * d.V', d.U' * x)) - logpFq((), (n//2,), D .^ 2 ./ 4)
end
function MeasureTheory.logdensity(d::VonMisesFisher{M,(:H, :P)}, x) where {M}
    n = size(x, 1)
    P = d.P
    return real(dot(d.H, P, x)) - logpFq((), (n//2,), (P^2) / 4)
end

StatsBase.mode(d::VonMisesFisher{<:Any,(:μ, :κ)}) = d.μ
StatsBase.mode(d::VonMisesFisher{<:Any,(:c,)}) = normalize(d.c)
StatsBase.mode(d::VonMisesFisher{<:Any,(:F,)}) = (F = svd(d.F); F.U * F.Vt)
StatsBase.mode(d::VonMisesFisher{<:Any,(:U, :D, :V)}) = d.U * d.V'
StatsBase.mode(d::VonMisesFisher{<:Any,(:H, :P)}) = d.H

# ₀F₁(p//2; κ²/4) = 2ᵛ Iᵥ(κ) / κᵛ / Γ(v + 1) for v = p/2 - 1
# Note that the usual vMF constant Cₚ(κ) is defined wrt the un-normalized Hausdorff
# measure, whereas we use the normalized Hausdorff measure here.
function lognorm_vmf(p, κ)
    iszero(κ) && return log(one(κ))
    ν = p//2 - 1
    r = logbesseli(ν, κ) + ν * (logtwo - log(κ))
    return r + loggamma(oftype(r, p//2))
end
