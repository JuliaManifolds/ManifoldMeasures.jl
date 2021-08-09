"""
    VonMisesFisher(M; params...)

The von Mises-Fisher (vMF) distribution on the `Sphere` or `Stiefel` manifold `M`.

# Constructors

    VonMisesFisher(M::AbstractSphere{𝔽}; c)
    VonMisesFisher(M::AbstractSphere{𝔽}; μ, κ)

Construct the vMF distribution on the `Sphere` parameterized either by the mean direction
``μ ∈ 𝔽𝕊ⁿ`` and concentration ``κ ∈ ℝ⁺`` or by the single vector ``c = κμ``.

The density of the vMF distribution on `Sphere(n, 𝔽)` with respect to the normalized
Hausdorff measure is

```math
p(x | μ, κ) = \\frac{κ^{(n-1)/2}}{I_{(n-1)/2}(κ)} \\exp(κ \\Re(μ^\\mathrm{H} x)),
```

where ``I_ν(z)`` is the modified Bessel function of the first kind.

    VonMisesFisher(M::Stiefel{n,k,𝔽}; F)
    VonMisesFisher(M::Stiefel{n,k,𝔽}; M, D, Vt)

Construct the (Matrix) vMF distribution on the `Stiefel(n,k,𝔽)` manifold parameterized
either by the matrix ``F ∈ 𝔽^{n × k}`` or by its SVD decomposition ``F = M * D * Vt``.

The density of the vMF distribution on `Stiefel(n, k, 𝔽)` with respect to the normalized
Hausdorff measure is

```math
p(x | F) = \\frac{\\exp(\\Re(\\operatorname{tr}(F^\\mathrm{T} x)))}{_0 F_1(\\frac{n}{2}; \\frac{1}{4} F^\\mathrm{H}F)},
```

Note that even though `Stiefel(n+1,1,𝔽)` and `Sphere(n,𝔽)` are equivalent, their densities here
are not equivalent for ``𝔽 ≠ ℝ``, because for the Stiefel manifold, the inner product without
conjugation is used.
"""
struct VonMisesFisher{M,N,T} <: ParameterizedMeasure{N}
    manifold::M
    par::NamedTuple{N,T}
end
VonMisesFisher(M::AbstractManifold; kwargs...) = VonMisesFisher(M, NamedTuple(kwargs))

const Langevin = VonMisesFisher

Manifolds.base_manifold(d::VonMisesFisher) = getfield(d, :manifold)

function MeasureTheory.basemeasure(μ::VonMisesFisher{<:Union{AbstractSphere,Stiefel}})
    return normalize(Hausdorff(base_manifold(μ)))
end

function MeasureTheory.logdensity(
    d::VonMisesFisher{AbstractSphere,(:μ, :κ)}, x::AbstractArray
)
    p = manifold_dimension(base_manifold(d)) + 1
    κ = d.κ
    return κ * real(dot(d.μ, x)) - logvmfnorm(p, κ)
end
function MeasureTheory.logdensity(d::VonMisesFisher{AbstractSphere,(:c,)}, x::AbstractArray)
    p = manifold_dimension(base_manifold(d)) + 1
    c = d.c
    κ = norm(c)
    return real(dot(c, x)) - logvmfnorm(p, κ)
end

# TODO: handle potential under/overflow
function logvmfnorm(p, κ)
    ν = p//2 - 1
    lognorm = ν * log(κ) - logbesseli(ν, κ)
    return ifelse(iszero(κ), zero(lognorm), lognorm)
end

function MeasureTheory.logdensity(
    d::VonMisesFisher{Stiefel{n,k},(:F,)}, x::AbstractMatrix
) where {n,k}
    F = d.F
    return real(dotu(F, x)) - logpFq((), (n//2,), rmul!(F'F, 1//4))
end
function MeasureTheory.logdensity(
    d::VonMisesFisher{Stiefel{n,k},(:M, :D, :Vt)}, x::AbstractMatrix
) where {n,k}
    D = d.D
    return real(dotu(D .* d.Vt, transpose(d.M') * x)) -
           logpFq((), (n//2,), Diagonal((D .^ 2) ./ 4))
end
