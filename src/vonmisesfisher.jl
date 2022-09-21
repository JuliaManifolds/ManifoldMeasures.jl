"""
    VonMisesFisher(M; params...)

The von Mises-Fisher (vMF) or Langevin distribution on the `Sphere` or `Stiefel` manifold `M`.

Given a matrix ``X ∈ 𝔽^{n × k}`` with IID entries ``X_{ij} ∼ \\mathrm{Normal}(F_{ij}, 1)``
for ``F ∈ 𝔽^{n × k}``, the restriction of the corresponding distribution in ``𝔽^{n × k}``
to the `Stiefel(n, k, 𝔽)` manifold, that is, the matrices for which ``X^\\mathrm{H} X = I_k``,
is the vMF distribution on the `Stiefel` manifold. The vMF distribution can also be specified
for any submanifold of the `Stiefel` manifold, including the `Sphere` and the `Circle`.

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
    return VonMisesFisher(Stiefel(n, k, 𝔽); params...)
end

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

function Base.show(io::IO, mime::MIME"text/plain", μ::VonMisesFisher)
    return show_manifold_measure(io, mime, μ)
end

Manifolds.base_manifold(d::VonMisesFisher) = getfield(d, :manifold)

MeasureTheory.insupport(μ::VonMisesFisher, x) = Manifolds.is_point(Manifolds.base_manifold(μ), x)

function MeasureTheory.basemeasure(μ::VonMisesFisher)
    return normalize(Hausdorff(base_manifold(μ)))
end

function MeasureTheory.logdensity_def(d::VonMises{ℝ,(:μ, :κ)}, x)
    κ = d.κ
    return κ * cos(only(x) - only(d.μ)) - logbesseli(0, κ)
end

function MeasureTheory.logdensity_def(d::VonMisesFisher{M,(:μ, :κ)}, x) where {M}
    p = manifold_dimension(base_manifold(d)) + 1
    κ = d.κ
    return κ * realdot(d.μ, x) - lognorm_vmf(p, κ)
end
function MeasureTheory.logdensity_def(d::VonMisesFisher{M,(:c,)}, x) where {M}
    p = manifold_dimension(base_manifold(d)) + 1
    c = d.c
    κ = norm(c)
    return realdot(c, x) - lognorm_vmf(p, κ)
end
function MeasureTheory.logdensity_def(d::VonMisesFisher{M,(:F,)}, x) where {M}
    n, _ = representation_size(base_manifold(d))
    F = d.F
    return realdot(F, x) - logpFq((), (n//2,), (F'F) / 4)
end
function MeasureTheory.logdensity_def(d::VonMisesFisher{M,(:U, :D, :V)}, x) where {M}
    n, _ = representation_size(base_manifold(d))
    D = Diagonal(d.D)
    return realdot(D * d.V', d.U' * x) - logpFq((), (n//2,), D .^ 2 ./ 4)
end
function MeasureTheory.logdensity_def(d::VonMisesFisher{M,(:H, :P)}, x) where {M}
    n, _ = representation_size(base_manifold(d))
    P = d.P
    return real(dot(d.H, P, x)) - logpFq((), (n//2,), (P^2) / 4)
end

StatsBase.mode(d::VonMisesFisher{<:Any,(:μ, :κ)}) = d.μ
StatsBase.mode(d::VonMisesFisher{<:Any,(:c,)}) = (c = d.c; c / norm(c))
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

#
# Random sampling
#

function Base.rand(rng::AbstractRNG, T::Type, d::VonMisesFisher)
    p = default_point(d, T)
    return Random.rand!(rng, p, d)
end

##
## von Mises
##

# Best, D. J., and Nicholas I. Fisher. "Efficient simulation of the von Mises distribution."
# Journal of the Royal Statistical Society: Series C (Applied Statistics) 28.2 (1979): 152-157.
# doi: 10.2307/2346732
function Base.rand(rng::AbstractRNG, T::Type, d::VonMises{ℝ,(:μ, :κ)})
    κ = T(d.κ)
    tκ = 2κ
    τ = 1 + sqrt(1 + tκ^2)
    ρ = (τ - sqrt(2τ)) / tκ
    r = (1 + ρ^2) / 2ρ
    f = zero(T)
    while true
        z = cospi(rand(rng, T))
        f = T((1 + r * z) / (r + z))
        c = κ * (r - f)
        u = rand(rng, T)
        (c * (2 - c) > u || log(c / u) + 1 ≥ c) && break
    end
    θ₀ = acos(f)
    θ = (rand(rng, (θ₀, -θ₀)))
    return mod2pi(θ + T(d.μ) + π) - π
end
function Base.rand(rng::AbstractRNG, T::Type, d::VonMises{ℂ,(:μ, :κ)})
    θ = rand(rng, T, VonMises(ℝ; μ=angle(d.μ), κ=d.κ))
    return cis(θ)
end
function Base.rand(rng::AbstractRNG, T::Type, d::VonMises{ℂ,(:c,)})
    c = d.c
    θ = rand(rng, T, VonMises(ℝ; μ=angle(c), κ=abs(c)))
    return cis(θ)
end

##
## von Mises-Fisher on the sphere
##

function Random.rand!(
    rng::AbstractRNG, p::AbstractArray, d::VonMisesFisher{<:AbstractSphere,(:μ, :κ)}
)
    n = manifold_dimension(base_manifold(d)) + 1
    return _rand_vmf_sphere!(rng, p, n, d.μ, d.κ)
end
function Random.rand!(
    rng::AbstractRNG, p::AbstractArray, d::VonMisesFisher{<:AbstractSphere,(:c,)}
)
    n = manifold_dimension(base_manifold(d)) + 1
    return _rand_vmf_sphere!(rng, p, n, d.c)
end

# Andrew T.A Wood (1994) Simulation of the von mises fisher distribution,
# Communications in Statistics - Simulation and Computation, 23:1, 157-164
# doi: 10.1080/0361091940881316
function _rand_vmf_sphere!(rng, p, n, μ, κ)
    eltype(p) <: Real && isone(n) && return _rand_vmf_0sphere!(rng, p, κ * μ)
    T = real(eltype(p))
    t = _rand_normal_scale_vmf_sphere(rng, n, T(κ))
    _rand_tangent_vmf_sphere!(rng, p)
    _combine_tangent_normal_sphere!(p, t)
    _reflect_from_xaxis_to_c!(p, μ, 1)
    return p
end
function _rand_vmf_sphere!(rng, p, n, c)
    eltype(p) <: Real && isone(n) && return _rand_vmf_0sphere!(rng, p, c)
    T = real(eltype(p))
    κ = T(norm(c))
    t = _rand_normal_scale_vmf_sphere(rng, n, κ)
    _rand_tangent_vmf_sphere!(rng, p)
    _combine_tangent_normal_sphere!(p, t)
    _reflect_from_xaxis_to_c!(p, c, κ)
    return p
end
function _rand_vmf_0sphere!(rng, p, c)
    p[1] = rand(rng, Bernoulli(; logitp=2 * c[1])) ? 1 : -1
    return p
end

# in the tangent-normal parameterization
# p = t x + √(1 - t²) [0; ξ], for x the x-axis, ξ ∈ 𝕊ⁿ⁻², and t ∈ [-1, 1],
# then ξ follows the normalized Hausdorff measure, and p(t) ∝ (1 - t^2)^((n-3)/2).
# so we draw t and ξ, compose p, then then transform it using the reflection that
# takes the x to μ.
# Method due to Wood, 1994. Adapted also for complex and quaternionic spheres.
function _rand_normal_scale_vmf_sphere(rng, n, κ)
    T = eltype(κ)
    twoκ = 2κ
    if n == 3
        # 2t+1 follows an exponential distribution truncated to [0, 1]
        # so we use inverse transform sampling
        u = rand(rng, T)
        return 1 + log(u + exp(-twoκ) * (1 - u)) / κ
    end
    m = T((n - 1)//2)
    a = twoκ / (n - 1)
    b = sqrt(a^2 + 1) - a
    x = (1 - b) / (1 + b)
    c = κ * x + (n - 1) * log1p(-x^2)
    βdist = Beta(m, m)

    z = rand(rng, T, βdist)
    t = (1 - (1 + b) * z) / (1 - (1 - b) * z)
    while κ * t + (n - 1) * log1p(-x * t) - c < log(rand(rng))
        z = rand(rng, T, βdist)
        t = (1 - (1 + b) * z) / (1 - (1 - b) * z)
    end
    return t
end
function _rand_tangent_vmf_sphere!(rng, p)
    randn!(rng, p)
    p[1] -= real(p[1])
    rdiv!(p, norm(p))
    return p
end
function _combine_tangent_normal_sphere!(p, t)
    rmul!(p, sqrt(1 - t^2))
    p[1] += t
    return p
end
# in-place apply Householder reflection p ↦ p - q 2𝕽⟨q,p⟩/‖q‖², for q=e₁-c/‖c‖
function _reflect_from_xaxis_to_c!(p, c, cnorm=norm(c))
    num = real(p[1]) - realdot(c, p) / cnorm
    den = cnorm - real(c[1])
    α = num / den
    p .+= c .* α
    p[1] -= α * cnorm
    return p
end

##
## von Mises-Fisher on the Stiefel manifold
##

function Random.rand!(
    rng::AbstractRNG, p::AbstractArray, d::VonMisesFisher{<:Stiefel{n,k},(:U, :D, :V)}
) where {n,k}
    dim𝔽 = real_dimension(number_system(base_manifold(d)))
    return _rand_vmf_stiefel!(rng, p, dim𝔽, n, k, d.U, d.D, d.V)
end
function Random.rand!(
    rng::AbstractRNG, p::AbstractArray, d::VonMisesFisher{<:Stiefel{n,k},(:F,)}
) where {n,k}
    U, D, V = svd(d.F)
    dim𝔽 = real_dimension(number_system(base_manifold(d)))
    return _rand_vmf_stiefel!(rng, p, dim𝔽, n, k, U, D, V)
end
function Random.rand!(
    rng::AbstractRNG, p::AbstractArray, d::VonMisesFisher{<:Stiefel{n,k},(:H, :P)}
) where {n,k}
    D, V = eigen(Hermitian(d.P))
    U = d.H * V
    dim𝔽 = real_dimension(number_system(base_manifold(d)))
    return _rand_vmf_stiefel!(rng, p, dim𝔽, n, k, U, D, V)
end

# Peter Hoff. Simulation of the Matrix Bingham—von Mises—Fisher Distribution,
# With Applications to Multivariate and Relational Data.
# Journal of Computational and Graphical Statistics. 18(2). 2009.
function _rand_vmf_stiefel!(rng, p, dim𝔽, n, k, U, D, V)
    if isone(k)
        _rand_vmf_sphere!(rng, vec(p), dim𝔽 * n, vec(U), D[1])
        rmul!(p, V[1]')
        return p
    end
    T = real(eltype(p))
    Z = fill!(similar(p), zero(T))
    Z₁ = @view Z[:, 1]
    U₁ = @view U[:, 1]
    y = similar(Z₁)
    z = similar(U₁)
    while true
        _rand_vmf_sphere!(rng, Z₁, dim𝔽 * n, U₁, D[1])
        lcrit = zero(T)
        for j in 2:k
            s = n - j + 1
            r = dim𝔽 * s
            @views begin
                N = _nullbasis(Z[:, 1:(j - 1)])
                Uⱼ = U[:, j]
                Zⱼ = Z[:, j]
                zⱼ = z[1:s]
                yⱼ = y[1:s]
            end
            Dⱼ = D[j]
            if Dⱼ > 0
                mul!(zⱼ, N', Uⱼ, Dⱼ, false)
                _rand_vmf_sphere!(rng, yⱼ, r, zⱼ)
                mul!(Zⱼ, N, yⱼ)
                nzⱼ = norm(zⱼ)
                ν = r//2 - 1
                lcrit += T(
                    logbesseli(ν, nzⱼ) - logbesseli(ν, Dⱼ) + ν * (log(Dⱼ) - log(nzⱼ))
                )
            else  # sample from uniform distribution, lcrit contribution is zero
                randn!(rng, yⱼ)
                mul!(Zⱼ, N, yⱼ, inv(norm(yⱼ)), false)
            end
        end
        log(rand(rng)) < lcrit && break
    end
    mul!(p, Z, V')
    return p
end

# basis N of null space of A, such that N'A=0
# compute basis of null space of matrix A
function _nullbasis(A)
    F = qr(A)
    rank = size(F.R, 1)
    return F.Q[:, (rank + 1):end]
end
