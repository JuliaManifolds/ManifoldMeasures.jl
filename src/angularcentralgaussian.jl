"""
    AngularCentralGaussian(M; params...)

The Angular Central Gaussian (ACG) distribution on the manifold ``M``.

Accepted manifolds are `Sphere`, `ProjectiveSpace`, `Stiefel`, `Grassmann`, `Rotations`
and `SpecialOrthogonal`.

For manifolds with matrix points, this is also called the Matrix Angular Central Gaussian
distribution.

# Constructors

    AngularCentralGaussian(M; P)
    AngularCentralGaussian(M; L)

For a manifold ``M ⊂ 𝔽^{n × k}``, construct the ACG distribution parameterized either by
the inverse ``P = Σ^{-1}`` of an ``n × n`` positive definite matrix ``Σ`` or by the lower
Cholesky factor ``L`` of ``Σ``, such that ``Σ = L L^\\mathrm{H}``.
"""
struct AngularCentralGaussian{M,N,T} <: ParameterizedMeasure{N}
    manifold::M
    par::NamedTuple{N,T}
end
AngularCentralGaussian(M; params...) = AngularCentralGaussian(M, NamedTuple(params))

function Base.show(io::IO, mime::MIME"text/plain", μ::AngularCentralGaussian)
    return show_manifold_measure(io, mime, μ)
end

Manifolds.base_manifold(d::AngularCentralGaussian) = getfield(d, :manifold)

function MeasureTheory.basemeasure(μ::AngularCentralGaussian)
    return normalize(Hausdorff(base_manifold(μ)))
end

# Chikuse, 2003 Eq. 2.4.3
# extended to complex Stiefel using Eq. 6 of
# Wróblewska J. A note on some extensions of the matrix angular central Gaussian distribution. 2020.
# https://arxiv.org/abs/2010.03243
function MeasureTheory.logdensity_def(μ::AngularCentralGaussian{<:Any,(:P,)}, x)
    M = base_manifold(μ)
    d = real_dimension(number_system(M))
    s = representation_size(M)
    n, k = length(s) > 1 ? s : (first(s), 1)
    P = μ.P
    logdetx′Px = isone(k) ? real(dot(x, P, x)) : real(logdet(x' * P * x))
    return d * (k * real(logdet(P)) - n * logdetx′Px) / 2
end
function MeasureTheory.logdensity_def(μ::AngularCentralGaussian{<:Any,(:L,)}, x)
    M = base_manifold(μ)
    d = real_dimension(number_system(M))
    s = representation_size(M)
    n, k = length(s) > 1 ? s : (first(s), 1)
    L = LowerTriangular(μ.L)
    z = L \ x
    logdetx′Px = isone(k) ? log(real(norm(z))) : real(logdet(z'z)) / 2
    return -d * (k * real(logdet(L)) + n * logdetx′Px)
end

function Base.rand(rng::AbstractRNG, T::Type, d::AngularCentralGaussian)
    p = default_point(d, T)
    return Random.rand!(rng, p, d)
end

function Random.rand!(
    rng::AbstractRNG, p::AbstractArray, d::AngularCentralGaussian{M,(:L,)}
) where {M}
    z = randn!(rng, p)
    return project!(base_manifold(d), p, d.L * z)
end
