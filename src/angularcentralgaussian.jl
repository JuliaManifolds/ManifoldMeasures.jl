"""
    AngularCentralGaussian(M; params...)

The Angular Central Gaussian (ACG) distribution on the manifold ``M``.

Accepted manifolds are `Sphere`, `ProjectiveSpace`, `Stiefel`, `Grassmann`, `Rotations`
and `SpecialOrthogonal`.

For manifolds with matrix points, this is also called the Matrix Angular Central Gaussian
distribution.

# Constructors

    AngularCentralGaussian(M; Σ⁻¹)
    AngularCentralGaussian(M; L)

For a manifold ``M ⊂ 𝔽^{n × k}``, construct the ACG distribution parameterized either by
the inverse of an ``n × n`` positive definite matrix ``Σ`` or by its lower Cholesky factor
``L``, such that ``Σ = L L^\\mathrm{T}``.
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
# TODO: check exponents for complex case, see https://arxiv.org/abs/2010.03243
function MeasureTheory.logdensity(d::AngularCentralGaussian{M,(:Σ⁻¹,)}, x) where {M}
    n = size(x, 1)
    k = size(x, 2)
    Σ⁻¹ = d.Σ⁻¹
    return (k * real(logdet(Σ⁻¹)) - n * log(real(dot(x, Σ⁻¹, x)))) / 2
end
function MeasureTheory.logdensity(d::AngularCentralGaussian{M,(:L,)}, x) where {M}
    n = size(x, 1)
    k = size(x, 2)
    L = LowerTriangular(d.L)
    return -k * real(logdet(L)) - n * log(norm(L \ x))
end

function Base.rand(rng, T, d::AngularCentralGaussian)
    p = default_point(d, T)
    return Random.rand!(rng, p, d)
end

function Random.rand!(
    rng::AbstractRNG, p::AbstractArray, d::AngularCentralGaussian{M,(:L,)}
) where {M}
    z = randn!(rng, p)
    y = lmul!(LowerTriangular(d.L), z)
    return project!(base_manifold(d), p, y)
end
