const ACGManifolds = Union{Sphere,ProjectiveSpace,Stiefel,Grassmann}

struct AngularCentralGaussian{M,N,T} <: ParameterizedMeasure{N}
    manifold::M
    par::NamedTuple{N,T}
end

AngularCentralGaussian(M; kwargs...) = AngularCentralGaussian(M, kwargs)

Manifolds.base_manifold(d::AngularCentralGaussian) = getfield(d, :manifold)

function MeasureTheory.basemeasure(μ::AngularCentralGaussian{<:ACGManifolds})
    return normalize(Hausdorff(base_manifold(μ)))
end

# Chikuse, 2003 Eq. 2.4.3
function MeasureTheory.logdensity(
    d::AngularCentralGaussian{<:ACGManifolds,(:Σ⁻¹,)}, x::AbstractArray
)
    s = representation_size(base_manifold(d))
    n, k = length(s) == 1 ? first(s), 1 : s
    Σ⁻¹ = d.Σ⁻¹
    return -n//2 * log(real(dot(x, Σ⁻¹, x))) + k//2 * real(logdet(Σ⁻¹))
end
function MeasureTheory.logdensity(
    d::AngularCentralGaussian{<:ACGManifolds,(:L,)}, x::AbstractArray
)
    s = representation_size(base_manifold(d))
    n, k = length(s) == 1 ? first(s), 1 : s
    L = d.L
    z = LowerTriangular(L) \ x
    return -n//2 * log(real(dot(z, z))) - k * real(logdet(L))
end

function Random.rand!(
    rng::AbstractRNG, p::AbstractArray, d::AngularCentralGaussian{<:ACGManifolds,(:L,)}
)
    z = randn!(rng, p)
    y = lmul!(LowerTriangular(d.L), z)
    return project!(base_manifold(d), p, y)
end
