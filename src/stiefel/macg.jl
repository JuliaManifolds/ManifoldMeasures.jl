struct MatrixAngularCentralGaussian{N,T,n,k} <: ParameterizedMeasure{N}
    par::NamedTuple{N,T}
    manifold::Stiefel{n,k,ℝ}
end

function MatrixAngularCentralGaussian(n, k; kwargs...)
    par = NamedTuple(kwargs)
    return MatrixAngularCentralGaussian(par, Stiefel(n, k))
end

Manifolds.base_manifold(d::MatrixAngularCentralGaussian) = getfield(d, :manifold)

function MeasureTheory.basemeasure(μ::MatrixAngularCentralGaussian{(:Σ⁻¹,)})
    return normalize(Hausdorff(base_manifold(μ), μ.Σ⁻¹))
end
function MeasureTheory.basemeasure(μ::MatrixAngularCentralGaussian{(:L,)})
    return normalize(Hausdorff(base_manifold(μ), μ.L))
end

# Chikuse, 2003 Eq. 2.4.3
function MeasureTheory.logdensity(
    d::MatrixAngularCentralGaussian{(:Σ⁻¹,)}, x::AbstractMatrix
)
    n, k = representation_size(base_manifold(d))
    Σ⁻¹ = d.Σ⁻¹
    return -n//2 * log(dot(x, Σ⁻¹, x)) + k//2 * logdet(Σ⁻¹)
end
function MeasureTheory.logdensity(d::MatrixAngularCentralGaussian{(:L,)}, x::AbstractMatrix)
    n, k = representation_size(base_manifold(d))
    L = d.L
    z = LowerTriangular(L) \ x
    return -n//2 * log(dot(z, z)) - k * logdet(L)
end

function Base.rand(rng::AbstractRNG, ::Type, d::MatrixAngularCentralGaussian{(:L,)})
    L = d.L
    z = randn!(rng, similar(d.L, representation_size(base_manifold(d))))
    y = lmul!(LowerTriangular(L), z)
    return y * (Symmetric(y'y))^(-1//2)
end
