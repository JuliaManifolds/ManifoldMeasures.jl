struct VonMisesFisher{M,N,T} <: ParameterizedMeasure{N}
    manifold::M
    par::NamedTuple{N,T}
end
function VonMisesFisher(M::AbstractManifold; kwargs...)
    return VonMisesFisher(M, kwargs)
end

const Langevin = VonMisesFisher

Manifolds.base_manifold(d::VonMisesFisher) = getfield(d, :manifold)

function MeasureTheory.basemeasure(μ::VonMisesFisher)
    return normalize(Hausdorff(base_manifold(μ)))
end

# parameterizations with F ∈ 𝔽ⁿˣᵏ whose SVD decomposition is F = M D Vt.
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
    return real(dotu(D .* d.Vt, d.M' * x)) - logpFq((), (n//2,), Diagonal((D .^ 2) ./ 4))
end
