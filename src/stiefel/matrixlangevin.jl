struct MatrixLangevin{N,T,n,k} <: ParameterizedMeasure{N}
    par::NamedTuple{N,T}
    manifold::Stiefel{n,k,ℝ}
end
function MatrixLangevin(n, k; kwargs...)
    par = NamedTuple(kwargs)
    return MatrixLangevin(par, Stiefel(n, k))
end

const MatrixVonMisesFisher = MatrixLangevin

Manifolds.base_manifold(d::MatrixLangevin) = getfield(d, :manifold)

function MeasureTheory.basemeasure(μ::MatrixLangevin)
    return normalize(Hausdorff(base_manifold(μ)))
end

function MeasureTheory.logdensity(d::MatrixLangevin{(:F,)}, x::AbstractMatrix)
    n, _ = representation_size(base_manifold(d))
    F = d.F
    return tr_At_B(F, x) - log(pFq((), (n//2,), rmul!(F'F, 1//4)))
end

function Random.rand!(rng::AbstractRNG, p::AbstractMatrix, d::MatrixLangevin{(:F,)})
    F = d.F
    trΛ = sum(svdvals(F))
    μ = Hausdorff(base_manifold(d), F)
    while true
        rand!(rng, p, μ)
        rand(rng) < exp(tr_At_B(F, x) - trΛ) && break
    end
    return p
end
