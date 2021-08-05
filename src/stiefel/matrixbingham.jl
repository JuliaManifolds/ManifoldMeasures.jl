struct MatrixBingham{N,T,n,k} <: ParameterizedMeasure{N}
    par::NamedTuple{N,T}
    manifold::Stiefel{n,k,ℝ}
end
function MatrixBingham(n, k; kwargs...)
    par = NamedTuple(kwargs)
    return MatrixBingham(par, Stiefel(n, k))
end

Manifolds.base_manifold(d::MatrixBingham) = getfield(d, :manifold)

function MeasureTheory.basemeasure(μ::MatrixBingham{(:B,)})
    return normalize(Hausdorff(base_manifold(μ), μ.B))
end

function MeasureTheory.logdensity(d::MatrixBingham{(:B,)}, x::AbstractMatrix)
    B = d.B
    n, k = representation_size(base_manifold(d))
    return dot(x, B, x) - log(pFq((k//2,), (n//2,), B))
end
