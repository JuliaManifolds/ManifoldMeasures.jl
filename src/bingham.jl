const BinghamManifolds = Union{Sphere,ProjectiveSpace,Stiefel,Grassmann}

struct Bingham{M,N,T} <: ParameterizedMeasure{N}
    manifold::M
    par::NamedTuple{N,T}
end
Bingham(M::AbstractManifold; kwargs...) = Bingham(M, NamedTuple(kwargs))

Manifolds.base_manifold(d::Bingham) = getfield(d, :manifold)

function MeasureTheory.basemeasure(μ::Bingham{<:BinghamManifolds})
    return normalize(Hausdorff(base_manifold(μ)))
end

function MeasureTheory.logdensity(d::Bingham{<:BinghamManifolds,(:B,)}, x::AbstractArray)
    s = representation_size(base_manifold(d))
    n, k = length(s) == 1 ? (first(s), 1) : s
    B = d.B
    return real(dot(x, B, x)) - logpFq((k//2,), (n//2,), B)
end
