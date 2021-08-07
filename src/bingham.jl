struct Bingham{M,N,T} <: ParameterizedMeasure{N}
    manifold::M
    par::NamedTuple{N,T}
end
Bingham(M::AbstractManifold; kwargs...) = Bingham(M, kwargs)

Manifolds.base_manifold(d::Bingham) = getfield(d, :manifold)

MeasureTheory.basemeasure(μ::Bingham) = normalize(Hausdorff(base_manifold(μ)))

function MeasureTheory.logdensity(
    d::Bingham{M,(:B,)}, x::AbstractArray
) where {M<:Union{Sphere,ProjectiveSpace,Stiefel,Grassmann}}
    n = size(x, 1)
    k = size(x, 2)
    B = d.B
    return real(dot(x, B, x)) - logpFq((k//2,), (n//2,), B)
end
