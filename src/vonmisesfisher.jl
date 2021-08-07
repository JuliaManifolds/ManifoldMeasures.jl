struct VonMisesFisher{M,N,T} <: ParameterizedMeasure{N}
    manifold::M
    par::NamedTuple{N,T}
end
VonMisesFisher(M::AbstractManifold; kwargs...) = VonMisesFisher(M, NamedTuple(kwargs))

const Langevin = VonMisesFisher

Manifolds.base_manifold(d::VonMisesFisher) = getfield(d, :manifold)

function MeasureTheory.basemeasure(μ::VonMisesFisher{<:Union{AbstractSphere,Stiefel}})
    return normalize(Hausdorff(base_manifold(μ)))
end

# parameterizations with c ∈ 𝔽ⁿ⁺¹ , where c = κ μ for μ ∈ 𝕊ⁿ, κ ∈ ℝ⁺
function MeasureTheory.logdensity(
    d::VonMisesFisher{AbstractSphere,(:μ, :κ)}, x::AbstractArray
)
    p = manifold_dimension(base_manifold(d)) + 1
    κ = d.κ
    return κ * real(dot(d.μ, x)) - logvmfnorm(p, κ)
end
function MeasureTheory.logdensity(d::VonMisesFisher{AbstractSphere,(:c,)}, x::AbstractArray)
    p = manifold_dimension(base_manifold(d)) + 1
    c = d.c
    κ = norm(c)
    return real(dotu(c, x)) - logvmfnorm(p, κ)
end

# TODO: handle potential under/overflow
function logvmfnorm(p, κ)
    ν = p//2 - 1
    lognorm = ν * log(κ) - logbesseli(ν, κ)
    return ifelse(iszero(κ), zero(lognorm), lognorm)
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
