# The Hausdorff measure generalizes the notion of volume or area to a manifold.
# For manifolds embedded in real space, it can be intuitively understoof as the addition
# of an infinitesimal dimension and the limit of the volume in the embedded space as the
# infinitesimal dimension goes to 0.
# A more useful description comes from the area formula. Given an m-
# dimensional manifold M and a Lipschitz map f: ℝᵐ → M ⊆ ℝⁿ that embeds M in ℝⁿ, then the Hausdorff
# measure dμ(x) on M is is the pushforward of the Lebesgue measure through f, given as
# dμ(x) = det(G(x))^(-1/2) dλ(f⁻¹(x)), where G(x) is the matrix representation of the metric tensor
# in M, given by G(x) = J(x)ᵀ J(x), where J is the Jacobian of f.
# This is the un-normalized Hausdorff measure, which integrates to the area/volume of M.
struct Hausdorff{M<:AbstractManifold,A} <: PrimitiveMeasure
    manifold::M
    atlas::A  # optional, when the Hausdorff measure is constructed from a Lebesgue measure
end
Hausdorff(M::AbstractManifold) = Hausdorff(M, nothing)

function Base.show(io::IO, ::MIME"text/plain", μ::Hausdorff)
    return print(io, "Hausdorff", (μ.manifold, μ.atlas))
end

Manifolds.base_manifold(μ::Hausdorff) = μ.manifold

MeasureTheory.logdensity(::Hausdorff, x) = zero(eltype(x))

function Random.rand!(rng::AbstractRNG, p, μ::WeightedMeasure{S,<:Hausdorff}) where {S}
    return rand!(rng, p, μ.base)
end

# Stiefel

# In general, given matrix z ∈ ℝ^(n × k) with IID std normal elements, p=z(z'z)^(-1/2) is drawn
# from the Hausdorff measure on St(n,k).
# This implementation uses the unique QR decomposition z=QR where R[i,i] > 0, where Q is then
# drawn from the Hausdorff measure on St(n,k).
# See Theorem 2.3.19 of Gupta AK, Nagar DK. Matrix variate distributions. CRC Press; 2018
function Random.rand!(rng::AbstractRNG, p::AbstractMatrix, ::Hausdorff{<:Stiefel})
    randn!(rng, p)
    Q, _ = qr_unique!(p)
    return Q
end

function MeasureTheory.logdensity(
    ::T, ::T, x::AbstractMatrix
) where {M<:Stiefel,T<:Hausdorff{M}}
    return zero(eltype(x))
end

# Chikuse, 2003 Eq. 1.4.8
function logmass(::Hausdorff{Stiefel{n,k,ℝ}}) where {n,k}
    halfn = n//2
    return k * logtwo + (k * halfn) * logπ - logmvgamma(k, halfn)
end

# Sphere

function Base.rand!(rng::AbstractRNG, p::AbstractArray, μ::Hausdorff{<:AbstractSphere})
    return normalize!(randn!(rng, p))
end

function logmass(μ::Hausdorff{AbstractSphere{𝔽}}) where {𝔽}
    n = manifold_dimension(base_manifold(μ))
    ν = number_dimension(𝔽) * (n + 1)//2
    return ν * log2π - loggamma(ν)
end

# ProjectiveSpace

function Base.rand!(rng::AbstractRNG, p::AbstractArray, μ::Hausdorff{<:AbstractProjectiveSpace})
    return normalize!(randn!(rng, p))
end

function logmass(μ::Hausdorff{AbstractProjectiveSpace{𝔽}}) where {𝔽}
    n = manifold_dimension(μ)
    return logmass(Hausdorff(Sphere(n, 𝔽))) - logmass(Hausdorff(Sphere(0, 𝔽)))
end
