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

"""
    Hausdorff{M,A} <: PrimitiveMeasure

The un-normalized Hausdorff measure on a manifold.

The Hausdorff measure generalizes the notion of area or volume to a manifold that is embedded
in a metric space. That is, the mass of the measure over some region of the manifold is the
area/volume of that region in the embedded space.

# Constructors

    Hausdorff(M::AbstractManifold)

Constructs the Hausdorff measure for the manifold `M` using the default embedding of the
manifold.
"""
struct Hausdorff{M,A} <: PrimitiveMeasure
    manifold::M
    atlas::A  # optional, when the Hausdorff measure is constructed from a Lebesgue measure
end
Hausdorff(M) = Hausdorff(M, nothing)

function Base.show(io::IO, mime::MIME"text/plain", μ::Hausdorff)
    print(io, "Hausdorff")
    return show(io, mime, (μ.manifold, μ.atlas))
end

Manifolds.base_manifold(μ::Hausdorff) = μ.manifold

MeasureTheory.logdensity(::Hausdorff, x) = zero(eltype(x))

function Base.rand(rng::AbstractRNG, T::Type, μ::Normalized{<:Hausdorff})
    p = default_point(μ, T)
    return Random.rand!(rng, p, μ)
end

# Stiefel

# In general, given matrix z ∈ 𝔽^(n × k) with IID std normal elements, p=z(z'z)^(-1/2) is drawn
# from the Hausdorff measure on St(n,k,𝔽).
# This implementation uses the unique QR decomposition z=QR where R[i,i] is real and positive
# for all i, so that Q ~ Hausdorff(St(n,k,𝔽)).
# See Theorem 2.3.19 of Gupta AK, Nagar DK. Matrix variate distributions. CRC Press; 2018
function Random.rand!(
    rng::AbstractRNG, p::AbstractMatrix, ::Normalized{<:Hausdorff{<:Stiefel}}
)
    randn!(rng, p)
    Q, _ = qr_unique!(p)
    return Q
end

# Chikuse, 2003 Eq. 1.4.8
function logmass(::Hausdorff{Stiefel{n,k,ℝ}}) where {n,k}
    halfn = n//2
    return k * logtwo + (k * halfn) * logπ - logmvgamma(k, halfn)
end
# vol(St(n,k,𝔽)) = vol(𝔽𝕊ⁿ⁻¹) * vol(St(n-1,k-1,𝔽)), Chikuse, 2003 §1.4.4
# then, vol(St(n,k,𝔽)) = 2ᵏ π^(k(2n-k+1)d/4) / ∏ᵢ₌₁ᵏΓ(d(n-k+i)/2)),
# where d = real_dimension(𝔽)
function logmass(::Hausdorff{Stiefel{n,k,ℂ}}) where {n,k}
    a = n - k
    lfact = logfactorial(a)
    lm = k * logtwo + (k * (2n - k + 1)//2) * logπ - lfact
    for _ in 1:(k - 1)
        a += 1
        lfact += log(a)
        lm -= lfact
    end
    return lm
end
function logmass(::Hausdorff{Stiefel{n,k,ℍ}}) where {n,k}
    a = 2(n - k) + 1
    lfact = logfactorial(a)
    lm = k * logtwo + (k * (a + k)) * logπ - lfact
    for _ in 1:(k - 1)
        a += 2
        lfact += log(a) + log(a - 1)
        lm -= lfact
    end
    return lm
end

# Grassmann

function Random.rand!(
    rng::AbstractRNG, p::AbstractMatrix, ::Normalized{<:Hausdorff{Grassmann{n,k,𝔽}}}
) where {n,k,𝔽}
    return rand!(rng, p, normalize(Hausdorff(Stiefel(n, k, 𝔽))))
end

# because Gr(n,k,𝔽) = St(n,k,𝔽) / U(k,𝔽),
# then vol(Gr(n,k,𝔽)) = vol(St(n,k,𝔽)) / vol(St(k,k,𝔽))
#                     = π^(k((n-k)d/2)) ∏ᵢ₌₁ᵏ Γ(id/2) / Γ((n-k)d/2 + id/2),
# where d = real_dimension(𝔽)
function logmass(::Hausdorff{Grassmann{n,k,ℝ}}) where {n,k}
    return k * (n - k)//2 * logπ + logmvgamma(k, k//2) - logmvgamma(k, n//2)
end
function logmass(::Hausdorff{Grassmann{n,k,ℂ}}) where {n,k}
    a = n - k
    lfact1 = logfactorial(a)
    lfact2 = log(1)
    lm = k * a * logπ + lfact2 - lfact1
    for i in 1:(k - 1)
        a += 1
        lfact1 += log(a)
        lfact2 += log(i)
        lm += lfact2 - lfact1
    end
    return lm
end
function logmass(::Hausdorff{Grassmann{n,k,ℍ}}) where {n,k}
    a = 2(n - k) + 1
    b = 1
    lfact1 = logfactorial(a)
    lfact2 = log(b)
    lm = 2 * k * (n - k) * logπ + lfact2 - lfact1
    for _ in 1:(k - 1)
        a += 2
        b += 2
        lfact1 += log(a) + log(a - 1)
        lfact2 += log(b) + log(b - 1)
        lm += lfact2 - lfact1
    end
    return lm
end

# Circle

default_point(::Normalized{<:Hausdorff{Circle{ℝ}}}, T::Type) = zero(float(real(T)))
default_point(::Normalized{<:Hausdorff{Circle{ℂ}}}, T::Type) = zero(complex(float(real(T))))

function Random.rand!(rng::AbstractRNG, p::Real, ::Normalized{<:Hausdorff{Circle{ℝ}}})
    return rand(rng, typeof(p)) * twoπ - π
end
function Random.rand!(rng::AbstractRNG, p::Complex, ::Normalized{<:Hausdorff{Circle{ℂ}}})
    return sign(randn(rng, typeof(p)))
end

logmass(::Hausdorff{<:Circle}) = log2π

# Sphere

function Random.rand!(
    rng::AbstractRNG, p::AbstractArray, μ::Normalized{<:Hausdorff{<:AbstractSphere}}
)
    return normalize!(randn!(rng, p))
end

function logmass(μ::Hausdorff{<:AbstractSphere{𝔽}}) where {𝔽}
    m = prod(representation_size(base_manifold(μ)))
    ν = real_dimension(𝔽) * m//2
    return logtwo + ν * logπ - loggamma(ν)
end

# ProjectiveSpace

function Random.rand!(
    rng::AbstractRNG, p::AbstractArray, μ::Normalized{<:Hausdorff{<:AbstractProjectiveSpace}}
)
    return normalize!(randn!(rng, p))
end

# because 𝔽ℙⁿ = 𝔽𝕊ⁿ / 𝔽𝕊⁰, then vol(𝔽ℙⁿ) = vol(𝔽𝕊ⁿ) / vol(𝔽𝕊⁰)
function logmass(μ::Hausdorff{<:AbstractProjectiveSpace{ℝ}})
    m = prod(representation_size(base_manifold(μ)))
    ν = m//2
    return ν * logπ - loggamma(ν)
end
function logmass(μ::Hausdorff{<:AbstractProjectiveSpace{ℂ}})
    m = prod(representation_size(base_manifold(μ)))
    n = m - 1
    return n * logπ - logfactorial(n)
end
function logmass(μ::Hausdorff{<:AbstractProjectiveSpace{ℍ}})
    m = prod(representation_size(base_manifold(μ)))
    n = m - 1
    return 2n * logπ - loggamma(2(n + 1))
end

# Rotations/SpecialOrthogonal

function Random.rand!(
    rng::AbstractRNG,
    p::AbstractMatrix,
    ::Normalized{<:Hausdorff{<:Union{Rotations,SpecialOrthogonal}}},
)
    # draw p ~ O(n)
    rand!(rng, p, Hausdorff(Stiefel(n, n)))
    # fix to p ~ SO(n)
    if det(p) < 0
        @views p[:, [1, 2]] = p[:, [2, 1]]
    end
    return p
end

# since O(n) = St(n,n) is essentially two copies of SO(n), then vol(SO(n)) = vol(St(n,n)) / 2
function logmass(::Hausdorff{<:Union{Rotations{n},SpecialOrthogonal{n}}}) where {n}
    return logmass(Hausdorff(Stiefel(n, n))) - logtwo
end
