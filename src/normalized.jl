"""
    logmass(μ::AbstractMeasure)

Compute the logarithm of the total mass of the measure `μ` over its manifold `M`, that is
`μ(M) = ∫_M dμ(x)`.
"""
function logmass end

# TODO: use some exponential wrapper
mass(μ) = exp(logmass(μ))

struct Normalized{M} <: AbstractMeasure
    base::M
end

function Base.show(io::IO, mime::MIME"text/plain", μ::Normalized)
    print(io, "Normalized(")
    show(io, mime, basemeasure(μ))
    return print(io, ")")
end

MeasureTheory.basemeasure(μ::Normalized) = μ.base

function MeasureTheory.logdensity(μ::Normalized, x)
    ν = basemeasure(μ)
    ℓ = float(logdensity(ν, x))
    return ℓ - oftype(ℓ, logmass(ν))
end

logmass(μ::Normalized) = false

LinearAlgebra.normalize(μ::Normalized) = μ
# Warning! Type-piracy! ☠️
LinearAlgebra.normalize(μ::AbstractMeasure) = Normalize(μ)
