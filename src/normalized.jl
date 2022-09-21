"""
    logmass(μ::AbstractMeasure)

Compute the logarithm of the total mass of the measure `μ` over its manifold `M`, that is
``μ(M) = ∫_M \\mathrm{d}μ(x)``.
"""
function logmass end

# TODO: use some exponential wrapper
mass(μ) = exp(logmass(μ))

struct Normalized{M} <: MeasureBase.AbstractMeasure
    base::M
end

function Base.show(io::IO, mime::MIME"text/plain", μ::Normalized)
    print(io, "Normalized(")
    show(io, mime, MeasureBase.basemeasure(μ))
    return print(io, ")")
end

MeasureBase.basemeasure(μ::Normalized) = μ.base

function MeasureBase.logdensity_def(μ::Normalized, x)
    ν = MeasureBase.basemeasure(μ)
    ℓ = float(MeasureBase.logdensityof(ν, x))
    return ℓ - oftype(ℓ, logmass(ν))
end

logmass(μ::Normalized) = false

LinearAlgebra.normalize(μ::Normalized) = μ
# Warning! Type-piracy! ☠️
LinearAlgebra.normalize(μ::MeasureBase.AbstractMeasure) = Normalized(μ)
