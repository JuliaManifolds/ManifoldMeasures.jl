# alias for equivalent uniform measures
# TODO: If we generalize Haar for G-manifolds, then we can add Haar under left-action of O(n)
# here.
const UniformStiefelMeasures{n,k,P} = Hausdorff{Stiefel{n,k,ℝ},P}

# In general, given matrix z ∈ ℝ^(n × k) with IID std normal elements, p=z(z'z)^(-1/2) is drawn
# from the Hausdorff measure on St(n,k).
# This implementation uses the unique QR decomposition z=QR where R[i,i] > 0, where Q is then
# drawn from the Hausdorff measure on St(n,k).
# See Theorem 2.3.19 of Gupta AK, Nagar DK. Matrix variate distributions. CRC Press; 2018
function Base.rand(rng::AbstractRNG, ::Type, μ::Hausdorff{<:Stiefel{n,k,ℝ}}) where {n,k}
    p = randn!(rng, similar(μ.point))
    Q, R = qr(p)
    p .= Matrix(Q) .* sign.(view(R, diagind(R))')
    return p
end

function MeasureTheory.logdensity(
    ::UniformStiefelMeasures{n,k}, ::UniformStiefelMeasures{n,k}, x::AbstractMatrix
) where {n,k}
    return zero(eltype(x))
end

# Chikuse, 2003 Eq. 1.4.8
volume(::Stiefel{n,k,ℝ}) where {n,k} = 2^k * π^(k * n//2) / mvgamma(k, n//2)
