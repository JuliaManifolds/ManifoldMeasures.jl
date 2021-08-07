# tr(A'B)
tr_At_B(A, B) = sum(x -> dot(x...), zip(eachcol(A), eachcol(B)))

# hypergeometric functions with matrix arguments
pFq((), (), x) = exp(tr(x)) # ₀F₀
# TODO: Implement ₀F₁
# see P Koev, A Edelman. The efficient evaluation of the hypergeometric function of a
# matrix argument. Math. Comp. 75 (2006), 833-846
# pFq(a::NTuple{p}, b::NTuple{q}, x)

function pFq(()::Tuple{}, (α,)::NTuple{1}, z)
    x = sqrt(z)
    return besseli(α - 1, 2x) * gamma(α) * x^(1 - α)
end

logpFq(a, b, z) = log(pFq(a, b, z))
function logpFq(()::Tuple{}, (α,)::NTuple{1}, z)
    x = sqrt(z)
    return logbesseli(α - 1, 2x) + loggamma(α) + (1 - α) * log(x)
end
