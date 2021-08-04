# multivariate gamma function
# Chikuse, 2003 Eq. 1.5.7
mvgamma(m, a) = π^(m * (m - 1)//4) * prod(i -> gamma(a - (i - 1)//2), 1:m)

# tr(A'B)
tr_At_B(A, B) = sum((Ai, Bi) -> dot(Ai, Bi), zip(eachcol(A), eachcol(B)))

# hypergeometric functions with matrix arguments
pFq((), (), x) = exp(tr(x)) # ₀F₀
# TODO: Implement ₀F₁
# see P Koev, A Edelman. The efficient evaluation of the hypergeometric function of a
# matrix argument. Math. Comp. 75 (2006), 833-846
# pFq(a::NTuple{p}, b::NTuple{q}, x)
