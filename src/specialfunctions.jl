logbesseli(ν, x) = log(besselix(ν * one(real(x)), x)) + abs(real(x))

# hypergeometric functions with matrix arguments
pFq((), (), x) = exp(tr(x)) # ₀F₀
# TODO: Implement ₀F₁
# see P Koev, A Edelman. The efficient evaluation of the hypergeometric function of a
# matrix argument. Math. Comp. 75 (2006), 833-846
# pFq(a::NTuple{p}, b::NTuple{q}, x)

logpFq(a, b, z) = log(pFq(a, b, z))
