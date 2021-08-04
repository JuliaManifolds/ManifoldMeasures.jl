# multivariate gamma function
# Chikuse, 2003 Eq. 1.5.7
mvgamma(m, a) = π^(m * (m - 1)//4) * prod(i -> gamma(a - (i - 1)//2), 1:m)

