function qr_unique!(A)
    Q, R = LinearAlgebra.qr(A)
    s = @views sign.(real.(R[diagind(R)]))
    A .= Q .* s'
    R .*= s
    return A, R
end

function svd_unique(A)
    F = LinearAlgebra.svd(A)
    U = F.U
    U1 = @view U[1, :]
    F.Vt .*= sign.(U1)
    F.U .*= sign.(U1')
    U1 .= real.(U1)
    return F
end
