function qr_unique!(A::AbstractMatrix)
    Q, R = qr(A)
    Rdiag = @views R[diagind(R)]
    A .= Matrix(Q) .* sign.(transpose(Rdiag))
    R .*= sign.(Rdiag)
    return A, R
end

function svd_unique(A::AbstractMatrix)
    F = svd(A)
    U = F.U
    U1 = @view U[1, :]
    F.Vt .*= sign.(U1)
    F.U .*= sign.(U1')
    U1 .= real.(U1)
    return F
end
