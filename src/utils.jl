# dotu, aka sum(x .* y)
function dotu(x, y)
    return sum(zip(x, y)) do (xᵢ, yᵢ)
        return xᵢ * yᵢ
    end
end
dotu(A::AbstractArray{<:Real}, B) = dot(A, B)
