using ManifoldMeasures
using Test

@testset "ManifoldMeasures.jl" begin
    include("hausdorff.jl")
    include("vonmisesfisher.jl")
end
