using ManifoldMeasures, Manifolds, MeasureTheory, Random
using Test
using ManifoldMeasures: mass, logmass
using Manifolds: ℝ

@testset "Hausdorff" begin
    @testset "basic" begin
        M = Sphere(2)
        d = Hausdorff(M)
        @test d isa Hausdorff{Sphere{2,ℝ},Nothing}
        @test base_manifold(d) === M
        @test d.atlas isa Nothing
        @test sprint(show, "text/plain", d) === "Hausdorff($M, nothing)"
        p = normalize(randn(3))
        @test iszero(logdensity(d, p))
    end
end
