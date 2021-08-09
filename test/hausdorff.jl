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

    @testset "manifolds" begin
        @testset "Sphere" begin
            @test mass(Hausdorff(Sphere(0))) ≈ 2
            @test mass(Hausdorff(Sphere(1))) ≈ 2π
            @test mass(Hausdorff(Sphere(2))) ≈ 4π
            @test mass(Hausdorff(Sphere(3))) ≈ 2π^2
            for n in 0:10
                @test mass(Hausdorff(Sphere(n, ℂ))) ≈ mass(Hausdorff(Sphere(2n + 1)))
                @test mass(Hausdorff(Sphere(n, ℍ))) ≈ mass(Hausdorff(Sphere(4n + 3)))
            end
        end

        @testset "ProjectiveSpace" begin
            @testset "n=$n" for n in 0:10
                @test mass(Hausdorff(ProjectiveSpace(n))) ≈ mass(Hausdorff(Sphere(n))) / 2
                @test mass(Hausdorff(ProjectiveSpace(n, ℂ))) ≈
                      mass(Hausdorff(Sphere(n, ℂ))) / mass(Hausdorff(Sphere(0, ℂ)))
                @test mass(Hausdorff(ProjectiveSpace(n, ℍ))) ≈
                      mass(Hausdorff(Sphere(n, ℍ))) / mass(Hausdorff(Sphere(0, ℍ)))
            end
        end

        @testset "Stiefel" begin
            function mass_stiefel_recursive(::Stiefel{n,1,𝔽}) where {n,𝔽}
                return mass(Hausdorff(Sphere(n - 1, 𝔽)))
            end
            function mass_stiefel_recursive(::Stiefel{n,k,𝔽}) where {n,k,𝔽}
                return mass(Hausdorff(Sphere(n - 1, 𝔽))) *
                       mass_stiefel_recursive(Stiefel(n - 1, k - 1, 𝔽))
            end
            @testset "Stiefel($n, $k, $𝔽)" for n in 1:10, k in 1:2:n, 𝔽 in (ℝ, ℂ, ℍ)
                k == 1 && @test mass(Hausdorff(Stiefel(n, k, 𝔽))) ≈
                      mass(Hausdorff(Sphere(n - 1, 𝔽)))
                @test mass(Hausdorff(Stiefel(n, k, 𝔽))) ≈
                      mass_stiefel_recursive(Stiefel(n, k, 𝔽))
            end
        end

        @testset "Grassmann" begin
            @testset "Grassmann($n, $k, $𝔽)" for n in 1:10, k in 1:2:n, 𝔽 in (ℝ, ℂ, ℍ)
                k == 1 && @test mass(Hausdorff(Grassmann(n, k, 𝔽))) ≈
                      mass(Hausdorff(ProjectiveSpace(n - 1, 𝔽)))
                @test mass(Hausdorff(Grassmann(n, k, 𝔽))) ≈
                      mass(Hausdorff(Stiefel(n, k, 𝔽))) / mass(Hausdorff(Stiefel(k, k, 𝔽)))
            end
        end

        @testset "$MT" for MT in (Rotations, SpecialOrthogonal)
            @test mass(Hausdorff(MT(1))) ≈ 1
            @test mass(Hausdorff(MT(2))) ≈ 2π
            @test mass(Hausdorff(MT(3))) ≈ 8π^2
        end
    end
end
