using LinearAlgebra, ManifoldMeasures, Manifolds, MeasureTheory, Random, StatsBase
using Test
using Manifolds: ℝ

function test_vMF(d, μ; atol=1e-2, check_logdensity=true, estimate_mode=true)
    M = base_manifold(d)
    x = rand(normalize(basemeasure(d)))

    # check mode is μ
    @test mode(d) ≈ μ

    if check_logdensity
        v = project(M, μ, x)
        @test logdensity(d, μ) ≥ logdensity(d, exp(M, μ, v, 1e-5))
        @test logdensity(d, μ) ≥ logdensity(d, exp(M, μ, v, -1e-5))
    end

    # check rand returns the right type
    @test is_point(M, rand(d))
    @test real(eltype(rand(d))) <: Float64
    @test real(eltype(rand(Float32, d))) <: Float32

    if estimate_mode
        n = 100_000
        xs = [rand(d) for _ in 1:n]
        @test all(x -> is_point(M, x), xs)
        if d isa VonMises{ℝ}
            z̄ = mean(cis, xs)
            μ̂ = angle(z̄)
        else
            d isa VonMises{ℂ}
            z̄ = mean(xs)
            μ̂ = project(M, z̄)
        end
        @test norm(μ̂ - μ) ≤ atol
    end
end

@testset "von Mises-Fisher" begin
    # TODO: use frequentist tests from Mardia & Jupp to check samplers
    @testset "VonMises" begin
        @testset "real (μ, κ)" begin
            @test VonMises(; μ=0.5, κ=1.0) isa VonMisesFisher{Circle{ℝ},(:μ, :κ)}
            @testset for κ in (1e-2, 10, 1e5)
                μ = rand() * 2π - π
                test_vMF(VonMises(; μ=μ, κ=κ), μ; estimate_mode=κ == 10)
            end
        end
        @testset "complex (μ, κ)" begin
            @test VonMises(ℂ; μ=im, κ=1.0) isa VonMisesFisher{Circle{ℂ},(:μ, :κ)}
            @testset for κ in (1e-1, 50, 1e5)
                μ = sign(randn(ComplexF64))
                test_vMF(VonMises(ℂ; μ=μ, κ=κ), μ; estimate_mode=κ == 10)
            end
        end
        @testset "complex (c,)" begin
            @test VonMises(ℂ; c=3 + 4im) isa VonMisesFisher{Circle{ℂ},(:c,)}
            @testset for κ in (1e-1, 50, 1e5)
                c = randn(ComplexF64)
                test_vMF(VonMises(ℂ; c=c), sign(c); estimate_mode=κ == 10)
            end
        end
    end

    @testset "VonMisesFisher" begin
        @testset "Sphere" begin
            @testset "Sphere($n, $𝔽)" for 𝔽 in (ℝ, ℂ), n in (0, 1, 4)
                T = 𝔽 === ℂ ? ComplexF64 : Float64
                @test VonMisesFisher(n + 1, 𝔽) === VonMisesFisher(Sphere(n, 𝔽))
                @testset "(μ, κ)" begin
                    @testset for κ in (1e-1, 50, 1e5)
                        μ = normalize(randn(T, n + 1))
                        test_vMF(
                            VonMisesFisher(n + 1, 𝔽; μ=μ, κ=κ), μ; estimate_mode=κ == 10
                        )
                    end
                end
                @testset "(c,)" begin
                    @testset for κ in (1e-1, 50, 1e5)
                        c = randn(T, n + 1)
                        test_vMF(
                            VonMisesFisher(n + 1, 𝔽; c=c),
                            normalize(c);
                            estimate_mode=κ == 10,
                        )
                    end
                end
            end
        end
    end
end
