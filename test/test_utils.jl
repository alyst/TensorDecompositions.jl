@testset "Utilities" begin

    T = rand_kruskal3(2, (10, 20, 30), true)
    @test size(T) == (10, 20, 30)

    @testset "khatrirao()" begin
        A = randn(5,4)
        @test_throws DimensionMismatch TensorDecompositions.khatrirao!(randn(5,5), A)
        @test_throws DimensionMismatch TensorDecompositions.khatrirao!(randn(4,5), A)
        @test TensorDecompositions.khatrirao!(randn(5,4), A) == A
        @test TensorDecompositions.khatrirao(A) == A

        @test_throws DimensionMismatch TensorDecompositions.khatrirao!(randn(30,4), rand(2,4), rand(5, 4), rand(3, 5))
        @test_throws DimensionMismatch TensorDecompositions.khatrirao!(randn(30,5), rand(2,4), rand(5, 4), rand(3, 4))
        @test_throws DimensionMismatch TensorDecompositions.khatrirao!(randn(29,4), rand(2,4), rand(5, 4), rand(3, 4))

        @test @inferred(TensorDecompositions.khatrirao!(randn(30,4), randn(2,4), randn(5, 4), randn(3, 4))) isa Matrix{Float64}

        @test_throws DimensionMismatch TensorDecompositions.khatrirao(rand(2,4), rand(5, 4), rand(3, 5))
        X = @inferred TensorDecompositions.khatrirao(rand(2,4), rand(5, 4), rand(3, 4))
        @test X isa Matrix{Float64}
        @test size(X) == (30, 4)
    end

    @testset "_row_unfold()" begin
        res = @inferred TensorDecompositions._row_unfold(T, 1)
        @test size(res) == (10, 600)

        res = @inferred TensorDecompositions._row_unfold(T, 2)
        @test size(res) == (20, 300)

        res = @inferred TensorDecompositions._row_unfold(T, 3)
        @test size(res) == (30, 200)
    end

    @testset "_col_unfold()" begin
        res = @inferred TensorDecompositions._col_unfold(T, 1)
        @test size(res) == (600, 10)

        res = @inferred TensorDecompositions._col_unfold(T, 2)
        @test size(res) == (300, 20)

        res = @inferred TensorDecompositions._col_unfold(T, 3)
        @test size(res) == (200, 30)
    end

    @testset "tensorcontractmatrices()" begin
        factors = TensorDecompositions._random_factors(size(T), (5, 2, 6))
        res = @inferred tensorcontractmatrices(T, factors)
        @test size(res) == (5, 2, 6)

        res2 = @inferred tensorcontractmatrices(T, factors[[2, 1]], [2, 1])
        @test size(res2) == (5, 2, 30)
    end

    @testset "tensorcontractmatrices!() with TensorOpHelper" begin
        helper0 = TensorDecompositions.TensorOpHelper{Float64}(use_pool=false)
        @test helper0 isa TensorDecompositions.SimpleTensorOpHelper
        @test TensorDecompositions.arraypool(helper0) === nothing
        helper = TensorDecompositions.TensorOpHelper{Float64}()
        @test helper isa TensorDecompositions.SimpleTensorOpHelper
        @test eltype(helper) == Float64
        @test TensorDecompositions.arraypool(helper) isa TensorDecompositions.ArrayPool
        factors = TensorDecompositions._random_factors(size(T), (5, 2, 6))
        dest = @inferred TensorDecompositions.acquire!(helper, (5, 2, 6))
        res = @inferred tensorcontractmatrices!(dest, T, factors, 1:3, transpose=false, helper=helper)
        @test res === dest
        res2 = @inferred tensorcontractmatrices(T, factors, 1:3, transpose=false)
        @test res == res2
        TensorDecompositions.release!(helper, dest)

        dest2 = @inferred TensorDecompositions.acquire!(helper, (5, 20, 6))
        res3 = @inferred tensorcontractmatrices!(dest2, T, factors[[1,3]], [1,3], transpose=false, helper=helper)
        @test res3 === dest2
        res4 = @inferred tensorcontractmatrices(T, factors[[1,3]], [1,3], transpose=false)
        @test res4 == res3
        TensorDecompositions.release!(helper, dest2)
    end

end
