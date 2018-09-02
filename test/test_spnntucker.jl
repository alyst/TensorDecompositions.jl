using TensorDecompositions: factor

@testset "Sparse (semi-)nonnegative Tucker decomposition" begin
Random.seed!(12345)

@testset "nonnegative decomposition" begin
    # An example of nonnegative and semi-nonnegative Tucker decomposition
    tucker_orig = rand_tucker((4, 5, 6), (40, 50, 60), factors_nonneg=true, core_nonneg=true)
    tnsr_orig = compose(tucker_orig)

    tnsr_max = maximum(tnsr_orig)
    tucker_orig.core ./= tnsr_max
    tnsr_orig ./= tnsr_max

    tnsr = add_noise(tnsr_orig, 0.6, true)

    # Solve the problem
    @time tucker_spnn = spnntucker(tnsr, size(tucker_orig.core), tol=1E-5, ini_decomp=:hosvd,
                             core_nonneg=true,
                             max_iter=1000, verbose=true, lambdas=fill(0.01, 4))

    # Reporting
    @test rel_residue(tucker_spnn) < 0.05
    @info "Relative error of decomposition : $(rel_residue(tucker_spnn))"
end

@testset "semi-nonnegative decomposition" begin
    tucker_orig = rand_tucker((4, 5, 6), (40, 50, 60), factors_nonneg=true, core_nonneg=false)
    tnsr_orig = compose(tucker_orig)

    tnsr_max = maximum(tnsr_orig)
    tucker_orig.core ./=  tnsr_max
    tnsr_orig ./= tnsr_max

    tnsr = add_noise(tnsr_orig, 0.6, false)

    # Solve the problem
    @time tucker_spnn = spnntucker(tnsr, size(tucker_orig.core), tol=1E-5,
                             ini_decomp=:hosvd, adaptive_steps=true,
                             core_nonneg=false,
                             max_iter=1000, verbose=true, lambdas=fill(0.01, 4))

    # Reporting
    @test rel_residue(tucker_spnn) < 0.05
    @info("Relative error of decomposition : $(rel_residue(tucker_spnn))")
end

@testset "pseudo-PCA decomposition" begin
    tucker_orig = rand_tucker((4, 5, 6), (40, 50, 60), factors_nonneg=true, core_nonneg=true)
    tnsr_orig = compose(tucker_orig)

    tnsr_max = maximum(tnsr_orig)
    tucker_orig.core ./= tnsr_max
    tnsr_orig ./= tnsr_max

    tnsr = add_noise(tnsr_orig, 0.6, false)

    # Solve the problem
    @time tucker_spnn = spnntucker(tnsr, size(tucker_orig.core), tol=1E-4, ini_decomp=:hosvd,
                             core_nonneg=false, mus=fill(0.01, 3), bounds=[1.0, 1.0, 1.0, Inf], Lmin=1.0,
                             max_iter=1000, verbose=true, lambdas=fill(0.01, 4))

    # Reporting
    @test rel_residue(tucker_spnn) < 0.05
    @info("Relative error of decomposition : $(rel_residue(tucker_spnn))")
end

@testset "weighted, nonnegative decomposition" begin
    # An example of nonnegative and semi-nonnegative Tucker decomposition
    tucker_orig = rand_tucker((4, 5, 6), (40, 50, 60), factors_nonneg=true, core_nonneg=true)
    tnsr_orig = compose(tucker_orig)

    tnsr_max = maximum(tnsr_orig)
    tucker_orig.core ./= tnsr_max
    tnsr_orig ./= tnsr_max

    tnsr = add_noise(tnsr_orig, 0.6, true)

    # Solve the problem
    @time tucker_spnn = spnntucker(tnsr, size(tucker_orig.core), tol=1E-4, ini_decomp=:hosvd,
                                   core_nonneg=true, tensor_weights=rand(size(tnsr)...), #fill(1.0, size(tnsr)...),
                                   max_iter=1000, verbose=true, lambdas=fill(0.01, 4))

    # Reporting
    @test rel_residue(tucker_spnn) < 0.05
    @info("Relative error of decomposition : $(rel_residue(tucker_spnn))")
end

@testset "fixed factors" begin
    # An example of nonnegative and semi-nonnegative Tucker decomposition
    tucker_orig = rand_tucker((4, 5, 6), (40, 50, 60), factors_nonneg=true, core_nonneg=true)
    tnsr_orig = compose(tucker_orig)

    tnsr_max = maximum(tnsr_orig)
    tucker_orig.core ./= tnsr_max
    tnsr_orig ./= tnsr_max

    tnsr = add_noise(tnsr_orig, 0.6, true)

    # Solve the problem
    @time tucker_spnn = spnntucker(tnsr, size(tucker_orig.core), tol=1E-4, ini_decomp=:hosvd,
                                   core_nonneg=true,
                                   fixed_factors=[nothing, factor(tucker_orig, 2), nothing],
                                   max_iter=1000, verbose=true, lambdas=fill(0.01, 4))

    @test rel_residue(tucker_spnn) < 0.05
    s2 = factor(tucker_spnn, 2)[1, 1] / factor(tucker_orig, 2)[1, 1]
    @test factor(tucker_spnn, 2) ≈ factor(tucker_orig, 2) * s2 # the factor is fixed up to a scale
    s1 = factor(tucker_spnn, 1)[1, 1] / factor(tucker_orig, 1)[1, 1]
    @test factor(tucker_spnn, 1) ≉ factor(tucker_orig, 1) * s1 # the other factor not
    @info "Relative error of decomposition : $(rel_residue(tucker_spnn))"
end

end
