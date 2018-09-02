"""
Metrics of sparse (semi-)nonnegative Tucker decomposition progress.
"""
struct SPNNTuckerMetrics
    sqr_residue::Float64          # residue, i.e. 0.5 norm(tnsr - recomposed)^2
    rel_residue::Float64          # residue relative to the ||tnsr||
    rel_residue_delta::Float64    # residue delta relative to the current residue

    function SPNNTuckerMetrics(sqr_residue::Float64, prev_sqr_residue::Float64, tnsr_nrm::Float64)
        sqr_residue < -1E-10*tnsr_nrm^2 && @warn("Negative residue: $sqr_residue")
        sqr_residue = max(0.0, sqr_residue)
        new(sqr_residue, sqrt(2*sqr_residue)/tnsr_nrm, abs(sqr_residue-prev_sqr_residue)/(prev_sqr_residue+1E-5))
    end

end

"""
Determine projection type
"""
function _spnntucker_projection_type(is_nonneg::Bool, lambda::Number, bound::Number)
    if is_nonneg
        if lambda > 0.0
            # regularization
            return isfinite(bound) ? :NonnegRegBounded : :NonnegReg
        else
            return isfinite(bound) ? :NonnegBounded : :Nonneg
        end
    else
        if lambda > 0.0
            # regularization
            return isfinite(bound) ? :SignedRegBounded : :SignedReg
        else
            return isfinite(bound) ? :SignedBounded : :Unbounded
        end
    end
end

_spnntucker_project(::Type{Val{PRJ}}, x, lambda, bound) where PRJ = throw(ArgumentError("Unknown project type: $PRJ"))

_spnntucker_project(::Type{Val{:Nonneg}}, x, lambda, bound) = max(x, 0.0)
_spnntucker_project(::Type{Val{:NonnegReg}}, x, lambda, bound) = max(x - lambda, 0.0)
_spnntucker_project(::Type{Val{:NonnegBounded}}, x, lambda, bound) = clamp(x, 0.0, bound)
_spnntucker_project(::Type{Val{:NonnegRegBounded}}, x, lambda, bound) = clamp(x - lambda, 0.0, bound)

_spnntucker_project(::Type{Val{:Unbounded}}, x, lambda, bound) = x
_spnntucker_project(::Type{Val{:SignedReg}}, x, lambda, bound) = x > lambda ? x - lambda : (x < -lambda ? x + lambda : 0.0)
_spnntucker_project(::Type{Val{:SignedBounded}}, x, lambda, bound) = x > bound ? bound : (x < -bound ? -bound : x)
_spnntucker_project(::Type{Val{:SignedRegBounded}}, x, lambda, bound) = clamp(x > lambda ? x - lambda : (x < -lambda ? x + lambda : 0.0), -bound, bound)

"""
Helper object for spnntucker().
"""
mutable struct SPNNTuckerHelper{T<:Number, N} <: TensorOpHelper{T}
    tnsr::Array{T, N}
    tnsr_weights::Union{Array{T, N}, Nothing}   # weights of `tnsr` elements
    wtnsr::Array{T, N}          # weighted `tnsr` (or `tnsr` if no weights)
    wtnsr_nrm::Float64          # norm of the weighted tnsr
    proj_types::Vector{Type}
    lambdas::Vector{T}
    mus::Vector{T}
    bounds::Vector{T}
    Lmin::Float64
    #tmp_core_unfold::Vector{Matrix{T}}
    StepMultMin::Float64        # minimal step adjustment multiplier, 1 == no adaptation
    StepMult::Vector{Float64}   # step adjustment multipliers, != 1
    arr_pool::ArrayPool{T}

    function SPNNTuckerHelper(tnsr::Array{T,N}, is_nonneg::Vector{Bool},
                              lambdas::Vector{Float64}, mus::Vector{Float64}, bounds::Vector{T},
                              Lmin::Float64, StepMultMin::Float64;
                              tensor_weights::Union{Array{T,N}, Nothing} = nothing,
                              verbose::Bool=false) where {T, N}
        verbose && @info("Precomputing input tensor unfoldings...")
        tnsr_dims = size(tnsr)
        if tensor_weights === nothing
            wtnsr = tnsr
        else
            (tnsr_dims == size(tensor_weights)) || throw(DimensionMismatch("Shapes of tnsr and its weights do not match"))
            w_min, w_max = extrema(tensor_weights)
            (w_min >= 0) || throw(ArgumentError("Tensor weights should be non-negative"))
            isfinite(w_max) || throw(ArgumentError("Tensor weights not finite"))
            wtnsr = tnsr .* (tensor_weights ./ w_max)
        end
        # determine projection types
        proj_types = Type[Val{_spnntucker_projection_type(is_nonneg[i], lambdas[i] + (i <= N ? mus[i] : 0.0), bounds[i])} for i in 1:(N+1)]
        new{T,N}(tnsr, tensor_weights, wtnsr, norm(wtnsr),
                 proj_types, lambdas, mus, bounds,
                 Lmin, StepMultMin, fill(1.0, N+1),
                 ArrayPool{T}()
        )
    end
end

is_adaptive_steps(helper::SPNNTuckerHelper) = helper.StepMultMin < 1.0

"""
Internal state of Tucker decomposition for `spnntucker()`
"""
mutable struct SPNNTuckerState{T,N} <: TensorDecomposition{T, N}
    tucker::Tucker{T,N}
    factor2s::Vector{Matrix{T}}     # factor squares
    wtnsrXfactors_low::Vector{Array{T, N}}
    factor2_nrms::Vector{Float64}   # Frobenius norms of factor squares
    ortho_penalty::Vector{Float64}
    resid::Float64
    L::Vector{Float64}              # Lipschitz constants

    function SPNNTuckerState(tucker::Tucker{T,N}, helper::SPNNTuckerHelper{T,N}, init::Bool=false) where {T,N}
        decomp = new{T,N}(tucker,
                 [Matrix{T}(undef, size(f, 2), size(f, 2)) for f in factors(tucker)],
                 Vector{Array{T,N}}(), zeros(Float64, N), zeros(Float64, N), NaN,
                 fill(NaN, N+1))
        if init
            #verbose && @info("Calculating factors squares...")
            for (i, f) in enumerate(factors(tucker))
                ff = mul!(decomp.factor2s[i], f', f)
                decomp.factor2_nrms[i] = norm(ff)
            end
            #verbose && @info("Calculating initial residue...")
            delta = helper.wtnsr .- compose(decomp.tucker)
            if helper.tnsr_weights !== nothing
                delta .*= helper.tnsr_weights
            end
            decomp.resid = 0.5*sum(abs2, delta) + _spnntucker_reg_penalty(decomp, helper)
        end
        return decomp
    end

end

core(decomp::SPNNTuckerState) = core(decomp.tucker)
factor(decomp::SPNNTuckerState, i::Int) = factor(decomp.tucker, i)
factors(decomp::SPNNTuckerState) = factors(decomp.tucker)
factors(decomp::SPNNTuckerState, ixs) = factors(decomp.tucker, ixs)

"""
Copy the `n`-th factor and the core using the `src` state.
"""
function copy_core_and_factor!(dest::SPNNTuckerState{T,N}, src::SPNNTuckerState{T,N}, n::Int) where {T,N}
    copyto!(dest.tucker.core, src.tucker.core)
    copyto!(dest.tucker.factors[n], src.tucker.factors[n])
    copyto!(dest.factor2s[n], src.factor2s[n]) # restore factor square, core update needs it
    dest.factor2_nrms[n] = src.factor2_nrms[n]
    dest.ortho_penalty[n] = src.ortho_penalty[n]
    dest.resid = src.resid
    dest.L[N+1] = src.L[N+1]
    dest.L[n] = src.L[n]
    return dest
end

function update_tensorXfactors_low!(decomp::SPNNTuckerState{T,N}, tnsr::Array{T,N}) where {T,N}
    if isempty(decomp.wtnsrXfactors_low)
        decomp.wtnsrXfactors_low = [Array{T,N}(undef, ntuple(i -> size(i <= n ? core(decomp) : tnsr, i), N)) for n in 1:N]
    end

    tensorcontractmatrix!(decomp.wtnsrXfactors_low[1], tnsr,
                          factor(decomp, 1), 1)
    for n in 2:N
        tensorcontractmatrix!(decomp.wtnsrXfactors_low[n],
                              decomp.wtnsrXfactors_low[n-1], factor(decomp, n), n)
    end
    return decomp
end

function _spnntucker_reg_penalty(decomp::SPNNTuckerState{T,N}, helper::SPNNTuckerHelper{T, N}) where {T,N}
    res = 0.0
    for i in 1:N
        helper.lambdas[i] > 0.0 && (res += helper.lambdas[i] * sum(abs, factor(decomp, i)))
        helper.mus[i] > 0.0 && (res += 0.5 * helper.mus[i] * decomp.ortho_penalty[i])
    end
    helper.lambdas[N+1] > 0.0 && (res += helper.lambdas[N+1] * sum(abs, core(decomp)))
    return res
end

# update core tensor of `decomp`
function _spnntucker_update_core!(prj::Type{Val{PRJ}},
    decomp::SPNNTuckerState{T,N}, src_core::StridedArray{T,N}, n::Integer,
    helper::SPNNTuckerHelper{T,N}
) where {T,N,PRJ}
    # for weighted tensor this L is not accurate, but it should be fine
    decomp.L[N+1] = max(prod(decomp.factor2_nrms), helper.Lmin)
    @assert isfinite(decomp.L[N+1]) "Non-finite L[N+1]"
    s = (helper.StepMult[N+1]/decomp.L[N+1])
    s_lambda = (helper.lambdas[N+1]/decomp.L[N+1])
    bound = helper.bounds[N+1]

    if helper.tnsr_weights === nothing
        tensorXfactors_all = n < N ?
            tensorcontractmatrices!(acquire!(helper, size(core(decomp))),
                                    decomp.wtnsrXfactors_low[n], factors(decomp, (n+1):N), (n+1):N, helper=helper) :
            decomp.wtnsrXfactors_low[N]
        core_grad = tensorcontractmatrices!(acquire!(helper, size(core(decomp))), src_core,
                                            decomp.factor2s, 1:N, transpose=false, helper=helper)
        @inbounds core(decomp) .= _spnntucker_project.(prj, src_core .- s .* (core_grad .- tensorXfactors_all),
                                                       s_lambda, bound)
        (n < N) && release!(helper, tensorXfactors_all) # not acquired if n < N
        release!(helper, core_grad)
    else
        # restore tensor from decomposition replacing the core and n-th factor from src
	    # FIXME store core*factors(dest, 1:(n-1))
    	wdecomp_delta = tensorcontractmatrices!(acquire!(helper, size(helper.wtnsr)), src_core,
        	                                    factors(decomp), 1:N, transpose=true, helper=helper)
	    @inbounds wdecomp_delta .= (wdecomp_delta .- helper.tnsr) .* helper.tnsr_weights
        core_grad = tensorcontractmatrices!(acquire!(helper, size(core(decomp))), wdecomp_delta,
                                            factors(decomp), 1:N, transpose=false, helper=helper)
        @inbounds core(decomp) .= _spnntucker_project.(prj, src_core .- s .* core_grad, s_lambda, bound)
        release!(helper, wdecomp_delta)
        release!(helper, core_grad)
    end
    return decomp
end

# update n-th factor matrix of `decomp`
# return new residual
function _spnntucker_update_factor!(prj::Type{Val{PRJ}},
    decomp::SPNNTuckerState{T,N}, src_factor::StridedMatrix{T}, n::Int,
    helper::SPNNTuckerHelper{T,N}
) where {T,N,PRJ}
    dest_factor = factor(decomp, n)
    bound = helper.bounds[n]

    all_but_n = [1:(n-1); (n+1):N]
    core_ndim = size(core(decomp), n)
    cXtf_size = ntuple(i -> i != n ? size(helper.wtnsr, i) : core_ndim, N)
    coreXtfactor = tensorcontractmatrices!(acquire!(helper, cXtf_size),
                                           core(decomp),
                                           factors(decomp, all_but_n), all_but_n, transpose=true, helper=helper)
    coreXtfactor2 = tensorcontract!(1, coreXtfactor, 1:N, 'N',
                                    coreXtfactor, [1:(n-1); N+1; (n+1):N], 'N',
                                    0, acquire!(helper, (core_ndim, core_ndim)), [n, N+1], method=:BLAS)::Matrix{T}
    tnsrXcoreXtfactor = tensorcontract!(1, helper.wtnsr, 1:N, 'N',
                                        coreXtfactor, [1:(n-1); N+1; (n+1):N], 'N',
                                        0, acquire!(helper, size(dest_factor)), [n, N+1], method=:BLAS)::Matrix{T}
    # update Lipschitz constant
    decomp.L[n] = max(helper.Lmin, norm(coreXtfactor2))
    s = (helper.StepMult[n]/decomp.L[n])

    if helper.tnsr_weights === nothing
        # update n-th factor matrix
        if helper.mus[n] == 0.0
            factorXcoreXtfactor2 = mul!(acquire!(helper, size(src_factor)), src_factor, coreXtfactor2)
            @assert size(dest_factor) == size(src_factor) == size(factorXcoreXtfactor2) == size(tnsrXcoreXtfactor)
            lambda = s*helper.lambdas[n]
            @inbounds dest_factor .= _spnntucker_project.(prj, src_factor .- s .* (factorXcoreXtfactor2 .- tnsrXcoreXtfactor),
                                                          lambda, bound)
            release!(helper, factorXcoreXtfactor2)
        else
            # SVD-like case, enforce "orthogonality" of dest_factor columns
            coreXtfactor_mtx = _col_unfold(copyto!(acquire!(helper, size(coreXtfactor)), coreXtfactor), n)
            wtnsr_delta_mtx = _col_unfold(copyto!(acquire!(helper, size(helper.wtnsr)), helper.wtnsr), n)
            # multiply wtnsr_delta_mtx = wtnsr_delta_mtx - coreXfactor*dest_factor
            # FIXME use proper character (instead of 'N') when T is complex
            BLAS.gemm!('N', 'T', one(T), coreXtfactor_mtx, dest_factor, -one(T), wtnsr_delta_mtx)
            tmp_col = acquire!(helper, size(src_factor, 1))
            dest_factor_cols_sum = acquire!(helper, (size(dest_factor, 1), 1))
            sum!(dest_factor_cols_sum, dest_factor)
            fXf_mtx = mul!(acquire!(helper, (size(dest_factor, 1), size(dest_factor, 1))), dest_factor, dest_factor')
            total_L2 = 0.0
            for j in 1:size(src_factor, 2)
                # column-wise update of dest factor
                # subtract j-th dest_factor column × j-th row from coreXtfactor2 from factorXcoreXtfactor2
                src_col = view(src_factor, :, j)
                dest_col = view(dest_factor, :, j)
                cXtf_row = view(coreXtfactor_mtx, :, j)
                @inbounds for i1 in eachindex(dest_col), i2 in eachindex(cXtf_row)
                    wtnsr_delta_mtx[i2, i1] -= dest_col[i1]*cXtf_row[i2]
                end
                # subtract j-th dest_factor column × itself from fXf
                @inbounds dest_factor_cols_sum .-= dest_col # exclude dest_col from L
                @inbounds for i1 in eachindex(dest_col), i2 in eachindex(dest_col)
                    fXf_mtx[i2, i1] -= dest_col[i1]*dest_col[i2]
                end

                # update Lipschitz constant for orthogonalization term
                L2 = sum(abs2, dest_factor_cols_sum) *
                     abs2(dot(src_col, view(dest_factor_cols_sum, :, 1)))
                total_L2 += L2
                L = max(helper.Lmin, sqrt(L2))

                # update dest_col
                mul!(tmp_col, wtnsr_delta_mtx', cXtf_row)
                mul!(dest_col, fXf_mtx, src_col)
                b = max(helper.Lmin, sum(abs2, cXtf_row))
                k1 = 1.0/(b+helper.mus[n]*L)
                k2 = k1*helper.mus[n]
                lambda = k1*helper.lambdas[n]
                bound = helper.bounds[n]
                dest_col .= _spnntucker_project.(prj, k2.*(src_col.*L .- dest_col) .- k1.*tmp_col, lambda, bound)
                # add updated j-th dest_factor column × j-th row from coreXtfactor2 from factorXcoreXtfactor2
                @inbounds for i1 in eachindex(dest_col), i2 in eachindex(cXtf_row)
                    wtnsr_delta_mtx[i2, i1] += dest_col[i1]*cXtf_row[i2]
                end
                # add updated j-th dest_factor column × itself from fXf
                @inbounds dest_factor_cols_sum .+= dest_col # add updated dest_col to the L
                @inbounds for i1 in eachindex(dest_col), i2 in eachindex(dest_col)
                    fXf_mtx[i2, i1] += dest_col[i1]*dest_col[i2]
                end
            end
            # update dest_factor-dependent fields of decomp
            decomp.L[n] += helper.mus[n]*sqrt(total_L2)
            ortho_penalty = 0.0
            @inbounds for j1 in 1:size(dest_factor, 2)
                dest_col1 = view(dest_factor, :, j1)
                @inbounds for j2 in (j1+1):size(dest_factor, 2)
                    ortho_penalty += dot(dest_col1, view(dest_factor, :, j2))^2
                end
            end
            decomp.ortho_penalty[n] = 2*ortho_penalty
            release!(helper, wtnsr_delta_mtx)
            release!(helper, coreXtfactor_mtx)
            release!(helper, fXf_mtx)
            release!(helper, tmp_col)
            release!(helper, dest_factor_cols_sum)
        end
        dest_factor2 = mul!(decomp.factor2s[n], dest_factor', dest_factor)
        decomp.factor2_nrms[n] = norm(dest_factor2)
        factor2XcoreXtfactor2 = dot(dest_factor2, coreXtfactor2)
        factorXtnsrXcoreXtfactor = dot(dest_factor, tnsrXcoreXtfactor)
        release!(helper, coreXtfactor)
        release!(helper, coreXtfactor2)
        release!(helper, tnsrXcoreXtfactor)
        decomp.resid = 0.5*(factor2XcoreXtfactor2-2*factorXtnsrXcoreXtfactor+abs2(helper.wtnsr_nrm)) +
               _spnntucker_reg_penalty(decomp, helper)
    else
        method = contractmethod(nothing, helper)
        wdecomp_delta = tensorcontractmatrix!(acquire!(helper, size(helper.wtnsr)), coreXtfactor,
                                              src_factor, n, transpose=true, method=method)
        @assert size(wdecomp_delta) == size(helper.tnsr) == size(helper.tnsr_weights)
        @inbounds wdecomp_delta .= (wdecomp_delta .- helper.tnsr) .* helper.tnsr_weights
        tnsrXfactors = tensorcontractmatrices!(acquire!(helper, ntuple(i -> i != n ? size(core(decomp), i) : size(helper.wtnsr, i), N)),
                                               wdecomp_delta,
                                               factors(decomp, all_but_n), all_but_n, transpose=false, helper=helper)
        factor_grad = TensorOperations.contract!(
                1, tnsrXfactors, Val{:N}, core(decomp), Val{:N}, 0,
                acquire!(helper, size(dest_factor)),
                (n,), ntuple(i -> i<n ? i : (i+1), N-1),
                (n,), ntuple(i -> i<n ? i : (i+1), N-1),
                (1, 2), Val{method})

        lambda = s*helper.lambdas[n]
        @assert size(src_factor) == size(factor_grad)
        @inbounds dest_factor .= _spnntucker_project.(prj, src_factor .- s .* factor_grad,
                                                      lambda, bound)
        release!(helper, coreXtfactor)
        release!(helper, coreXtfactor2)
        release!(helper, tnsrXfactors)
        release!(helper, factor_grad)
        # recalculate the residue using the updated dest_factor
        tensorcontractmatrix!(wdecomp_delta, coreXtfactor, dest_factor, n, transpose=true, method=method)
        @inbounds wdecomp_delta .= (wdecomp_delta .- helper.tnsr) .* helper.tnsr_weights
        release!(helper, wdecomp_delta)
        # update dest factor square (used only for L calculation in the weighted case)
        dest_factor2 = mul!(decomp.factor2s[n], dest_factor', dest_factor)
        decomp.factor2_nrms[n] = norm(dest_factor2)
        decomp.resid = 0.5*sum(abs2, wdecomp_delta) + _spnntucker_reg_penalty(decomp, helper)
    end
    # for weighted tensor this L is not accurate, but it should be fine
    decomp.L[N+1] = max(prod(decomp.factor2_nrms), helper.Lmin)
    return decomp
end

function _spnntucker_update_proxy_factor!(
    proxy::Tucker{T,N}, cur::Tucker{T,N}, prev::Tucker{T,N},
    n::Integer, w::Number
) where {T,N}
    @assert size(proxy.factors[n]) == size(cur.factors[n]) == size(prev.factors[n])
    @inbounds proxy.factors[n] .= cur.factors[n] .+ w .* (cur.factors[n] .- prev.factors[n])
    return proxy.factors[n]
end

function _spnntucker_update_proxy_core!(
    proxy::Tucker{T,N}, cur::Tucker{T,N}, prev::Tucker{T,N}, w::Number
) where {T,N}
    @assert size(proxy.core) == size(cur.core) == size(prev.core)
    @inbounds proxy.core .= cur.core .+ w .* (cur.core .- prev.core)
    return proxy
end

"""
Sparse (semi-)nonnegative Tucker decomposition

Decomposes nonnegative tensor `tnsr` into optionally nonnegative `core` tensor
and sparse nonnegative factor matrices `factors`.

 * `tnsr` nonnegative `N`-mode tensor to decompose
 * `core_dims` size of a core densor
 * `core_nonneg` if true, the output core tensor is nonnegative
 * `tensor_weights` if not `nothing`, the weights of `tnsr` elements in the residual error
 * `tol` the target error of decomposition relative to the Frobenius norm of `tnsr`
 * `max_iter` maximum number of iterations if error stays above `tol`
 * `max_time` max running time
 * `lambdas` `N+1` vector of non-negative sparsity regularizer coefficients for the factor matrices and the core tensor
 * `Lmin` lower bound for Lipschitz constant for the gradients of the residual error eqn{l(Z,U) = fnorm(tnsr - ttl(Z, U))` by `Z` and each `U`
 * `rw` controls the extrapolation weight
 * `bounds` `N+1` vector of the maximal absolute values allows for the elements of core tensor and factor matrices (effective only if the regularization is disabled)
 * `ini_decomp` initial decomposition, if equals to `:hosvd`, `hosvd()` is used to generate the starting decomposition, if `nothing`, a random decomposition is used
 * `verbose` more output algorithm progress

Returns:
  * `Tucker` decomposition object with additional properties:
    * `:converged` method convergence indicator
    * `:rel_residue` the Frobenius norm of the residual error `l(Z,U)` plus regularization penalty (if any)
    * `:niter` number of iterations
    * `:nredo` number of times `core` and `factor` were recalculated to avoid the increase in objective function
    * `:iter_diag` convergence info for each iteration, see `SPNNTuckerMetrics`

The function uses the alternating proximal gradient method to solve the following optimization problem:
 deqn{min 0.5 |tnsr - Z times_1 U_1 ldots times_K U_K |_{F^2} +
 sum_{n=1}^{K} lambda_n |U_n|_1 + lambda_{K+1} |Z|_1, ;text{where; Z geq 0, U_i geq 0.}
 If `core_nonneg` is `FALSE`, core tensor `Z` is allowed to have negative
 elements and eqn{z_{i,j}=max(0,z_{i,j}-lambda_{K+1}/L_{K+1}}) rule is replaced by eqn{z_{i,j}=sign(z_{i,j})max(0,|z_{i,j}|-lambda_{K+1}/L_{K+1})}.
 The method stops if either the relative improvement of the error is below the tolerance `tol` for 3 consequitive iterations or
 both the relative error improvement and relative error (wrt the `tnsr` norm) are below the tolerance.
 Otherwise it stops if the maximal number of iterations or the time limit were reached.

The implementation is based on ntds_fapg() MATLAB code by Yangyang Xu and Wotao Yin.

See Y. Xu, "Alternating proximal gradient method for sparse nonnegative Tucker decomposition", Math. Prog. Comp., 7, 39-70, 2015.
See http://www.caam.rice.edu/~optimization/bcu/`
"""
function spnntucker(tnsr::StridedArray{T, N}, core_dims::NTuple{N, Int};
                    tensor_weights::Union{StridedArray{T, N}, Nothing} = nothing,
                    core_nonneg::Bool=true, factor_nonneg::Union{AbstractVector{Bool}, Nothing} = nothing,
                    tol::Float64=1e-4, hosvd_init::Bool=false,
                    max_iter::Int=500, max_time::Float64=0.0,
                    lambdas::Vector{Float64} = fill(0.0, N+1),
                    mus::Vector{Float64} = fill(0.0, N),
                    Lmin::Float64 = 1.0, adaptive_steps::Bool=false, step_mult_min::Float64=1E-3,
                    rw::Float64=0.9999,
                    bounds::Vector{Float64} = fill(Inf, N+1), ini_decomp = nothing,
                    verbose::Bool=false) where {T,N}
    start_time = time()

    # define "kernel" functions for "fixing" the core tensor after iteration

    if ini_decomp === nothing
        verbose && @info("Generating random initial factor matrices and core tensor estimates...")
        ini_decomp = Tucker(ntuple(i -> randn(size(tnsr, i), core_dims[i]), N), randn(core_dims...))
        rescale_ini = true
    elseif ini_decomp == :hosvd
        verbose && @info("Using High-Order SVD to get initial decomposition...")
        # "solve" Z = tnsr x_1 U_1' ... x_K U_K'
        ini_decomp = hosvd(tnsr, core_dims, pad_zeros=true)
        rescale_ini = true
    elseif isa(ini_decomp, Tucker{T,N})
        ini_decomp = deepcopy(ini_decomp)
        rescale_ini = false
    else
        throw(ArgumentError("Incorrect ini_decomp value"))
    end

    #verbose && @info("Initializing helper object...")
    helper = SPNNTuckerHelper(tnsr,
                              Bool[factor_nonneg !== nothing ? factor_nonneg : fill(true, N); core_nonneg],
                              lambdas, mus, bounds,
                              Lmin, adaptive_steps ? step_mult_min : 1.0,
                              tensor_weights=tensor_weights, verbose = verbose)
    verbose && @info("|tensor|=$(helper.wtnsr_nrm)")

    if rescale_ini
        verbose && @info("Rescaling initial decomposition...")
        rescale!(ini_decomp, helper.wtnsr_nrm)
    end
    decomp0 = SPNNTuckerState(ini_decomp, helper, true)   # previous
    verbose && @info("Initial residue=$(decomp0.resid)")
    decomp = deepcopy(decomp0)                            # current decomposition
    decomp_p = deepcopy(decomp0.tucker)   # proxy decomposition

    # Iterations of block-coordinate update
    # iteratively updated variables:
    # GradU: gradients with respect to each component matrix of U
    # GradZ: gradient with respect to Z
    t0 = fill(1.0, N+1)
    t = deepcopy(t0)

    iter_diag = Vector{SPNNTuckerMetrics}()
    nstall = 0
    nredo = 0
    nnoredo = zeros(Int, N+1)
    converged = false

    #verbose && @info("Starting iterations...")
    pb = Progress(max_iter, "Alternating proximal gradient iterations ")
    niter = 1
    resid_prev = NaN
    while !converged
        resid_prev = decomp.resid
        update_tensorXfactors_low!(decomp, helper.wtnsr)

        any_redone = false
        for n in N:-1:1
            # --- correction and extrapolation ---
            t[n] = (1.0+sqrt(1.0+4.0*t0[n]^2))/2.0
            #verbose && info("Updating proxy factors $n...")
            t[N+1] = (1.0+sqrt(1.0+4.0*t0[N+1]^2))/2.0
            # -- update the core tensor and n-th factor proxies --
            _spnntucker_update_proxy_factor!(decomp_p, decomp.tucker, decomp0.tucker, n,
                                             min((t0[n]-1)/t[n], rw*sqrt(isfinite(decomp.L[n]) && isfinite(decomp0.L[n]) ? decomp0.L[n]/decomp.L[n] : 1.0)))
            _spnntucker_update_proxy_core!(decomp_p, decomp.tucker, decomp0.tucker,
                                           min((t0[N+1]-1)/t[N+1], rw*sqrt(isfinite(decomp.L[N+1]) && isfinite(decomp0.L[N+1]) ? decomp0.L[N+1]/decomp.L[N+1] : 1.0)))
            t0[n] = t[n]
            t0[N+1] = t[N+1]
            copy_core_and_factor!(decomp0, decomp, n)

            # try to make a step using extrapolated decompositon (Zm,Um)
            _spnntucker_update_core!(helper.proj_types[N+1], decomp, core(decomp_p), n, helper)
            _spnntucker_update_factor!(helper.proj_types[n], decomp, factor(decomp_p, n), n, helper)
            redone = decomp.resid > decomp0.resid
            while niter > 1 && decomp.resid > decomp0.resid
                # extrapolated Zm,Um decomposition lead to residual norm increase,
                # revert extrapolation and make a step using Z0,U0 to ensure
                # objective function is decreased
                # re-update to make objective nonincreasing
                copy_core_and_factor!(decomp, decomp0, n)
                _spnntucker_update_core!(helper.proj_types[n], decomp, core(decomp0), n, helper)
                _spnntucker_update_factor!(helper.proj_types[n], decomp, factor(decomp0, n), n, helper)
                if decomp.resid > decomp0.resid
                    verbose && @warn("$niter: residue increase at redo step")
                    if is_adaptive_steps(helper)
                        # reduce core and n-th factor steps by 0.9 and 0.8, resp.
                        StepMultCore = max(0.9 * helper.StepMult[N+1], helper.StepMultMin)
                        StepMultFactor = max(0.8 * helper.StepMult[n], helper.StepMultMin)
                        if (StepMultCore == helper.StepMultMin) && (StepMultFactor == helper.StepMultMin)
                            verbose && @warn("$niter: adaptive step multipliers reached their minimum")
                            break
                        end
                        helper.StepMult[N+1] = StepMultCore
                        helper.StepMult[n] = StepMultFactor
                    else
                        break
                    end
                end
            end
            if redone
                nnoredo[n] = 0
                nnoredo[N+1] = 0
                any_redone = true
                nredo += 1
            else
                nnoredo[n] += 1
            end

            # update StepMult[n]
            if is_adaptive_steps(helper) && helper.StepMult[n] < 1.0 && (nnoredo[n] >= 3) && mod(nnoredo[n], 3) == 0
                # increase StepMult for the n-th factor after 3 successful iterations
                helper.StepMult[n] = min(helper.StepMult[n] * 1.1, 1.0)
                verbose && @info("Increasing $(n)-th factor step multiplier: $(helper.StepMult[n])")
            end
        end
        any_redone || (nnoredo[N+1] += 1)

        # --- diagnostics, reporting, stopping checks ---

        #verbose && @info("Storing statistics...")
        cur_metrics = SPNNTuckerMetrics(decomp0.resid, resid_prev, helper.wtnsr_nrm)
        push!(iter_diag, cur_metrics)

        if is_adaptive_steps(helper) && helper.StepMult[N+1] < 1.0 && (nnoredo[N+1] >= 3) && mod(nnoredo[N+1], 3) == 0
            # increase StepMult for the core tensor after 3 successful iterations
            helper.StepMult[N+1] = min(helper.StepMult[N+1] * 1.05, 1.0)
            verbose && @info("Increasing core tensor step multiplier: $(helper.StepMult[N+1])")
        end

        update!(pb, niter, showvalues=[(Symbol("|resid|"), sqrt(2*decomp.resid)),
                                       (Symbol("|resid|/|T|"), cur_metrics.rel_residue),
                                       (Symbol("1-|resid[i]|^2/|resid[i-1]|^2"), cur_metrics.rel_residue_delta)])

        # check stopping criterion
        adj_tol = tol * prod(helper.StepMult)^(1/length(helper.StepMult))
        niter += 1
        nstall = cur_metrics.rel_residue_delta < adj_tol ? nstall + 1 : 0
        if nstall >= 3 || cur_metrics.rel_residue < adj_tol
            verbose && (cur_metrics.rel_residue == 0.0) && @info("Residue is zero. Exact decomposition was found")
            verbose && (nstall >= 3) && @info("Decrease of the relative error is below $adj_tol $nstall times in a row")
            verbose && (cur_metrics.rel_residue < adj_tol) && @info("Relative error is $(cur_metrics.rel_residue) times below input tensor norm")
            verbose && @info("spnntucker() converged in $niter iteration(s), $nredo redo steps")
            converged = true
            finish!(pb)
            break
        elseif (max_time > 0) && ((time() - start_time) > max_time)
            cancel(pb, "Maximal time exceeded, might be not an optimal solution")
            verbose && @info("Final relative error $(cur_metrics.rel_residue)")
            break
        elseif niter == max_iter
            cancel(pb, "Maximal number of iterations reached, might be not an optimal solution")
            verbose && @info("Final relative error $(cur_metrics.rel_residue)")
            break
        end
    end # iterations
    finish!(pb)

    res = decomp0.tucker
    res.props[:niter] = niter
    res.props[:nredo] = nredo
    res.props[:converged] = converged
    res.props[:rel_residue] = 2*sqrt(decomp0.resid-_spnntucker_reg_penalty(decomp0, helper))/helper.wtnsr_nrm
    res.props[:iter_diag] = iter_diag
    return res
end
