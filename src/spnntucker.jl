"""
State of sparse (semi-)nonnegative Tucker decomposition
"""
struct SPNNTuckerState
    sqr_residue::Float64          # residue, i.e. 0.5 norm(tnsr - recomposed)^2
    rel_residue::Float64          # residue relative to the ||tnsr||
    rel_residue_delta::Float64    # residue delta relative to the current residue

    function SPNNTuckerState(sqr_residue::Float64, prev_sqr_residue::Float64, tnsr_nrm::Float64)
        sqr_residue < -1E-10*tnsr_nrm^2 && @warn("Negative residue: $sqr_residue")
        sqr_residue = max(0.0, sqr_residue)
        new(sqr_residue, sqrt(2*sqr_residue)/tnsr_nrm, abs(sqr_residue-prev_sqr_residue)/(prev_sqr_residue+1E-5))
    end

end

"""
Helper object for spnntucker().
"""
struct SPNNTuckerHelper{T<:Number, N} <: TensorOpHelper{T}
    tnsr::Array{T, N}
    tnsr_nrm::Float64
    core_dims::NTuple{N,Int}
    tnsrXfactors_low::Vector{Array{T, N}}
    lambdas::Vector{T}
    bounds::Vector{T}
    Lmin::Float64
    #tmp_core_unfold::Vector{Matrix{T}}
    L::Vector{Float64}   # previous Lipschitz constants
    L0::Vector{Float64}
    arr_pool::ArrayPool{T}

    function SPNNTuckerHelper(tnsr::Array{T,N}, core_dims::NTuple{N,Int},
                              lambdas::Vector{Float64}, bounds::Vector{T},
                              Lmin::Float64; verbose::Bool=false) where {T, N}
        verbose && @info("Precomputing input tensor unfoldings...")
        tnsr_dims = size(tnsr)
        new{T,N}(tnsr, norm(tnsr), core_dims,
                 [Array{T,N}(undef, ntuple(i -> i <= n ? core_dims[i] : tnsr_dims[i], N)) for n in 1:N],
                 lambdas, bounds,
                 Lmin, fill(1.0, N+1), fill(1.0, N+1), ArrayPool{T}()
        )
    end
end

function _spnntucker_update_tensorXfactors_low!(helper::SPNNTuckerHelper{T,N}, decomp::Tucker{T,N}) where {T,N}
    tensorcontractmatrix!(helper.tnsrXfactors_low[1], helper.tnsr,
                          factor(decomp, 1), 1)
    for n in 2:N
        tensorcontractmatrix!(helper.tnsrXfactors_low[n],
                              helper.tnsrXfactors_low[n-1], factor(decomp, n), n)
    end
    return helper
end

function _spnntucker_factor_grad_components!(helper::SPNNTuckerHelper{T,N}, decomp::Tucker{T,N}, n::Int) where {T,N}
    all_but_n = [1:(n-1); (n+1):N]
    cXtf_size = (size(helper.tnsr)[1:n-1]..., helper.core_dims[n], size(helper.tnsr)[(n+1):N]...)
    coreXtfactor = tensorcontractmatrices!(acquire!(helper, cXtf_size),
                                           core(decomp),
                                           factors(decomp, all_but_n), all_but_n, transpose=true, helper=helper)
    cXtf2 = tensorcontract!(1, coreXtfactor, 1:N, 'N',
                            coreXtfactor, [1:(n-1); N+1; (n+1):N], 'N',
                            0, acquire!(helper, (helper.core_dims[n], helper.core_dims[n])), [n, N+1], method=:BLAS)
    tXcXtf = tensorcontract!(1, helper.tnsr, 1:N, 'N',
                             coreXtfactor, [1:(n-1); N+1; (n+1):N], 'N',
                             0, acquire!(helper, size(factor(decomp, n))), [n, N+1], method=:BLAS)
    release!(helper, coreXtfactor)
    return cXtf2, tXcXtf
end

function _spnntucker_reg_penalty(decomp::Tucker{T,N}, lambdas::Vector{T}) where {T,N}
    res = 0.0
    for i in 1:N
        res += lambdas[i] > 0.0 ? (lambdas[i] * sum(abs, factor(decomp, i))) : 0.0
    end
    return res + (lambdas[N+1] > 0.0 ? (lambdas[N+1] * sum(abs, core(decomp))) : 0.0)
end

_spnntucker_project(::Type{Val{PRJ}}, x, lambda, bound) where {PRJ} = throw(ArgumentError("Unknown project type: $PRJ"))

_spnntucker_project(::Type{Val{:Nonneg}}, x, lambda, bound) = max(x, 0.0)
_spnntucker_project(::Type{Val{:NonnegReg}}, x, lambda, bound) = max(x - lambda, 0.0)
_spnntucker_project(::Type{Val{:NonnegBounded}}, x, lambda, bound) = clamp(x, 0.0, bound)

_spnntucker_project(::Type{Val{:Unbounded}}, x, lambda, bound) = x
_spnntucker_project(::Type{Val{:SignedReg}}, x, lambda, bound) = x > lambda ? x - lambda : (x < -lambda ? x + lambda : 0.0)
_spnntucker_project(::Type{Val{:SignedBounded}}, x, lambda, bound) = x > bound ? bound : (x < -bound ? -bound : x)

# update core tensor of dest
function _spnntucker_update_core!(prj::Type{Val{PRJ}},
    helper::SPNNTuckerHelper{T,N}, dest::Tucker{T,N}, src::Tucker{T,N},
    src_factor2s::Vector{Matrix{T}}, n::Integer) where {T,N,PRJ}

    tensorXfactors_all = n < N ?
        tensorcontractmatrices!(acquire!(helper, helper.core_dims),
                                helper.tnsrXfactors_low[n], dest.factors[(n+1):N], (n+1):N, helper=helper) :
        helper.tnsrXfactors_low[N]
    s = (1.0/helper.L[N+1])
    core_grad = tensorcontractmatrices!(acquire!(helper, helper.core_dims), core(src), src_factor2s, helper=helper)
    s_lambda = (helper.lambdas[N+1]/helper.L[N+1])::Float64
    bound = helper.bounds[N+1]
    dest.core .= _spnntucker_project.(prj, src.core .- s .* (core_grad .- tensorXfactors_all),
                                      s_lambda, bound)
    (n < N) && release!(helper, tensorXfactors_all) # not acquired if n < N
    release!(helper, core_grad)
    return dest
end

# update n-th factor matrix of dest
# return new residual
function _spnntucker_update_factor!(
    helper::SPNNTuckerHelper{T,N}, dest::Tucker{T,N}, src::Tucker{T,N},
    dest_factor2s::Vector{Matrix{T}}, n::Int
) where {T,N}
    src_factor = factor(src, n)
    dest_factor = factor(dest, n)
    coreXtfactor2, tnsrXcoreXtfactor = _spnntucker_factor_grad_components!(helper, dest, n)
    factorXcoreXtfactor2 = mul!(acquire!(helper, size(src_factor)), src_factor, coreXtfactor2)

    # update Lipschitz constant
    helper.L0[n] = helper.L[n]
    helper.L[n] = max(helper.Lmin, norm(coreXtfactor2))
    s = (1.0/helper.L[n])
    # update n-th factor matrix

    lambda = helper.lambdas[n]
    bound = helper.bounds[n]
    @assert size(dest_factor) == size(src_factor) == size(factorXcoreXtfactor2) == size(tnsrXcoreXtfactor)
    @inbounds if lambda == 0.0 && isfinite(bound)
        dest_factor .= clamp.(src_factor .- s .* (factorXcoreXtfactor2 .- tnsrXcoreXtfactor), 0.0, bound)
    else
        dest_factor .= max.(src_factor .- s .* (factorXcoreXtfactor2 .- tnsrXcoreXtfactor .+ lambda), 0.0)
    end
    dest_factor2 = mul!(dest_factor2s[n], dest_factor', dest_factor)
    factor2XcoreXtfactor2 = dot(dest_factor2, coreXtfactor2)
    factorXtnsrXcoreXtfactor = dot(dest_factor, tnsrXcoreXtfactor)
    release!(helper, coreXtfactor2)
    release!(helper, tnsrXcoreXtfactor)
    release!(helper, factorXcoreXtfactor2)

    return 0.5*(factor2XcoreXtfactor2-2*factorXtnsrXcoreXtfactor+helper.tnsr_nrm^2) +
            _spnntucker_reg_penalty(dest, helper.lambdas)
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
    * `:iter_diag` convergence info for each iteration, see `SPNNTuckerState`

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
                    core_nonneg::Bool=true, tol::Float64=1e-4, hosvd_init::Bool=false,
                    max_iter::Int=500, max_time::Float64=0.0,
                    lambdas::Vector{Float64} = fill(0.0, N+1),
                    Lmin::Float64 = 1.0, rw::Float64=0.9999,
                    bounds::Vector{Float64} = fill(Inf, N+1), ini_decomp = nothing,
                    verbose::Bool=false) where {T,N}
    start_time = time()

    # define "kernel" functions for "fixing" the core tensor after iteration
    core_bound = bounds[N+1]
    core_lambda = lambdas[N+1]
    if core_nonneg
        if core_lambda > 0.0
            # regularization
            projection_type = Val{:NonnegReg}
        elseif isfinite(core_bound)
            projection_type = Val{:NonnegBounded}
        else
            projection_type = Val{:Nonneg}
        end
    else
        if core_lambda > 0.0
            # regularization
            projection_type = Val{:SignedReg}
        elseif isfinite(core_bound)
            projection_type = Val{:SignedBounded}
        else
            projection_type = Val{:Unbounded}
        end
    end

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
        rescale_ini = false
    else
        throw(ArgumentError("Incorrect ini_decomp value"))
    end

    #verbose && @info("Initializing helper object...")
    helper = SPNNTuckerHelper(tnsr, core_dims, lambdas, bounds, Lmin, verbose = verbose)
    verbose && @info("|tensor|=$(helper.tnsr_nrm)")

    verbose && @info("Rescaling initial decomposition...")
    decomp0 = deepcopy(ini_decomp)
    if rescale_ini
        rescale!(decomp0, helper.tnsr_nrm)
    end
    decomp = deepcopy(decomp0)     # current decomposition
    decomp_p = deepcopy(decomp0)   # proxy decomposition

    #verbose && @info("Calculating factors squares...")
    factor2s0 = Matrix{T}[f'f for f in factors(decomp0)]
    factor2s = deepcopy(factor2s0)
    factor2_nrms = norm.(factor2s)

    #verbose && @info("Calculating initial residue...")
    resid = resid0 = 0.5*sum(abs2, tnsr .- compose(decomp0)) + _spnntucker_reg_penalty(decomp0, lambdas)
    resid = resid0 # current residual error
    verbose && @info("Initial residue=$resid0")

    # Iterations of block-coordinate update
    # iteratively updated variables:
    # GradU: gradients with respect to each component matrix of U
    # GradZ: gradient with respect to Z
    t0 = fill(1.0, N+1)
    t = deepcopy(t0)

    iter_diag = Vector{SPNNTuckerState}()
    nstall = 0
    nredo = 0
    converged = false

    #verbose && @info("Starting iterations...")
    pb = Progress(max_iter, "Alternating proximal gradient iterations ")
    niter = 1
    while !converged
        update!(pb, niter)

        residn0 = resid
        _spnntucker_update_tensorXfactors_low!(helper, decomp0)

        for n in N:-1:1
            # -- update the core tensor Z --
            helper.L0[N+1] = helper.L[N+1]
            helper.L[N+1] = max(helper.Lmin, prod(factor2_nrms))

            # try to make a step using extrapolated decompositon (Zm,Um)
            _spnntucker_update_core!(projection_type, helper, decomp, decomp_p, factor2s, n)
            residn = _spnntucker_update_factor!(helper, decomp, decomp_p, factor2s, n)
            if residn > residn0
                # extrapolated Zm,Um decomposition lead to residual norm increase,
                # revert extrapolation and make a step using Z0,U0 to ensure
                # objective function is decreased
                nredo += 1
                # re-update to make objective nonincreasing
                copyto!(factor2s[n], factor2s0[n]) # restore factor square, core update needs it
                _spnntucker_update_core!(projection_type, helper, decomp, decomp0, factor2s, n)
                residn = _spnntucker_update_factor!(helper, decomp, decomp0, factor2s, n)
                verbose && residn > residn0 && @warn("$niter: residue increase at redo step")
            end
            # --- correction and extrapolation ---
            t[n] = (1.0+sqrt(1.0+4.0*t0[n]^2))/2.0
            #verbose && @info("Updating proxy factors $n...")
            _spnntucker_update_proxy_factor!(decomp_p, decomp, decomp0, n, min((t0[n]-1)/t[n], rw*sqrt(helper.L0[n]/helper.L[n])))
            t[N+1] = (1.0+sqrt(1.0+4.0*t0[N+1]^2))/2.0
            #verbose && @info("Updating proxy core $n...")
            _spnntucker_update_proxy_core!(decomp_p, decomp, decomp0, min((t0[N+1]-1)/t[N+1], rw*sqrt(helper.L0[N+1]/helper.L[N+1])))

            #verbose && @info("Storing updated core and factors...")
            copyto!(decomp0.core, decomp.core)
            copyto!(decomp0.factors[n], decomp.factors[n])
            copyto!(factor2s0[n], factor2s[n])
            factor2_nrms[n] = norm(factor2s[n])
            t0[n] = t[n]
            t0[N+1] = t[N+1]
            residn0 = residn
        end

        # --- diagnostics, reporting, stopping checks ---
        resid0 = resid
        resid = residn0

        #verbose && @info("Storing statistics...")
        cur_state = SPNNTuckerState(resid, resid0, helper.tnsr_nrm)
        push!(iter_diag, cur_state)

        # check stopping criterion
        niter += 1
        nstall = cur_state.rel_residue_delta < tol ? nstall + 1 : 0
        if nstall >= 3 || cur_state.rel_residue < tol
            verbose && (cur_state.rel_residue == 0.0) && @info("Residue is zero. Exact decomposition was found")
            verbose && (nstall >= 3) && @info("Relative error below $tol $nstall times in a row")
            verbose && (cur_state.rel_residue < tol) && @info("Relative error is $(cur_state.rel_residue) times below input tensor norm")
            verbose && @info("spnntucker() converged in $niter iteration(s), $nredo redo steps")
            converged = true
            finish!(pb)
            break
        elseif (max_time > 0) && ((time() - start_time) > max_time)
            cancel(pb, "Maximal time exceeded, might be not an optimal solution")
            verbose && @info("Final relative error $(cur_state.rel_residue)")
            break
        elseif niter == max_iter
            cancel(pb, "Maximal number of iterations reached, might be not an optimal solution")
            verbose && @info("Final relative error $(cur_state.rel_residue)")
            break
        end
    end # iterations
    finish!(pb)

    res = decomp0
    res.props[:niter] = niter
    res.props[:nredo] = nredo
    res.props[:converged] = converged
    res.props[:rel_residue] = 2*sqrt(resid-_spnntucker_reg_penalty(decomp, lambdas))/helper.tnsr_nrm
    res.props[:iter_diag] = iter_diag
    return res
end
