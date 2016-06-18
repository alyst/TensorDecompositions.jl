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
mutable struct SPNNTuckerHelper{T<:Number, N} <: TensorOpHelper{T}
    tnsr::Array{T, N}
    tnsr_weights::Union{Array{T, N}, Nothing}   # weights of `tnsr` elements
    wtnsr::Array{T, N}          # weighted `tnsr` (or `tnsr` if no weights)
    wtnsr_nrm::Float64          # norm of the weighted tnsr
    core_dims::NTuple{N,Int}
    wtnsrXfactors_low::Vector{Array{T, N}}
    lambdas::Vector{T}
    bounds::Vector{T}
    Lmin::Float64
    #tmp_core_unfold::Vector{Matrix{T}}
    L::Vector{Float64}          # previous Lipschitz constants
    L0::Vector{Float64}
    StepMultMin::Float64        # minimal step adjustment multiplier, 1 == no adaptation
    StepMult::Vector{Float64}   # step adjustment multipliers, != 1
    arr_pool::ArrayPool{T}

    function SPNNTuckerHelper(tnsr::Array{T,N}, core_dims::NTuple{N,Int},
                              lambdas::Vector{Float64}, bounds::Vector{T},
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
        new{T,N}(tnsr, tensor_weights, wtnsr, norm(wtnsr), core_dims,
                 [Array{T,N}(undef, ntuple(i -> i <= n ? core_dims[i] : tnsr_dims[i], N)) for n in 1:N],
                 lambdas, bounds,
                 Lmin, fill(1.0, N+1), fill(1.0, N+1),
                 StepMultMin, fill(1.0, N+1),
                 ArrayPool{T}()
        )
    end
end

is_adaptive_steps(helper::SPNNTuckerHelper) = helper.StepMultMin < 1.0

function _spnntucker_update_tensorXfactors_low!(helper::SPNNTuckerHelper{T,N}, decomp::Tucker{T,N}) where {T,N}
    tensorcontractmatrix!(helper.wtnsrXfactors_low[1], helper.wtnsr,
                          factor(decomp, 1), 1)
    for n in 2:N
        tensorcontractmatrix!(helper.wtnsrXfactors_low[n],
                              helper.wtnsrXfactors_low[n-1], factor(decomp, n), n)
    end
    return helper
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
    src_factor2s::Vector{Matrix{T}}, n::Integer
) where {T,N,PRJ}
    s = (helper.StepMult[N+1]/helper.L[N+1])
    s_lambda = (helper.lambdas[N+1]/helper.L[N+1])
    bound = helper.bounds[N+1]

    if helper.tnsr_weights === nothing
        tensorXfactors_all = n < N ?
            tensorcontractmatrices!(acquire!(helper, helper.core_dims),
                                    helper.wtnsrXfactors_low[n], dest.factors[(n+1):N], (n+1):N, helper=helper) :
            helper.wtnsrXfactors_low[N]
        core_grad = tensorcontractmatrices!(acquire!(helper, helper.core_dims), core(src), src_factor2s, helper=helper)
        dest.core .= _spnntucker_project.(prj, src.core .- s .* (core_grad .- tensorXfactors_all),
                                          s_lambda, bound)
        (n < N) && release!(helper, tensorXfactors_all) # not acquired if n < N
        release!(helper, core_grad)
    else
        # restore tensor from decomposition replacing the core and n-th factor from src
	    # FIXME store core*factors(dest, 1:(n-1))
    	wdecomp_delta = tensorcontractmatrices!(acquire!(helper, size(helper.wtnsr)), core(src),
        	                                    [factor(i != n ? dest : src, i) for i in 1:N], 1:N, transpose=true, helper=helper)
        # subtract tnsr and weight
	    wdecomp_delta .= (wdecomp_delta .- helper.tnsr) .* helper.tnsr_weights
        # convert back to core dimensions
        core_grad = tensorcontractmatrices!(acquire!(helper, helper.core_dims), wdecomp_delta,
                                            [factor(i != n ? dest : src, i) for i in 1:N], 1:N, transpose=false, helper=helper))
        dest.core .= _spnntucker_project.(prj, src.core .- s .* core_grad, s_lambda, bound)
        release!(helper, wdecomp_delta)
        release!(helper, core_grad)
    end
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
    lambda = helper.lambdas[n]
    bound = helper.bounds[n]
    helper.L0[n] = helper.L[n]

    all_but_n = [1:(n-1); (n+1):N]
    cXtf_size = ntuple(i -> i != n ? size(helper.wtnsr, i) : helper.core_dims[n], N)
    coreXtfactor = tensorcontractmatrices!(acquire!(helper, cXtf_size),
                                           core(dest),
                                           factors(dest, all_but_n), all_but_n, transpose=true, helper=helper)
    coreXtfactor2 = tensorcontract!(1, coreXtfactor, 1:N, 'N',
                                    coreXtfactor, [1:(n-1); N+1; (n+1):N], 'N',
                                     0, acquire!(helper, (helper.core_dims[n], helper.core_dims[n])), [n, N+1], method=:BLAS)
    helper.L[n] = max(helper.Lmin, norm(coreXtfactor2))
    s = (helper.StepMult[n]/helper.L[n])

    if helper.tnsr_weights === nothing
        tnsrXcoreXtfactor = tensorcontract!(1, helper.wtnsr, 1:N, 'N',
                                            coreXtfactor, [1:(n-1); N+1; (n+1):N], 'N',
                                            0, acquire!(helper, size(factor(dest, n))), [n, N+1], method=:BLAS)::Matrix{T}
        factorXcoreXtfactor2 = mul!(acquire!(helper, size(src_factor)), src_factor, coreXtfactor2)

        # update Lipschitz constant
        # update n-th factor matrix
        @assert size(dest_factor) == size(src_factor) == size(factorXcoreXtfactor2) == size(tnsrXcoreXtfactor)
        @inbounds if lambda == 0 && isfinite(bound)
             dest_factor .= min.(src_factor .- s .* (factorXcoreXtfactor2 .- tnsrXcoreXtfactor), bound)
        else
             dest_factor .= max.(src_factor .- s .* (factorXcoreXtfactor2 .- tnsrXcoreXtfactor .+ lambda), zero(T))
        end
        dest_factor2 = mul!(dest_factor2s[n], dest_factor', dest_factor)
        factor2XcoreXtfactor2 = dot(dest_factor2, coreXtfactor2)
        factorXtnsrXcoreXtfactor = dot(dest_factor, tnsrXcoreXtfactor)
        release!(helper, coreXtfactor)
        release!(helper, coreXtfactor2)
        release!(helper, tnsrXcoreXtfactor)
        release!(helper, factorXcoreXtfactor2)

        return 0.5*(factor2XcoreXtfactor2-2*factorXtnsrXcoreXtfactor+helper.wtnsr_nrm^2) +
               _spnntucker_reg_penalty(dest, helper.lambdas)
    else
        method = contractmethod(nothing, helper)
        wdecomp_delta = tensorcontractmatrix!(acquire!(helper, size(helper.wtnsr)), coreXtfactor,
                                              src_factor, n, transpose=true, method=method)
        @assert size(wdecomp_delta) == size(helper.tnsr) == size(helper.tnsr_weights)
        @inbounds wdecomp_delta .= (wdecomp_delta .- helper.tnsr) .* helper.tnsr_weights
        tnsrXfactors = tensorcontractmatrices!(acquire!(helper, ntuple(i -> i != n ? helper.core_dims[i] : size(helper.wtnsr, i), N)),
                                               wdecomp_delta,
                                               factors(dest, all_but_n), all_but_n, transpose=false, helper=helper)
        factor_grad = TensorOperations.contract!(
                1, tnsrXfactors, Val{:N}, core(dest), Val{:N}, 0,
                acquire!(helper, size(dest_factor)),
                (n,), ntuple(i -> i<n ? i : (i+1), N-1),
                (n,), ntuple(i -> i<n ? i : (i+1), N-1),
                (1, 2), Val{method})

        @assert size(src_factor) == size(factor_grad)
        @inbounds if lambda == 0 && isfinite(bound)
             dest_factor .= min.(src_factor .- s .* factor_grad, bound)
        else
             dest_factor .= max.(src_factor .- s .* (factor_grad .+ lambda), zero(T))
        end
        release!(helper, coreXtfactor)
        release!(helper, coreXtfactor2)
        release!(helper, tnsrXfactors)
        release!(helper, factor_grad)
        # recalculate the residue using the updated dest_factor
        tensorcontractmatrix!(wdecomp_delta, coreXtfactor, dest_factor, n, transpose=true, method=method)
        @inbounds wdecomp_delta .= (wdecomp_delta .- helper.tnsr) .* helper.tnsr_weights
        release!(helper, wdecomp_delta)
        # update dest factor square (used only for L calculation in the weighted case)
        mul!(dest_factor2s[n], dest_factor', dest_factor)
        return 0.5*sum(abs2, wdecomp_delta) + _spnntucker_reg_penalty(dest, helper.lambdas)
    end
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
                    tensor_weights::Union{StridedArray{T, N}, Nothing} = nothing,
                    core_nonneg::Bool=true, tol::Float64=1e-4, hosvd_init::Bool=false,
                    max_iter::Int=500, max_time::Float64=0.0,
                    lambdas::Vector{Float64} = fill(0.0, N+1),
                    Lmin::Float64 = 1.0, adaptive_steps::Bool=false, step_mult_min::Float64=1E-3,
                    rw::Float64=0.9999,
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
    helper = SPNNTuckerHelper(tnsr, core_dims, lambdas, bounds,
                              Lmin, adaptive_steps ? step_mult_min : 1.0,
                              tensor_weights=tensor_weights, verbose = verbose)
    verbose && @info("|tensor|=$(helper.wtnsr_nrm)")

    decomp0 = deepcopy(ini_decomp)
    if rescale_ini
        verbose && @info("Rescaling initial decomposition...")
        rescale!(decomp0, helper.wtnsr_nrm)
    end
    decomp = deepcopy(decomp0)     # current decomposition
    decomp_p = deepcopy(decomp0)   # proxy decomposition

    #verbose && @info("Calculating factors squares...")
    factor2s0 = Matrix{T}[f'f for f in factors(decomp0)]
    factor2s = deepcopy(factor2s0)
    factor2_nrms = norm.(factor2s)

    #verbose && @info("Calculating initial residue...")
    resid = resid0 = 0.5*sum(abs2, helper.tnsr_weights === nothing ?
                                   tnsr .- compose(decomp0) :
                                   helper.tnsr_weights .* (tnsr .- compose(decomp0))) +
            _spnntucker_reg_penalty(decomp0, lambdas)
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
    nnoredo = zeros(Int, N+1)
    converged = false

    #verbose && @info("Starting iterations...")
    pb = Progress(max_iter, "Alternating proximal gradient iterations ")
    niter = 1
    while !converged
        update!(pb, niter)

        residn0 = resid
        _spnntucker_update_tensorXfactors_low!(helper, decomp0)

        any_redone = false
        for n in N:-1:1
            # -- update the core tensor Z --
            helper.L0[N+1] = helper.L[N+1]
            helper.L[N+1] = max(prod(factor2_nrms), helper.Lmin)

            # try to make a step using extrapolated decompositon (Zm,Um)
            _spnntucker_update_core!(projection_type, helper, decomp, decomp_p, factor2s, n)
            residn = _spnntucker_update_factor!(helper, decomp, decomp_p, factor2s, n)
            redone = residn > residn0
            while residn > residn0
                # extrapolated Zm,Um decomposition lead to residual norm increase,
                # revert extrapolation and make a step using Z0,U0 to ensure
                # objective function is decreased
                # re-update to make objective nonincreasing
                copyto!(factor2s[n], factor2s0[n]) # restore factor square, core update needs it
                _spnntucker_update_core!(projection_type, helper, decomp, decomp0, factor2s, n)
                residn = _spnntucker_update_factor!(helper, decomp, decomp0, factor2s, n)
                if residn > residn0
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
            # update StepMult[n]
            if is_adaptive_steps(helper) && helper.StepMult[n] < 1.0 && (nnoredo[n] >= 3) && mod(nnoredo[n], 3) == 0
                # increase StepMult for the n-th factor after 3 successful iterations
                helper.StepMult[n] = min(helper.StepMult[n] * 1.1, 1.0)
                verbose && @info("Increasing $(n)-th factor step multiplier: $(helper.StepMult[n])")
            end
        end
        any_redone || (nnoredo[N+1] += 1)

        # --- diagnostics, reporting, stopping checks ---
        resid0 = resid
        resid = residn0

        #verbose && @info("Storing statistics...")
        cur_state = SPNNTuckerState(resid, resid0, helper.wtnsr_nrm)
        push!(iter_diag, cur_state)

        if is_adaptive_steps(helper) && helper.StepMult[N+1] < 1.0 && (nnoredo[N+1] >= 3) && mod(nnoredo[N+1], 3) == 0
            # increase StepMult for the core tensor after 3 successful iterations
            helper.StepMult[N+1] = min(helper.StepMult[N+1] * 1.05, 1.0)
            verbose && @info("Increasing core tensor step multiplier: $(helper.StepMult[N+1])")
        end

        # check stopping criterion
        adj_tol = tol * prod(helper.StepMult)^(1/length(helper.StepMult))
        niter += 1
        nstall = cur_state.rel_residue_delta < adj_tol ? nstall + 1 : 0
        if nstall >= 3 || cur_state.rel_residue < adj_tol
            verbose && (cur_state.rel_residue == 0.0) && @info("Residue is zero. Exact decomposition was found")
            verbose && (nstall >= 3) && @info("Decrease of the relative error is below $adj_tol $nstall times in a row")
            verbose && (cur_state.rel_residue < adj_tol) && @info("Relative error is $(cur_state.rel_residue) times below input tensor norm")
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
    res.props[:rel_residue] = 2*sqrt(resid-_spnntucker_reg_penalty(decomp, lambdas))/helper.wtnsr_nrm
    res.props[:iter_diag] = iter_diag
    return res
end
