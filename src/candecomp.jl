function candecomp(T::StridedArray,
                   r::Integer;
                   algo::String="als",
                   tol::Float64=1e-5,
                   maxiter::Integer=100,
                   compute_res::Bool=true,
                   random_init::Bool=true,
                   verbose::Bool=true) 
    @compat algos = Dict{String, Function}( "als" => _candecomp_als, 
                                            "sgsd" => _candecomp_sgsd)
    haskey(algos, algo) || error(string("Algorithm ", algo," does not exist."))
    num_modes = _check_tensor(T, r)
    factors = random_init ? _random_init(size(T), r) : hosvd(T, r, compute_core=false).factors      
    return algos[algo](T, r, num_modes, factors, tol, maxiter, compute_res, verbose)
end

function _candecomp_als(T::StridedArray,
                        r::Integer,
                        num_modes::Integer,
                        factors::Vector{Matrix{Float64}},
                        tol::Float64,
                        maxiter::Integer,
                        compute_res::Bool,
                        verbose::Bool)

    gram = [F'F for F in factors]
    T_norm = vecnorm(T) 
    T_flat = _unfold(T, num_modes)'
    T_size = size(T)
    niters = 0
    converged = false
    res = T_norm 
    lbds = Array(Float64, r)
    V = Array(Float64, div(length(T), minimum(T_size)), r) 
    while !converged && niters < maxiter
        VB = 0
        for i in 1:num_modes
            idx = [num_modes:-1:i + 1, i - 1:-1:1]
            VB = prod(T_size[idx]) 
            @inbounds V[1:VB, :] = reduce(_KhatriRao, factors[idx])
            @inbounds factors[i] = _unfold(T, i) * V[1:VB, :] / reduce(.*, gram[idx])
            lbds = sum(abs(factors[i]), 1)
            factors[i] ./= lbds
            gram[i] = factors[i]'factors[i]
        end
        res_old = res 
        @inbounds res = vecnorm(V[1:VB, :] * (factors[num_modes] .* lbds)' - T_flat) 
        converged = abs(res - res_old) < tol * res_old 
        niters += 1
    end

    verbose && _iter_status(converged, niters, maxiter)

    return Tucker(T, factors, lbds, compute_res=compute_res)
end

function _candecomp_sgsd(T::StridedArray,
                         r::Integer,
                         num_modes::Integer,
                         factors::Vector{Matrix{Float64}},
                         tol::Float64,
                         maxiter::Integer,
                         compute_res::Bool,
                         verbose::Bool)
 
    num_modes == 3 || error("This algorithm only applies to 3-mode tensors.")

    (n1, n2, n3) = size(T)
    IB = (min(n1 - 1, r), (n2 == r) ? 2 : 1)
    Q = qr(factors[1], thin=false)[1]
    Z = qr(fliplr(factors[2]), thin=false)[1]
    q = Array(Float64, n1, n1)
    z = Array(Float64, n2, n2)
    R = tensorcontract(tensorcontract(T, [1, 2, 3], Q, [1, 4], [4, 2, 3]), [1, 2, 3], Z, [2, 4], [1, 4, 3])

    res = vecnorm(T)
    converged = false
    niters = 0
    while !converged && niters < maxiter
        q = eye(n1)::Matrix{Float64}
        z = eye(n2)::Matrix{Float64}

        for i in 1:IB[1]
            @inbounds q[:, i:n1]::Matrix{Float64} *= svd(q[:, i:n1]' * slice(R, :, n2 - r + i, :))[1]
        end
        R = tensorcontract(R, [1, 2, 3], q, [1, 4], [4, 2, 3])
        for i in r:-1:IB[2]
            @inbounds z[:, 1:n2 - r + i]::Matrix{Float64} *= fliplr(svd(slice(R, i, :, :)' * z[:, 1:n2 - r + i])[3])
        end
        Q *= q
        Z *= z
        R = tensorcontract(tensorcontract(T, [1, 2, 3], Q, [1, 4], [4, 2, 3]), [1, 2, 3], Z, [2, 4], [1, 4, 3])

        res_old = res
        res = vecnorm(tril(squeeze(sum(R .^ 2, 3), 3), n2 - r - 1))
        converged = abs(res - res_old) < tol 
        niters += 1
    end

    verbose && _iter_status(converged, niters, maxiter)

    R = R[1:r, n2-r+1:n2, :]
    M = cat(3, eye(r, r), eye(r, r))
    for i in r - 1:-1:1, j = i + 1:r
        d = i + 1:j - 1
        @inbounds M[i, j, :] = hcat(R[j, j, :][:], R[i, i, :][:]) \ (R[i, j, :][:] - mapslices(R3 -> sum(M[i, d, 1] * (diag(R3)[d] .* M[d, j, 2])), R, [1, 2])[:])
    end
    
    factors[1] = Q[:, 1:r] * M[:, :, 1] 
    factors[2] = Z[:, n2 - r + 1:n2] * M[:, :, 2]'
    factors[3] = _unfold(T, 3) * _KhatriRao(factors[2], factors[1]) / ((factors[2]'factors[2]) .* (factors[1]'factors[1]))
    
    lbds = ones(1, r)
    for i in 1:num_modes
        lbd = mapslices(vecnorm, factors[i], 1) 
        factors[i] ./= lbd
        lbds .*= lbd
    end

    return Tucker(T, factors, lbds, compute_res=compute_res)
end

