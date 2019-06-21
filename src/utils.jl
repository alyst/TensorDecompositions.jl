# utilities

abstract type TensorOpHelper{T<:Number} end

Base.eltype(helper::TensorOpHelper{T}) where T = T
arraypool(helper::TensorOpHelper) = helper.arr_pool
acquire!(helper::TensorOpHelper{T}, dims) where T =
    arraypool(helper) !== nothing ? acquire!(arraypool(helper), dims) : Array{T}(undef, dims)
release!(helper::TensorOpHelper{T}, arr::Array{T}) where T =
    (arraypool(helper) !== nothing) && release!(arraypool(helper), arr)

arraypool(helper::Nothing) = nothing

struct SimpleTensorOpHelper{T,P} <: TensorOpHelper{T}
    arr_pool::P

    # default constructor to resolve type instability
    SimpleTensorOpHelper{T}(pool::Union{Nothing, ArrayPool{T}}) where T =
        new{T,typeof(pool)}(pool)
end

TensorOpHelper{T}(pool::Union{ArrayPool{T}, Nothing}) where T =
    SimpleTensorOpHelper{T}(pool)

TensorOpHelper{T}(; use_pool::Bool = true) where T =
    TensorOpHelper{T}(use_pool ? ArrayPool{T}() : nothing)

function tensorcontractmatrix!(dest::StridedArray{T,N}, src::StridedArray{T,N},
                               mtx::StridedMatrix{T}, n::Int;
                               transpose::Bool=false) where {T,N}
    #@info "TTM: dest=$(size(dest)) src=$(size(src)) mtx=$(size(mtx)) n=$n transpose=$transpose"
    TensorOperations.contract!(1, src, :N, mtx, :N, 0, dest,
                               ntuple(i -> i<n ? i : (i+1), N-1), (n,),
                               transpose ? (1,) : (2,), transpose ? (2,) : (1,),
                               ntuple(i -> i<n ? i : (i==n ? N : i-1), N))
end

function tensorcontractmatrix(tnsr::StridedArray{T,N}, mtx::StridedMatrix{T}, n::Int;
                              transpose::Bool=false,
                              helper::Union{TensorOpHelper{T}, Nothing} = nothing) where {T, N}
    dest_size = ntuple(i -> i != n ? size(tnsr, i) : size(mtx, transpose ? 1 : 2), N)
    tensorcontractmatrix!(helper !== nothing ? acquire!(helper, dest_size) : Array{T, N}(undef, dest_size),
                          tnsr, mtx, n, transpose=transpose)
end

"""
Contract N-mode tensor and M matrices.

  * `dest` array to hold the result
  * `src`  source tensor to contract
  * `matrices` matrices to contract
  * `modes` corresponding modes of matrices to contract
  * `helper` `TensorOpHelper` to use for getting intermediate tensors
  * `transpose` if true, matrices are contracted along their columns
"""
function tensorcontractmatrices!(dest::StridedArray{T,N}, src::StridedArray{T,N},
                                 matrices::Any, modes::Any = ntuple(i -> i, N);
                                 transpose::Bool=false,
                                 helper::Union{TensorOpHelper{T}, Nothing} = nothing) where {T, N}
    length(matrices) == length(modes) ||
        throw(ArgumentError("The number of matrices doesn't match the length of mode sequence"))
    local tmp::Array{T,N}
    res_size = collect(size(src))
    for i in eachindex(matrices)
        mode = modes[i]
        mtx = matrices[i]
        res_size[mode] = size(mtx, transpose ? 1 : 2)
        res_dims = ntuple(k -> res_size[k], N)
        dest_i = i < length(matrices) ? (helper !== nothing ? acquire!(helper, res_dims) : Array{T, N}(undef, res_dims)) : dest
        tensorcontractmatrix!(dest_i, i > 1 ? tmp : src, mtx, mode,
                              transpose=transpose)
        (i > 1) && (helper !== nothing) && release!(helper, tmp)
        tmp = dest_i
    end
    return dest
end

"""
Contract N-mode tensor and M matrices.

If `pool` is provided, the resulting tensor is acquired from the pool.

  * `tensor` tensor to contract
  * `matrices` matrices to contract
  * `modes` corresponding modes of matrices to contract
  * `pool` `ArrayPool` to use for getting intermediate and resulting tensors
  * `transpose` if true, matrices are contracted along their columns
"""
function tensorcontractmatrices(tensor::StridedArray{T,N}, matrices::Any,
                                modes::Any = 1:length(matrices);
                                transpose::Bool=false,
                                helper::Union{TensorOpHelper{T}, Nothing} = nothing) where {T, N}
    length(matrices) == length(modes) ||
        throw(ArgumentError("The number of matrices doesn't match the length of mode sequence"))
    new_size = collect(size(tensor))
    for (mtx, mode) in zip(matrices, modes)
        new_size[mode] = size(mtx, transpose ? 1 : 2)
    end
    dest_dims = ntuple(i -> new_size[i], N)
    return tensorcontractmatrices!(helper !== nothing ? acquire!(helper, dest_dims) : Array{T,N}(undef, dest_dims),
                                   tensor, matrices, modes,
                                   transpose=transpose, helper=helper)
end

"""
Generates random factor matrices for Tucker/CANDECOMP etc decompositions.

  * `orig_dims` original tensor dimensions
  * `core_dims` core tensor dimensions

Returns:
  * a vector of `N` (orig[n], core[n])-sized matrices
"""
_random_factors(orig_dims::NTuple{N, Int}, core_dims::NTuple{N, Int}) where {N} =
    Matrix{Float64}[randn(o_dim, c_dim) for (o_dim, c_dim) in zip(orig_dims, core_dims)]

"""
Generates random factor matrices for Tucker/CANDECOMP decompositions if core tensor is `r^N` hypercube.

Returns:
  * a vector of `N` (orig[n], r)-sized matrices
"""
_random_factors(dims::NTuple{N, Int}, r::Integer) where {N} =
    _random_factors(dims, ntuple(_ -> r, N))

"""
    khatrirao!(dest::AbstractMatrix{T}, mtxs::NTuple{N, <:AbstractMatrix{T}})

In-place Khatri-Rao matrices product (column-wise Kronecker product) calculation.
"""
@generated function khatrirao!(dest::AbstractMatrix{T},
                               mtxs::NTuple{N, <:AbstractMatrix{T}}) where {N, T}
    (N === 1) && return quote
        size(dest) == size(mtxs[1]) ||
            throw(DimensionMismatch("Output and single input matrix have different sizes ($(size(dest)) and $(size(mtxs[1])))"))
        return copyto!(dest, mtxs[1])
    end
    # generate the code for looping over the matrices 2:N
    _innerloop = Base.Cartesian.lreplace(:(desti[offsj_k + 1] = destij_k), :k, N)
    for k in N:-1:2
        _innerloop = Base.Cartesian.lreplace(quote
            for j_k in axes(mtxs[k], 1)
                destij_k = destij_{k-1}*coli_k[j_k]
                offsj_k = offsj_{k-1}*size(mtxs[k], 1) + j_k - 1
                $_innerloop
            end
        end, :k, k)
    end
    # main code
    quote
    # dimensions check
    ncols = size(dest, 2)
    for i in 1:length(mtxs)
        (size(mtxs[i], 2) == ncols) ||
            throw(DimensionMismatch("Output matrix and input matrix #$i have different number of columns ($ncols and $(size(mtxs[i], 2)))"))
    end
    nrows = prod(@ntuple($N, i -> size(mtxs[i], 1)))
    size(dest, 1) == nrows ||
        throw(DimensionMismatch("Output matrix rows and the expected number of rows do not match ($(size(dest, 1)) and $nrows)"))
    # multiplication
    @inbounds for i in axes(dest, 2)
        @nexprs($N, k -> (coli_k = view(mtxs[k], :, i)))
        desti = view(dest, :, i)
        for j_1 in axes(mtxs[1], 1)
            destij_1 = coli_1[j_1]
            offsj_1 = j_1 - 1
            $_innerloop
        end
    end
    return dest
    end
end

khatrirao!(dest::AbstractMatrix, mtxs::AbstractVector) =
    khatrirao!(dest, tuple(mtxs...))
khatrirao!(dest::AbstractMatrix, mtxs...) = khatrirao!(dest, tuple(mtxs...))

"""
    khatrirao(mtxs::NTuple{N, <:AbstractMatrix{T}})

Calculates Khatri-Rao product of a sequence of matrices (column-wise Kronecker product).
"""
function khatrirao(mtxs::NTuple{N, <:AbstractMatrix{T}}) where {N, T}
    (N === 1) && return copy(first(mtxs))
    ncols = size(first(mtxs), 2)
    for i in 2:length(mtxs)
        (size(mtxs[i], 2) == ncols) ||
            throw(DimensionMismatch("Input matrices have different number of columns ($ncols and $(size(mtxs[i], 2)))"))
    end
    return khatrirao!(Matrix{T}(undef, prod(ntuple(i -> size(mtxs[i], 1), N)), ncols), mtxs)
end

khatrirao(mtxs::AbstractVector) = khatrirao(tuple(mtxs...))
khatrirao(mtxs...) = khatrirao(tuple(mtxs...))

"""
Unfolds the tensor into matrix, such that the specified
group of modes becomes matrix rows and the other one becomes columns.

  * `row_modes` vector of modes to be unfolded as rows
  * `col_modes` vector of modes to be unfolded as columns
"""
function _unfold(tnsr::StridedArray, row_modes::Vector{Int}, col_modes::Vector{Int})
    length(row_modes) + length(col_modes) == ndims(tnsr) ||
        throw(ArgumentError("column and row modes should be disjoint subsets of 1:$(ndims(tnsr))"))

    dims = size(tnsr)
    return reshape(permutedims(tnsr, [row_modes; col_modes]),
                   prod(dims[row_modes]), prod(dims[col_modes]))
end

function _unfold(tnsr::StridedArray, row_modes::NTuple{NR, Int}, col_modes::NTuple{NC, Int}) where {NR, NC}
    length(row_modes) + length(col_modes) == ndims(tnsr) ||
        throw(ArgumentError("column and row modes should be disjoint subsets of 1:$(ndims(tnsr))"))

    return reshape(permutedims(tnsr, ntuple(i -> i <= NR ? row_modes[i] : col_modes[i - NR], ndims(tnsr))),
                   prod(ntuple(i -> size(tnsr, row_modes[i]), NR)),
                   prod(ntuple(i -> size(tnsr, col_modes[i]), NC)))
end

"""
Unfolds the tensor into matrix such that the specified mode becomes matrix row.
"""
_row_unfold(tnsr::StridedArray, mode::Integer) =
    _unfold(tnsr, (mode,), ntuple(i -> i < mode ? i : i+1, ndims(tnsr)-1))

"""
Unfolds the tensor into matrix such that the specified mode becomes matrix column.
"""
_col_unfold(tnsr::StridedArray, mode::Integer) =
    _unfold(tnsr, ntuple(i -> i < mode ? i : i+1, ndims(tnsr)-1), (mode,))

function _iter_status(converged::Bool, niters::Integer, maxiter::Integer)
    converged ? @info("Algorithm converged after $(niters) iterations.") :
                @warn("Maximum number $(maxiter) of iterations exceeded.")
end

_check_sign(v::StridedVector) = sign(v[findmax(abs.(v))[2]]) * v

"""
Checks the validity of the core tensor dimensions.
"""
function _check_tensor(tnsr::StridedArray{T, N}, core_dims::NTuple{N, Int}) where {T<:Real,N}
    ndims(tnsr) > 2 || throw(ArgumentError("This method does not support scalars, vectors, or matrices input."))
    for i in 1:N
        0 < core_dims[i] <= size(tnsr, i) ||
            throw(ArgumentError("core_dims[$i]=$(core_dims[i]) given, 1 <= core_dims[$i] <= size(tensor, $i) = $(size(tnsr, i)) expected."))
    end
    #isreal(T) || throw(ArgumentError("This package currently only supports real-number-valued tensors."))
    return N
end

"""
Checks the validity of the core tensor dimensions, where core tensor is `r^N` hypercube.
"""
_check_tensor(tensor::StridedArray{<:Number, N}, r::Integer) where N =
    _check_tensor(tensor, ntuple(_ -> r, N))
