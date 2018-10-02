"""
Tucker decomposition of a N-mode tensor.
"""
struct Tucker{T<:Number, N} <: TensorDecomposition{T, N}
    factors::NTuple{N, Matrix{T}}   # factor matrices
    core::Array{T, N}               # core tensor
    props::Dict{Symbol, Any}        # extra properties

    Tucker{T, N}(factors::NTuple{N, Matrix{T}}, core::StridedArray{T, N}) where {T<:Number, N} =
        new{T, N}(factors, core, Dict{Symbol, Any}())
end

Tucker(factors::NTuple{N, Matrix{T}}, core::StridedArray{T, N}) where {T<:Number, N} =
    Tucker{T, N}(factors, core)

function Tucker(factors::AbstractArray{<:StridedMatrix{T}}, core::StridedArray{T, N}) where {T<:Number, N}
    length(factors) == N || throw(ArgumentError("Number of factor matrices do not match the core dimensions"))
    Tucker{T, N}(tuple(factors...), core)
end

"""
Returns the core tensor of Tucker decomposition.
"""
core(decomp::Tucker) = decomp.core

"""
Returns the specified factor matrix of Tucker decomposition.
"""
factor(decomp::Tucker, i::Int) = decomp.factors[i]

"""`factors(decomp::Tucker, [ixs])`
Returns the factor matrices of Tucker decomposition.
"""
factors(decomp::Tucker) = decomp.factors
factors(decomp::Tucker, ixs) = decomp.factors[ixs]

"""
Composes a full tensor from Tucker decomposition.
"""
compose(decomp::Tucker{T}; helper::Union{TensorOpHelper{T}, Nothing} = nothing) where {T,N} =
    tensorcontractmatrices(core(decomp), factors(decomp),
                           transpose=true, helper=helper)

compose!(dest::StridedArray{T,N}, decomp::Tucker{T,N};
         helper::Union{TensorOpHelper{T}, Nothing} = nothing) where {T,N} =
    tensorcontractmatrices!(dest, core(decomp), factors(decomp),
                            transpose=true, helper=helper)

"""
Scale the factors and core of the initial decomposition.
Each decompositon component is scaled proportional to the number of its elements.
After scaling, `|decomp|=s`
"""
function rescale!(decomp::Tucker, s::Number)
    total_length = length(decomp.core) + sum(length, decomp.factors) # total elements in the decomposition
    for F in decomp.factors
        F .*= s^(length(F)/total_length)/norm(F)
    end
    decomp.core .*= s^(length(decomp.core)/total_length)/norm(decomp.core)
    return decomp
end
