#=
# Use GPU acceleration to evaluate the entire pseudospectrum at once 
=#
__precompile__()
module gpusvd
export sigma
    using ProgressMeter
    using ThreadsX
    using CUDA
    using LinearAlgebra
    using GenericLinearAlgebra
    using BlockArrays
    function sigma(Z::Matrix, L::Matrix)::Matrix{Float64}
        # Distribute a shifted matrix and find the smallest singular values
        sig_out = similar(Z)
        ndim = size(Z)[1] # number of shifted matrices
        Ndim = size(L)[1] # square size of each shifted matrix

        #= 
        # To maximize GPU acceleration, build a large diagonal matrix made of
        # all the shifted matrices. Send this large diagonal matrix to the GPU
        # and perform SVD, returning the singular values. For each block that is
        # returned, find the minimum singular value and store in the corresponding
        # entry of sigma
        =#

        # Total number of matrices: one matrix for every value in Z, each matrix is 
        # the size of L
        # Include progress bar for long calculations
        p = Progress(length(Z), dt=0.1, desc="Computing pseudospectrum...", 
        barglyphs=BarGlyphs("[=> ]"), barlen=50)
        Dmatrix = Vector{Matrix}(undef, length(Z))
        ThreadsX.foreach(Iterators.product(1:ndim, 1:ndim)) do (i,j)
            Dmatrix[(i - 1) * ndim + j] = L - Z[i,j] .* I
            next!(p)
        end
        finish!(p)
        # Turn into one large block-diagonal and apply svdvals 
        vals = svdvals(Diagonal(Dmatrix))
        # Find the minimum singular value from each block and rearrange
        # into the same form as Z
        sig_out = reshape(minimum.(vals), size(Z))
        # Row-major vs colum major means return the transpose of the result 
        return sig_out'

        # The GPU needs column-major data; pull the values from the resultant shift
        # matrix to fill the current block, ap/prepend with zeros to match the total
        # matrix dimensions
        #=
        bar = Diagonal([L - Z[i,j] .* I for i in 1:ndim for j in 1:ndim])
        bar = reshape(bar, (length(Z), length(Z)))
        bar = Matrix(mortar(bar))
        print("Shifted matrix: "); show(bar); println("")
        vals = svdvals(bar)
        print("SVD result: "); show(vals); println("")
        vals = reshape(vals, (Ndim, length(Z)))
        mins = [minimum(@view vals[:,i]) for i in 1:length(Z)]
        print("SVD vals: "); show(mins); println("")
        # Include progress bar for long calculations
        p = Progress(length(Z), dt=0.5, desc="Computing pseudospectrum...", barglyphs=BarGlyphs("[=> ]"), barlen=50)
        # Automatic load balancing, false sharing protection
        ThreadsX.foreach(Iterators.product(1:size(Z)[1], 1:size(Z)[2])) do (i,j)
            # Shift along the diagonal by a value in Z, take the smallest singular value
            foo[i,j] = real(minimum(GenericLinearAlgebra.svdvals!(L - Z[i,j] .* I))) # I is an automatically sized identity matrix
            next!(p)
        end
        return foo
        finish!(p)
        =#
    end
end