#=
# Use GPU acceleration to evaluate the entire pseudospectrum at once 
=#
__precompile__()
module gpusvd
export sigma
export pspec
    using ProgressMeter
    using ThreadsX
    using CUDA
    using LinearAlgebra
    using GenericLinearAlgebra
    using BlockArrays
    using BlockDiagonals
    using SparseArrays
    function sigma(Z::Matrix, L::Matrix)::Matrix{Float64}
        # Distribute a shifted matrix and find the smallest singular values
        sig_out = similar(Z)
        ndim = size(Z)[1] # number of shifted matrices
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
    end

    # Calculate the pseudospectrum using a large, block diagonal matrix
    function sigma(G::Matrix, Ginv::Matrix, Z::Matrix, L::Matrix)
        # Distribute a shifted matrix and find the smallest singular values
        sig_out = similar(Z)
        ndim = size(Z)[1] # number of shifted matrices
        # Include progress bar for long calculations
        p = Progress(length(Z), dt=0.1, desc="Computing pseudospectrum...", 
        barglyphs=BarGlyphs("[=> ]"), barlen=50)
        Dmatrix = Vector{Matrix}(undef, length(Z))
        ThreadsX.foreach(Iterators.product(1:ndim, 1:ndim)) do (i,j)
            # Calculate the shifted matrix
            Lshift = L - Z[i,j] .* I
            # Calculate the adjoint
            Lshift_adj = Ginv * adjoint(Lshift) * G
            Dmatrix[(i - 1) * ndim + j] = Lshift_adj * Lshift
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

    end

    # Calculate the pseudospectrum using the Gram matrices
    function pspec(G::Matrix, Ginv::Matrix, Z::Matrix, L::Matrix)
        ##########################################################################
        ##########################################################################
        #=
        #  NOTE: it appears that there is not an SVD algorithm that returns
        #  unsorted singular values. That means that stacking all shifted matrices
        #  into a large block diagonal and solving the entire system destroys
        #  the location information required. Only if each L operator is 
        #  very large (> 10,000) would GPU methods produce a speedup. Unless a GPU
        #  algorithm can be found that does not sort the singular values, this
        #  avenue is a dead-end. The code below performs the stacking and GPU solve,
        #  and is kept in case a new SVD algorithm is found
        =#
        #=
        ndim = size(Z)[1] # number of shifted matrices
        Dmatrix = Vector{Matrix}(undef, length(Z))
        ThreadsX.foreach(Iterators.product(1:ndim, 1:ndim)) do (i,j)
            # Calculate the shifted matrix
            Lshift = L - Z[i,j] .* I
            # Calculate the adjoint and map the product to the 
            # diagonal entry of Dmatrix
            Lshift_adj = Ginv * adjoint(Lshift) * G
            Dmatrix[(i - 1) * ndim + j] = Lshift_adj * Lshift
            next!(p)
        end
        finish!(p)
        # Construct a BlockDiagonal representation from the vector of matrices
        foo = BlockDiagonal([Dmatrix[i] for i in 1:length(Dmatrix)])
        # Convert to sparse format and send to the GPU as Compressed Storage Column (CSC) data
        d_foo = CUSPARSE.CuSparseMatrixCSC(sparse(foo))
        # Perform the SVD decomposition on the GPU and return the singular values
        svds = CUSOLVER.svdvals(d_foo)
        # Reshape the minimum values of each block
        svds = reshape(minimum.(svds), (length(Dmatrix), length(Dmatrix)))
        return svds
        =#
        ##########################################################################
        ##########################################################################

        sig = similar(Z)
        # Include progress bar for long calculations
        p = Progress(length(Z), dt=0.1, desc="Computing pseudospectrum...", 
        barglyphs=BarGlyphs("[=> ]"), barlen=50)
        # Automatic load balancing, false sharing protection
        ThreadsX.foreach(Iterators.product(1:size(Z)[1], 1:size(Z)[2])) do (i,j)
            # Calculate the shifted matrix
            Lshift = L - Z[i,j] .* I
            # Calculate the adjoint
            Lshift_adj = Ginv * adjoint(Lshift) * G
            # Calculate the pseudospectrum
            sig[i,j] = minimum(GenericLinearAlgebra.svdvals(Lshift_adj * Lshift))
            next!(p)
        end
        return sig
        finish!(p)
    end 
end