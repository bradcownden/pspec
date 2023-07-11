__precompile__()

#=
# Take a perturbation array and calculate the perturbed 
# eigenvalues 
=#

module pert

export vpert

using LinearAlgebra
using GenericLinearAlgebra
using ThreadsX
using DelimitedFiles

    # Normalize, add the perturbation to L, then compute the
    # eigenvalues of the perturbed matrix. Write to file.
    function vpert(epsilon::AbstractFloat, dV::Vector, w, x::Vector, G::Matrix, Ginv::Matrix, L::Matrix)
        # Check vector lengths
        if length(dV) != length(x)
            println("\nERROR: Collocation vector and perturbation vector must be the same length. Got len(dV) = ", length(dV), " while len(x) = ", length(x))
        end
        # Check w function arguments
        try
            w(x,1)
        catch e
            println("\nERROR: Function w must be of the form w(x,i) and return a single value")
            return nothing
        end

        N = length(x)
        # Construct delta L
        deltaV = diagm([dV[i] / w(x,i) for i in eachindex(x)])
        dL_upper = zeros(eltype(x), (N, 2*N))
        dL_lower = reduce(hcat, [deltaV, zeros(eltype(x), (N,N))])
        deltaL = vcat(dL_upper, dL_lower)
        # Scale by the norm
        nvals = GenericLinearAlgebra.svdvals((Ginv * transpose(deltaL) * G) * deltaL)
        norm = minimum([v for v in nvals if v != 0])
        L += (-1im * epsilon) .* (deltaL ./ sqrt(norm))
        # Find the perturbed eigenvalues
        pvals = ThreadsX.sort!(GenericLinearAlgebra.eigvals(L), alg=ThreadsX.StableQuickSort, by = abs)
        print("Done! Perturbed eigenvalues = "); show(pvals); println("")
        # Write to file
        write(pvals)
    end

    # Write out the perturbed eigenvalues
    function write(data::Vector)
        # Check for data directory; create if abscent
        isdir("./data") ? nothing : mkdir("./data")
        # Construct file name
        this_size = Integer((length(data)-2)/2)
        fname = "./data/pEigenvals_N" * string(this_size) * "P" 
        if eltype(data) == BigFloat || eltype(data) == Complex{BigFloat}
            fname *= string(precision(BigFloat)) * ".txt"
        else
            fname *= "64.txt"
        end
        open(fname, "w") do io
            writedlm(io, length(data))
            # Caution: \t character automatically added to file between real and imaginary parts
            writedlm(io, hcat(real.(data), imag.(data)))
            println("Wrote data to ", split(io.name," ")[2][1:end-1])
        end
    end

end