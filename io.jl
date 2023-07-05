__precompile__()

#= 
#  Module for writing output files for pspec
=#

module io

using DelimitedFiles

export writeData

    # Write vector data to file
    function writeData(data::Vector)
        # Check for data directory; create if abscent
        isdir("./data") ? nothing : mkdir("./data")
        # Construct file name
        fname = "./data/jEigenvals_N" * string(length(data)) * "P"
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

    function writeData(data::Matrix, x::Vector)
        # Check for data directory; create if abscent
        isdir("./data") ? nothing : mkdir("./data")
        # Construct file name
        fname = "./data/jpspec_N" * string(length(x)) * "P"
        if eltype(x) == BigFloat || eltype(x) == Complex{BigFloat}
            fname *= string(precision(BigFloat)) * ".txt"
        else
            fname *= "64.txt"
        end
        open(fname, "w") do io
            writedlm(io, adjoint([inpts.xmin, inpts.xmax, inpts.ymin, inpts.ymax, inpts.xgrid]))
            writedlm(io, hcat(size(data)))
            writedlm(io, data)
            println("Wrote data to ", split(io.name," ")[2][1:end-1])
        end
    end

end
