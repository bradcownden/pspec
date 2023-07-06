__precompile__()

#=
#  Perform Clenshaw-Curtis quadrature on a function: calculate the quadrature points
#  and weights, then return a matrix whose diagonal is the partial sum
=#

module quad

export Gram
export quadrature_GC
export quadrature_GL

using LinearAlgebra
using ThreadsX
using BlockArrays
using LoopVectorization

    # Takes a function f which it itself a function of an input array and 
    # index; returns quadrature matrix for GC points
    function quadrature_GC(f, x::Vector)
        # Quadrature points
        tquad = Vector{eltype(x)}(undef, length(x))
        @tturbo for i in eachindex(x)
            tquad[i] = 0.5*pi*(2*i - 1)/length(x)
        end
        # Quadrature weights
        Nfl = trunc(Int, floor(length(x))/2)
        wts = Vector{eltype(x)}(undef, length(x))
        sums = Vector{eltype(x)}(undef, length(x))
        # Use the identity T_n(cos x) = cos(n x) to simplify the calculation of weights
        ThreadsX.foreach(eachindex(x)) do i
            insum = zero(eltype(x))
            @turbo for j in 1:Nfl
                insum += cos(2 * j * tquad[i])/(4*j^2 - 1)
            end
            sums[i] = insum
            wts[i] = 2 * (1 - 2 * sums[i]) / length(x)
        end
        # Partial sums are the weights times the function evaluated at the 
        # quadrature points
        foo = Vector{eltype(x)}(undef,length(x))
        ThreadsX.map!(i->wts[i]* f(cos.(tquad), i), foo, 1:length(foo))
        return reshape(Diagonal(foo), (length(x), length(x)))
    end

    # Takes a function f which it itself a function of an input array and 
    # index; returns quadrature matrix for GL points
    function quadrature_GL(f, x::Vector)
        # Quadrature points
        tquad = Vector{eltype(x)}(undef, length(x))
        @turbo for i in eachindex(x)
            tquad = cos(pi * i / length(x))
        end
        kappa = [i == 1 ? 2 : i == length(x) ? 2 : 1 for i in eachindex(x)]
        #ThreadsX.map!(i-> 0.5 * pi * (2*i - 1) / length(x), tquad,1:length(x))
        # Quadrature weights
        Nfl = trunc(Int, floor(length(x))/2)
        wts = Vector{eltype(x)}(undef, length(x))
        sums = Vector{eltype(x)}(undef, length(x))
        # Use the identity T_n(cos x) = cos(n x) to simplify the calculation of weights
        ThreadsX.foreach(eachindex(x)) do i
            sums[i] = ThreadsX.sum(2*j == length(x) ? cos(2 * j * tquad[i])/(4*j^2 - 1) : 2 * cos(2 * j * tquad[i])/(4*j^2 - 1) for j in 1:Nfl)
            wts[i] = 2 * (1 - 2 * sums[i]) / (kappa[i] * length(x))
        end
        # Partial sums are the weights times the function evaluated at the 
        # quadrature points
        foo = Vector{eltype(x)}(undef,length(x))
        ThreadsX.map!(i->wts[i]* f(cos.(tquad), i), foo, 1:length(foo))
        return reshape(Diagonal(foo), (length(x), length(x)))
    end

    # Takes a function f which it itself a function of an input array and 
    # index; returns quadrature matrix for LGR points
    function quadrature_LGR(f, x::Vector)
        # Quadrature points
        tquad = Vector{eltype(x)}(undef, length(x))
        ThreadsX.map!(i-> 0.5 * pi * (2*i - 1) / length(x), tquad,1:length(x))
        # Quadrature weights
        Nfl = trunc(Int, floor(length(x))/2)
        wts = Vector{eltype(x)}(undef, length(x))
        sums = Vector{eltype(x)}(undef, length(x))
        # Use the identity T_n(cos x) = cos(n x) to simplify the calculation of weights
        ThreadsX.foreach(1:length(x)) do i
            sums[i] = ThreadsX.sum(cos(2 * j * tquad[i])/(4*j^2 - 1) for j in 1:Nfl)
            wts[i] = 2 * (1 - 2 * sums[i]) / length(x)
        end
        # Partial sums are the weights times the function evaluated at the 
        # quadrature points
        foo = Vector{eltype(x)}(undef,length(x))
        ThreadsX.map!(i->wts[i]* f(cos.(tquad), i), foo, 1:length(foo))
        return reshape(Diagonal(foo), (length(x), length(x)))
    end

    # Takes a function f which it itself a function of an input array and 
    # index; returns quadrature matrix for RGR points
    function quadrature_RGR(f, x::Vector)
        # Quadrature points
        tquad = Vector{eltype(x)}(undef, length(x))
        ThreadsX.map!(i-> 0.5 * pi * (2*i - 1) / length(x), tquad,1:length(x))
        # Quadrature weights
        Nfl = trunc(Int, floor(length(x))/2)
        wts = Vector{eltype(x)}(undef, length(x))
        sums = Vector{eltype(x)}(undef, length(x))
        # Use the identity T_n(cos x) = cos(n x) to simplify the calculation of weights
        ThreadsX.foreach(1:length(x)) do i
            sums[i] = ThreadsX.sum(cos(2 * j * tquad[i])/(4*j^2 - 1) for j in 1:Nfl)
            wts[i] = 2 * (1 - 2 * sums[i]) / length(x)
        end
        # Partial sums are the weights times the function evaluated at the 
        # quadrature points
        foo = Vector{eltype(x)}(undef,length(x))
        ThreadsX.map!(i->wts[i]* f(cos.(tquad), i), foo, 1:length(foo))
        return reshape(Diagonal(foo), (length(x), length(x)))
    end

    # Use quadrature to construct Gram matrices an inverses
    # Functions w, p, q are from Sturm Louiville operators and are themselves functions
    # that take an array of points and an index. D is the derivative matrix from
    # Chebyshev discretization
    function Gram(w, p, q, D::Matrix, x::Vector, abscissa::String)
        # Construct via reshaping of diagonal blocks
        if abscissa == "GC"
            G1 = reduce(hcat, [Transpose(D) * (0.5 .* quadrature_GC(p, x)) * D + 0.5 .* quadrature_GC(q, x), zeros(eltype(x), (length(x), length(x)))])
            G2 = reduce(hcat, [zeros(eltype(x), (length(x), length(x))), 0.5 .* quadrature_GC(w, x)])
        elseif abscissa == "GL"
            G1 = reduce(hcat, [Transpose(D) * (0.5 .* quadrature_GL(p, x)) * D + 0.5 .* quadrature_GL(q, x), zeros(eltype(x), (length(x), length(x)))])
            G2 = reduce(hcat, [zeros(eltype(x), (length(x), length(x))), 0.5 .* quadrature_GL(w, x)])
        elseif abscissa == "LGR"
            G1 = reduce(hcat, [Transpose(D) * (0.5 .* quadrature_LGR(p, x)) * D + 0.5 .* quadrature_LRG(q, x), zeros(eltype(x), (length(x), length(x)))])
            G2 = reduce(hcat, [zeros(eltype(x), (length(x), length(x))), 0.5 .* quadrature_LGR(w, x)])
        elseif abscissa == "RGR"
            G1 = reduce(hcat, [Transpose(D) * (0.5 .* quadrature_RGR(p, x)) * D + 0.5 .* quadrature_RGR(q, x), zeros(eltype(x), (length(x), length(x)))])
            G2 = reduce(hcat, [zeros(eltype(x), (length(x), length(x))), 0.5 .* quadrature_RGR(w, x)])
        else
            println("\nERROR: Unrecognized abscissa in Gram(). Must be one of 'GC', 'GL', 'LGR', or 'RGR'")
            G1 = zeros((2*length(x), 2*length(x)))
            G2 = similar(G1)
        end
        G = vcat(G1, G2)
    return G, inv(G)
    end
end