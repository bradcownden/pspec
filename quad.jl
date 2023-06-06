#=
#  Perform Clenshaw-Curtis quadrature on a function: calculate the quadrature points
#  and weights, then return a matrix whose diagonal is the partial sum
=#
__precompile__()
module quad
export quadrature
    using LinearAlgebra
    using ThreadsX
    # Takes a function f which it itself a function of an input array and an index
    # Array input is for data type consistancy
    function quadrature(f, x::Vector)::Matrix
        # Quadrature points
        tquad = Vector{eltype(x)}(undef, length(x))
        ThreadsX.map!(i-> 0.5 * pi * (2*i - 1) / length(x), tquad,1:length(x))
        # Quadrature weights
        Nfl = Int(floor(length(x))/2)
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
end