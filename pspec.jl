using Base.Threads
using BenchmarkTools
using DelimitedFiles
using ThreadsX
using Base.MPFR
using ProgressMeter
using Parameters
using Distributed
@everywhere using LinearAlgebra
@everywhere using GenericLinearAlgebra

include("./gpusvd.jl") 
include("./quad.jl")
import .gpusvd, .quad

#####################
#= Debug Verbosity =#
#####################

# Debug 0: no debugging information
# Debug 1: function timings and matrix inversion check
# Debug 2: outputs from 1 plus matrix outputs and quadrature check
const debug = 0

#######################################################
#= Psuedospectrum calculation leveraging parallelism =#
#######################################################

####################
#= I/O and Inputs =#
####################

@with_kw mutable struct Inputs
    N::Int64 = 4
    xmin::Float64 = -1
    xmax::Float64 = 1
    ymin::Float64 = -1
    ymax::Float64 = 1
    xgrid::Int64 = 2
    ygrid::Int64 = 2
end

function readInputs(f::String)::Inputs
    # Open the specified file and read the inputs into a dictionary
    data = Dict{SubString, Any}()
    if isfile(f)
        open(f) do file
            for line in readlines(file)
                data[split(chomp(line),"=")[1]] = split(chomp(line), "=")[2]
            end
        end
    else
        println(""); println("ERROR: couldn't find input file ", f)
    end
    # Create struct input values
    inpts = Inputs()
    for k in collect(keys(data)) 
        if k == "spectral_N" 
            inpts.N = parse(Int64, get(data, k,nothing))
            if inpts.N < 4
                println("WARNING: number of spectral modes must be N > 3. " *
                "Defaulting to N = 4.")
                inpts.N = 4
            end
        elseif k == "p_gridx" 
            inpts.xgrid = parse(Int64, get(data, k,nothing))
            if inpts.xgrid < 1
                inpts.xgrid = 1
            end
        elseif k == "p_gridy"
            inpts.ygrid = parse(Int64, get(data, k, nothing))
            if inpts.ygrid < 1
                inpts.ygrid = 1
            end
        elseif k == "xgrid_min"
            inpts.xmin = parse(Float64, get(data, k, nothing))
        elseif k == "xgrid_max"
            inpts.xmax = parse(Float64, get(data, k, nothing))
        elseif k == "ygrid_min"
            inpts.ymin = parse(Float64, get(data, k, nothing))
        elseif k == "ygrid_max"
            inpts.ymax = parse(Float64, get(data, k, nothing))           
        else
            println(""); println("\nERROR: unexpected entry in input file: ", k)
        end
    end
    return inpts
end


function writeData(inpts::Inputs, data::Vector)
    open("jEigenvals_N" * string(inpts.N) * "P" * string(P) * ".txt", "w") do io
        writedlm(io, length(data))
        # Caution: \t character automatically added to file between real and imaginary parts
        writedlm(io, hcat(real.(data), imag.(data)))
        println("Wrote data to ", split(io.name," ")[2][1:end-1])
    end
end

function writeData(inpts::Inputs, data::Matrix)
    open("jpspec_N" * string(inpts.N) * "P" * string(P) * ".txt", "w") do io
        writedlm(io, adjoint([inpts.xmin, inpts.xmax, inpts.ymin, inpts.ymax, inpts.xgrid]))
        writedlm(io, hcat(size(data)))
        writedlm(io, data)
        println("Wrote data to ", split(io.name," ")[2][1:end-1])
    end
end

#####################
#= Basis functions =#
#####################

function basis(N::Integer, P::Integer)# Gauss-Chebyshev collocation points
    # Set the precision of all subsequent operations based on the
    # specified digits of precision
    if P > 64
        setprecision(P)
        foo = Vector{BigFloat}(undef, N)
        ThreadsX.map!(i->cos(pi*(2*i-1)/(2*N)),foo,1:N)
        return foo
    else
        foo = Vector{Float64}(undef, N)
        ThreadsX.map!(i->cos(pi*(2*i-1)/(2*N)),foo,1:N)
        return foo
    end
end

function make_D(x::Array, N::Integer)# Make the derivative matrix
    foo = Matrix{eltype(x)}(undef, (N,N))
    # Automatic load balancing, false sharing protection
    ThreadsX.foreach(Iterators.product(1:N, 1:N)) do (i,j)
        if i != j
            foo[i,j] = (-1.)^(i+j) * sqrt((1. - x[j] * x[j]) /
            (1. - x[i] * x[i])) / (x[i] - x[j])
        else
            foo[i,j] = 0.5 * x[i] / (1. - x[i] * x[i])
        end
    end
    return foo
end

function make_DD(x::Array, N::Integer, D::Matrix) # Make the second derivative matrix
    foo = Matrix{eltype(x)}(undef, (N,N))
    # Automatic load balancing, false sharing protection
    ThreadsX.foreach(Iterators.product(1:N, 1:N)) do (i,j)
        if i == j
            foo[i,j] = x[j] * x[j] / (1. - x[j] * x[j]) ^ 2 - (N * N - 1.) / (3. * (1. - x[j] * x[j]))
        else
            foo[i,j] = D[i,j] * (x[i] / (1. - x[i] * x[i]) - 2. / (x[i] - x[j]))
        end
    end
    return foo
end

###############
#= Operators =#
###############

function L1(x::Array, D::Matrix, DD::Matrix) # Make the L1 operator
    N = length(x)
    foo = Matrix{eltype(x)}(undef, (N,N))
    # Automatic load balancing, false sharing protection
    ThreadsX.foreach(1:N) do i
        foo[i,:] = pp(x,i) .* D[i,:] + p(x,i) .* DD[i,:] # Dot operator applies addition to every element
        foo[i,i] -= V(x,i)
    end
    return foo
end

function L2(x::Array, D::Matrix) # Make the L2 operator
    N = length(x)
    foo = Matrix{eltype(x)}(undef, (N,N))
    # Automatic load balancing, false sharing protection
    ThreadsX.foreach(1:N) do i
        foo[i,:] = (2 * gamma(x,i)) .* D[i,:] # Dot operator applies addition to every element
        foo[i,i] += gammap(x,i)
    end
    return foo
end

###############################
#= Sturm-Louiville functions =#
###############################

function s(x::Array, i::Integer)
    return sqrt(1 - x[i]^2) # Automatic datatype matching
end

function w(x::Array, i::Integer) # Automatic datatype matching
    return x[i]
end

function p(x::Array, i::Integer) # Automatic datatype matching
    return (x[i]-1)^2 * (x[i] + 1) * (3 - x[i] + 2 * s(x,i))
end

function pp(x::Array, i::Integer) # Automatic datatype matching
    return (x[i] - 1) * (1 + 2 * x[i]) * (3 - x[i] + 2 * s(x,i))
end

function gamma(x::Array, i::Integer) # Automatic datatype matching
    return (1 + x[i] + s(x,i)) * (3 - x[i] + 2 * s(x,i)) /
    sqrt((-3 + x[i] - 2 * s(x,i))/(x[i]-1))
end

function gammap(x::Array, i::Integer) # Automatic datatype matching
    return 1 / sqrt((-3 + x[i] - 2 * s(x,i))/(x[i]-1))
end

function V(x::Array, i::Integer) # Automatic datatype matching
    return 3 * (1 - 4 * s(x,i) + x[i] * (34 - 15 * x[i] + 44 * s(x,i))) /
    (8 * (1 + s(x,i)) * (3 - x[i] + 2 * s(x,i)))
end

##############################
#= Psuedospectrum functions =#
##############################

function make_Z(inputs::Inputs, x::Vector)
    Nsteps = inputs.xgrid
    xvals = Vector{eltype(x)}(undef, Nsteps+1)
    yvals = Vector{eltype(x)}(undef, Nsteps+1)
    dx = (inputs.xmax - inputs.xmin)/Nsteps
    dy = (inputs.ymax - inputs.ymin)/Nsteps
    # Construct vectors of displacements
    ThreadsX.map!(i->inputs.xmin + i*dx,xvals,0:Nsteps)
    ThreadsX.map!(i->inputs.ymin + i*dy,yvals,0:Nsteps)
    # Meshgrid matrix
    foo = Matrix{Complex{eltype(x)}}(undef, (Nsteps+1, Nsteps+1))
    # Automatic load balancing, false sharing protection
    ThreadsX.foreach(Iterators.product(1:size(foo)[1], 1:size(foo)[2])) do (i,j) # Index i is incremented first
        foo[i,j] = xvals[i] + yvals[j]*1im
    end
    return foo
end

##################################################
#= Serial psuedospectrum for timing =#
##################################################

function serial_sigma(G::Matrix, Ginv::Matrix, Z::Matrix, L::Matrix)
    foo = similar(Z)
     # Include progress bar for long calculations
     p = Progress(length(Z), dt=0.1, desc="Computing pseudospectrum...", 
     barglyphs=BarGlyphs("[=> ]"), barlen=50)
     for i in 1:size(Z)[1]
        for j in 1:size(Z)[2]
            # Calculate the shifted matrix
            Lshift = L - Z[i,j] .* I
            # Calculate the adjoint
            Lshift_adj = Ginv * adjoint(Lshift) * G
            # Calculate the pseudospectrum
            foo[i,j] = minimum(GenericLinearAlgebra.svdvals(Lshift_adj * Lshift))
            next!(p)
        end
    end
    return foo
    finish!(p)
end

################################
#= Distributed pseudospectrum =#
################################

# Requires workers to have already been spawned
@everywhere function pspec(G::Matrix, Ginv::Matrix, Z::Matrix, L::Matrix)
    if nprocs() > 1
        # Calculate all the shifted matrices
        ndim = size(Z)[1]
        println("Constructing shifted matrices...")
        foo = pmap(i -> (L - Z[i] .* LinearAlgebra.I), eachindex(Z))
        # Apply svd to (Lshift)^\dagger Lshift
        println("Constructing adjoint products...")
        bar = pmap(x -> (Ginv * adjoint(x) * G) * x, foo)
        println("Calculating SVDs...")
        sig = pmap(GenericLinearAlgebra.svdvals!, bar)
        # Reshape and return sigma
        return reshape(minimum.(sig), (ndim, ndim))
    else
        println("No workers have been spawned");
        return 1
    end 
end

##########
#= Main =#
##########

if length(ARGS) != 1
    println("Usage:")
    println("julia -t M pspec.jl P")
    println("M (int): the number of tasks to launch in parallel regions")
    println("P (int): digits of precision for calculations - default is 32")
    println("NOTE: Requires the file 'Inputs.txt' to be in the current directory")
    println("")
    exit()
else
    P = parse(Int64, ARGS[1])
    if P < 32
        P = -1
    end
end

if nthreads() > 1
    println("Number of threads: ", nthreads())
end
if P > 0
    setprecision(P)
    println("Precision set to ", Base.precision(BigFloat), " bits")
end

# Read the inputs from a file
inputs = readInputs("./Inputs.txt")
N = inputs.N

# Construct basis functions
x = basis(N, P)
D = make_D(x, N)
DD = make_DD(x, N, D)

# Construct L1 and L2 operators
L = L1(x, D, DD)
LL = L2(x, D)

# Stack the matrices
Lupper = reduce(hcat, [zeros(eltype(x), (N,N)), Matrix{Complex{eltype(x)}}(I,N,N)]) # Match the data type of the collocation array
Llower = reduce(hcat, [L, LL])
BigL = vcat(Lupper, Llower)
BigL = BigL .* -1im # Automatic type matching

# Find the eigenvalues
println("Computing eigenvalues...")
vals = ThreadsX.sort!(GenericLinearAlgebra.eigvals(BigL), alg=ThreadsX.StableQuickSort, by = abs)
print("Done! Eigenvalues = "); show(vals); println("")

# Write eigenvalues to file
writeData(inputs, vals)


##################################
#= Calculate the Psuedospectrum =#
##################################

# Make the meshgrid
Z = make_Z(inputs, x)
# Construct the Gram matrices
G, Ginv = quad.Gram(w, p, V, D, x)

# Debug
if debug > 1
    print("Collocation points = ", size(x), " "); show(x); println("")
    print("D = ", size(D), " "); show(D); println("")
    print("DD = ", size(DD), " "); show(DD); println("")
    print("L1 = ", size(L), " "); show(L); println("")
    print("L2 = ", size(LL), " "); show(LL); println("")
    print("L = ", size(BigL), " "); show(BigL); println("")
    print("Z = ", size(Z), " "); show(Z); println("")
    function integrand(x::Vector, i::Int)
        return cos(2 .* x[i]) * sin(x[i])
    end
    print("Integral[Cos(2x)Sin(x),{x,-1,1}] = 0: ")
    show(sum(diag(quad.quadrature(integrand,x)))); println("")
end

# Debug/timing
if debug > 0
    print("Ginv * G = I: "); println(isapprox(Ginv * G, I))
    # Serial Timing
    println("Timing for serial sigma:")
    @btime serial_sigma(G, Ginv, Z, BigL)
    # Large, block matrix
    println("Timing for gpusvd.sigma:")
    @btime gpusvd.sigma(G, Ginv, Z, BigL)
    # Threaded over individual shifted matrices
    println("Timing for gpusvd.pspec:")
    @btime gpusvd.pspec(G, Ginv, Z, BigL)
    # Multiproc method
    addprocs(nthreads())
    @everywhere using GenericLinearAlgebra
    println("Timing for Distributed pspec:")
    @btime pspec(G, Ginv, Z, BigL)
    rmprocs(workers())
end

# Calculate the sigma matrix. Rough benchmarking favours multiprocessor
# methods if N > 50 and grid > 10
println("Computing the psuedospectrum...")
if N >= 50 && inputs.xgrid >= 10
    addprocs(nthreads())
    @everywhere using GenericLinearAlgebra # Send to workers after spawn
    sig = pspec(G, Ginv, Z, BigL)
    rmprocs(workers())
else
    sig = gpusvd.pspec(G, Ginv, Z, BigL)
end

# Debug
if debug > 1
    ssig = serial_sigma(G, Ginv, Z, BigL)
    print("Parallel/Serial calculation match: "); println(isapprox(ssig, sig))
end

print("Done! Sigma = "); show(sig); println("")

# Write Psuedospectrum to file
writeData(inputs, sig)



