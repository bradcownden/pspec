using Base.Threads
using BenchmarkTools
using DelimitedFiles
using ThreadsX
using Base.MPFR
using GenericLinearAlgebra
using ProgressMeter
using Parameters
using Distributed
@everywhere using LinearAlgebra

@everywhere include("./gpusvd.jl") 
include("./quad.jl")
@everywhere import .gpusvd, .quad

# Debug
const debug = 1

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


function writeData(inpts::Inputs, data::Vector)::Nothing
    open("jEigenvals_N" * string(inpts.N) * "P" * string(P) * ".txt", "w") do io
        writedlm(io, length(data))
        # Caution: \t character automatically added to file between real and imaginary parts
        writedlm(io, hcat(real.(data), imag.(data)))
        println("Wrote data to ", split(io.name," ")[2][1:end-1])
    end
end

function writeData(inpts::Inputs, data::Matrix)::Nothing
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

function basis(N::Integer)::Vector{Float64} # Gauss-Chebyshev collocation points
    foo = Vector{Float64}(undef, N)
    ThreadsX.map!(i->cos(pi*(2*i-1)/(2*N)),foo,1:N)
    return foo
end

function derivative(i::Integer, j::Integer, x::Array)::Float64 # Calculate an element of the first derivative matrix
    if i != j
        return (-1.)^(i+j) * sqrt((1. - x[j] * x[j]) /
        (1. - x[i] * x[i])) / (x[i] - x[j])
    else
        return 0.5 * x[i] / (1. - x[i] * x[i])
    end
end

function dderivative(i::Integer, j::Integer, x::Array, D::Matrix)::Float64 # Calculate an element of the second derivative matrix
    if i == j
        return x[j] * x[j] / (1. - x[j] * x[j]) ^ 2 - (N * N - 1.) / (3. * (1. - x[j] * x[j]))
    else
        return D[i,j] * (x[i] / (1. - x[i] * x[i]) - 2. / (x[i] - x[j]))
    end
end

function make_D(x::Array, N::Integer)::Matrix{Float64} # Make the derivative matrix
    foo = Matrix{Float64}(undef, (N,N))
    # Automatic load balancing, false sharing protection
    ThreadsX.foreach(Iterators.product(1:N, 1:N)) do (i,j)
        foo[i,j] = derivative(i, j, x)
    end
    return foo
end

function make_DD(x::Array, N::Integer, D::Matrix)::Matrix{Float64} # Make the second derivative matrix
    foo = Matrix{Float64}(undef, (N,N))
    # Automatic load balancing, false sharing protection
    ThreadsX.foreach(Iterators.product(1:N, 1:N)) do (i,j)
        foo[i,j] = dderivative(i, j, x, D)
    end
    return foo
end

#########################################
#= BigFloat-compatible basis functions =#
#########################################

function BF_basis(N::Integer)::Vector{BigFloat} # BigFloat version: Gauss-Chebyshev collocation points
    foo = Vector{BigFloat}(undef, N)
    ThreadsX.map!(i->BigFloat(cos(pi*(2*i-1)/(2*N))),foo,1:N)
    return foo
end

function BF_derivative(i::Integer, j::Integer, x::Array)::BigFloat # BigFloat version: Calculate an element of the first derivative matrix
    if i != j
        return BigFloat((-1.)^(i+j) * sqrt((1. - x[j] * x[j]) /
        (1. - x[i] * x[i])) / (x[i] - x[j]))
    else
        return BigFloat(0.5 * x[i] / (1. - x[i] * x[i]))
    end
end

function BF_dderivative(i::Integer, j::Integer, x::Array, D::Matrix)::BigFloat # BigFloat version: Calculate an element of the second derivative matrix
    if i == j
        return BigFloat(x[j] * x[j] / (1. - x[j] * x[j]) ^ 2 - (N * N - 1.) / (3. * (1. - x[j] * x[j])))
    else
        return BigFloat(D[i,j] * (x[i] / (1. - x[i] * x[i]) - 2. / (x[i] - x[j])))
    end
end


function BF_make_DD(x::Array, N::Integer, D::Matrix)::Matrix{BigFloat} # BigFloat version: Make the second derivative matrix
    foo = Matrix{BigFloat}(undef, (N,N))
    # Automatic load balancing, false sharing protection
    ThreadsX.foreach(Iterators.product(1:N, 1:N)) do (i,j)
        foo[i,j] = BF_dderivative(i, j, x, D)
    end
    return foo
end

function BF_make_D(x::Array, N::Integer)::Matrix{BigFloat} # BigFloat version: Make the derivative matrix
    foo = Matrix{BigFloat}(undef, (N,N))
    # Automatic load balancing, false sharing protection
    ThreadsX.foreach(Iterators.product(1:N, 1:N)) do (i,j)
        foo[i,j] = BF_derivative(i, j, x)
    end
    return foo
end

###############
#= Operators =#
###############

function L1(x::Array, D::Matrix, DD::Matrix)::Matrix{Float64} # Make the L1 operator
    N = length(x)
    foo = Matrix{ComplexF64}(undef, (N,N))
    # Automatic load balancing, false sharing protection
    ThreadsX.foreach(1:N) do i
        foo[i,:] = pp(x,i) .* D[i,:] + p(x,i) .* DD[i,:] # Dot operator applies addition to every element
        foo[i,i] -= V(x,i)
    end
    return foo
end

function L2(x::Array, D::Matrix)::Matrix{Float64} # Make the L2 operator
    N = length(x)
    foo = Matrix{ComplexF64}(undef, (N,N))
    # Automatic load balancing, false sharing protection
    ThreadsX.foreach(1:N) do i
        foo[i,:] = (2 * gamma(x,i)) .* D[i,:] # Dot operator applies addition to every element
        foo[i,i] += gammap(x,i)
    end
    return foo
end

###################################
#= BigFloat-compatible Operators =#
###################################

function BF_L1(x::Array, D::Matrix, DD::Matrix)::Matrix{Complex{BigFloat}} # BigFloat version: Make the L1 operator
    N = length(x)
    foo = Matrix{Complex{BigFloat}}(undef, (N,N))
    # Automatic load balancing, false sharing protection
    ThreadsX.foreach(1:N) do i
        foo[i,:] = BigFloat(pp(x,i)) .* D[i,:] + BigFloat(p(x,i)) .* DD[i,:] # Dot operator applies addition to every element
        foo[i,i] -= BigFloat(V(x,i))
    end
    return foo
end

function BF_L2(x::Array, D::Matrix)::Matrix{Complex{BigFloat}} # BigFloat version: Make the L2 operator
    N = length(x)
    foo = Matrix{Complex{BigFloat}}(undef, (N,N))
    # Automatic load balancing, false sharing protection
    ThreadsX.foreach(1:N) do i
        foo[i,:] = BigFloat(2 * gamma(x,i)) .* D[i,:] # Dot operator applies addition to every element
        foo[i,i] += BigFloat(gammap(x,i))
    end
    return foo
end

###############################
#= Sturm-Louiville functions =#
###############################

function s(x::Array, i::Integer)
    return sqrt(1 - x[i]^2)
end

function w(x::Array, i::Integer)
    return x[i]
end

function p(x::Array, i::Integer)
    return (x[i]-1)^2 * (x[i] + 1) * (3 - x[i] + 2 * s(x,i))
end

function pp(x::Array, i::Integer)
    return (x[i] - 1) * (1 + 2 * x[i]) * (3 - x[i] + 2 * s(x,i))
end

function gamma(x::Array, i::Integer)
    return (1 + x[i] + s(x,i)) * (3 - x[i] + 2 * s(x,i)) /
    sqrt((-3 + x[i] - 2 * s(x,i))/(x[i]-1))
end

function gammap(x::Array, i::Integer)
    return 1 / sqrt((-3 + x[i] - 2 * s(x,i))/(x[i]-1))
end

function V(x::Array, i::Integer)
    return 3 * (1 - 4 * s(x,i) + x[i] * (34 - 15 * x[i] + 44 * s(x,i))) /
    (8 * (1 + s(x,i)) * (3 - x[i] + 2 * s(x,i)))
end

##############################
#= Psuedospectrum functions =#
##############################

function make_Z(xmin, xmax, ymin, ymax, Nsteps::Integer)
    xvals = Vector{eltype(xmin)}(undef, Nsteps+1)
    yvals = Vector{eltype(xmin)}(undef, Nsteps+1)
    dx = (xmax - xmin)/Nsteps
    dy = (ymax - ymin)/Nsteps
    # Construct vectors of displacements
    ThreadsX.map!(i->xmin + i*dx,xvals,0:Nsteps)
    ThreadsX.map!(i->ymin + i*dy,yvals,0:Nsteps)
    # Meshgrid matrix
    foo = Matrix{Complex}(undef, (Nsteps+1, Nsteps+1))
    # Automatic load balancing, false sharing protection
    ThreadsX.foreach(Iterators.product(1:size(foo)[1], 1:size(foo)[2])) do (i,j) # Index i is incremented first
        foo[i,j] = xvals[i] + yvals[j]*1im
    end
    return foo
end

function sigma(Z::Matrix, L::Matrix)
    # Distribute a shifted matrix and find the smallest singular values
    foo = similar(Z)
    # Automatic load balancing, false sharing protection
    ThreadsX.foreach(Iterators.product(1:size(Z)[1], 1:size(Z)[2])) do (i,j)
        # Shift along the diagonal by a value in Z, take the smallest singular value
        foo[i,j] = real(minimum(svd(L - Z[i,j] .* I).S)) # I is an automatically sized identity matrix
    end
    return foo
end

##################################################
#= BigFloat-compatible psuedospectrum functions =#
##################################################

function BF_make_Z(xmin, xmax, ymin, ymax, Nsteps::Integer)::Matrix{Complex{BigFloat}}
    xvals = Vector{BigFloat}(undef, Nsteps+1)
    yvals = Vector{BigFloat}(undef, Nsteps+1)
    dx = (xmax - xmin)/Nsteps
    dy = (ymax - ymin)/Nsteps
    # Construct vectors of displacements
    ThreadsX.map!(i->xmin + i*dx,xvals,0:Nsteps)
    ThreadsX.map!(i->ymin + i*dy,yvals,0:Nsteps)
    # Meshgrid matrix
    foo = Matrix{Complex{BigFloat}}(undef, (Nsteps+1, Nsteps+1))
    # Automatic load balancing, false sharing protection
    ThreadsX.foreach(Iterators.product(1:size(foo)[1], 1:size(foo)[2])) do (i,j) # Index i is incremented first
        foo[i,j] = xvals[i] + yvals[j]*1im
    end
    return foo
end

function BF_sigma(Z::Matrix, L::Matrix)::Matrix{BigFloat}
    # Distribute a shifted matrix and find the smallest singular values
    foo = similar(Z)
    # Include progress bar for long calculations
    p = Progress(length(Z), dt=0.1, desc="Computing pseudospectrum...", 
    barglyphs=BarGlyphs("[=> ]"), barlen=50)
    # Automatic load balancing, false sharing protection
    ThreadsX.foreach(Iterators.product(1:size(Z)[1], 1:size(Z)[2])) do (i,j)
        # Shift along the diagonal by a value in Z, take the smallest singular value
        foo[i,j] = real(minimum(GenericLinearAlgebra.svdvals!(L - Z[i,j] .* I))) # I is an automatically sized identity matrix
        next!(p)
    end
    return foo
    finish!(p)
end
    
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
        foo = pmap(i -> (L - Z[i] .* LinearAlgebra.I), eachindex(Z))
        # Apply svd to (Lshift)^\dagger Lshift
        bar = pmap(x -> (Ginv * adjoint(x) * G) * x, foo)
        sig = pmap(svdvals, bar)
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

# Split into BigFloat and regular cases
if P > 0
    x = BF_basis(N)
    D = BF_make_D(x,N)
    DD = BF_make_DD(x,N,D)
    L = BF_L1(x, D, DD)
    LL = BF_L2(x,D)
else
    x = basis(N)
    D = make_D(x,N)
    DD = make_DD(x,N,D)
    L = L1(x, D, DD)
    LL = L2(x,D)
end

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
#writeData(inputs, vals)


##################################
#= Calculate the Psuedospectrum =#
##################################

# Make the meshgrid
grid = inputs.xgrid
Z = make_Z(inputs.xmin,inputs.xmax,inputs.ymin,inputs.ymax,grid)
#Z = BF_make_Z(xmin,xmax,ymin,ymax,grid)

# Debug
if debug > 2
    print("Collocation points = "); show(x); println("")
    print("D = "); show(D); println("")
    print("DD = "); show(DD); println("")
    print("L1 = "); show(L); println("")
    print("L2 = "); show(LL); println("")
    print("L = "); show(BigL); println("")
    print("Z = "); show(Z); println("")
    function integrand(x::Vector, i::Int)
        return cos(2 .* x[i]) * sin(x[i])
    end
    print("Integral[Cos(2x)Sin(x),{x,-1,1}] = 0: ")
    show(sum(diag(quad.quadrature(integrand,x)))); println("")
end

# Construct the Gram matrices
G, Ginv = quad.Gram(w, p, V, D, x)

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
    # Multiproc methods
    addprocs(nthreads())
    println("Timing for Distributed pspec:")
    @btime pspec(G, Ginv, Z, BigL)
    rmprocs(workers())
end

# Calculate the sigma matrix
println("Computing the psuedospectrum...")
if N >= 40 && grid >= 10
    addprocs(nthreads())
    sig = pspec(G, Ginv, Z, L)
    rmprocs(workers())
else
    sig = gpusvd.pspec(G, Ginv, Z, BigL)
end

# Debug
if debug > 0
    ssig = serial_sigma(G, Ginv, Z, BigL)
    print("Parallel/Serial calculation match: "); println(isapprox(ssig, sig))
end

print("Done! Sigma = "); show(sig); println(size(sig))

# Write Psuedospectrum to file
#writeData(inputs, sig)



