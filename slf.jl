__precompile__()

#=
#  Module for specifying functions present in the Sturm-Louiville form of
#  the L operators. Simplification of the ratios of f(x)/w(x) is assumed to
#  have been already performed (functions will not be scaled by a density)
=#

module slf

export w, p, pp, gamma, gammap, V, scale

    function s(x::Array, i::Integer) # Automatic datatype matching
        return sqrt(1 - x[i]^2) 
    end

    function z(x::Array, i::Integer) # Automatic datatype matching
        return exp(atanh(x[i]))
    end

    function w(x::Array, i::Integer) # Automatic datatype matching
        return 1 / (s(x,i) * (3 - x[i] + 2 * s(x,i)))
    end

    function p(x::Array, i::Integer) # Automatic datatype matching
        return -(x[i]^2 - 1) / z(x,i)
    end
    
    function pp(x::Array, i::Integer) # Automatic datatype matching
        return (x[i] - 1) * (1 + 2 * x[i]) / s(x,i)
    end
    
    function gamma(x::Array, i::Integer) # Automatic datatype matching
        return (1 + z(x,i)) / sqrt(1 + (1 + z(x,i))^2)
    end
    
    function gammap(x::Array, i::Integer) # Automatic datatype matching
        return - z(x,i) / ((x[i]^2 - 1) * (2 + z(x,i) * (2 + z(x,i)))^(3/2))
    end
    
    function V(x::Array, i::Integer) # Automatic datatype matching
        return -3 * s(x,i) * (1 - 4 * s(x,i) + x[i] * (34 - 15 * x[i] + 44 * s(x,i))) /
        (4 * (x[i] - 1) * (1 + x[i] + s(x,i))^2 * (-3 + x[i] - 2 * s(x,i))^2)
    end

end

