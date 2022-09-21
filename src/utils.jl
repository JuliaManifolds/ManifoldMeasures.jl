function show_manifold_measure(io, mime, μ)
    name = nameof(typeof(μ))
    fields = (getfield(μ, n) for n in fieldnames(typeof(μ)) if n !== :par)
    field_string = sprint(show, mime, Tuple(fields))[2:(end - 1)]
    if endswith(field_string, ",")
        field_string = field_string[1:(end - 1)]
    end
    print(io, name, "(", field_string)
    par = MeasureBase.params(μ)
    if !isempty(par)
        par_string = sprint(show, mime, par)[2:(end - 1)]
        if endswith(par_string, ",")
            par_string = par_string[1:(end - 1)]
        end
        print(io, "; ", par_string)
    end
    return print(io, ")")
end
