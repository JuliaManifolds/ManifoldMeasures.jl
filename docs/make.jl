using ManifoldMeasures
using Documenter

DocMeta.setdocmeta!(ManifoldMeasures, :DocTestSetup, :(using ManifoldMeasures); recursive=true)

makedocs(;
    modules=[ManifoldMeasures],
    authors="Seth Axen <seth.axen@gmail.com> and contributors",
    repo="https://github.com/sethaxen/ManifoldMeasures.jl/blob/{commit}{path}#{line}",
    sitename="ManifoldMeasures.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://sethaxen.github.io/ManifoldMeasures.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/sethaxen/ManifoldMeasures.jl",
    devbranch="main",
)
