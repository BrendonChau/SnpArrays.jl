using Documenter, SnpArrays

makedocs(
    format = Documenter.HTML(canonical = "https://OpenMendel.github.io/SnpArrays.jl/stable"),
    sitename = "SnpArrays.jl",
    authors = "Hua Zhou",
    checkdocs = :exports,
    warnonly = [:missing_docs, :cross_references],
    pages = [
        "SnpArrays.jl Tutorial" => "index.md",
        "Linear Algebra Benchmarks" => "linalg.md",
        "API Reference" => "api.md"
    ]
)

deploydocs(
    repo = "github.com/OpenMendel/SnpArrays.jl.git"
)
