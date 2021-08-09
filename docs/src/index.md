```@meta
CurrentModule = ManifoldMeasures
```

# ManifoldMeasures

ManifoldMeasures extends Manifolds.jl to use MeasureTheory.jl to implement generic measures on manifolds as well as common specific measures, such as those that appear in directional statistics.

## Goals

The implementations are designed with probabilistic programming in mind but are also generally useful.
This includes the following goals:

  - Constructors should be lightweight, so that arguments are not checked, and unnecessary normalization constants are not computed.
  - Log-densities should be compatible with Julia's automatic differentiation (AD) frameworks.
    This implies avoiding problematic patterns like `try`..`catch` blocks or mutation.
    When necessary or significantly more efficient, we define custom ChainRules.jl-compatible AD rules.
  - Log-densities and constructors should _try_ to be friendly for symbolic computation, which implies wrapping any control flow in functions like `ifelse` that can be symbolically overloaded.
    Note that currently we don't do any symbolic overloading.
  - Implement the most general forms of measures.
    For example, if a complex or quaternionic generalization of a measure are known or are straightforward, then the generalization is implemented.
