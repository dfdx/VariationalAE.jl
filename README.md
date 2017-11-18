# VariationalAE

[![Build Status](https://travis-ci.org/dfdx/VariationalAE.jl.svg?branch=master)](https://travis-ci.org/dfdx/VariationalAE.jl)

Implementation of Variational Autoencoder using [XGrad.jl](https://github.com/dfdx/XGrad.jl).

## Installation

```
Pkg.add("Distributions")
Pkg.add("GradDescent")
Pkg.add("MLDataUtils")
Pkg.add("StatsBase")
Pkg.add("Espresso"); Pkg.checkout("Espresso")
Pkg.clone("https://github.com/dfdx/XGrad.jl")
Pkg.clone("https://github.com/dfdx/VariationalAE.jl")
```

## Examples

See `examples/` directory.