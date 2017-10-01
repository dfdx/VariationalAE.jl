
import GradDescent: Optimizer

struct ModelOptimizer{M, O <: Optimizer}
    model::Type{M}
    optimizers::Dict{Symbol, O}
end

function ModelOptimizer(::Type{M}, opt::Optimizer) where M
    optimizers = Dict{Symbol, Optimizer}()
    for fld in fieldnames(M)
        optimizers[fld] = deepcopy(opt)
    end
    return ModelOptimizer(M, optimizers)
end

Base.show(io::IO, opt::ModelOptimizer) = print(io, "ModelOptimizer($(opt.model))")


ModelOptimizer(m, opt::Optimizer) = ModelOptimizer(typeof(m), opt)


function update_params!(m_opt::ModelOptimizer, m, grad)
    @assert(typeof(m) == typeof(grad), "Wrong gradient type: model is $(typeof(m)), " *
            "but gradient is $(typeof(grad))")
    for fld in fieldnames(m)
        # TODO: this will break on scalar parameters
        theta = getfield(m, fld)
        theta .-= update(m_opt.optimizers[fld], getfield(grad, fld))
        setfield!(m, fld, theta)
    end
end

