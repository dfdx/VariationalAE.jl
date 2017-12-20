
# main file of Variational Autoencoder
# based on Auto-Encoding Variational Bayes by D. Kingma and M. Welling ([K&W])
# https://arxiv.org/abs/1312.6114
# see also http://vdumoulin.github.io/morphing_faces/
# see also https://jmetzen.github.io/2015-11-27/vae.html

using Distributions
using GradDescent
using MLDataUtils
using StatsBase
using XGrad


logistic(x) = 1 ./ (1 + exp.(-x))
@diffrule logistic(x::Number) x (logistic(x) .* (1 .- logistic(x)) .* ds)

softplus(x) = log(exp(x) + 1)
@diffrule softplus(x::Number) x logistic(x) .* ds


include("model.jl")
include("modelopt.jl")


model_params(m) = [getfield(m, f) for f in fieldnames(m)]
model_named_params(m) = [f => getfield(m, f) for f in fieldnames(m)]


function xavier_init(dim_in, dim_out; c=1)
    low = -c * sqrt(6.0 / (dim_in + dim_out))
    high = c * sqrt(6.0 / (dim_in + dim_out))
    return rand(Uniform(low, high), dim_in, dim_out)
end


function StatsBase.fit!(m::VAE{T}, X::AbstractMatrix{Float64};
              n_epochs=10, batch_size=100, opt=Adam(α=0.001)) where T
    mem = Dict()
    m_opt = ModelOptimizer(typeof(m), opt)
    for epoch in 1:n_epochs
        print("Epoch $epoch: ")
        epoch_cost = 0
        t = @elapsed for (i, x) in enumerate(eachbatch(X, size=batch_size))
            eps = typeof(x)(rand(Normal(0, 1), size(m.We3, 1), batch_size))
            cost, dm, deps, dx = xgrad(vae_cost, mem=mem, m=m, eps=eps, x=x)
            update_params!(m_opt, m, dm)
            epoch_cost += cost
        end
        println("avg_cost=$(epoch_cost / (size(X,2) / batch_size)), elapsed=$t")
    end
    return m
end


function generate(m::VAE, z::AbstractVector)
    return decode(m, z)
end


function generate(m::VAE, id::Int)
    z = zeros(size(m.Wd1, 2))
    z[id] = 1.0
    hd1 = tanh.(m.Wd1 * z .+ m.bd1)
    hd2 = tanh.(m.Wd2 * hd1 .+ m.bd2)
    return logistic.(m.Wd3 * hd2 .+ m.bd3)
end


function reconstruct(m::VAE, x::AbstractVector)
    x = reshape(x, length(x), 1)
    mu, _ = encode(m, x)
    z = mu
    x_rec = decode(m, z)
    return x_rec
end

