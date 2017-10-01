
# main file of Variational Autoencoder
# based on Auto-Encoding Variational Bayes by D. Kingma and M. Welling ([K&W])
# https://arxiv.org/abs/1312.6114
# see also http://vdumoulin.github.io/morphing_faces/
# see also https://jmetzen.github.io/2015-11-27/vae.html

using Distributions
using XDiff
using GradDescent
using MLDatasets
using MLDataUtils
using ImageView


logistic(x) = 1 ./ (1 + exp.(-x))
@scalardiff logistic(x::Number) 1 (logistic(x) .* (1 .- logistic(x)))

softplus(x) = log(exp(x) + 1)
@scalardiff softplus(x::Number) 1 logistic(x)


# variational autoencoder with Gaussian observed and latent variables
mutable struct VAE{T}
    # encoder / recognizer
    We1::AbstractMatrix{T}  # encoder: layer 1 weight
    be1::AbstractVector{T}  # encoder: layer 1 bias
    We2::AbstractMatrix{T}  # encoder: layer 2 weight
    be2::AbstractVector{T}  # encoder: layer 2 bias
    We3::AbstractMatrix{T}  # encoder: layer 3 mu weight
    be3::AbstractVector{T}  # encoder: layer 3 mu bias
    We4::AbstractMatrix{T}  # encoder: layer 3 log(sigma^2) weight
    be4::AbstractVector{T}  # encoder: layer 3 log(sigma^2) bias
    # decoder / generator
    Wd1::AbstractMatrix{T}  # decoder: layer 1 weight
    bd1::AbstractVector{T}  # decoder: layer 1 bias
    Wd2::AbstractMatrix{T}  # decoder: layer 2 weight
    bd2::AbstractVector{T}  # decoder: layer 2 bias
    Wd3::AbstractMatrix{T}  # decoder: layer 3 weight
    bd3::AbstractVector{T}  # decoder: layer 3 bias
end

function Base.show(io::IO, m::VAE{T}) where T
    print(io, "VAE{$T}($(size(m.We1,2)), $(size(m.We1,1)), $(size(m.We2,1)), " *
          "$(size(m.We3,1)), $(size(m.Wd1,1)), $(size(m.Wd2,1)), $(size(m.Wd3,1)))")
end


VAE{T}(n_inp, n_he1, n_he2, n_z, n_hd1, n_hd2, n_out) where T =
    VAE{T}(
        # encoder
        xavier_init(n_he1, n_inp),
        zeros(n_he1),
        xavier_init(n_he2, n_he1),
        zeros(n_he2),
        xavier_init(n_z, n_he2),
        zeros(n_z),
        xavier_init(n_z, n_he2),
        zeros(n_z),
        # decoder
        xavier_init(n_hd1, n_z),
        zeros(n_hd1),
        xavier_init(n_hd2, n_hd1),
        zeros(n_hd2),
        xavier_init(n_out, n_hd2),
        zeros(n_out)
    )


include("cost.jl")



model_params(m) = [getfield(m, f) for f in fieldnames(m)]
model_named_params(m) = [f => getfield(m, f) for f in fieldnames(m)]



function xavier_init(dim_in, dim_out; c=1)
    low = -c * sqrt(6.0 / (dim_in + dim_out))
    high = c * sqrt(6.0 / (dim_in + dim_out))
    return rand(Uniform(low, high), dim_in, dim_out)
end


function fit(m::VAE{T}, X::AbstractMatrix{Float64};
             n_epochs=10, batch_size=100, opt=Adam(α=0.001)) where T
    # compile gradient
    x1 = X[:, 1:batch_size]
    eps = typeof(x1)(rand(Normal(0, 1), size(m.We3, 1), batch_size))  # eps has size (n_inp, n_z)
    println("Compiling gradient...")
    g = xdiff(vae_cost; model_named_params(m)..., eps=eps, x=x1)
    # fit to data
    opt = Adam(α=1.0)
    mem = Dict()
    theta = model_params(m)
    optimizers = [deepcopy(opt) for i=1:length(theta)]
    for epoch in 1:n_epochs
        println("Epoch: $epoch")
        for (i, x) in enumerate(eachbatch(X, size=batch_size))
            # partial_fit(m, x, g; mem=mem)
            # eps = typeof(x)(randn(size(m.We3, 1), batch_size))  # eps has size (n_inp, n_z)
            eps = zeros(size(m.We3, 1), batch_size)
            dvals = Base.invokelatest(g, theta..., eps, x)
            cost = dvals[1]            
            println("cost = $cost")
            deltas = dvals[2:end-2]
            for j=1:length(deltas)
                # delta = update(optimizers[j], deltas[j])
                delta = 0.001 * deltas[j]
                theta[j] .-= delta
            end
        end
    end
end


function generate(m::VAE, z::AbstractVector)
    hd1 = tanh.(m.Wd1 * z .+ m.bd1)
    hd2 = tanh.(m.Wd2 * hd1 .+ m.bd2)
    return logistic.(m.Wd3 * hd2 .+ m.bd3)
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
    he1 = tanh.(m.We1 * x .+ m.be1)
    he2 = tanh.(m.We2 * he1 .+ m.be2)
    mu = m.We3 * he2 .+ m.be3
    log_sigma2 = m.We4 * he2 .+ m.be4
    z = mu .+ sqrt.(exp.(log_sigma2))
    hd1 = tanh.(m.Wd1 * z .+ m.bd1)
    hd2 = tanh.(m.Wd2 * hd1 .+ m.bd2)
    x_rec = logistic.(m.Wd3 * hd2 .+ m.bd3)
    return x_rec
end


function showit(x)
    reshape(x, 28, 28) |> imshow
end


function run()
    m = VAE{Float64}(784, 500, 500, 20, 500, 500, 784)
    X, _ = MNIST.traindata()
    X = reshape(X, 784, 60000)
    @time fit(m, X; n_epochs=10)

    reconstruct(m, X[:, 100]) |> showit
    for i=1:5
        generate(m, i) |> showit
    end

    m2 = deepcopy(m)
    
end

