
# main file of Variational Autoencoder
# based on Auto-Encoding Variational Bayes by D. Kingma and M. Welling ([K&W])
# https://arxiv.org/abs/1312.6114
# see also http://vdumoulin.github.io/morphing_faces/

using Base.LinAlg.BLAS
using Distributions
import StatsBase.fit
import StatsBase.coef
import StatsBase: sample, sample!

typealias Mat{T} AbstractArray{T, 2}
typealias Vec{T} AbstractArray{T, 1}

# variational autoencoder with Gaussian observed and latent variables
type VAE{T,XD}
    # encoding
    # W[1] - 1st layer, W[2] - 2nd layer (mu), W[3] 3rd layer (sigma2)
    W::NTuple{Matrix{T},3} 
    b::NTuple{Vector{T},3}
end


function encode{T}(vae::VAE{T,Normal}, x::Vector{T})
    # [K&W] Eq. 12
    h = tanh(vae.W[1] * x .+ vae.b[1])
    mu = vae.W[2] * h .+ vae.b[2]
    sigma2 = vae.W[3] * h .+ vae.b[3]
end
