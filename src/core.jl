
# main file of Variational Autoencoder
# based on Auto-Encoding Variational Bayes by D. Kingma and M. Welling
# https://arxiv.org/abs/1312.6114
# see also http://vdumoulin.github.io/morphing_faces/

using Base.LinAlg.BLAS
using Distributions
import StatsBase.fit
import StatsBase.coef
import StatsBase: sample, sample!

typealias Mat{T} AbstractArray{T, 2}
typealias Vec{T} AbstractArray{T, 1}
typealias Gaussian Normal


type VAE{T, XD}
    pz::MvNormal         # prior distribution p(z)
    px::XD               # prior distribution p(x)
    ePhi::Matrix{T}      # parameters of q(z|x) (encoding)
    dPhi::Matrix{T}      # parameters of q(x|z) (decoding)
    
end
