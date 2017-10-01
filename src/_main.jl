
include("core.jl")




function _main()
    We1 = randn(500, 784); be1 = randn(500);
    We2 = randn(500, 500); be2 = randn(500);
    We3 = randn(20, 500); be3 = randn(20);
    We4 = randn(20, 500); be4 = randn(20);
    Wd1 = randn(500, 20); bd1 = randn(500);
    Wd2 = randn(500, 500); bd2 = randn(500);
    Wd3 = randn(784, 500); bd3 = randn(784);
    x = randn(784, 100); eps = rand(Normal(0, 1),  20)

    inputs = [:We1 => We1, :be1 => be1, :We2 => We2, :be2 => be2, :We3 => We3, :be3 => be3,
              :We4 => We4, :be4 => be4,
              :Wd1 => Wd1, :bd1 => bd1, :Wd2 => Wd2, :bd2 => bd2, :Wd3 => Wd3, :bd3 => bd3,
              :eps => eps, :x => x]
    vals = [inp[2] for inp in inputs]
    
    vae_cost(vals...)
    dvae_cost = xdiff(vae_cost; inputs...)


    ex = quote
        # encoder
        he1 = tanh.(We1 * x .+ be1)
        he2 = tanh.(We2 * he1 .+ be2)
        mu = We3 * he2 .+ be3
        log_sigma2 = We4 * he2 .+ be4
        z = mu .+ sqrt.(exp.(log_sigma2)) .* eps
        # decoder
        hd1 = tanh.(Wd1 * z .+ bd1)
        hd2 = tanh.(Wd2 * hd1 .+ bd2)
        x_rec = logistic.(Wd3 * hd2 .+ bd3)
        # loss
        rec_loss = sum(x .* log.(1e-10 + x_rec) + (1 - x) .* log.(1e-10 + 1 - x_rec))
        latent_loss = -0.5 * sum(1 + log_sigma2 .- mu .^ 2 - exp.(log_sigma2))
        cost = sum(rec_loss + latent_loss)
    end
    dex = xdiff(ex; inputs...)
end
