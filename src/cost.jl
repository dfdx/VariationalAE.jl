
function vae_cost(We1, be1, We2, be2, We3, be3, We4, be4, Wd1, bd1, Wd2, bd2, Wd3, bd3, eps, x)
    dummy = 42.0
    # encoder
    he1 = softplus.(We1 * x .+ be1)
    he2 = softplus.(We2 * he1 .+ be2)
    mu = We3 * he2 .+ be3
    log_sigma2 = We4 * he2 .+ be4
    z = mu .+ sqrt.(exp.(log_sigma2)) .* eps
    # decoder
    hd1 = softplus.(Wd1 * z .+ bd1)
    hd2 = softplus.(Wd2 * hd1 .+ bd2)
    x_rec = logistic.(Wd3 * hd2 .+ bd3)
    # loss
    rec_loss = sum(x .* log.(1e-10 + x_rec) + (1 - x) .* log.(1e-10 + 1 - x_rec), 1)
    # rec_loss = rec_loss_ ./ length(rec_loss_)
    KLD = -0.5 * sum(1 + log_sigma2 .- mu .^ 2 - exp.(log_sigma2), 1)
    ELBO = mean(rec_loss .+ KLD)
end
