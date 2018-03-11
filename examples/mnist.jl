
using MLDatasets
using ImageView
using CuArrays
# using VariationalAE


function show_pic(x)
    reshape(x, 28, 28)' |> imshow
end


function show_recon(m, x)
    x_ = reconstruct(m, x)
    show_pic(x)
    show_pic(x_)
end


function run()
    m = VAE{Float32}(784, 500, 500, 20, 500, 500, 784) |> to_cuda

    X, _ = MNIST.traindata()
    X = convert(Matrix{Float32}, reshape(X, 784, 60000))
    @time m = fit!(m, X; n_epochs=2, cuda=true)

    # check reconstructed image
    for i=1:2:10
        show_recon(m, X[:, i])
    end

    # check learned features
    for i=1:2:10
        generate(m, i) |> show_pic
    end
end




function main()
    ex = quote 
        (mu, log_sigma2) = encode(m, x)
        z = mu .+ sqrt.(exp.(log_sigma2)) .* eps
        x_rec = decode(m, z)
        rec_loss = -(sum(x .* log.(1.0e-10 + x_rec) + (1 - x) .* log.((1.0e-10 + 1) - x_rec), 1))
        KLD = -0.5 * sum(((1 + log_sigma2) .- mu .^ 2) - exp.(log_sigma2), 1)
        cost = mean(rec_loss .+ KLD)
    end
    m = VAE{Float32}(784, 500, 500, 20, 500, 500, 784) |> to_cuda
    x = cu(rand(Float32, 784, 100))
    inputs = [:m => m, :eps => cu(rand(Float32, 20, 100)), :x => x]
    g = ExGraph(ex; inputs...)

    encode(m, x)
end
