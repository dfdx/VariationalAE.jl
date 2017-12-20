
using MLDatasets
using ImageView
using VariationalAE


function show_pic(x)
    reshape(x, 28, 28)' |> imshow
end


function show_recon(m, x)
    x_ = reconstruct(m, x)
    show_pic(x)
    show_pic(x_)
end


function run()
    m = VAE{Float64}(784, 500, 500, 20, 500, 500, 784)
    X, _ = MNIST.traindata()
    X = reshape(X, 784, 60000)
    @time m = fit!(m, X; n_epochs=2)

    # check reconstructed image
    show_recon(m, X[:, 100])

    # check learned features
    for i=1:2:10
        generate(m, i) |> show_pic
    end
end
