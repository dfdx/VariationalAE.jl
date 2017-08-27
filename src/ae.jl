
using Distributions
using XDiff
using GradDescent
using MLDatasets
using MLDataUtils


logistic(x) = 1 ./ (1 + exp.(-x))
@scalardiff logistic(x::Number) 1 (logistic(x) .* (1 .- logistic(x)))


function autoencoder(We1, We2, Wd, b1, b2, x)
    firstLayer = logistic(We1 * x .+ b1)
    encodedInput = logistic(We2 * firstLayer .+ b2)
    reconstructedInput = logistic(Wd * encodedInput)
    return reconstructedInput
end


function autoencoder_cost(We1, We2, Wd, b1, b2, x)
    firstLayer = logistic(We1 * x .+ b1)
    encodedInput = logistic(We2 * firstLayer .+ b2)
    reconstructedInput = logistic(Wd * encodedInput)
    cost = sum((reconstructedInput .- x) .^ 2.0)
    return cost
end


function xavier_init(dim_in, dim_out; c=1)
    low = -c * sqrt(6.0 / (dim_in + dim_out))
    high = c * sqrt(6.0 / (dim_in + dim_out))
    return rand(Uniform(low, high), dim_in, dim_out)
end


function show_component(Wd, i)
    z = zeros(250)
    z[2] =  1.0
    xr = logistic(Wd *  z)
    imshow(reshape(xr, 28, 28))
end


function main()
    batch_size = 100
    n_epochs = 10

    We1 = xavier_init(500, 784); We2 = xavier_init(250, 500); Wd = xavier_init(784, 250)
    b1 = zeros(500); b2 = zeros(250)
    theta = (We1, We2, Wd, b1, b2)
    X, _ = MNIST.traindata()
    X = reshape(X, 784, 60000)
    x = X[:, 1:batch_size]
    g = xdiff(autoencoder_cost; We1=We1, We2=We2, Wd=Wd, b1=b1, b2=b2, x=x)
    # fit data
    # opt = Adam(α=1.0)
    # mem = Dict()
    # theta = model_params(m)
    opt = Adam(α=0.001)
    optimizers = [deepcopy(opt) for i=1:length(theta)]
    for epoch in 1:n_epochs
        println("Epoch: $epoch")
        for (i, x) in enumerate(eachbatch(X, size=batch_size))
            # partial_fit(m, x, g; mem=mem)
            # eps = typeof(x)(randn(size(m.We3, 1), batch_size))  # eps has size (n_inp, n_z)
            dvals = g(theta..., x)
            cost = dvals[1]
            println("cost = $cost")
            deltas = dvals[2:end-2]
            for j=1:length(deltas)
                delta = update(optimizers[j], deltas[j])
                # delta = 0.0001 * deltas[j]
                theta[j] .-= delta
            end
        end
    end


    x = X[:, 1020]
    xr = autoencoder(We1, We2, Wd, b1, b2, x)
    imshow(reshape(x, 28, 28))
    imshow(reshape(xr, 28, 28))


    
end
