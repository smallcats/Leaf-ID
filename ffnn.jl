#=
Implements a feed-forward neural network. 
  Uses functions from nn.jl. 
  The weights should be stored as an array of matrices, one for each layer. 
  The matrix for a layer should be of size (c,n) 
    where c is the size of the current layer and n is the size of the next layer.
=#

include("nn.jl")

""" Tested
Forward propagates m samples through a feed-forward neural network, with a linear final layer.

fprop(X,W,act,keep) --> Y

Inputs:
  W is the weights, 
  X is a (m,feat) matrix where feat is the number of features,
  act is the activation function.

Option: keep 
  \"-y\" outputs of each layer will be kept,
  \"-n\" outputs of each layer will not be kept.

Outputs:
  If \"-y\" is active then the output will be a 1-d array of length equal to the number of non-input layers containing 2-d arrays of size (m,n_l) where n_l is the size of layer l, each representing the outputs of a layer for all samples.
  If \"-n\" is active then the output will be a 2-d array of the outputs of the output layer.
"""
function fprop(X, W, act, keep = "-n")
  layers = length(W)
  if keep == "-y"
    Y = Array{Array{Float64,2},1}(layers)
    Y[1] = fproplayer(X, W[1], act)
    for l = 2:layers-1
      Y[l] = fproplayer(Y[l-1], W[l], act)
    end
    Y[layers] = fproplayer(Y[layers-1], W[layers], x->x)
    return Y
  else
    for l = 1:layers-1
      X = fproplayer(X,W[l],act)
    end
    return fproplayer(X, W[layers],x->x)
  end
end

"""
Backpropagates one run of a feed-forward neural network. Currently assumes the error function is square-error, and all layers use a sigmoid activation function.

  bprop(Y, t, W) --> gradE

Inputs:
  Y is an array of length equal to the number of non-input layers containing 2d arrays with one row of the outputs of the network.
  t is a (1,n) array of the target output where n is the number of output features.
  W is the weights given as an array of 2d arrays.

Outputs:
  gradE is an array of the partials of error with respect to weights
"""
function bprop(Y, t, W)
  layers = length(W)
  bv = Array{Array{Float64,2},1}(layers)
  der = [y .* (1-y) for y in Y]
  bv[end] = Y[end] - t
  for l = 1:layers-1
    bv[end-l] = bproplayer(der[end-l], bv[end-l+1], W[end-l+1])
  end
  [getpartial(Y[k], bv[k]) for k = 1:layers]
end
