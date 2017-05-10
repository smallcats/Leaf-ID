#=
Functions for a small cnn with one convolutional layer and one fully connected layer with one output neuron.
The intention is to train it to recognize whether an object is in a photo, and then look at the output of the convolutional layer to find the relevant pixels for that classification.
=#

include("nn.jl")

"""
Forward propagation.

fprop(X, k, w, b) --> (Y,h)

X is a sample (size (a,b)), 
k is the convolutional kernel weights (size (n,n)),
w is the weights of the fully connected layer (size (a-n+1, b-n+1)),
b is the bias of the fully connected layer.

Y is a length 2 array with the first entry is X and the second the output of the convolutional layer.
h is the output (i.e. the hypothesis).
"""
function fprop(X, k, w, b)
  Y = Array{Array{Float64,2},1}(2)
  Y[1] = X
  Y[2] = convfproplayer(Y[1],k,sigmoid)
  h = sigmoid(sum(Y[2] .* w) + b)
  (Y,h)
end

"""
Backpropagation.

bprop(Y,h,w,t) --> (gradE, db)

Y is the output of a forward propagation,
w is the weights of the fully connected layer,
t is the target.

gradE is an array of length 2 containing the gradient of each layer wrt the weights.
  gradE[1] is the partials wrt the weights of the kernel,
  gradE[2] is the partials wrt the weights of the fully connected layer,
db is the partial with respect to the bias.
"""
function bprop(Y,h,w,t)
  D2 = h - t
  D1 = D2*(w.*(Y[2].*(1-Y[2])))
  gradE = Array{Array{Float64,2},1}(2)
  db = D2
  gradE[2] = D2 * Y[2]
  convsize = size(k,1)
  gradE[1] = Array{Float64,2}((convsize,convsize))
  for r = 1:convsize
    for s = 1:convsize
      gradE[1][r,s] = sum(D1 .* Y[1][r:end-convsize+r, s:end-convsize+s])
    end
  end
  (gradE,db)
end

"""
Trains the nn for one step.

train1(X,k,w,b,t,velocity=0,alpha=0.9,epsilon=0.1) --> (knew,wnew,bnew,velocity)
"""
function train1(X,k,w,b,t,velocity=0,alpha=0.9,epsilon=0.1)
  allw = vcat(k[:],w[:],b)
  allw,velocity = nesterovbig(allw, velocity,alpha)
  k = reshape(allw[1:length(k)], size(k))
  w = reshape(allw[length(k)+1:length(k)+length(w)], size(w))
  b = allw[end]
  Y,h = fprop(X,k,w,b)
  gradE,db = bprop(Y,h,w,t)
  allgrad = vcat(gradE[1][:],gradE[2][:],db)
  allw,velocity = nesterovsmall(allw,velocity,allgrad,epsilon)
  knew = reshape(allw[1:length(k)], size(k))
  wnew = reshape(allw[length(k)+1:length(k)+length(w)], size(w))
  bnew = allw[end]
  (knew,wnew,bnew,velocity)
end

"""

"""
function train(X,kinit,winit,binit,t,alpha=0.9,epsilon=0.1)

end
