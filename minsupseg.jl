#=
Functions for a small cnn with one convolutional layer and one fully connected layer with one output neuron.
The intention is to train it to recognize whether an object is in a photo, and then look at the output of the convolutional layer to find the relevant pixels for that classification.
The cost function used is cross-entropy.
=#

include("nn.jl")

"""
Forward propagation.

fprop(X, k, w, b) --> (Y,h)

X is a sample (size (3,a,b)), 
k is the convolutional kernel weights (size (3,n,n)),
w is the weights of the fully connected layer (size (a-n+1, b-n+1)),
b is the bias of the fully connected layer.

Y is the output of the convolutional layer.
h is the output (i.e. the hypothesis).
"""
function fprop(X, k, w, b)
  Y = sigmoid(+((convfproplayer(X[i,:,:],k[i,:,:],x->x) for i = 1:3)...))
  h = sigmoid(sum(Y .* w) + b)
  (Y,h)
end

"""
Backpropagation.

bprop(X,Y,h,w,k,t) --> (gradEk,gradEw,gradEb)

X is the input sample,
Y is the output of the hidden layer from forward propagation,
w is the weights of the fully connected layer,
t is the target.

gradEk is the partials wrt the weights of the kernel,
gradEw is the partials wrt the weights of the fully connected layer,
gradEb is the partial with respect to the bias.
"""
function bprop(X,Y,h,w,k,t)
  D2 = h - t
  D1 = D2*(w.*(Y.*(1-Y)))
  gradEb = D2
  gradEw = D2 * Y
  convsize = size(k,1)
  gradEk = Array{Float64,3}((3,convsize,convsize))
  for l = 1:3
    for r = 1:convsize
      for s = 1:convsize
        gradEk[l,r,s] = sum(D1 .* X[l,r:end-convsize+r, s:end-convsize+s])
      end
    end
  end
  (gradEk,gradEw,gradEb)
end

"""
Numerically estimates the gradient of cost at the input X with target t. Used to check bprop.

numericGrad(X,w,k,b,t) --> (gradEk,gradEw,gradEb)
"""
function numericGrad(X,w,k,b,t, epsilon)
  gradEw = Array{Float64,2}(size(w))
  gradEk = Array{Float64,3}(size(k))
  for l = 1:3
    for m = 1:size(k,1)
      for n = 1:size(k,2)
        pos = zeros(size(k))
        pos[l,m,n] += epsilon
        kpos = k + pos
        kneg = k - pos
        ig,outpos = fprop(X,kpos,w,b)
        ig,outneg = fprop(X,kneg,w,b)
        costpos = -(t*log(outpos)+(1-t)*log(1-outpos))
        costneg = -(t*log(outneg)+(1-t)*log(1-outneg))
        gradEk[l,m,n] = (costpos - costneg)/(2*epsilon)
      end
    end
  end
  for l = 1:size(w,1)
    for m = 1:size(w,2)
      pos = zeros(size(w))
      pos[l,m] += epsilon
      wpos = w + pos
      wneg = w - pos
      ig,outpos = fprop(X,k,wpos,b)
      ig,outneg = fprop(X,k,wneg,b)
      costpos = -(t*log(outpos)+(1-t)*log(1-outpos))
      costneg = -(t*log(outneg)+(1-t)*log(1-outneg))
      gradEw[l,m] = (costpos - costneg)/(2*epsilon)
    end
  end
  bpos = b + epsilon
  bneg = b - epsilon
  ig,outpos = fprop(X,k,w,bpos)
  ig,outneg = fprop(X,k,w,bneg)
  costpos = -(t*log(outpos)+(1-t)*log(1-outpos))
  costneg = -(t*log(outneg)+(1-t)*log(1-outneg))
  gradEb = (costpos - costneg)/(2*epsilon)
  (gradEk, gradEw, gradEb)
end

"""
Trains the nn for one step.

train1(X,k,w,b,t,velocity=0,alpha=0.9,epsilon=0.1) --> (knew,wnew,bnew,velocity)
"""
function train1(X,k,w,b,t,velocity=zeros(length(k)+length(w)+1),alpha=0.9,epsilon=0.1)
  allw = vcat(k[:],w[:],b)
  allw,velocity = nesterovbig(allw, velocity,alpha)
  k = reshape(allw[1:length(k)], size(k))
  w = reshape(allw[length(k)+1:length(k)+length(w)], size(w))
  b = allw[end]
  Y,h = fprop(X,k,w,b)
  gradEk,gradEw,gradEb = bprop(X,Y,h,w,k,t)
  allgrad = vcat(gradEk[:],gradEw[:],gradEb)
  allw,velocity = nesterovsmall(allw,velocity,allgrad,epsilon)
  knew = reshape(allw[1:length(k)], size(k))
  wnew = reshape(allw[length(k)+1:length(k)+length(w)], size(w))
  bnew = allw[end]
  (knew,wnew,bnew,velocity)
end

"""
Trains the nn.

train(Xsamp,tsamp, maxiter, kinit,winit,binit,alpha=0.9,epsilon=0.1) --> (k,w,b)

Xsamp is a 4d array of samples. X[i,:,:,:] is the ith sample.
tsamp gives the classification target (0 or 1). tsamp[i] is the target of the ith sample.
kinit, winit and binit default to small normally distributed random values.
maxiter is the maximum number of iterations (default: size(X,1))
"""
function train(X,t,maxiter=size(X,1),kinit=0.01*randn(3,3,3),winit=0.01*randn(size(X,3)-2,size(X,4)-2),binit=0.01*randn(),alpha=0.9,epsilon=0.1)
  k,w,b,v = train1(X[1,:,:,:], kinit,winit,binit,t[1], zeros(length(k)+length(w)+1), alpha,epsilon)
  for i=2:maxiter
    k,w,b,v = train1(X[i,:,:,:], k,w,b,t[i],v,alpha,epsilon)
  end
  (k,w,b)
end

#Tests:
#=
X = Array{Float64,3}(3, 5, 7)
X[1,:,:]=[0 1 1 1 2 1 1;0 1 1 2 2 2 1;1 1 2 2 2 1 1;1 1 2 2 1 1 0; 1 1 1 1 0 0 0]/10
X[2,:,:]=[-1 -3 2 1 0 -1 3;0 0 0 0 -4 0 0;1 0 1 0 -1 0 -1;1 1 2 1 1 1 -5; -3 1 5 0 0 2 0]/10
X[3,:,:]=[0 -1 -1 -1 -2 -1 -1;4 -1 -1 -2 0 -2 -1;-1 3 -2 -2 -2 -1 -1;-1 -1 -2 -2 -1 -1 0; -1 -1 -1 -1 0 0 0]/10

k = Array{Float64,3}(3,3,3)

k[1,:,:] = [1 1 1;1 1 1;1 1 1]/10
k[2,:,:] = [-1 -1 -1;-1 -1 0;0 1 0]/10
k[3,:,:] = [-1 -1 -1;-1 -1 -1;-1 -1 -1]/10

w = [1 1 1 1 1;1 1 1 1 1;1 1 1 1 1]/10

b = -3/10

t = 1

------------------------

X = randn(3,5,7)
k = randn(3,3,3)
w = randn(3,5)
b = randn()
t = 0
=#
