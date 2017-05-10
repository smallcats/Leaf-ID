#=
General functions for neural networks.
=#

""" Tested
Forward propagate m samples with activation function act, weights w and features X.

fproplayer(X,w,act) --> out

act is a function Float64 -> Float64 that acts coordinate-wise on arrays, size(w) = (n+1,p) (bias weights in the first row), size(X) = (m,n).
m is the number of samples, n is the size of the input layer, p is the size of the output layer.
"""
function fproplayer(X,w,act)
  Xbias = hcat(ones(size(X,1)),X)
  act(Xbias*w)
end

""" Tested
The sigmoid function, acts coordinate-wise on arrays.

sigmoid(z) --> out
"""
function sigmoid(z)
  1./(1 + exp(-z))
end

""" Tested
Creates a matrix of dimension (d1, d1-length(arr)+1) which is 0 except on the \"thick diagonal\" where it has entries given by arr.

For example:
thickdiag([1,2,3],5) -> [1 0 0;2 1 0;3 2 1;0 3 2;0 0 3]
"""
function thickdiag(arr,d1)
  alen = length(arr)
  d2 = 1+d1-alen
  td = zeros(d1, d2)
  for i = 1:d2
    td[i:i+alen-1,i] = arr
  end
  td
end

""" Tested
Convolutional forward propagation with activation function act, convolution kernel k and features X.

convfproplayer(X,k,act) --> out

act is a function Float64 -> Float64 that acts coordinate-wise on arrays, k the matrix giving the kernel of convolution and X is the feature matrix (so, 2D data).
"""
function convfproplayer(X,k,act)
  csize = size(k,1)
  nextlayer = zeros(size(X,1)-csize+1, size(X,2)-csize+1)
  for i = 1:csize
    nextlayer += X[i:end-csize+i,:]*thickdiag(k[i,:], size(X,2))
  end
  act(nextlayer)
end

""" Tested
Performs a pooling layer.

poolfproplayer(X, poolsize, [option]) --> out

Use option \"-a\" to get average pooling rather than max pooling
"""
function poolfproplayer(X, poolsize, maxav="-m")
  p = poolsize-1
  newsize = (size(X,1)-p, size(X,2)-p)
  Y = zeros(newsize)
  for k = 1:newsize[1]
    for j = 1:newsize[2]
      if maxav == "-a"
        Y[k,j]=sum(X[k:p+k, j:p+j])/(poolsize^2)
      else
        Y[k,j]=maximum(X[k:p+k, j:p+j])
      end
    end
  end
  Y
end

"""
Backpropagates from the next layer to the current layer.

bproplayer(der, bvnext, w) --> bv

Given derivatives of the current layer's output after activation wrt its values before activation (der),
  backpropagated values for the next layer (bvnext),
  and outgoing weights without bias weights (w)
  returns the backpropagated values for the current layer mulitiplied by der.

size(der) = (m,n), size(bvnext) = (m,l), size(w) = (n,l), size of the output is (m,n),
  where l is the size of the next layer, n is the size of the current layer and m is the number of samples.
"""
function bproplayer(der, bvnext, w)
  (bvnext*w').*der
end

""" 
Gets partials wrt weights from backpropagated values in a fully connected layer.

getpartial(y, bvnext) --> dE/dw

Takes the outputs to the current layer (y),
  and the backpropagated values to the next layer (bvnext),
  and returns the partials of the weights.

size(y) = (m,n), size(bvnext) = (m,p), size of the output is (m,n+1,p)
  where n is the size fo the current layer, m is the number of samples and p is the size of the next layer.
"""
function getpartial(y, bvnext)
  m = size(y,1)
  y = hcat(ones(m,1), y)
  out = Array{Float64,3}(m, size(y,2), size(p,2)) 
  for k = 1:m
    out[k,:,:] = y[[k],:]'*bvnext[[k],:]
  end
  out
end

"""
Performs a softmax layer.

softmax(X,w) --> out

X is the output of the previous layer, w is the weights to the current layer.

size(X) = (m,n), size(w) = (n+1,p)
  where m is the number of samples, n is the size of the previous layer, and p is the size of the current layer.
"""
function softmax(X,w)
  z = hcat(ones(size(X,1),1),X)*w
  yunnorm = exp(z)
  yunnorm./sum(yunnorm)
end

"""
Gives the average cross-entropy cost of a hypothesis over m samples compared to a target value.

crossentropy(h,t) --> out

h is the hypothesis with size(h) = (m,n).
t is the target with size(t) = (m,n).

The output cross-entropy is size (m,1).
"""
function crossentropy(h,t)
  sum(t.*log(h))/m
end

"""
Take a big Nesterov step. w is the weights, 
  velocity the accumulated gradient, 
  alpha the step size (i.e. friction)

nesterovbig(w, velocity, alpha = 0.9) --> (newweights, newvelocity)
"""
function nesterovbig(w, velocity, alpha = 0.9)
  newv = alpha*velocity
  (w + newv, newv)
end

"""
Take a small Nesterov step. w is weights, 
  velocity the accumulated gradient, 
  gradE the gradient of error with respect to weights (at the current position, after a big Nesterov step), 
  epsilon the learning rate (i.e. accumulation rate).

nesterovsmall(w,velocity,gradE,epsilon=0.1) --> (newweights, newvelocity)
"""
function nesterovsmall(w, velocity, gradE, epsilon = 0.1)
  delta = epsilon*gradE
  (w - delta, velocity - delta)
end
