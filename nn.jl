"""
Forward propagate m samples with activation function act, weights w and features X.
act is a function Float64 -> Float64 that acts coordinate-wise on arrays, size(w) = (n+1,p) (bias weights in the first row), size(X) = (m,n).
m is the number of samples, n is the size of the input layer, p is the size of the output layer.
"""
function fproplayer(X,w,act)
  Xbias = hcat(ones(size(X,1)),X)
  act(Xbias*w)
end

"""
The sigmoid function, acts coordinate-wise on arrays.
"""
function sigmoid(z)
  1./(1 + exp(-z))
end

"""
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

"""
Convolutional forward propagation with activation function act, convolution kernel k and features X.
at is a function Float64 -> Float64 that acts coordinate-wise on arrays, k the matrix giving the kernel of convolution and X is the feature matrix (so, 2D data).
"""
function convfproplayer(X,k,act)
  csize = size(k,1)
  nextlayer = zeros(size(X,1)-csize+1, size(X,2)-csize+1)
  for i = 1:csize
    nextlayer += X[i:end-csize+i,:]*thickdiag(k[i,:], size(X,1))
  end
  act(nextlayer)
end

"""
Given derivatives of the next layer's output after activation wrt its values before activation (der),
  backpropagated values for the next layer (bv),
  and outgoing weights without bias weights (w)
  returns the backpropagated values for the current layer.

size(dernext) = (1,l), size(bvnext) = (1,l), size(w) = (n,l) 
  where l is the size of the next layer and n is the size of the current layer.
"""
function bproplayer(dernext, bvnext, w)
  (dernext.*bvnext)*w'
end
