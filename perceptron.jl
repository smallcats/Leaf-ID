#In comments m = number of samples, n = number of features, p = number of perceptron models

"""
Evaluates size(Ws,2) perceptrons on X with each row a sample, and each column a feature.
Ws are the weights (each perceptron's weight in a column) with the first weight giving the bias.
size(X) = (m,n), size(Ws) = (n+1,p)

Output has size (m,p)
"""
function pcpteval(X,Ws)
  Xbias = hcat(ones(size(X,1),1),X)
  Xbias*Ws
end

"""
Trains length(y) perceptrons for one sample x with labels y.

size(x) = (1,n), size(y) = (1,p), size(winit) = (n+1,p)
Output is a tuple of weights of size (n+1,p) and correct classifications
"""
function pcpttrainstep(x,y,winit=zeros(size(x,2)+1,length(y)))
  classification = pcpteval(x,winit) .> 0
  correct = classification .== y
  (winit + transpose(hcat(1,x))*(~correct .* (-2*classification + 1)), correct)
end

"""
&& applied iteratively over an array.
"""
function arrayand(arr)
  #and over a whole array
  out = true
  for k in arr
    out = k && out
  end
  out
end

"""
Trains size(Y,2) perceptrons on size(X,1) samples in parallel.

size(X) = (m,n), size(Y) = (m,p), size(winit) = (n+1,p)
Output is weights of size (n+1,p)
"""
function pcpttrain(X,Y,winit=zeros(size(X,2)+1, size(Y,2)),maxstep=10000)
  (m,p) = size(Y)
  w = winit
  #println("w: $(size(w))")
  #println("X: $(size(X))")
  #println("Y: $(size(Y))")
  steps = 0
  correct = fill(false, (m,p))
  sep = false
  while !sep
    sampnum = steps%m+1
    print("\rstep: $steps")
    w,correct[sampnum,:]  = pcpttrainstep(X[[sampnum],:],Y[[sampnum],:],w)
    if steps > maxstep break end
    sep = arrayand(correct)
    steps += 1
  end
  (w,correct)
end
