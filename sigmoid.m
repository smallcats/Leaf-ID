function s = sigmoid(z)
  #Sigmoid function. Acts componentwise for vectors and matrices.
  s = 1./(1+exp(-z));