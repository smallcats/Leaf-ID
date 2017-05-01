function H = lrPrediction(theta,X)
  #For parameters theta, and matrix X whose rows are examples, returns the 
  #prediction of a logistic regression model.
  
  #size(theta) = [n+1,p], size(X) = [m,n], size(H) = [m,p]
  #m = number of samples, n = number of features, p = number of outputs 
  
  #useful variable
  m = size(X,1);
  
  #find prediction
  H = sigmoid([ones(m,1),X]*theta);
end