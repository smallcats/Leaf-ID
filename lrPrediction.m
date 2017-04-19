function h = lrPrediction(theta,X)
  #For parameters theta, and matrix X whose rows are examples, returns the 
  #prediction of a logistic regression model.
  
  #size(theta) = [n+1,1], size(X) = [m,n], size(h) = [m,1]
  
  #useful variable
  m = size(X,1);
  
  #find prediction
  h = sigmoid([ones(m,1),X]*theta);