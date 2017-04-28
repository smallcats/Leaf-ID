function [J, J_grad] = lrCost(theta,lambda,X,y)
  #X is a matrix where each column is a feature of the dataset, and each row a
  #sample. y is a vector with labels for the data in X, again samples are in 
  #rows. Theta gives the parameters of the logistic regression model as a column
  #vector. Then J is returned as the cost of the parameters theta for input X 
  #and output y. J_grad is returned as the gradient wrt the parameters theta.
  
  #size(X) = [m,n], size(theta) = [n+1,1], size(y) = [m,1], size(J_grad) = [n+1,1]
  #m = number of samples, n = number of features
  
  #useful variables
  m = length(y);
  theta_lin = theta(2:end);
  X_aff = [ones(m,1),X];
  
  #run logistic reg with parameters theta on X
  h = lrPrediction(theta, X);
  
  #find cost and gradient
  J = (-(y'*log(h) + (1-y)'*log(1-h)) + lambda*(theta_lin'*theta_lin)/2)/m;
  J_grad = (X_aff'*(h-y) + lambda*[0;theta_lin])/m;
end