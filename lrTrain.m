function theta = lrTrain(X,y,lambda,theta_init)
  #Trains logistic regression by minimizing lrCost on training set X with labels
  #Y with regularization parameter lambda and starting with theta_init as 
  #initial model parameters. The first entry of theta(_init) is the bias term.
  #Uses fminunc to minimize.
  
  #size(X) = [m,n], size(Y) = [m,1], size(theta_init) = [n+1,1]
  
  options = optimset('GradObj', 'on', 'MaxIter', 50);
  theta = fminunc(@(t)lrCost(t,lambda,X,y), theta_init, options);
end