function theta = train1vAll(init_theta, lambda, X, y, num_classes)
  #Trains p logistic regression models 1 vs all, for training data X and labels 
  #y (integers 1 to p) with regularization lambda and initial model parameters 
  #init_theta. Regularization and initial parameters are kept fixed for training
  #each model. 
 
  #Inputs:
  #  init_theta : matrix size [n+1,1]
  #  lambda : real number
  #  X : matrix size [m, n+1]
  #  y : matrix size [m,1]

  #Outputs:
  #  theta : matrix size [n+1,p]
  theta = zeros(length(init_theta),num_classes);
  Y = destructure(y,num_classes);
  for model_num = 1:num_classes
    theta(:,model_num) = lrTrain(X, Y(:,model_num), lambda, init_theta);
    fprintf('\rTrained %d models',model_num);
    fflush(1);
  endfor
  fprintf('\n');
end