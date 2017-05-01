function result = testModel(Xtest,ytest,theta)
  #Returns the accuracy of the logistic regression model with parameters given
  #by theta on the data Xtest with labels ytest.
  
  #Inputs:
  #  Xtest - size [m,n] matrix
  #  ytest - size [m,1] matrix
  #  theta - size [n+1,1] matrix
  #m = number of samples, n = number of features (pixels in preprocessed images)

  #Outputs:
  #  result - double
  
  prediction = lrPrediction(theta,Xtest);
  [certainty, class_pred] = max(prediction,[],2);
  class_pred -= 1;
  correct = class_pred == ytest;
  result = sum(correct)/length(correct);