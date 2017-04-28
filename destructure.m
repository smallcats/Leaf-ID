function Y = destructure(y, num_classes)
  #Inputs:
  # y : A vector of integers 0 to num_classes.
  # num_classes : An integer.
  
  #Output:
  # Y : A matrix with size [length(y),num_classes] in which each row has one 
  #     entry of 1 and the rest 0. The entry 1 in the ith row is in column j+1,
  #     where the ith entry of y is j.
  
  m = length(y);
  Y = zeros(m,num_classes);
  for i = 1:m
    Y(i,y(i)+1) = 1;
  endfor
end