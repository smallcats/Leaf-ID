#Script for loading and putting data in a usable format. Divides data into three
#matrices, X, for training, X_cv for cross-validation, and X_test for testing,
#along with y, y_cv and y_test.
#Each sample inhabits a row of X, unrolled.

clear all;

pkg load image

#Example usage:

%dirpath = 'leafsnap-dataset/dataset/images/field';
%num_sp = 2;
%newsize = [30,40];
%[species,leaves,leaf_ind,tot_leaves] = load_leaf_paths(dirpath, num_sp);
%[leafimgs] = load_images(species,leaves,leaf_ind,tot_leaves,dirpath);
%[cleafimgs] = clean_img_data(leafimgs,tot_leaves,newsize);
%X = unroll(cleafimgs, tot_leaves);
%Xnorm = normalize(X);
%step = int8(tot_leaves/9);
%[Xtrain, ytrain, Xcv, ycv, Xtest,ytest] = sampleSets(Xnorm,step,leaf_ind,tot_leaves,num_sp);

#Load leaf paths
function [species,leaves,leaf_ind,tot_leaves] = load_leaf_paths(dirpath, num_sp)
  tempcell = readdir(dirpath)(3:end);
  for filenum = 1:length(tempcell)
    if isdir(strcat(dirpath,'/',tempcell{filenum}))
      species{end+1} = tempcell{filenum};
    endif
  endfor
  leaves = cell(num_sp,1);
  num_leaves = zeros(num_sp,1);
  leaf_ind = zeros(num_sp,1);
  fprintf('\n');
  for sp = 1:num_sp
    tempcell = readdir(strcat(dirpath,'/',species{sp}))(3:end);
    for filenum = 1:length(tempcell)
      if tempcell{filenum}(end-3:end) == '.jpg'
        leaves{sp}{end+1} = tempcell{filenum};
      endif
    endfor
    num_leaves(sp) = length(leaves{sp});
    leaf_ind(sp:end) += num_leaves(sp);
    fprintf('\rSpecies loaded: %d', sp);
    fflush(1); 
  endfor
  fprintf('\n');
  leaf_ind = [0;leaf_ind];
  tot_leaves = sum(num_leaves);
end

#Load images
function [leafimgs] = load_images(species,leaves,leaf_ind,tot_leaves,dirpath)
  leafimgs = cell(tot_leaves,1);
  sp = 1;
  fprintf('\n');
  for leaf = 1:tot_leaves
    #find species of leaf
    if leaf > leaf_ind(sp+1)
      sp += 1;
    endif
    #load leaf image
    fprintf('\rLoading leaf image %d from species %d', leaf,sp);
    fflush(1);
    leafimgs{leaf} = imread(strcat(dirpath,'/',species{sp},'/',leaves{sp}{leaf-leaf_ind(sp)}));
  endfor
  fprintf('\n');
end
  

#clean dataset - ensure all imgs have the same size, and decrease size of data
#newsize = [30,40];
function [cleafimgs] = clean_img_data(leafimgs,tot_leaves,newsize)
  cleafimgs = leafimgs;
  for k = 1:tot_leaves
    fprintf('\rProcessing leaf %d',k);
    fflush(1);
    if size(cleafimgs{k})(1) > size(cleafimgs{k})(2)
      cleafimgs{k} = permute(cleafimgs{k},[2,1,3]);
    endif
    cleafimgs{k} = imresize(cleafimgs{k}, newsize);
  endfor
  fprintf('\n');
end
  
#unroll into X
function X = unroll(leafimgs, tot_leaves)
  X = zeros(tot_leaves, 3600);
  for k = 1:tot_leaves
    X(k,:) = leafimgs{k}(:)';
  endfor
end
    
#Normalize
function Xnorm = normalize(X)
  Xnorm = X/(max(max(X))/2);
  Xnorm = Xnorm - mean(mean(Xnorm));
end
  
#step = int8(tot_leaves/9);

function [Xtrain, ytrain, Xcv, ycv, Xtest,ytest] = sampleSets(X,step,leaf_ind,tot_leaves,num_sp)
  perm = randperm(tot_leaves);
  Xrand = X(perm,:);
  Xcv = Xrand(1:step,:);
  Xtest = Xrand(step+1:2*step,:);
  Xtrain = Xrand(2*step+1:end,:);

  #Labels
  y = ones(tot_leaves,1);
  for k = 2:num_sp
    y(leaf_ind(k)+1:leaf_ind(k+1)) *= k;
  endfor
  y -= 1;
  y = y(perm);
  ycv = y(1:step);
  ytest = y(step+1:2*step);
  ytrain = y(2*step+1:end);
end

dirpath = 'leafsnap-dataset/dataset/images/field';
num_sp = 10;
newsize = [30,40];
[species,leaves,leaf_ind,tot_leaves] = load_leaf_paths(dirpath, num_sp);
[leafimgs] = load_images(species,leaves,leaf_ind,tot_leaves,dirpath);
[cleafimgs] = clean_img_data(leafimgs,tot_leaves,newsize);
X = unroll(cleafimgs, tot_leaves);
Xnorm = normalize(X);
step = int16(tot_leaves/9);
[Xtrain, ytrain, Xcv, ycv, Xtest,ytest] = sampleSets(Xnorm,step,leaf_ind,tot_leaves,num_sp);
clear dirpath num_sp newsize species leaves leaf_ind tot_leaves leafimgs cleafimgs X Xnorm step;
